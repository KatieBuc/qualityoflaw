import fitz
import pytest

from automation.src.input_process.__main__ import main
from automation.src.input_process.config import parse_input_process_config
from automation.src.input_process.extract import ExtractionManifest
from automation.src.input_process.pipeline import analyze, check_all
from automation.tests.test_input_process_quality import statute

REPEATED = "Salinan resmi peraturan daerah kabupaten contoh nomor lima"


@pytest.fixture
def config(tmp_path):
    cfg = parse_input_process_config(
        {
            "project": "demo",
            "input_process": {
                "raw_files_dir": str(tmp_path / "raw_files"),
                "cleaned_txt_dir": str(tmp_path / "preprocessed" / "cleaned_txt"),
                "log_path": str(tmp_path / "input_process.log"),
            },
            "paths": {"small_scale_stems": ["A"]},
        }
    )
    cfg.raw_files_dir.mkdir()
    cfg.cleaned_txt_dir.mkdir(parents=True)
    return cfg


def put(config, stem, fmt, text):
    """Write `cleaned_txt/<stem>.txt` the way stage 1 would have, or the way the user would.

    fmt pdf / html / txt: our own extraction (recorded in the manifest).
    fmt ocr: text the user supplied (no manifest entry).
    """
    path = config.cleaned_txt_dir / f"{stem}.txt"
    path.write_text(text, encoding="utf-8")
    manifest = ExtractionManifest(config.manifest_path)
    if fmt == "ocr":
        manifest.forget(stem)
    else:
        manifest.record(stem, fmt, path)
    manifest.save()
    return path


def image_only_pdf(config, name):
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()
    doc.save(str(config.raw_files_dir / name))
    doc.close()


def files(config):
    return sorted(p.name for p in config.cleaned_txt_dir.glob("*.txt"))


def text_of(config, stem):
    return (config.cleaned_txt_dir / f"{stem}.txt").read_text(encoding="utf-8")


def rules(entry):
    return {f["rule"]: f for f in entry["failed"]}


def log_lines(config):
    return config.log_path.read_text(encoding="utf-8").splitlines() if config.log_path.exists() else []


# --------------------------------------------------------------------------- one folder, nothing moves
def test_every_result_stays_in_cleaned_txt_whatever_the_verdict(config):
    image_only_pdf(config, "SCAN.pdf")
    put(config, "SCAN", "pdf", "")
    put(config, "BAD", "txt", "Pasal 1\npendek.")
    put(config, "GOOD", "txt", statute())

    report = check_all(config)

    assert files(config) == ["BAD.txt", "GOOD.txt", "SCAN.txt"]                     # nothing moved, nothing deleted
    assert report["summary"] == {"files": 3, "passed": 1, "needs_ocr": 1, "eliminated": 1}
    assert text_of(config, "BAD") == "Pasal 1\npendek." and text_of(config, "SCAN") == ""   # failed files untouched


def test_a_clean_text_passes_and_is_left_exactly_as_it_is(config):
    put(config, "ACEH_BIREUEN", "txt", statute())
    before = text_of(config, "ACEH_BIREUEN")

    outcome = analyze(config).outcomes["ACEH_BIREUEN"]
    report = check_all(config)

    assert report["summary"] == {"files": 1, "passed": 1, "needs_ocr": 0, "eliminated": 0}
    assert outcome.status == "PASS" and outcome.file == "ACEH_BIREUEN.txt"
    assert text_of(config, "ACEH_BIREUEN") == before


def test_a_passing_text_with_fixes_is_rewritten_in_place_and_stays_stable(config):
    chrome = "\n".join(["Beranda", "Kebijakan Privasi", "Hubungi Kami"] * 6)
    put(config, "WEB", "html", statute() + "\n" + chrome)
    outcome = analyze(config).outcomes["WEB"]

    check_all(config)

    assert outcome.status == "PASS" and "strip_boilerplate" in outcome.fixes_applied
    assert "P1_web_chrome" in outcome.findings_before and "P1_web_chrome" not in outcome.findings_after
    assert "Beranda" not in text_of(config, "WEB")
    fixed = text_of(config, "WEB")

    second = analyze(config).outcomes["WEB"]                       # still our extraction, so still judged as html
    assert second.status == "PASS" and second.source_format == "html" and second.fixes_applied == []
    check_all(config)
    assert text_of(config, "WEB") == fixed                         # the second run changes nothing


def test_a_fix_that_removes_too_much_is_discarded(config):
    put(config, "REPEAT", "txt", statute() + "\n" + "\n".join([REPEATED] * 70))  # > 30 % of the letters
    outcome = analyze(config).outcomes["REPEAT"]

    check_all(config)

    assert outcome.fix_rejected.startswith("K2_letter_loss")
    assert "strip_boilerplate" not in outcome.fixes_applied
    assert text_of(config, "REPEAT").count(REPEATED) == 70


def test_an_eliminated_file_lists_the_indicators_it_failed_with_value_and_limit(config):
    put(config, "BAD", "txt", statute() + "\n" + "�" * 400)
    put(config, "TINY", "txt", "Pasal 1\nsingkat.")

    report = check_all(config)

    assert report["summary"]["eliminated"] == 2 and report["summary"]["passed"] == 0
    bad = next(e for e in report["eliminated"] if e["stem"] == "BAD")
    assert bad["source_format"] == "txt"
    garbage = rules(bad)["C2_garbage_chars"]  # shown under its own name, not "R4_C2_..."
    assert garbage["value"] > 0.003 and garbage["limit"] == 0.003
    tiny = rules(next(e for e in report["eliminated"] if e["stem"] == "TINY"))
    assert tiny["R1_letters"]["value"] < 1000 and tiny["R1_letters"]["limit"] == 1000


# --------------------------------------------------------------------------- OCR
def test_a_scanned_pdf_needs_ocr_and_the_file_to_replace_is_its_own_name(config):
    image_only_pdf(config, "SCAN.pdf")
    put(config, "SCAN", "pdf", "")

    report = check_all(config)

    assert report["eliminated"] == [] and report["summary"]["needs_ocr"] == 1
    [entry] = report["needs_ocr"]
    assert set(entry) == {"stem", "pdf", "expected_ocr_file", "failed"}
    assert (entry["stem"], entry["pdf"], entry["expected_ocr_file"]) == ("SCAN", "SCAN.pdf", "SCAN.txt")
    scanned = rules(entry)["S1_scanned_pdf"]
    assert scanned["value"] == 1.0 and scanned["limit"] == 0.5


def test_replacing_the_file_with_ocr_text_is_checked_on_the_next_run(config):
    image_only_pdf(config, "SCAN.pdf")
    put(config, "SCAN", "pdf", "")
    assert check_all(config)["summary"]["needs_ocr"] == 1

    (config.cleaned_txt_dir / "SCAN.txt").write_text(statute(), encoding="utf-8")   # the user's OCR result
    outcome = analyze(config).outcomes["SCAN"]
    report = check_all(config)

    assert report["needs_ocr"] == [] and report["summary"]["passed"] == 1
    assert outcome.source_format == "ocr" and outcome.status == "PASS"
    assert files(config) == ["SCAN.txt"]


def test_ocr_output_that_is_still_unusable_is_eliminated_not_sent_back_to_ocr(config):
    image_only_pdf(config, "SCAN.pdf")
    put(config, "SCAN", "ocr", "Pasal 1\nteks pendek hasil OCR.")

    report = check_all(config)

    assert report["needs_ocr"] == []
    assert [e["stem"] for e in report["eliminated"]] == ["SCAN"]
    assert report["eliminated"][0]["source_format"] == "ocr"


def test_your_text_is_never_judged_as_needing_ocr_even_when_the_original_is_a_pdf(config):
    image_only_pdf(config, "SCAN.pdf")
    path = put(config, "SCAN", "pdf", "")
    path.write_text("saya ketik sendiri", encoding="utf-8")   # you saved your own text over it: no longer our extraction

    outcome = analyze(config).outcomes["SCAN"]

    assert outcome.source_format == "ocr" and outcome.status == "ELIMINATE"


def test_a_text_pdf_passes_without_ocr(config):
    doc = fitz.open()
    for chunk in (statute(8, 1), statute(8, 9)):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(30, 30, 565, 810), chunk, fontsize=5)
    doc.save(str(config.raw_files_dir / "GOOD.pdf"))
    doc.close()
    put(config, "GOOD", "pdf", statute(16))

    outcome = analyze(config).outcomes["GOOD"]

    assert outcome.status == "PASS" and outcome.metrics["pdf_pages"] == 2


# --------------------------------------------------------------------------- every run checks every text
def test_a_text_you_edit_is_judged_again_on_the_next_run(config):
    put(config, "A_POLICY", "txt", statute())
    assert check_all(config)["summary"]["passed"] == 1

    (config.cleaned_txt_dir / "A_POLICY.txt").write_text("Pasal 1\npendek.", encoding="utf-8")

    report = check_all(config)
    assert report["summary"] == {"files": 1, "passed": 0, "needs_ocr": 0, "eliminated": 1}


def test_a_second_run_changes_nothing_and_gives_the_same_counts(config):
    put(config, "GOOD", "txt", statute())
    put(config, "BAD", "txt", "Pasal 1\npendek.")
    first = check_all(config)
    before = {name: text_of(config, name[:-4]) for name in files(config)}
    second = check_all(config)

    assert first["summary"] == second["summary"] == {"files": 2, "passed": 1, "needs_ocr": 0, "eliminated": 1}
    assert before == {name: text_of(config, name[:-4]) for name in files(config)}


def test_the_log_only_run_rewrites_nothing(config):
    chrome = "\n".join(["Beranda", "Kebijakan Privasi", "Hubungi Kami"] * 6)
    put(config, "WEB", "html", statute() + "\n" + chrome)
    before = text_of(config, "WEB")

    report = check_all(config, write_cleaned=False)

    assert report["summary"]["passed"] == 1 and text_of(config, "WEB") == before
    assert any("PASS WEB" in l for l in log_lines(config))


def test_cleaned_texts_are_the_reference_corpus_for_the_unique_word_rule(config):
    """A lone OCR file is compared with the other texts in the folder, not with nothing."""
    for i in range(12):
        put(config, f"C{i}", "txt", statute())
    image_only_pdf(config, "OCRFILE.pdf")
    put(config, "OCRFILE", "ocr", statute())

    outcome = analyze(config, small_scale=False).outcomes["OCRFILE"]

    assert "corpus_unique_word_ratio" in outcome.metrics


def test_small_scale_judges_only_the_configured_stems_but_keeps_the_whole_corpus(config):
    for i in range(12):
        put(config, f"C{i}", "txt", statute())
    image_only_pdf(config, "A.pdf")
    put(config, "A", "ocr", statute())

    result = analyze(config, small_scale=True)

    assert list(result.outcomes) == ["A"]
    assert "corpus_unique_word_ratio" in result.outcomes["A"].metrics         # the other 12 texts were the corpus


def test_exact_duplicates_are_flagged_internally_but_both_kept(config):
    put(config, "A_ONE", "txt", statute())
    put(config, "B_TWO", "txt", statute())
    outcomes = analyze(config).outcomes

    check_all(config)

    assert files(config) == ["A_ONE.txt", "B_TWO.txt"]
    assert outcomes["B_TWO"].duplicate_of == "A_ONE" and "duplicate_of=A_ONE" in outcomes["B_TWO"].review
    assert outcomes["A_ONE"].duplicate_of == ""


def test_threshold_overrides_change_the_outcome(config):
    short = "Pasal 1\nBupati menetapkan peraturan daerah ini dengan sebaik-baiknya demi kepentingan bersama."
    put(config, "SHORT", "txt", short)
    assert check_all(config)["summary"]["eliminated"] == 1

    relaxed = parse_input_process_config(
        {
            "project": "demo",
            "input_process": {
                "raw_files_dir": str(config.raw_files_dir),
                "cleaned_txt_dir": str(config.cleaned_txt_dir),
                "log_path": str(config.log_path),
                "thresholds": {"min_letters": 10, "min_prose_letters": 10},
            },
        }
    )
    assert check_all(relaxed)["summary"] == {"files": 1, "passed": 1, "needs_ocr": 0, "eliminated": 0}


# --------------------------------------------------------------------------- summary
def test_summary_has_counts_and_only_the_files_that_did_not_pass(config):
    put(config, "OK", "txt", statute())
    put(config, "BAD", "txt", "Pasal 1\npendek.")
    image_only_pdf(config, "SCAN.pdf")
    put(config, "SCAN", "pdf", "")

    report = check_all(config)

    assert set(report) == {"project", "summary", "needs_ocr", "eliminated", "warnings"}
    assert report["summary"] == {"files": 3, "passed": 1, "needs_ocr": 1, "eliminated": 1}
    for entry in report["needs_ocr"] + report["eliminated"]:
        for failed in entry["failed"]:
            assert set(failed) == {"rule", "value", "limit"}


def test_a_file_that_is_not_text_is_reported_not_checked(config):
    put(config, "OK", "txt", statute())
    assert check_all(config)["warnings"] == {}
    (config.cleaned_txt_dir / "stray.csv").write_text("not text", encoding="utf-8")

    assert check_all(config)["warnings"] == {"unrecognized_files": ["stray.csv"]}


# --------------------------------------------------------------------------- the log
def test_the_log_records_only_pass_and_elimination_and_warns_for_failed_files(config):
    put(config, "OK", "txt", statute())
    put(config, "BAD", "txt", "Pasal 1\npendek.")
    image_only_pdf(config, "SCAN.pdf")
    put(config, "SCAN", "pdf", "")

    check_all(config)

    lines = log_lines(config)
    assert all(l[:4].isdigit() and l[4] == "-" for l in lines)                        # timestamped
    assert any(" INFO " in l and "PASS OK (txt)" in l for l in lines)
    eliminated = next(l for l in lines if "ELIMINATED BAD" in l)
    assert " WARNING " in eliminated and "R1_letters" in eliminated and "cleaned_txt/BAD.txt" in eliminated
    needs_ocr = next(l for l in lines if "NEEDS_OCR SCAN" in l)
    assert " WARNING " in needs_ocr and "S1_scanned_pdf" in needs_ocr and "cleaned_txt/SCAN.txt" in needs_ocr
    assert not any(" WARNING " in l and "PASS" in l for l in lines)                    # a pass is never a warning
    kinds = {l.split()[3] for l in lines}
    assert kinds <= {"PASS", "NEEDS_OCR", "ELIMINATED", "QUALITY"}                       # nothing else is recorded


def test_the_log_is_appended_to_not_rewritten(config):
    put(config, "BAD", "txt", "Pasal 1\npendek.")
    check_all(config)
    first = len(log_lines(config))
    check_all(config)
    assert len(log_lines(config)) > first
    assert sum("ELIMINATED BAD" in l for l in log_lines(config)) == 2


# --------------------------------------------------------------------------- CLI
def _cli_config(config, tmp_path):
    cfg_file = tmp_path / "pipeline.yaml"
    cfg_file.write_text(
        "project: demo\ninput_process:\n"
        f"  raw_files_dir: {config.raw_files_dir.as_posix()}\n"
        f"  cleaned_txt_dir: {config.cleaned_txt_dir.as_posix()}\n"
        f"  log_path: {config.log_path.as_posix()}\n",
        encoding="utf-8",
    )
    return str(cfg_file)


def test_cli_run_extracts_checks_and_prints_the_ocr_instructions(config, tmp_path, capsys):
    (config.raw_files_dir / "GOOD.txt").write_text(statute(), encoding="utf-8")
    (config.raw_files_dir / "BAD.txt").write_text("Pasal 1\npendek.", encoding="utf-8")
    image_only_pdf(config, "SCAN.pdf")

    code = main(["--pipeline-config", _cli_config(config, tmp_path), "run"])

    out = capsys.readouterr().out
    assert code == 0
    assert "1 passed, 1 need OCR, 1 eliminated" in out
    assert "SCAN.txt" in out
    assert files(config) == ["BAD.txt", "GOOD.txt", "SCAN.txt"]
    assert any("NEEDS_OCR SCAN" in l for l in log_lines(config))


def test_the_whole_ocr_round_trip_is_extract_overwrite_check(config, tmp_path):
    image_only_pdf(config, "SCAN.pdf")
    cfg = _cli_config(config, tmp_path)
    assert main(["--pipeline-config", cfg, "run"]) == 0
    assert files(config) == ["SCAN.txt"] and text_of(config, "SCAN") == ""

    (config.cleaned_txt_dir / "SCAN.txt").write_text(statute(), encoding="utf-8")   # the OCR result
    assert main(["--pipeline-config", cfg, "run"]) == 0                              # extract keeps it, check judges it

    assert text_of(config, "SCAN").startswith(statute()[:40])
    assert not any("NEEDS_OCR SCAN" in l for l in log_lines(config)[-3:])
    assert any("PASS SCAN (ocr)" in l for l in log_lines(config))


# --------------------------------------------------------------------------- moving data made by an earlier layout
def test_migrate_layout_gathers_every_earlier_folder_into_cleaned_txt(config, tmp_path, monkeypatch):
    from automation.src.input_process.migrate import migrate_layout

    monkeypatch.setattr("automation.src.paths.DATA_ROOT", tmp_path / "data")   # config.raw_dir is data/<project>/raw
    raw = config.raw_dir
    old_cleaned, old_extracted = raw / "cleaned_txt", raw / "extracted_txt"
    old_cleaned.mkdir(parents=True)
    old_extracted.mkdir(parents=True)
    (old_cleaned / "DONE.txt").write_text("done", encoding="utf-8")
    (old_cleaned / "HAND__txt.txt").write_text("accepted by hand", encoding="utf-8")
    for name, text in (("P__pdf.txt", "pdf text"), ("P__txt.txt", "txt text"), ("Q__ocr.txt", "ocr text"),
                       ("Q__pdf.txt", ""), ("PLAIN.txt", "plain name")):
        (old_extracted / name).write_text(text, encoding="utf-8")
    (old_extracted / "DONE.txt").write_text("a stale copy", encoding="utf-8")        # cleaned_txt already has DONE
    (raw / "input_report.json").write_text("{}", encoding="utf-8")

    result = migrate_layout(config)

    assert files(config) == ["DONE.txt", "HAND.txt", "P.txt", "PLAIN.txt", "Q.txt"] and not old_cleaned.exists()
    assert text_of(config, "DONE") == "done"                                   # the one already in cleaned_txt wins
    assert (old_extracted / "DONE.txt").exists() and "extracted_txt/DONE.txt" in result.conflicts
    assert text_of(config, "P") == "txt text" and text_of(config, "Q") == "ocr text"   # txt over pdf, yours over all
    manifest = ExtractionManifest(config.manifest_path)
    assert manifest.machine_format("P", config.cleaned_txt_dir / "P.txt") == "txt"
    assert manifest.machine_format("Q", config.cleaned_txt_dir / "Q.txt") is None
    assert result.removed_reports == ["input_report.json"]
    assert migrate_layout(config).moved_extracted == []                           # idempotent


def test_migrate_layout_dry_run_changes_nothing(config, tmp_path, monkeypatch):
    from automation.src.input_process.migrate import migrate_layout

    monkeypatch.setattr("automation.src.paths.DATA_ROOT", tmp_path / "data")
    old_extracted = config.raw_dir / "extracted_txt"
    old_extracted.mkdir(parents=True)
    (old_extracted / "P__txt.txt").write_text("x", encoding="utf-8")

    result = migrate_layout(config, dry_run=True)

    assert result.moved_extracted == ["P__txt.txt -> P.txt"]
    assert files(config) == [] and (old_extracted / "P__txt.txt").exists()


# --------------------------------------------------------------------------- manifest rebuild
def test_manifest_rebuild_registers_texts_without_a_record_and_pdf_texts_can_need_ocr_again(config):
    from automation.src.input_process.migrate import rebuild_manifest

    image_only_pdf(config, "SCAN.pdf")
    (config.cleaned_txt_dir / "SCAN.txt").write_text("", encoding="utf-8")          # no record: judged as yours
    assert analyze(config).outcomes["SCAN"].status == "ELIMINATE"

    result = rebuild_manifest(config)

    assert result.registered == ["SCAN"]
    assert analyze(config).outcomes["SCAN"].status == "NEEDS_OCR"                    # now a machine-extracted PDF text
    entry = ExtractionManifest(config.manifest_path).data["SCAN"]
    assert entry["format"] == "pdf" and entry["source"] == "SCAN.pdf" and entry["source_sha256"]


def test_manifest_rebuild_picks_the_original_extract_would_pick(config):
    from automation.src.input_process.migrate import rebuild_manifest

    image_only_pdf(config, "A.pdf")
    (config.raw_files_dir / "A.txt").write_text("Pasal 1 isi.", encoding="utf-8")
    (config.cleaned_txt_dir / "A.txt").write_text("Pasal 1 isi.", encoding="utf-8")

    rebuild_manifest(config)

    entry = ExtractionManifest(config.manifest_path).data["A"]
    assert (entry["format"], entry["source"]) == ("txt", "A.txt")                      # txt outranks pdf


def test_manifest_rebuild_leaves_edited_texts_and_texts_without_an_original_alone(config):
    from automation.src.input_process.migrate import rebuild_manifest

    (config.raw_files_dir / "EDITED.txt").write_text("Pasal 1 asli.", encoding="utf-8")
    put(config, "EDITED", "txt", "Pasal 1 asli.")
    (config.cleaned_txt_dir / "EDITED.txt").write_text("my own correction", encoding="utf-8")   # entry no longer matches
    (config.cleaned_txt_dir / "ORPHAN.txt").write_text("no original anywhere", encoding="utf-8")

    result = rebuild_manifest(config)

    assert result.edited == ["EDITED"] and result.no_original == ["ORPHAN"] and result.registered == []
    assert "ORPHAN" not in ExtractionManifest(config.manifest_path).data


def test_manifest_rebuild_completes_entries_that_lack_the_original_and_is_idempotent(config):
    from automation.src.input_process.migrate import rebuild_manifest

    (config.raw_files_dir / "A.txt").write_text("Pasal 1 isi.", encoding="utf-8")
    put(config, "A", "txt", "Pasal 1 isi.")                                            # an entry from before the hashes

    first = rebuild_manifest(config)
    second = rebuild_manifest(config)

    assert first.source_added == ["A"] and first.registered == []
    assert second.up_to_date == ["A"] and second.source_added == [] and second.registered == []


def test_manifest_rebuild_dry_run_writes_nothing(config):
    from automation.src.input_process.migrate import rebuild_manifest

    (config.raw_files_dir / "A.txt").write_text("Pasal 1 isi.", encoding="utf-8")
    (config.cleaned_txt_dir / "A.txt").write_text("Pasal 1 isi.", encoding="utf-8")

    result = rebuild_manifest(config, dry_run=True)

    assert result.registered == ["A"] and not config.manifest_path.exists()


def test_cli_manifest_rebuild(config, tmp_path, capsys):
    (config.raw_files_dir / "A.txt").write_text("Pasal 1 isi.", encoding="utf-8")
    (config.cleaned_txt_dir / "A.txt").write_text("Pasal 1 isi.", encoding="utf-8")

    assert main(["--pipeline-config", _cli_config(config, tmp_path), "manifest-rebuild"]) == 0

    assert "Registered 1 text(s)" in capsys.readouterr().out
    assert ExtractionManifest(config.manifest_path).data["A"]["source"] == "A.txt"
