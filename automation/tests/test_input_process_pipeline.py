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
                "extracted_txt_dir": str(tmp_path / "extracted_txt"),
                "cleaned_txt_dir": str(tmp_path / "preprocessed" / "cleaned_txt"),
                "log_path": str(tmp_path / "input_process.log"),
            },
            "paths": {"small_scale_stems": ["A"]},
        }
    )
    cfg.raw_files_dir.mkdir()
    cfg.extracted_txt_dir.mkdir()
    return cfg


def put(config, stem, fmt, text):
    """Write `extracted_txt/<stem>.txt` the way stage 1 would have, or the way the user would.

    fmt pdf / html / txt: our own extraction (recorded in the manifest).
    fmt ocr: text the user supplied (no manifest entry).
    """
    path = config.extracted_txt_dir / f"{stem}.txt"
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


def cleaned(config):
    return sorted(p.name for p in config.cleaned_txt_dir.glob("*.txt")) if config.cleaned_txt_dir.is_dir() else []


def extracted(config):
    return sorted(p.name for p in config.extracted_txt_dir.glob("*.txt"))


def rules(entry):
    return {f["rule"]: f for f in entry["failed"]}


def log_lines(config):
    return config.log_path.read_text(encoding="utf-8").splitlines() if config.log_path.exists() else []


# --------------------------------------------------------------------------- sorting files
def test_a_clean_text_passes_into_cleaned_txt(config):
    put(config, "ACEH_BIREUEN", "txt", statute())
    outcome = analyze(config).outcomes["ACEH_BIREUEN"]  # analyze writes nothing

    report = check_all(config)

    assert cleaned(config) == ["ACEH_BIREUEN.txt"]
    assert report["summary"] == {"files": 1, "passed": 1, "needs_ocr": 0, "eliminated": 0}
    assert "Pasal 3" in (config.cleaned_txt_dir / "ACEH_BIREUEN.txt").read_text(encoding="utf-8")
    assert outcome.extracted_file == "ACEH_BIREUEN.txt" and outcome.cleaned_file == "ACEH_BIREUEN.txt"


def test_passed_files_are_not_listed_as_problems(config):
    put(config, "GOOD", "txt", statute())
    put(config, "TINY", "txt", "Pasal 1\nsingkat.")

    report = check_all(config)

    assert "GOOD" not in str(report["eliminated"]) + str(report["needs_ocr"])
    assert [e["stem"] for e in report["eliminated"]] == ["TINY"]


def test_an_eliminated_file_lists_the_indicators_it_failed_with_value_and_limit(config):
    put(config, "BAD", "txt", statute() + "\n" + "�" * 400)
    put(config, "TINY", "txt", "Pasal 1\nsingkat.")

    report = check_all(config)

    assert cleaned(config) == []
    assert report["summary"]["eliminated"] == 2
    bad = next(e for e in report["eliminated"] if e["stem"] == "BAD")
    assert bad["source_format"] == "txt"
    garbage = rules(bad)["C2_garbage_chars"]  # shown under its own name, not "R4_C2_..."
    assert garbage["value"] > 0.003 and garbage["limit"] == 0.003
    tiny = rules(next(e for e in report["eliminated"] if e["stem"] == "TINY"))
    assert tiny["R1_letters"]["value"] < 1000 and tiny["R1_letters"]["limit"] == 1000


def test_a_scanned_pdf_needs_ocr_and_the_file_to_replace_is_its_own_name(config):
    image_only_pdf(config, "SCAN.pdf")
    put(config, "SCAN", "pdf", "")

    report = check_all(config)

    assert cleaned(config) == [] and report["eliminated"] == []
    assert report["summary"]["needs_ocr"] == 1
    [entry] = report["needs_ocr"]
    assert set(entry) == {"stem", "pdf", "expected_ocr_file", "failed"}
    assert (entry["stem"], entry["pdf"], entry["expected_ocr_file"]) == ("SCAN", "SCAN.pdf", "SCAN.txt")
    scanned = rules(entry)["S1_scanned_pdf"]
    assert scanned["value"] == 1.0 and scanned["limit"] == 0.5


def test_replacing_the_file_with_ocr_text_is_checked_on_the_next_run(config):
    image_only_pdf(config, "SCAN.pdf")
    put(config, "SCAN", "pdf", "")
    assert check_all(config)["summary"]["needs_ocr"] == 1
    assert extracted(config) == ["SCAN.txt"]

    put(config, "SCAN", "ocr", statute())  # the user's OCR result replaces the empty extraction
    outcome = analyze(config).outcomes["SCAN"]
    report = check_all(config)

    assert report["needs_ocr"] == [] and cleaned(config) == ["SCAN.txt"]
    assert outcome.source_format == "ocr" and outcome.status == "PASS"
    assert extracted(config) == []


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


# --------------------------------------------------------------------------- fix + re-check
def test_boilerplate_is_fixed_and_the_text_is_measured_again(config):
    chrome = "\n".join(["Beranda", "Kebijakan Privasi", "Hubungi Kami"] * 6)
    put(config, "WEB", "html", statute() + "\n" + chrome)
    outcome = analyze(config).outcomes["WEB"]

    check_all(config)

    assert outcome.status == "PASS" and "strip_boilerplate" in outcome.fixes_applied
    assert "P1_web_chrome" in outcome.findings_before
    assert "P1_web_chrome" not in outcome.findings_after  # re-checked after fixing
    assert "Beranda" not in (config.cleaned_txt_dir / "WEB.txt").read_text(encoding="utf-8")


def test_a_fix_that_removes_too_much_is_discarded(config):
    put(config, "REPEAT", "txt", statute() + "\n" + "\n".join([REPEATED] * 70))  # > 30 % of the letters
    outcome = analyze(config).outcomes["REPEAT"]

    check_all(config)

    assert outcome.fix_rejected.startswith("K2_letter_loss")
    assert "strip_boilerplate" not in outcome.fixes_applied
    assert (config.cleaned_txt_dir / "REPEAT.txt").read_text(encoding="utf-8").count(REPEATED) == 70


def test_already_clean_text_needs_no_fix(config):
    put(config, "OK", "txt", statute())
    assert analyze(config).outcomes["OK"].fixes_applied == []


# --------------------------------------------------------------------------- the order: cleaned, extracted, original
def test_cleaned_txt_is_never_pruned(config):
    config.cleaned_txt_dir.mkdir(parents=True)
    (config.cleaned_txt_dir / "EARLIER.txt").write_text("from an earlier run", encoding="utf-8")
    put(config, "FINE", "txt", statute())

    check_all(config)

    assert cleaned(config) == ["EARLIER.txt", "FINE.txt"]
    assert (config.cleaned_txt_dir / "EARLIER.txt").read_text(encoding="utf-8") == "from an earlier run"


def test_a_policy_already_in_cleaned_txt_is_skipped_even_if_extracted_txt_has_it(config):
    config.cleaned_txt_dir.mkdir(parents=True)
    (config.cleaned_txt_dir / "DONE.txt").write_text("already cleaned", encoding="utf-8")
    put(config, "DONE", "ocr", "Pasal 1\npendek.")  # would be eliminated if it were checked

    report = check_all(config)

    assert report["summary"] == {"files": 1, "passed": 1, "needs_ocr": 0, "eliminated": 0}
    assert report["warnings"]["already_in_cleaned_txt"] == ["DONE.txt"]
    assert extracted(config) == ["DONE.txt"]                  # never deleted behind the user's back
    assert any("IGNORED extracted_txt/DONE.txt" in l and " WARNING " in l for l in log_lines(config))


def test_moving_a_file_by_hand_into_cleaned_txt_accepts_it_as_it_is(config):
    """The per-file exception: drag an eliminated file from extracted_txt to cleaned_txt. Same name, nothing to rename."""
    put(config, "BAD", "txt", "Pasal 1\npendek.")
    assert check_all(config)["summary"]["eliminated"] == 1

    config.cleaned_txt_dir.mkdir(parents=True, exist_ok=True)
    (config.extracted_txt_dir / "BAD.txt").rename(config.cleaned_txt_dir / "BAD.txt")  # the user's move
    report = check_all(config)

    assert cleaned(config) == ["BAD.txt"] and extracted(config) == []
    assert (config.cleaned_txt_dir / "BAD.txt").read_text(encoding="utf-8") == "Pasal 1\npendek."   # untouched
    assert report["eliminated"] == [] and report["summary"] == {"files": 1, "passed": 1, "needs_ocr": 0, "eliminated": 0}


def test_files_that_pass_leave_extracted_txt_and_only_unresolved_ones_stay(config):
    image_only_pdf(config, "SCAN.pdf")
    put(config, "SCAN", "pdf", "")
    put(config, "BAD", "txt", "Pasal 1\npendek.")
    put(config, "GOOD", "txt", statute())

    check_all(config)

    assert cleaned(config) == ["GOOD.txt"]
    assert extracted(config) == ["BAD.txt", "SCAN.txt"]
    assert "GOOD" not in ExtractionManifest(config.manifest_path).data       # a passed file's entry is dropped


def test_a_second_run_changes_nothing_and_counts_every_policy(config):
    put(config, "GOOD", "txt", statute())
    put(config, "BAD", "txt", "Pasal 1\npendek.")
    first = check_all(config)
    second = check_all(config)

    assert first["summary"] == second["summary"] == {"files": 2, "passed": 1, "needs_ocr": 0, "eliminated": 1}
    assert cleaned(config) == ["GOOD.txt"] and extracted(config) == ["BAD.txt"]


def test_the_log_only_run_does_not_move_or_delete_anything(config):
    put(config, "GOOD", "txt", statute())

    report = check_all(config, write_cleaned=False)

    assert report["summary"]["passed"] == 1                     # what would be in cleaned_txt
    assert extracted(config) == ["GOOD.txt"] and not config.cleaned_txt_dir.exists()
    assert any("PASS GOOD" in l for l in log_lines(config))


def test_cleaned_texts_are_the_reference_corpus_for_the_unique_word_rule(config):
    """A lone OCR file is compared with the cleaned corpus, not with nothing."""
    config.cleaned_txt_dir.mkdir(parents=True)
    for i in range(12):
        (config.cleaned_txt_dir / f"C{i}.txt").write_text(statute(), encoding="utf-8")
    image_only_pdf(config, "OCRFILE.pdf")
    put(config, "OCRFILE", "ocr", statute())

    outcome = analyze(config).outcomes["OCRFILE"]

    assert "corpus_unique_word_ratio" in outcome.metrics


def test_exact_duplicates_are_flagged_internally_but_both_kept(config):
    put(config, "A_ONE", "txt", statute())
    put(config, "B_TWO", "txt", statute())
    outcomes = analyze(config).outcomes

    check_all(config)

    assert cleaned(config) == ["A_ONE.txt", "B_TWO.txt"]
    assert outcomes["B_TWO"].duplicate_of == "A_ONE" and "duplicate_of=A_ONE" in outcomes["B_TWO"].review
    assert outcomes["A_ONE"].duplicate_of == ""


def test_small_scale_checks_only_the_configured_stems(config):
    put(config, "A", "txt", statute())
    put(config, "B", "txt", statute())

    report = check_all(config, small_scale=True)

    assert report["summary"]["files"] == 1 and cleaned(config) == ["A.txt"]


def test_threshold_overrides_change_the_outcome(config):
    short = "Pasal 1\nBupati menetapkan peraturan daerah ini dengan sebaik-baiknya demi kepentingan bersama."
    put(config, "SHORT", "txt", short)
    assert check_all(config)["summary"]["eliminated"] == 1

    relaxed = parse_input_process_config(
        {
            "project": "demo",
            "input_process": {
                "raw_files_dir": str(config.raw_files_dir),
                "extracted_txt_dir": str(config.extracted_txt_dir),
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


def test_warnings_appear_only_when_there_is_something_to_say(config):
    put(config, "OK", "txt", statute())
    assert check_all(config)["warnings"] == {}

    (config.raw_files_dir / "MISSING.pdf").write_bytes(b"%PDF")
    (config.extracted_txt_dir / "stray.csv").write_text("not text", encoding="utf-8")

    report = check_all(config)

    assert report["warnings"] == {"not_extracted": ["MISSING.pdf"], "unrecognized_files": ["stray.csv"]}


# --------------------------------------------------------------------------- the log
def test_the_log_records_time_file_and_result_and_warns_for_failed_files(config):
    put(config, "OK", "txt", statute())
    put(config, "BAD", "txt", "Pasal 1\npendek.")
    image_only_pdf(config, "SCAN.pdf")
    put(config, "SCAN", "pdf", "")

    check_all(config)

    lines = log_lines(config)
    assert all(l[:4].isdigit() and l[4] == "-" for l in lines)                        # timestamped
    assert any(" INFO " in l and "PASS OK (txt) -> cleaned_txt/OK.txt" in l for l in lines)
    eliminated = next(l for l in lines if "ELIMINATED BAD" in l)
    assert " WARNING " in eliminated and "R1_letters" in eliminated
    needs_ocr = next(l for l in lines if "NEEDS_OCR SCAN" in l)
    assert " WARNING " in needs_ocr and "S1_scanned_pdf" in needs_ocr and "extracted_txt/SCAN.txt" in needs_ocr
    assert not any(" WARNING " in l and "PASS" in l for l in lines)                    # a pass is never a warning


def test_the_log_is_appended_to_not_rewritten(config):
    put(config, "BAD", "txt", "Pasal 1\npendek.")
    check_all(config)
    first = len(log_lines(config))
    check_all(config)
    assert len(log_lines(config)) > first
    assert sum("QUALITY CHECK started" in l for l in log_lines(config)) == 2


# --------------------------------------------------------------------------- CLI
def test_cli_run_extracts_checks_and_prints_the_ocr_instructions(config, tmp_path, capsys):
    (config.raw_files_dir / "GOOD.txt").write_text(statute(), encoding="utf-8")
    (config.raw_files_dir / "BAD.txt").write_text("Pasal 1\npendek.", encoding="utf-8")
    image_only_pdf(config, "SCAN.pdf")
    cfg_file = tmp_path / "pipeline.yaml"
    cfg_file.write_text(
        "project: demo\ninput_process:\n"
        f"  raw_files_dir: {config.raw_files_dir.as_posix()}\n"
        f"  extracted_txt_dir: {config.extracted_txt_dir.as_posix()}\n"
        f"  cleaned_txt_dir: {config.cleaned_txt_dir.as_posix()}\n"
        f"  log_path: {config.log_path.as_posix()}\n",
        encoding="utf-8",
    )

    code = main(["--pipeline-config", str(cfg_file), "run"])

    out = capsys.readouterr().out
    assert code == 0
    assert "1 in cleaned_txt, 1 need OCR, 1 eliminated" in out
    assert "SCAN.txt" in out
    assert cleaned(config) == ["GOOD.txt"] and extracted(config) == ["BAD.txt", "SCAN.txt"]
    assert any("NEEDS_OCR SCAN" in l for l in log_lines(config))


def test_the_flow_extract_then_replace_then_check(config, tmp_path):
    """The whole OCR round trip with nothing but file names."""
    image_only_pdf(config, "SCAN.pdf")
    cfg_file = tmp_path / "pipeline.yaml"
    cfg_file.write_text(
        "project: demo\ninput_process:\n"
        f"  raw_files_dir: {config.raw_files_dir.as_posix()}\n"
        f"  extracted_txt_dir: {config.extracted_txt_dir.as_posix()}\n"
        f"  cleaned_txt_dir: {config.cleaned_txt_dir.as_posix()}\n"
        f"  log_path: {config.log_path.as_posix()}\n",
        encoding="utf-8",
    )
    assert main(["--pipeline-config", str(cfg_file), "run"]) == 0
    assert extracted(config) == ["SCAN.txt"] and cleaned(config) == []

    (config.extracted_txt_dir / "SCAN.txt").write_text(statute(), encoding="utf-8")   # the OCR result
    assert main(["--pipeline-config", str(cfg_file), "run"]) == 0

    assert cleaned(config) == ["SCAN.txt"] and extracted(config) == []


# --------------------------------------------------------------------------- tidy
def test_tidy_removes_extracted_files_of_policies_already_in_cleaned_txt(config):
    from automation.src.input_process.pipeline import tidy_extracted

    config.cleaned_txt_dir.mkdir(parents=True)
    (config.cleaned_txt_dir / "DONE.txt").write_text("x", encoding="utf-8")
    put(config, "DONE", "txt", "x")
    put(config, "OPEN", "txt", "x")

    assert tidy_extracted(config, dry_run=True) == ["DONE.txt"]
    assert extracted(config) == ["DONE.txt", "OPEN.txt"]                    # dry run

    assert tidy_extracted(config) == ["DONE.txt"]
    assert extracted(config) == ["OPEN.txt"]


# --------------------------------------------------------------------------- moving data made by the earlier layout
def test_migrate_layout_moves_cleaned_txt_renames_markers_and_records_our_extractions(config, tmp_path, monkeypatch):
    from automation.src.input_process.migrate import migrate_layout

    monkeypatch.setattr("automation.src.paths.DATA_ROOT", tmp_path / "data")   # config.raw_dir is data/<project>/raw
    raw = config.raw_dir
    old_cleaned = raw / "cleaned_txt"
    old_cleaned.mkdir(parents=True)
    (old_cleaned / "DONE.txt").write_text("done", encoding="utf-8")
    (old_cleaned / "HAND__txt.txt").write_text("accepted by hand", encoding="utf-8")
    for name, text in (("P__pdf.txt", "pdf text"), ("P__txt.txt", "txt text"), ("Q__ocr.txt", "ocr text"), ("Q__pdf.txt", "")):
        (config.extracted_txt_dir / name).write_text(text, encoding="utf-8")
    (raw / "input_report.json").write_text("{}", encoding="utf-8")

    result = migrate_layout(config)

    assert cleaned(config) == ["DONE.txt", "HAND.txt"] and not old_cleaned.exists()
    assert extracted(config) == ["P.txt", "Q.txt"]
    assert (config.extracted_txt_dir / "P.txt").read_text(encoding="utf-8") == "txt text"       # txt outranks pdf
    assert (config.extracted_txt_dir / "Q.txt").read_text(encoding="utf-8") == "ocr text"       # yours outranks all
    manifest = ExtractionManifest(config.manifest_path)
    assert manifest.machine_format("P", config.extracted_txt_dir / "P.txt") == "txt"
    assert manifest.machine_format("Q", config.extracted_txt_dir / "Q.txt") is None
    assert result.removed_reports == ["input_report.json"] and not (raw / "input_report.json").exists()
    assert migrate_layout(config).renamed == []                                                 # idempotent


def test_migrate_layout_dry_run_changes_nothing(config, tmp_path, monkeypatch):
    from automation.src.input_process.migrate import migrate_layout

    monkeypatch.setattr("automation.src.paths.DATA_ROOT", tmp_path / "data")

    (config.extracted_txt_dir / "P__txt.txt").write_text("x", encoding="utf-8")

    result = migrate_layout(config, dry_run=True)

    assert result.renamed == ["P__txt.txt -> P.txt"] and extracted(config) == ["P__txt.txt"]
