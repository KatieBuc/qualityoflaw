import json

import fitz
import pytest

from automation.src.input_process.__main__ import main
from automation.src.input_process.config import parse_input_process_config
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
                "cleaned_txt_dir": str(tmp_path / "cleaned_txt"),
                "report_path": str(tmp_path / "input_report.json"),
            },
            "paths": {"small_scale_stems": ["A"]},
        }
    )
    cfg.raw_files_dir.mkdir()
    cfg.extracted_txt_dir.mkdir()
    return cfg


def put(config, stem, fmt, text):
    path = config.extracted_txt_dir / f"{stem}__{fmt}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def image_only_pdf(config, name):
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()
    doc.save(str(config.raw_files_dir / name))
    doc.close()


def cleaned(config):
    return sorted(p.name for p in config.cleaned_txt_dir.glob("*.txt")) if config.cleaned_txt_dir.is_dir() else []


def rules(entry):
    return {f["rule"]: f for f in entry["failed"]}


# --------------------------------------------------------------------------- sorting files
def test_a_clean_text_passes_into_cleaned_txt_without_its_marker(config):
    put(config, "ACEH_BIREUEN", "txt", statute())

    report = check_all(config)

    assert cleaned(config) == ["ACEH_BIREUEN.txt"]
    assert report["summary"] == {"files": 1, "passed": 1, "needs_ocr": 0, "eliminated": 0}
    assert "Pasal 3" in (config.cleaned_txt_dir / "ACEH_BIREUEN.txt").read_text(encoding="utf-8")
    outcome = analyze(config).outcomes["ACEH_BIREUEN"]
    assert outcome.extracted_file == "ACEH_BIREUEN__txt.txt" and outcome.cleaned_file == "ACEH_BIREUEN.txt"


def test_passed_files_are_not_listed_in_the_report(config):
    put(config, "GOOD", "txt", statute())
    put(config, "TINY", "txt", "Pasal 1\nsingkat.")

    report = check_all(config)

    assert "GOOD" not in json.dumps(report)
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


def test_a_scanned_pdf_is_listed_as_needing_ocr_with_the_file_name_to_supply(config):
    image_only_pdf(config, "SCAN.pdf")
    put(config, "SCAN", "pdf", "")

    report = check_all(config)

    assert cleaned(config) == [] and report["eliminated"] == []
    assert report["summary"]["needs_ocr"] == 1
    [entry] = report["needs_ocr"]
    assert set(entry) == {"stem", "pdf", "expected_ocr_file", "failed"}
    assert (entry["stem"], entry["pdf"], entry["expected_ocr_file"]) == ("SCAN", "SCAN.pdf", "SCAN__ocr.txt")
    scanned = rules(entry)["S1_scanned_pdf"]
    assert scanned["value"] == 1.0 and scanned["limit"] == 0.5


def test_ocr_round_trip_supersedes_the_pdf_text_on_the_next_run(config):
    image_only_pdf(config, "SCAN.pdf")
    put(config, "SCAN", "pdf", "")
    assert check_all(config)["summary"]["needs_ocr"] == 1

    put(config, "SCAN", "ocr", statute())  # the user's OCR result
    report = check_all(config)

    assert report["needs_ocr"] == [] and cleaned(config) == ["SCAN.txt"]
    outcome = analyze(config).outcomes["SCAN"]
    assert outcome.source_format == "ocr" and outcome.status == "PASS"
    assert outcome.superseded == ["SCAN__pdf.txt"]


def test_ocr_output_that_is_still_unusable_is_eliminated_not_sent_back_to_ocr(config):
    image_only_pdf(config, "SCAN.pdf")
    put(config, "SCAN", "pdf", "")
    put(config, "SCAN", "ocr", "Pasal 1\nteks pendek hasil OCR.")

    report = check_all(config)

    assert report["needs_ocr"] == []
    assert [e["stem"] for e in report["eliminated"]] == ["SCAN"]
    assert report["eliminated"][0]["source_format"] == "ocr"


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

    check_all(config)
    outcome = analyze(config).outcomes["WEB"]

    assert outcome.status == "PASS" and "strip_boilerplate" in outcome.fixes_applied
    assert "P1_web_chrome" in outcome.findings_before
    assert "P1_web_chrome" not in outcome.findings_after  # re-checked after fixing
    assert "Beranda" not in (config.cleaned_txt_dir / "WEB.txt").read_text(encoding="utf-8")


def test_a_fix_that_removes_too_much_is_discarded(config):
    put(config, "REPEAT", "txt", statute() + "\n" + "\n".join([REPEATED] * 70))  # > 30 % of the letters

    check_all(config)
    outcome = analyze(config).outcomes["REPEAT"]

    assert outcome.fix_rejected.startswith("K2_letter_loss")
    assert "strip_boilerplate" not in outcome.fixes_applied
    assert (config.cleaned_txt_dir / "REPEAT.txt").read_text(encoding="utf-8").count(REPEATED) == 70


def test_already_clean_text_needs_no_fix(config):
    put(config, "OK", "txt", statute())
    assert analyze(config).outcomes["OK"].fixes_applied == []


# --------------------------------------------------------------------------- re-run behaviour
def test_cleaned_txt_mirrors_the_latest_result(config):
    config.cleaned_txt_dir.mkdir()
    (config.cleaned_txt_dir / "GONE.txt").write_text("old", encoding="utf-8")
    (config.cleaned_txt_dir / "NOW_BAD.txt").write_text("previously fine", encoding="utf-8")
    put(config, "NOW_BAD", "txt", "Pasal 1\nterlalu pendek.")
    put(config, "FINE", "txt", statute())

    check_all(config)

    assert cleaned(config) == ["FINE.txt"]


def test_report_only_leaves_cleaned_txt_alone_but_writes_the_report(config):
    put(config, "FINE", "txt", statute())

    report = check_all(config, write_cleaned=False)

    assert not config.cleaned_txt_dir.exists()
    assert json.loads(config.report_path.read_text(encoding="utf-8"))["summary"] == report["summary"]


def test_precedence_prefers_ocr_then_txt_then_html_then_pdf(config):
    put(config, "P", "pdf", statute())
    assert analyze(config).outcomes["P"].source_format == "pdf"
    put(config, "P", "html", statute())
    assert analyze(config).outcomes["P"].source_format == "html"
    put(config, "P", "txt", statute())
    assert analyze(config).outcomes["P"].source_format == "txt"
    put(config, "P", "ocr", statute())
    outcome = analyze(config).outcomes["P"]
    assert outcome.source_format == "ocr" and outcome.superseded == ["P__html.txt", "P__pdf.txt", "P__txt.txt"]


def test_a_supplied_txt_is_used_when_the_pdf_of_the_same_policy_is_scanned(config):
    """The real case in data/indonesia: 10 scanned PDFs already have a good .txt."""
    image_only_pdf(config, "BOTH.pdf")
    put(config, "BOTH", "pdf", "")
    put(config, "BOTH", "txt", statute())

    report = check_all(config)

    assert report["needs_ocr"] == [] and cleaned(config) == ["BOTH.txt"]
    outcome = analyze(config).outcomes["BOTH"]
    assert outcome.source_format == "txt" and outcome.superseded == ["BOTH__pdf.txt"]


def test_exact_duplicates_are_flagged_internally_but_both_kept(config):
    put(config, "A_ONE", "txt", statute())
    put(config, "B_TWO", "txt", statute())

    check_all(config)
    outcomes = analyze(config).outcomes

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
                "report_path": str(config.report_path),
                "thresholds": {"min_letters": 10, "min_prose_letters": 10},
            },
        }
    )
    assert check_all(relaxed)["summary"] == {"files": 1, "passed": 1, "needs_ocr": 0, "eliminated": 0}


# --------------------------------------------------------------------------- report
def test_report_has_only_counts_and_the_files_that_did_not_pass(config):
    put(config, "OK", "txt", statute())
    put(config, "BAD", "txt", "Pasal 1\npendek.")
    image_only_pdf(config, "SCAN.pdf")
    put(config, "SCAN", "pdf", "")

    report = check_all(config)

    assert set(report) == {"generated_at", "project", "summary", "needs_ocr", "eliminated"}
    assert report["project"] == "demo"
    assert set(report["summary"]) == {"files", "passed", "needs_ocr", "eliminated"}
    assert report["summary"] == {"files": 3, "passed": 1, "needs_ocr": 1, "eliminated": 1}
    for entry in report["needs_ocr"] + report["eliminated"]:
        for failed in entry["failed"]:
            assert set(failed) == {"rule", "value", "limit"}
    json.dumps(report)  # fully serialisable


def test_warnings_appear_only_when_there_is_something_to_say(config):
    put(config, "OK", "txt", statute())
    assert "warnings" not in check_all(config)

    (config.raw_files_dir / "MISSING.pdf").write_bytes(b"%PDF")
    (config.extracted_txt_dir / "stray.txt").write_text("no marker", encoding="utf-8")

    report = check_all(config)

    assert report["warnings"] == {"not_extracted": ["MISSING.pdf"], "unrecognized_files": ["stray.txt"]}


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
        f"  report_path: {config.report_path.as_posix()}\n",
        encoding="utf-8",
    )

    code = main(["--pipeline-config", str(cfg_file), "run"])

    out = capsys.readouterr().out
    assert code == 0
    assert "1 passed, 1 need OCR, 1 eliminated" in out
    assert "SCAN__ocr.txt" in out
    assert cleaned(config) == ["GOOD.txt"]
    assert json.loads(config.report_path.read_text(encoding="utf-8"))["needs_ocr"][0]["pdf"] == "SCAN.pdf"
