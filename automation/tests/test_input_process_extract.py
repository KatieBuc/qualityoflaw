import fitz  # PyMuPDF, used here only to build small PDF fixtures
import pytest

from automation.src.input_process.__main__ import main
from automation.src.input_process.config import parse_input_process_config
from automation.src.input_process.extract import (
    extract_all,
    ExtractionManifest,
    heading_set,
)

PASAL = "Pasal 1\nBupati melaksanakan sistem informasi perlindungan perempuan dan anak.\n"


def make_pdf(path, pages):
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(fitz.Rect(50, 50, 550, 780), text, fontsize=11)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def config(tmp_path):
    cfg = parse_input_process_config(
        {
            "project": "demo",
            "input_process": {
                "raw_files_dir": str(tmp_path / "raw_files"),
                "extracted_txt_dir": str(tmp_path / "extracted_txt"),
                "cleaned_txt_dir": str(tmp_path / "preprocessed" / "cleaned_txt"),
                "log_path": str(tmp_path / "raw" / "input_process.log"),
            },
            "paths": {"small_scale_stems": ["B"]},
        }
    )
    cfg.raw_files_dir.mkdir()
    return cfg


def test_txt_is_copied_as_utf8_under_its_own_name(config):
    (config.raw_files_dir / "A.txt").write_bytes("﻿Pasal 1 é\r\nisi".encode("utf-8"))
    (config.raw_files_dir / "L.txt").write_bytes("caf\xe9 hukum".encode("cp1252"))

    results = {r.source: r for r in extract_all(config)}

    out = config.extracted_txt_dir
    assert results["A.txt"].output == "A.txt"
    # BOM gone, CRLF kept (compared as bytes: read_text would translate the newline).
    assert (out / "A.txt").read_bytes() == "Pasal 1 é\r\nisi\n".encode("utf-8")
    assert (out / "L.txt").read_text(encoding="utf-8").startswith("café hukum")
    assert "cp1252" in results["L.txt"].detail


def test_pdf_text_is_extracted_per_page(config):
    make_pdf(config.raw_files_dir / "P.pdf", ["Pasal 1 isi satu.", "Pasal 2 isi dua."])
    make_pdf(config.raw_files_dir / "U.PDF", ["Pasal 9 huruf besar."])  # upper-case extension

    results = {r.source: r for r in extract_all(config)}

    text = (config.extracted_txt_dir / "P.txt").read_text(encoding="utf-8")
    assert "Pasal 1 isi satu." in text and "Pasal 2 isi dua." in text
    assert text.index("Pasal 1") < text.index("Pasal 2")
    assert results["P.pdf"].detail == "2 page(s)"
    assert (config.extracted_txt_dir / "U.txt").exists()


def test_image_only_pdf_is_written_empty_for_the_quality_check_to_judge(config):
    make_pdf(config.raw_files_dir / "S.pdf", ["", ""])  # no text layer

    results = extract_all(config)

    assert results[0].status == "extracted"
    assert (config.extracted_txt_dir / "S.txt").read_text(encoding="utf-8") == ""


def test_html_is_extracted_and_scripts_are_dropped(config):
    (config.raw_files_dir / "H.html").write_text(
        "<html><head><script>var x=1;</script></head><body><h1>Peraturan Daerah</h1>"
        "<p>Pasal 1</p><p>Bupati melaksanakan perlindungan perempuan dan anak korban kekerasan.</p>"
        "<p>Pasal 2</p><p>Setiap orang wajib melapor.</p></body></html>",
        encoding="utf-8",
    )
    (config.raw_files_dir / "H2.htm").write_text("<p>Pasal 1</p><p>isi</p>", encoding="utf-8")

    extract_all(config)

    text = (config.extracted_txt_dir / "H.txt").read_text(encoding="utf-8")
    assert "Pasal 1" in text and "Pasal 2" in text and "var x" not in text
    assert (config.extracted_txt_dir / "H2.txt").exists()


def test_html_falls_back_to_bs4_when_trafilatura_drops_headings(config, monkeypatch):
    from automation.src.input_process import extract as ex

    body = "".join(f"<p>Pasal {i}</p><p>isi pasal {i} lengkap.</p>" for i in range(1, 8))
    (config.raw_files_dir / "H.html").write_text(f"<body>{body}</body>", encoding="utf-8")
    monkeypatch.setattr(ex, "traf_extract", lambda markup: "isi pasal 1 lengkap.")  # lost every heading

    extract_all(config)

    text = (config.extracted_txt_dir / "H.txt").read_text(encoding="utf-8")
    assert "Pasal 7" in text


def test_existing_output_is_never_overwritten(config):
    (config.raw_files_dir / "A.txt").write_text("new", encoding="utf-8")
    config.extracted_txt_dir.mkdir()
    (config.extracted_txt_dir / "A.txt").write_text("hand-fixed", encoding="utf-8")

    results = extract_all(config)

    assert results[0].status == "skipped_existing"
    assert (config.extracted_txt_dir / "A.txt").read_text(encoding="utf-8") == "hand-fixed"


def test_corrupt_pdf_is_reported_and_the_rest_continue(config):
    (config.raw_files_dir / "BAD.pdf").write_bytes(b"not a pdf at all")
    (config.raw_files_dir / "OK.txt").write_text(PASAL, encoding="utf-8")
    (config.raw_files_dir / "X.docx").write_bytes(b"x")

    results = {r.source: r for r in extract_all(config)}

    assert results["BAD.pdf"].status == "failed" and results["BAD.pdf"].detail
    assert results["OK.txt"].status == "extracted"
    assert results["X.docx"].status == "unsupported"
    assert not (config.extracted_txt_dir / "BAD.txt").exists()


def test_same_stem_in_several_formats_is_extracted_once_txt_first(config):
    make_pdf(config.raw_files_dir / "A.pdf", ["Pasal 1 dari pdf."])
    (config.raw_files_dir / "A.html").write_text("<p>Pasal 1 dari html.</p>", encoding="utf-8")
    (config.raw_files_dir / "A.txt").write_text("Pasal 1 dari txt.", encoding="utf-8")
    make_pdf(config.raw_files_dir / "B.pdf", ["Pasal 1 dari pdf."])
    (config.raw_files_dir / "B.html").write_text("<p>Pasal 1 dari html.</p>", encoding="utf-8")

    results = extract_all(config)

    assert sorted(p.name for p in config.extracted_txt_dir.iterdir()) == ["A.txt", "B.txt"]
    assert "dari txt" in (config.extracted_txt_dir / "A.txt").read_text(encoding="utf-8")
    assert "dari html" in (config.extracted_txt_dir / "B.txt").read_text(encoding="utf-8")
    assert sorted(r.source for r in results if r.status == "skipped_duplicate") == ["A.html", "A.pdf", "B.pdf"]


def test_a_policy_in_cleaned_txt_is_not_extracted_again(config):
    (config.raw_files_dir / "A.txt").write_text(PASAL, encoding="utf-8")
    config.cleaned_txt_dir.mkdir(parents=True)
    (config.cleaned_txt_dir / "A.txt").write_text("already done", encoding="utf-8")

    results = extract_all(config)

    assert results[0].status == "skipped_done" and not config.extracted_txt_dir.exists()


def test_the_order_is_cleaned_then_extracted_then_the_original(config):
    for name in ("DONE", "WAITING", "NEW"):
        (config.raw_files_dir / f"{name}.txt").write_text(PASAL, encoding="utf-8")
    config.cleaned_txt_dir.mkdir(parents=True)
    (config.cleaned_txt_dir / "DONE.txt").write_text("done", encoding="utf-8")
    config.extracted_txt_dir.mkdir()
    (config.extracted_txt_dir / "WAITING.txt").write_text("your fix", encoding="utf-8")

    status = {r.source: r.status for r in extract_all(config)}

    assert status == {"DONE.txt": "skipped_done", "WAITING.txt": "skipped_existing", "NEW.txt": "extracted"}
    assert (config.extracted_txt_dir / "WAITING.txt").read_text(encoding="utf-8") == "your fix"


def test_the_manifest_tells_our_extraction_from_a_file_you_replaced(config):
    (config.raw_files_dir / "A.txt").write_text(PASAL, encoding="utf-8")
    extract_all(config)
    target = config.extracted_txt_dir / "A.txt"

    manifest = ExtractionManifest(config.manifest_path)
    assert manifest.machine_format("A", target) == "txt"

    target.write_text("OCR text you pasted", encoding="utf-8")
    assert ExtractionManifest(config.manifest_path).machine_format("A", target) is None


def test_extraction_is_logged_with_a_warning_for_a_failure(config):
    (config.raw_files_dir / "BAD.pdf").write_bytes(b"not a pdf")
    (config.raw_files_dir / "OK.txt").write_text(PASAL, encoding="utf-8")

    extract_all(config)

    lines = config.log_path.read_text(encoding="utf-8").splitlines()
    assert any(" INFO " in l and "EXTRACTED OK.txt" in l for l in lines)
    assert any(" WARNING " in l and "EXTRACTION FAILED BAD.pdf" in l for l in lines)
    assert lines[0][:4].isdigit()          # every line starts with a timestamp


def test_small_scale_limits_to_the_configured_stems(config):
    for name in ("A.txt", "B.txt"):
        (config.raw_files_dir / name).write_text(PASAL, encoding="utf-8")

    results = extract_all(config, small_scale=True)

    assert [r.source for r in results] == ["B.txt"]


def test_missing_raw_files_folder_explains_migrate_raw(tmp_path):
    cfg = parse_input_process_config(
        {"input_process": {"raw_files_dir": str(tmp_path / "nope")}}
    )
    with pytest.raises(FileNotFoundError, match="migrate-raw"):
        extract_all(cfg)


def test_heading_set_knows_indonesian_headings():
    assert heading_set("Pasal 12 dan BAB III serta Article 4") == {"pasal12", "babiii", "article4"}


def test_cli_extract_reports_counts_and_fails_on_a_failed_file(config, tmp_path, capsys):
    (config.raw_files_dir / "BAD.pdf").write_bytes(b"nope")
    (config.raw_files_dir / "OK.txt").write_text(PASAL, encoding="utf-8")
    cfg_file = tmp_path / "pipeline.yaml"
    cfg_file.write_text(
        f"project: demo\ninput_process:\n  raw_files_dir: {config.raw_files_dir.as_posix()}\n"
        f"  extracted_txt_dir: {config.extracted_txt_dir.as_posix()}\n"
        f"  cleaned_txt_dir: {config.cleaned_txt_dir.as_posix()}\n"
        f"  log_path: {config.log_path.as_posix()}\n",
        encoding="utf-8",
    )

    code = main(["--pipeline-config", str(cfg_file), "extract"])

    captured = capsys.readouterr()
    assert code == 1
    assert "1 extracted" in captured.out and "1 failed" in captured.out
    assert "BAD.pdf" in captured.err
