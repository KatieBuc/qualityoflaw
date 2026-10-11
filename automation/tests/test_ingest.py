import json
import re

from automation.src.ingest import NO_STRUCTURE, run_ingest_step
from automation.src.input_process.__main__ import main
from automation.src.input_process import heading_keywords as KW
from automation.tests.test_txt_to_md_keywords import INDONESIAN

ENGLISH = "\n".join(f"Article {n}\nSomething happens in article {n}.\n" for n in range(1, 5))


def levels(md):
    return {m.group(2): len(m.group(1)) for m in re.finditer(r"(?m)^(#{1,6}) (.+)$", md)}


def test_indonesian_gets_the_corpus_depths_and_no_auto_title(tmp_path):
    src, out = tmp_path / "txt", tmp_path / "md"
    src.mkdir()
    (src / "ACEH_X.txt").write_text(INDONESIAN, encoding="utf-8")

    run_ingest_step(src, out, language_for=lambda stem: "Indonesian")

    found = levels((out / "ACEH_X.cleaned.md").read_text(encoding="utf-8"))
    assert (found["BAB I"], found["Bagian Kesatu"], found["Paragraf 1"], found["Pasal 1"]) == (1, 2, 3, 4)
    assert "Contoh" not in found                      # no generated '# title'


def test_another_language_keeps_the_generic_depths(tmp_path):
    src, out = tmp_path / "txt", tmp_path / "md"
    src.mkdir()
    (src / "UK_X.txt").write_text(ENGLISH, encoding="utf-8")

    run_ingest_step(src, out, language_for=lambda stem: "English")

    found = levels((out / "UK_X.cleaned.md").read_text(encoding="utf-8"))
    assert found["Article 1"] == 2


def test_a_two_bagian_regulation_keeps_its_bagian_under_the_profile():
    short = INDONESIAN.replace("Paragraf 2", "Pasal 4C").replace("Bagian Ketiga", "Pasal 6B")
    from automation.src.ingest import convert_text

    with_profile, _ = convert_text(short, "X", language="Indonesian")
    without, _ = convert_text(short, "X", language="English")
    assert "# Bagian Kesatu" in with_profile.replace("## ", "# ") and levels(with_profile)["Bagian Kesatu"] == 2
    assert "Bagian Kesatu" not in levels(without) or levels(without)["Bagian Kesatu"] != 2


def test_existing_outputs_are_kept_unless_forced(tmp_path):
    src, out = tmp_path / "txt", tmp_path / "md"
    src.mkdir()
    (src / "A.txt").write_text(INDONESIAN, encoding="utf-8")
    run_ingest_step(src, out, language_for=lambda s: "Indonesian")
    (out / "A.cleaned.md").write_text("edited by hand", encoding="utf-8")

    again = run_ingest_step(src, out, language_for=lambda s: "Indonesian")
    assert again["summary"] == {"converted": 0, "kept": 1, "skipped": 0}
    assert (out / "A.cleaned.md").read_text(encoding="utf-8") == "edited by hand"

    forced = run_ingest_step(src, out, language_for=lambda s: "Indonesian", force=True)
    assert forced["summary"]["converted"] == 1 and "BAB I" in (out / "A.cleaned.md").read_text(encoding="utf-8")


def test_text_without_structure_is_skipped_and_reported(tmp_path):
    src, out = tmp_path / "txt", tmp_path / "md"
    src.mkdir()
    (src / "PROSE.txt").write_text("Just a paragraph of prose.\nAnd another one.\n", encoding="utf-8")
    report = tmp_path / "report.json"

    result = run_ingest_step(src, out, report_path=report)

    assert result["skipped"] == [{"stem": "PROSE", "reason": NO_STRUCTURE}]
    assert not (out / "PROSE.cleaned.md").exists()
    assert json.loads(report.read_text(encoding="utf-8"))["summary"]["skipped"] == 1


def test_stems_limit_the_run(tmp_path):
    src, out = tmp_path / "txt", tmp_path / "md"
    src.mkdir()
    for name in ("A", "B"):
        (src / f"{name}.txt").write_text(INDONESIAN, encoding="utf-8")
    run_ingest_step(src, out, stems={"B"}, language_for=lambda s: "Indonesian")
    assert [p.name for p in out.iterdir()] == ["B.cleaned.md"]


def test_cli_markdown_reads_the_configured_source(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("automation.src.paths.DATA_ROOT", tmp_path)
    gold = tmp_path / "demo" / "raw" / "gold"
    cleaned = tmp_path / "demo" / "raw" / "cleaned_txt"
    gold.mkdir(parents=True); cleaned.mkdir(parents=True)
    (gold / "FROM_GOLD.txt").write_text(INDONESIAN, encoding="utf-8")
    (cleaned / "FROM_CLEANED.txt").write_text(INDONESIAN, encoding="utf-8")
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text(
        "project: demo\ninput_process:\n  markdown_source: gold\n"
        f"  gold_dir: {gold.as_posix()}\n  cleaned_txt_dir: {cleaned.as_posix()}\n"
        f"  report_path: {(tmp_path / 'r.json').as_posix()}\n",
        encoding="utf-8",
    )

    assert main(["--pipeline-config", str(cfg), "markdown"]) == 0

    out = tmp_path / "demo" / "preprocessed" / "cleaned_markdown"
    assert [p.name for p in out.iterdir()] == ["FROM_GOLD.cleaned.md"]
    assert "1 converted" in capsys.readouterr().out
    assert KW.profile_for("Indonesian")["depth"]["CHAPTER"] == 1
