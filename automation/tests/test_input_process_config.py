from pathlib import Path

import pytest

from automation.src.constants import DATA_ROOT, PROJECT_ROOT
from automation.src.input_process.__main__ import main
from automation.src.input_process.config import parse_input_process_config
from automation.src.input_process.migrate import migrate_raw
from automation.src.paths import project_dirs


def test_default_folders_are_under_the_project_raw_dir():
    config = parse_input_process_config({"project": "demo"})
    raw = DATA_ROOT / "demo" / "raw"
    assert config.raw_files_dir == raw / "raw_files"
    assert config.extracted_txt_dir == raw / "extracted_txt"
    assert config.cleaned_txt_dir == raw / "cleaned_txt"
    assert config.report_path == raw / "input_report.json"
    assert config.language == "Indonesian"


def test_every_folder_can_be_overridden_in_the_config(tmp_path):
    config = parse_input_process_config(
        {
            "project": "demo",
            "input_process": {
                "language": "English",
                "raw_files_dir": "data/demo/originals",
                "extracted_txt_dir": str(tmp_path / "ex"),
                "cleaned_txt_dir": "data/demo/ok",
                "report_path": str(tmp_path / "report.json"),
                "thresholds": {"min_letters": 500},
            },
        }
    )
    assert config.raw_files_dir == PROJECT_ROOT / "data/demo/originals"
    assert config.extracted_txt_dir == tmp_path / "ex"
    assert config.cleaned_txt_dir == PROJECT_ROOT / "data/demo/ok"
    assert config.report_path == tmp_path / "report.json"
    assert config.language == "English" and config.thresholds == {"min_letters": 500}


def test_bad_thresholds_are_rejected():
    with pytest.raises(ValueError, match="thresholds"):
        parse_input_process_config({"input_process": {"thresholds": [1, 2]}})


def test_project_dirs_expose_the_stage_1_folders():
    dirs = project_dirs("demo")
    assert dirs.raw_files.name == "raw_files" and dirs.raw_files.parent == dirs.raw
    assert dirs.extracted_txt.name == "extracted_txt"
    assert dirs.cleaned_txt.name == "cleaned_txt"
    assert dirs.input_report.name == "input_report.json"


@pytest.fixture
def raw_project(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.paths.DATA_ROOT", tmp_path)
    raw = tmp_path / "demo" / "raw"
    raw.mkdir(parents=True)
    for name in ("A.txt", "B.pdf", "C.PDF", "D.html", "notes.csv"):
        (raw / name).write_text(name, encoding="utf-8")
    return raw


def config_for(tmp_path):
    return parse_input_process_config({"project": "demo"})


def test_migrate_raw_moves_sources_and_leaves_other_files(raw_project, tmp_path):
    result = migrate_raw(config_for(tmp_path))

    assert sorted(result.moved) == ["A.txt", "B.pdf", "C.PDF", "D.html"]
    assert result.ignored == ["notes.csv"]
    assert (raw_project / "raw_files" / "C.PDF").read_text(encoding="utf-8") == "C.PDF"
    assert (raw_project / "notes.csv").exists() and not (raw_project / "A.txt").exists()


def test_migrate_raw_never_overwrites(raw_project, tmp_path):
    target = raw_project / "raw_files"
    target.mkdir()
    (target / "A.txt").write_text("already here", encoding="utf-8")

    result = migrate_raw(config_for(tmp_path))

    assert result.skipped_existing == ["A.txt"]
    assert (target / "A.txt").read_text(encoding="utf-8") == "already here"
    assert (raw_project / "A.txt").exists()  # the unmoved original stays put


def test_migrate_raw_dry_run_changes_nothing(raw_project, tmp_path):
    result = migrate_raw(config_for(tmp_path), dry_run=True)
    assert len(result.moved) == 4 and not (raw_project / "raw_files").exists()


def test_cli_migrate_raw(raw_project, tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text("project: demo\n", encoding="utf-8")

    assert main(["--pipeline-config", str(cfg), "migrate-raw"]) == 0

    assert "Moved 4 file(s)" in capsys.readouterr().out
    assert (raw_project / "raw_files" / "B.pdf").exists()
