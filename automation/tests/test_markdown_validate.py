import pytest

from automation.src.markdown import validate as v

SRC = "# BAB I\n\n#### Pasal 1\n\nIsi.\n"
GOOD = "# CHAPTER I\n\n#### Article 1\n\nContent.\n"


def test_matching_headers_pass():
    assert v.validate_headers("A", SRC, GOOD).ok


def test_count_mismatch_fails():
    check = v.validate_headers("A", SRC, "# CHAPTER I\n\nContent.\n")
    assert not check.ok and "count" in check.problem


def test_hierarchy_mismatch_fails():
    check = v.validate_headers("A", SRC, "# CHAPTER I\n\n### Article 1\n\nContent.\n")
    assert not check.ok and "hierarchy" in check.problem


def test_missing_counterpart_fails(tmp_path):
    (tmp_path / "s").mkdir()
    (tmp_path / "t").mkdir()
    (tmp_path / "s" / "A.cleaned.md").write_text(SRC, encoding="utf-8")
    report = v.validate_directories(tmp_path / "s", tmp_path / "t")
    assert not report.ok and report.failed[0].problem == "no translation file"


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.paths.DATA_ROOT", tmp_path)
    pre = tmp_path / "demo" / "preprocessed"
    (pre / "cleaned_markdown").mkdir(parents=True)
    (pre / "translation_markdown").mkdir()
    (pre / "cleaned_markdown" / "A.cleaned.md").write_text(SRC, encoding="utf-8")
    (pre / "translation_markdown" / "A.md").write_text(GOOD, encoding="utf-8")
    return tmp_path / "demo"


def test_cli_reports_mismatches_and_moves_nothing(project, capsys):
    (project / "preprocessed" / "translation_markdown" / "A.md").write_text("no headings", encoding="utf-8")

    assert v.main(["--project", "demo"]) == 1

    assert "FAILED A" in capsys.readouterr().out
    assert (project / "preprocessed" / "cleaned_markdown" / "A.cleaned.md").exists()
    assert not (project / "processed").exists()


def test_cli_passes_for_matching_pairs_and_still_moves_nothing(project, capsys):
    assert v.main(["--project", "demo"]) == 0
    assert "1/1 pairs valid" in capsys.readouterr().out
    assert (project / "preprocessed" / "translation_markdown" / "A.md").exists()
    assert not (project / "processed").exists()
