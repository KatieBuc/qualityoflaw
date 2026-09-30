import logging

import pytest

from automation.src.config_loader import parse_small_scale_stems
from automation.src.constants import DEFAULT_SMALL_SCALE_STEMS, SMALL_SCALE_FILES
from automation.src.markdown.policy_files import markdown_policy_files
from automation.src.policy_files import filter_policy_files


def _touch(directory, *names):
    for name in names:
        (directory / name).write_text("x", encoding="utf-8")


def test_default_stems_derive_from_small_scale_files():
    assert DEFAULT_SMALL_SCALE_STEMS == tuple(name.removesuffix(".txt") for name in SMALL_SCALE_FILES)


def test_parse_defaults_when_absent():
    assert parse_small_scale_stems(None) == DEFAULT_SMALL_SCALE_STEMS


def test_parse_reads_explicit_stems():
    assert parse_small_scale_stems(["A", "B"]) == ("A", "B")


@pytest.mark.parametrize("bad", [[], "A", [""], [1], ["A", None]])
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError, match="small_scale_stems"):
        parse_small_scale_stems(bad)


def test_markdown_small_scale_on_cleaned_md_corpus(tmp_path):
    _touch(tmp_path, "ACEH_BIREUEN.cleaned.md", "OTHER.cleaned.md")
    files = markdown_policy_files(
        tmp_path, True, suffix=".cleaned.md", stems=("ACEH_BIREUEN",)
    )
    assert [f.name for f in files] == ["ACEH_BIREUEN.cleaned.md"]


def test_markdown_small_scale_on_plain_md_corpus(tmp_path):
    _touch(tmp_path, "94. Latvia - DomViol - Latvian (a).md", "OTHER.md")
    files = markdown_policy_files(
        tmp_path, True, suffix=".md", stems=("94. Latvia - DomViol - Latvian (a)",)
    )
    assert [f.name for f in files] == ["94. Latvia - DomViol - Latvian (a).md"]


def test_markdown_small_scale_defaults_to_localpolicies_stems(tmp_path):
    _touch(tmp_path, "ACEH_BIREUEN.cleaned.md", "OTHER.cleaned.md")
    files = markdown_policy_files(tmp_path, True)
    assert [f.name for f in files] == ["ACEH_BIREUEN.cleaned.md"]


def test_markdown_without_small_scale_returns_everything(tmp_path):
    _touch(tmp_path, "A.cleaned.md", "B.cleaned.md")
    assert len(markdown_policy_files(tmp_path, False, stems=("Z",))) == 2


def test_markdown_small_scale_matching_nothing_raises(tmp_path):
    _touch(tmp_path, "ACEH_BIREUEN.cleaned.md")
    with pytest.raises(FileNotFoundError, match="small_scale_stems"):
        markdown_policy_files(tmp_path, True, stems=("Latvia",))


def test_markdown_small_scale_on_empty_dir_does_not_raise(tmp_path):
    assert markdown_policy_files(tmp_path, True, stems=("Latvia",)) == []


def test_partial_match_warns_but_selects(tmp_path, caplog):
    _touch(tmp_path, "A.cleaned.md", "B.cleaned.md")
    with caplog.at_level(logging.WARNING):
        files = markdown_policy_files(tmp_path, True, stems=("A", "MISSING"))
    assert [f.name for f in files] == ["A.cleaned.md"]
    assert "MISSING" in caplog.text


def test_txt_small_scale_uses_stems(tmp_path):
    _touch(tmp_path, "A.txt", "B.txt")
    assert [f.name for f in filter_policy_files(tmp_path, True, ("B",))] == ["B.txt"]


def test_txt_small_scale_matching_nothing_raises(tmp_path):
    _touch(tmp_path, "A.txt")
    with pytest.raises(FileNotFoundError, match="small_scale_stems"):
        filter_policy_files(tmp_path, True, ("Z",))
