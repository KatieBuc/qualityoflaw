from pathlib import Path

import pytest

from automation.src.criteria import CRITERIA_FILES, EXPECTED_INDICATOR_IDS
from automation.src.constants import PROMPTS_ROOT
from automation.src.rag.prompt_builder import (
    format_criteria_list,
    format_criteria_with_candidates,
    format_rubric_block,
    parse_criteria_lines,
)
from automation.src.rag.retriever import RetrievedChunk

CRITERION = "4.1 | Information and education about VAW | Does the policy include education about VAW?"


def write(tmp_path: Path, *lines: str) -> str:
    path = tmp_path / "criteria.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def test_rubric_lines_attach_to_their_criterion(tmp_path):
    path = write(
        tmp_path,
        CRITERION,
        "Code Yes if any of the following apply:",
        "campaigns about violence against women;",
        "Code No if the policy refers only to:",
        "education about women's rights.",
        "4.2 | Gender equality education | Does the policy educate about gender equality?",
        "Code Yes if it covers gender equality.",
    )

    first, second = parse_criteria_lines(path)

    assert (first.id, first.indicator) == ("4.1", "Information and education about VAW")
    assert first.rubric == [
        "Code Yes if any of the following apply:",
        "campaigns about violence against women;",
        "Code No if the policy refers only to:",
        "education about women's rights.",
    ]
    assert second.id == "4.2"
    assert second.rubric == ["Code Yes if it covers gender equality."]


def test_blank_and_comment_lines_are_ignored(tmp_path):
    path = write(tmp_path, "# a comment", "", CRITERION, "", "# another", "Code Yes if X.", "")

    (criterion,) = parse_criteria_lines(path)

    assert criterion.rubric == ["Code Yes if X."]


def test_rubric_line_with_two_pipes_is_not_read_as_a_criterion(tmp_path):
    # Without the id-shape guard this line would split into three fields and
    # silently become a fifth criterion.
    path = write(tmp_path, CRITERION, "Code Yes if it mentions a | b | c.")

    criteria = parse_criteria_lines(path)

    assert len(criteria) == 1
    assert criteria[0].rubric == ["Code Yes if it mentions a | b | c."]


def test_lines_before_any_criterion_are_reported_not_silently_dropped(tmp_path, caplog):
    path = write(tmp_path, "This file describes dimension 4.", CRITERION)

    with caplog.at_level("WARNING"):
        criteria = parse_criteria_lines(path)

    assert len(criteria) == 1
    assert criteria[0].rubric == []
    assert "before the first criterion" in caplog.text


def test_criteria_without_rubric_render_unchanged(tmp_path):
    path = write(tmp_path, CRITERION)

    (criterion,) = parse_criteria_lines(path)

    assert criterion.rubric == []
    assert format_rubric_block(criterion) == []
    assert format_criteria_list([criterion]) == criterion.heading


def test_rubric_is_rendered_under_its_criterion(tmp_path):
    path = write(tmp_path, CRITERION, "Code Yes if X.", "Code No if Y.")

    rendered = format_criteria_list(parse_criteria_lines(path))

    assert rendered.splitlines() == [
        "- 4.1 (Information and education about VAW): Does the policy include education about VAW?",
        "  Coding rules:",
        "    Code Yes if X.",
        "    Code No if Y.",
    ]


def test_candidates_are_listed_after_the_rules(tmp_path):
    path = write(tmp_path, CRITERION, "Code Yes if X.")
    candidates = {"4.1": [RetrievedChunk(chunk_id=7, text="The policy educates about VAW.", score=0.9)]}

    lines = format_criteria_with_candidates(parse_criteria_lines(path), candidates).splitlines()

    assert lines[1] == "  Coding rules:"
    assert lines[2] == "    Code Yes if X."
    # Rules come before evidence: know the rule, then read what it applies to.
    assert lines[3].startswith("  [4.1-7.0]")


def test_missing_candidates_still_reported_alongside_rules(tmp_path):
    path = write(tmp_path, CRITERION, "Code Yes if X.")

    lines = format_criteria_with_candidates(parse_criteria_lines(path), {}).splitlines()

    assert "  Coding rules:" in lines
    assert "  Candidate passages: (none retrieved)" in lines


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
def test_shipped_criteria_files_parse_to_the_expected_indicators(version):
    found = {
        criterion.id
        for name in CRITERIA_FILES
        for criterion in parse_criteria_lines(
            str(PROMPTS_ROOT / "quality_eval" / version / name)
        )
    }

    assert found == set(EXPECTED_INDICATOR_IDS)


@pytest.mark.parametrize("version,has_rubric", [("v1", False), ("v2", False), ("v3", True), ("v4", True)])
def test_rubric_reaches_the_prompt_for_versions_that_have_one(version, has_rubric):
    criteria = [
        criterion
        for name in CRITERIA_FILES
        for criterion in parse_criteria_lines(
            str(PROMPTS_ROOT / "quality_eval" / version / name)
        )
    ]
    rubric_lines = sum(len(criterion.rubric) for criterion in criteria)

    if has_rubric:
        # v3/v4 carry 451 non-empty lines in total; 56 are criteria and the
        # remaining 395 are coding rules that used to be discarded entirely.
        assert rubric_lines == 395
        # 18 of the 56 criteria have no rubric written for them yet (a gap in
        # the criteria files, not in parsing) — those render exactly as they
        # did before, question only.
        assert sum(1 for criterion in criteria if criterion.rubric) == 38
    else:
        assert rubric_lines == 0
