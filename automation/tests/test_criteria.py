from automation.src.constants import AUTOMATION_ROOT
from automation.src.criteria import EXPECTED_INDICATOR_COUNT, EXPECTED_INDICATOR_IDS, validate_criteria_files


def test_expected_indicator_count():
    assert EXPECTED_INDICATOR_COUNT == 56
    assert len(EXPECTED_INDICATOR_IDS) == 56


def test_criteria_files_match_schema():
    ok, issues = validate_criteria_files(AUTOMATION_ROOT / "prompts" / "quality_eval" / "v1")
    assert ok, issues
