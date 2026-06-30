import json
from pathlib import Path

import pytest

from quality_eval.v1.criteria import (
    EXPECTED_INDICATOR_COUNT,
    EXPECTED_INDICATOR_IDS,
    validate_criteria_files,
)
from quality_eval.v1.prompts.prompt_loader import generate_judge_prompt
from quality_eval.evaluate_accuracy import (
    deduplicate_reports,
    filter_golden_to_evaluated,
    load_llm_results,
    parse_report_timestamp,
)
from quality_eval.run_eval_new import validate_completeness, resolve_policy_paths


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "v1" / "prompts"
TRANSLATED_POLICY_DIR = Path(__file__).resolve().parents[1] / "v1" / "translated_policy"
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_expected_indicator_count():
    assert EXPECTED_INDICATOR_COUNT == 56
    assert len(EXPECTED_INDICATOR_IDS) == 56


def test_criteria_files_match_schema():
    ok, issues = validate_criteria_files(PROMPTS_DIR)
    assert ok, issues


def test_generate_judge_prompt_substitutes_placeholders(tmp_path):
    criteria_file = tmp_path / "criteria.txt"
    template_file = tmp_path / "template.txt"
    policy_file = tmp_path / "policy.txt"

    criteria_file.write_text("1.1 | Domestic violence | does the policy include Domestic violence?\n", encoding="utf-8")
    template_file.write_text(
        "CRITERIA:\n{{CRITERIA_LIST}}\nPOLICY:\n{{POLICY_TEXT}}",
        encoding="utf-8",
    )
    policy_file.write_text("Sample policy text.", encoding="utf-8")

    prompt = generate_judge_prompt(
        criteria_file_path=str(criteria_file),
        template_file_path=str(template_file),
        policy_file_path=str(policy_file),
    )

    assert "1.1 (Domestic violence)" in prompt
    assert "Sample policy text." in prompt
    assert "{{CRITERIA_LIST}}" not in prompt
    assert "{{POLICY_TEXT}}" not in prompt


def test_validate_completeness_detects_missing():
    partial = {"1.1": {"id": "1.1", "included": "Yes"}}
    is_complete, issues = validate_completeness(partial)
    assert not is_complete
    assert any("Missing indicators" in issue for issue in issues)


def test_validate_completeness_passes_for_full_set():
    full = {indicator_id: {"id": indicator_id} for indicator_id in EXPECTED_INDICATOR_IDS}
    is_complete, issues = validate_completeness(full)
    assert is_complete
    assert issues == []


def test_parse_report_timestamp_from_filename():
    ts = parse_report_timestamp(
        "27062026223404-gpt-4o-ACEH_BIREUEN.json",
        {},
    )
    assert ts.year == 2026
    assert ts.month == 6
    assert ts.day == 27


def test_parse_report_timestamp_from_metadata():
    ts = parse_report_timestamp(
        "report.json",
        {"evaluated_at": "2026-06-27T22:34:04+00:00"},
    )
    assert ts.year == 2026
    assert ts.hour == 22


def test_deduplicate_reports_keeps_latest():
    older = (
        parse_report_timestamp("01012026000000-gpt-4o-ACEH_BIREUEN.json", {}),
        "older.json",
        {"policy_file": "ACEH_BIREUEN.txt", "evaluation_results": {"1.1": {"id": "1.1", "included": "No"}}},
    )
    newer = (
        parse_report_timestamp("02012026000000-gpt-4o-ACEH_BIREUEN.json", {}),
        "newer.json",
        {"policy_file": "ACEH_BIREUEN.txt", "evaluation_results": {"1.1": {"id": "1.1", "included": "Yes"}}},
    )
    deduped = deduplicate_reports([older, newer])
    assert len(deduped) == 1
    assert deduped[0][1] == "newer.json"


def test_load_llm_results_deduplicates_folder(tmp_path):
    older_report = {
        "policy_file": "ACEH_BIREUEN.txt",
        "evaluation_results": {"1.1": {"id": "1.1", "included": "No"}},
    }
    newer_report = {
        "policy_file": "ACEH_BIREUEN.txt",
        "evaluation_results": {"1.1": {"id": "1.1", "included": "Yes"}},
    }
    (tmp_path / "01012026000000-gpt-4o-ACEH_BIREUEN.json").write_text(
        json.dumps(older_report), encoding="utf-8"
    )
    (tmp_path / "02012026000000-gpt-4o-ACEH_BIREUEN.json").write_text(
        json.dumps(newer_report), encoding="utf-8"
    )

    df = load_llm_results(str(tmp_path))
    assert len(df) == 1
    assert df.iloc[0]["pred_value"] == 1.0


def test_filter_golden_to_latest_year():
    import pandas as pd

    golden_df = pd.DataFrame(
        [
            {"filename": "A.txt", "year": 2020, "indicator_id": "1.1", "value": 0.0},
            {"filename": "A.txt", "year": 2022, "indicator_id": "1.1", "value": 1.0},
            {"filename": "B.txt", "year": 2021, "indicator_id": "1.1", "value": 0.0},
        ]
    )
    filtered = filter_golden_to_evaluated(golden_df, ["A.txt", "B.txt"])
    assert len(filtered) == 2
    assert filtered[filtered["filename"] == "A.txt"].iloc[0]["value"] == 1.0


def test_resolve_policy_paths_file(tmp_path):
    policy = tmp_path / "ACEH_BIREUEN.txt"
    policy.write_text("policy text", encoding="utf-8")
    paths = resolve_policy_paths(str(policy))
    assert paths == [str(policy)]


def test_resolve_policy_paths_folder(tmp_path):
    (tmp_path / "A.txt").write_text("a", encoding="utf-8")
    (tmp_path / "B.txt").write_text("b", encoding="utf-8")
    (tmp_path / "ignore.json").write_text("{}", encoding="utf-8")
    paths = resolve_policy_paths(str(tmp_path))
    assert len(paths) == 2
    assert all(path.endswith(".txt") for path in paths)


def test_load_llm_results_on_sample_policy():
    sample = TRANSLATED_POLICY_DIR / "ACEH_BIREUEN.txt"
    if not sample.exists():
        pytest.skip("Sample translated policy not available")

    result_dir = Path(__file__).resolve().parents[1] / "v1" / "result" / "gpt-4o"
    if not result_dir.exists():
        pytest.skip("Sample result directory not available")

    df = load_llm_results(str(result_dir))
    assert not df.empty
    assert df["filename"].nunique() <= len(list(result_dir.glob("*.json")))
