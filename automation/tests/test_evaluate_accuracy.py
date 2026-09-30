import json
from pathlib import Path

import pandas as pd
import pytest

from automation.src.evaluate_accuracy import (
    apply_manual_overwrites,
    build_error_analysis_df,
    calculate_metrics,
    deduplicate_reports,
    filter_golden_to_evaluated,
    find_unmatched_indicator_pairs,
    load_golden_dataframe,
    load_llm_results,
    load_manual_overwrites,
    parse_report_timestamp,
)


def test_golden_indicator_id_preserves_dot_ten_suffix(tmp_path):
    csv_path = tmp_path / "golden.csv"
    csv_path.write_text(
        "fullname,filename,year,indicator_id,value,indicator_value\n"
        "ACEH BIREUEN,ACEH_BIREUEN.txt,2022,1.1,1.0,Domestic violence\n"
        "ACEH BIREUEN,ACEH_BIREUEN.txt,2022,1.10,0.0,Technology-facilitated violence\n",
        encoding="utf-8",
    )

    golden_df = load_golden_dataframe(str(csv_path))
    ids = set(golden_df["indicator_id"])
    assert ids == {"1.1", "1.10"}


def test_build_error_analysis_df_labels_and_error_type():
    errors_df = pd.DataFrame(
        [
            {
                "fullname": "ACEH BIREUEN",
                "filename": "ACEH_BIREUEN.txt",
                "indicator_id": "1.1",
                "indicator_value": "Domestic violence",
                "value": 0.0,
                "pred_value": 1.0,
                "evidence": "Pasal 5 ayat (2) ...",
                "rationale": "The judge found an explicit reference.",
            },
        ]
    )
    output = build_error_analysis_df(errors_df)
    assert output.iloc[0]["error_type"] == "false_positive"
    assert output.iloc[0]["golden_label"] == "No"
    assert output.iloc[0]["pred_label"] == "Yes"
    assert output.iloc[0]["evidence"] == "Pasal 5 ayat (2) ..."
    assert output.iloc[0]["rationale"] == "The judge found an explicit reference."
    assert output.iloc[0]["discrepancy_root_cause"] == "Reference only"


def test_find_unmatched_indicator_pairs():
    golden_filtered = pd.DataFrame(
        [
            {"filename": "A.txt", "indicator_id": "1.1", "fullname": "A", "indicator_value": "x", "value": 1.0},
            {"filename": "A.txt", "indicator_id": "1.2", "fullname": "A", "indicator_value": "y", "value": 0.0},
        ]
    )
    llm_df = pd.DataFrame(
        [
            {"filename": "A.txt", "indicator_id": "1.1", "pred_value": 1.0},
            {"filename": "A.txt", "indicator_id": "1.3", "pred_value": 0.0},
        ]
    )
    unmatched = find_unmatched_indicator_pairs(golden_filtered, llm_df)
    assert len(unmatched) == 2
    assert set(unmatched["match_status"]) == {"golden_only", "llm_only"}


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


def test_load_manual_overwrites_missing_path_returns_empty():
    assert load_manual_overwrites(None) == {}


def test_load_manual_overwrites_missing_file_returns_empty(tmp_path):
    assert load_manual_overwrites(tmp_path / "does_not_exist.yaml") == {}


def test_load_manual_overwrites_parses_yaml(tmp_path):
    path = tmp_path / "manual_overwrites.yaml"
    path.write_text(
        "A.txt:\n  '4.4': 1\n  '5.1': 0\n"
        "B.txt:\n  '1.2': 1\n",
        encoding="utf-8",
    )

    overwrites = load_manual_overwrites(path)
    assert overwrites == {
        "A.txt": {"4.4": 1.0, "5.1": 0.0},
        "B.txt": {"1.2": 1.0},
    }


def test_apply_manual_overwrites_replaces_matching_value():
    golden_df = pd.DataFrame(
        [
            {"filename": "A.txt", "indicator_id": "4.4", "value": 0.0},
            {"filename": "A.txt", "indicator_id": "5.1", "value": 0.0},
            {"filename": "B.txt", "indicator_id": "4.4", "value": 0.0},
        ]
    )
    overwrites = {"A.txt": {"4.4": 1.0}}

    result = apply_manual_overwrites(golden_df, overwrites)

    assert result[(result["filename"] == "A.txt") & (result["indicator_id"] == "4.4")]["value"].iloc[0] == 1.0
    # unrelated rows are untouched
    assert result[(result["filename"] == "A.txt") & (result["indicator_id"] == "5.1")]["value"].iloc[0] == 0.0
    assert result[(result["filename"] == "B.txt") & (result["indicator_id"] == "4.4")]["value"].iloc[0] == 0.0


def test_apply_manual_overwrites_no_match_is_ignored_not_raised():
    golden_df = pd.DataFrame([{"filename": "A.txt", "indicator_id": "4.4", "value": 0.0}])
    # "9.9" doesn't exist for A.txt — should warn (printed) but not raise or add rows.
    result = apply_manual_overwrites(golden_df, {"A.txt": {"9.9": 1.0}})
    assert len(result) == 1
    assert result.iloc[0]["value"] == 0.0


def test_apply_manual_overwrites_empty_overwrites_returns_same_df():
    golden_df = pd.DataFrame([{"filename": "A.txt", "indicator_id": "4.4", "value": 0.0}])
    assert apply_manual_overwrites(golden_df, {}) is golden_df


def test_calculate_metrics_applies_manual_overwrites(tmp_path):
    csv_path = tmp_path / "golden.csv"
    csv_path.write_text(
        "fullname,filename,year,indicator_id,value,indicator_value\n"
        "ACEH BIREUEN,ACEH_BIREUEN.txt,2022,4.4,0.0,Prevention through education\n"
        "ACEH BIREUEN,ACEH_BIREUEN.txt,2022,5.1,0.0,NGO collaboration\n",
        encoding="utf-8",
    )
    llm_df = pd.DataFrame(
        [
            {"filename": "ACEH_BIREUEN.txt", "indicator_id": "4.4", "pred_value": 1.0},
            {"filename": "ACEH_BIREUEN.txt", "indicator_id": "5.1", "pred_value": 0.0},
        ]
    )

    errors_df, _, metrics_summary, _merged = calculate_metrics(
        str(csv_path), llm_df, manual_overwrites={"ACEH_BIREUEN.txt": {"4.4": 1.0}}
    )

    # Golden value for 4.4 corrected from 0.0 to 1.0 before comparison, so the
    # LLM's "Yes" prediction now matches it — no error, full accuracy.
    assert errors_df.empty
    assert metrics_summary["overall"]["accuracy"] == 1.0
