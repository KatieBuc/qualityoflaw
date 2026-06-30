import json
from pathlib import Path

import pandas as pd
import pytest

from automation.src.evaluate_accuracy import (
    build_error_analysis_df,
    deduplicate_reports,
    filter_golden_to_evaluated,
    find_unmatched_indicator_pairs,
    load_golden_dataframe,
    load_llm_results,
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
            },
        ]
    )
    output = build_error_analysis_df(errors_df)
    assert output.iloc[0]["error_type"] == "false_positive"
    assert output.iloc[0]["golden_label"] == "No"
    assert output.iloc[0]["pred_label"] == "Yes"


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
