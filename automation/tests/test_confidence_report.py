import json
import math

import pandas as pd
import pytest

from automation.src.config_loader import ConfidenceReportConfig
from automation.src.confidence_report import (
    STATUS_NO_DATA,
    STATUS_OK,
    build_confidence_report,
    build_low_confidence_frame,
    coverage_row,
    prepare_frame,
    write_confidence_report,
)


def make_frame(rows: list[dict]) -> pd.DataFrame:
    """rows: (indicator_id, golden, predicted, confidence)."""
    return pd.DataFrame(
        [
            {
                "fullname": f"Policy {index}",
                "filename": row.get("filename", f"POLICY_{index}.txt"),
                "indicator_id": row["indicator_id"],
                "indicator_value": "indicator",
                "value": row["value"],
                "pred_value": row["pred_value"],
                "confidence": row["confidence"],
                "answer_logprob": (
                    None if row["confidence"] is None else math.log(row["confidence"])
                ),
                "p_yes": row.get("p_yes"),
                "p_no": row.get("p_no"),
                "confidence_source": row.get(
                    "confidence_source",
                    "unavailable" if row["confidence"] is None else "logprobs",
                ),
                "rationale": "because",
            }
            for index, row in enumerate(rows)
        ]
    )


@pytest.fixture
def sample_df():
    # 6 predictions, 3 of them wrong. Two of the three errors sit below 0.9.
    return make_frame(
        [
            {"indicator_id": "1.1", "value": 1.0, "pred_value": 1.0, "confidence": 0.99},
            {"indicator_id": "1.2", "value": 0.0, "pred_value": 0.0, "confidence": 0.95},
            {"indicator_id": "1.3", "value": 1.0, "pred_value": 1.0, "confidence": 0.60},
            {"indicator_id": "4.1", "value": 0.0, "pred_value": 1.0, "confidence": 0.55},
            {"indicator_id": "4.2", "value": 1.0, "pred_value": 0.0, "confidence": 0.70},
            {"indicator_id": "4.3", "value": 0.0, "pred_value": 1.0, "confidence": 0.98},
        ]
    )


def test_coverage_row_counts_errors_below_threshold(sample_df):
    conf_df = prepare_frame(sample_df)

    row = coverage_row(conf_df, 0.9, "absolute")

    assert row["flagged"] == 3          # 0.60, 0.55, 0.70
    assert row["errors_captured"] == 2  # 4.1 and 4.2
    assert row["coverage_rate"] == pytest.approx(2 / 3)
    assert row["precision"] == pytest.approx(2 / 3)
    assert row["remaining_errors"] == 1
    # Base error rate is 3/6, so precision of 2/3 is a 1.33x lift.
    assert row["lift"] == pytest.approx((2 / 3) / 0.5)
    assert row["accuracy_on_unflagged"] == pytest.approx(2 / 3)


def test_coverage_row_with_nothing_flagged_reports_null_precision(sample_df):
    conf_df = prepare_frame(sample_df)

    row = coverage_row(conf_df, 0.1, "absolute")

    assert row["flagged"] == 0
    assert row["coverage_rate"] == 0.0
    assert row["precision"] is None
    assert row["lift"] is None


def test_coverage_row_with_zero_errors_reports_null_coverage():
    conf_df = prepare_frame(
        make_frame(
            [
                {"indicator_id": "1.1", "value": 1.0, "pred_value": 1.0, "confidence": 0.5},
                {"indicator_id": "1.2", "value": 0.0, "pred_value": 0.0, "confidence": 0.6},
            ]
        )
    )

    row = coverage_row(conf_df, 0.9, "absolute")

    assert row["flagged"] == 2
    assert row["coverage_rate"] is None
    assert row["precision"] == 0.0


def test_report_headline_numbers(sample_df):
    config = ConfidenceReportConfig(thresholds=(0.7, 0.9), quantiles=(0.5,), primary_threshold=0.9)

    report = build_confidence_report(sample_df, config, run_id="20260821_102245", model="gpt-5.2")

    assert report["status"] == STATUS_OK
    assert report["run_id"] == "20260821_102245"
    assert report["confidence_source"]["availability"] == 1.0
    assert report["baseline"]["error_count"] == 3
    assert report["baseline"]["accuracy"] == pytest.approx(0.5)
    assert report["coverage"]["primary"]["coverage_rate"] == pytest.approx(2 / 3)
    assert [row["threshold"] for row in report["coverage"]["by_absolute_threshold"]] == [0.7, 0.9]


def test_quantile_cutoffs_are_derived_from_the_distribution(sample_df):
    config = ConfidenceReportConfig(quantiles=(0.5,), primary_threshold=0.9)

    report = build_confidence_report(sample_df, config)

    (quantile_row,) = report["coverage"]["by_quantile"]
    assert quantile_row["kind"] == "quantile"
    assert quantile_row["quantile"] == 0.5
    # Median of the six confidences: (0.70 + 0.95) / 2.
    assert quantile_row["threshold"] == pytest.approx(0.825)
    assert quantile_row["flagged"] == 3


def test_breakdowns_split_by_dimension_and_error_type(sample_df):
    report = build_confidence_report(sample_df, ConfidenceReportConfig(primary_threshold=0.9))

    by_dimension = report["breakdowns"]["by_dimension"]
    assert by_dimension["scope_of_violence"]["errors"] == 0
    assert by_dimension["primary_prevention"]["errors"] == 3
    assert by_dimension["primary_prevention"]["errors_captured"] == 2

    by_error_type = report["breakdowns"]["by_error_type"]
    assert by_error_type["false_positive"]["errors"] == 2
    assert by_error_type["false_negative"]["errors"] == 1
    assert by_error_type["false_negative"]["coverage_rate"] == pytest.approx(1.0)


def test_calibration_and_discrimination_are_populated(sample_df):
    report = build_confidence_report(sample_df, ConfidenceReportConfig(calibration_bins=5))

    calibration = report["calibration"]
    assert sum(row["count"] for row in calibration["bins"]) == 6
    assert 0.0 <= calibration["expected_calibration_error"] <= 1.0
    assert calibration["brier_score"] == pytest.approx(
        sum((c - k) ** 2 for c, k in zip([0.99, 0.95, 0.60, 0.55, 0.70, 0.98], [1, 1, 1, 0, 0, 0])) / 6
    )

    discrimination = report["discrimination"]
    assert 0.0 <= discrimination["auroc_error_detection"] <= 1.0
    assert len(discrimination["risk_coverage_curve"]) == 10


def test_flagged_sample_is_the_least_confident_first(sample_df):
    report = build_confidence_report(sample_df, ConfidenceReportConfig(flagged_sample_size=2))

    sample = report["flagged_sample"]
    assert [entry["confidence"] for entry in sample] == [0.55, 0.60]
    assert sample[0]["correct"] is False
    assert sample[0]["pred_label"] == "Yes"
    assert sample[0]["golden_label"] == "No"


def test_legacy_run_without_confidence_degrades_cleanly():
    df = make_frame(
        [
            {"indicator_id": "1.1", "value": 1.0, "pred_value": 0.0, "confidence": None},
            {"indicator_id": "1.2", "value": 0.0, "pred_value": 0.0, "confidence": None},
        ]
    )

    report = build_confidence_report(df)

    assert report["status"] == STATUS_NO_DATA
    assert report["confidence_source"]["availability"] == 0.0
    assert report["confidence_source"]["source_counts"] == {"unavailable": 2}
    assert report["baseline"]["error_count"] == 1
    assert report["coverage"] is None
    assert report["calibration"] is None
    assert report["flagged_sample"] == []


def test_partial_availability_reports_only_the_scored_subset():
    df = make_frame(
        [
            {"indicator_id": "1.1", "value": 1.0, "pred_value": 0.0, "confidence": 0.4},
            {"indicator_id": "1.2", "value": 0.0, "pred_value": 1.0, "confidence": None},
        ]
    )

    report = build_confidence_report(df, ConfidenceReportConfig(primary_threshold=0.9))

    assert report["confidence_source"]["availability"] == 0.5
    assert report["baseline"]["error_count"] == 2
    # Only the scored row participates in coverage.
    assert report["coverage"]["primary"]["errors_captured"] == 1
    assert report["coverage"]["primary"]["coverage_rate"] == 1.0


def test_low_confidence_frame_sorted_and_filtered(sample_df):
    flagged = build_low_confidence_frame(sample_df, 0.9)

    assert list(flagged["confidence"]) == [0.55, 0.60, 0.70]
    assert list(flagged["correct"]) == [False, True, False]
    assert "error_type" in flagged.columns

    assert build_low_confidence_frame(sample_df, 0.1).empty


def add_method_scores(df: pd.DataFrame, **methods: list[float]) -> pd.DataFrame:
    out = df.copy()
    for name, values in methods.items():
        out[f"conf_{name}"] = values
    return out


def test_each_method_is_scored_separately(sample_df):
    # margin ranks the three errors (rows 4.1, 4.2, 4.3) lowest; the raw
    # probability in `confidence` does not.
    df = add_method_scores(
        sample_df,
        logprobs=[0.99, 0.95, 0.60, 0.55, 0.70, 0.98],
        margin=[0.99, 0.98, 0.97, 0.10, 0.15, 0.20],
    )

    report = build_confidence_report(
        df, ConfidenceReportConfig(primary_threshold=0.9), primary_method="logprobs"
    )

    assert set(report["methods"]) == {"logprobs", "margin"}
    assert report["primary_method"] == "logprobs"
    assert report["confidence_source"]["methods_present"] == ["logprobs", "margin"]

    # Margin flags exactly the three wrong answers, logprobs only two of them.
    assert report["methods"]["margin"]["coverage"]["primary"]["coverage_rate"] == 1.0
    assert report["methods"]["logprobs"]["coverage"]["primary"]["coverage_rate"] == pytest.approx(2 / 3)


def test_method_comparison_ranks_by_discrimination(sample_df):
    df = add_method_scores(
        sample_df,
        logprobs=[0.99, 0.95, 0.60, 0.55, 0.70, 0.98],
        margin=[0.99, 0.98, 0.97, 0.10, 0.15, 0.20],
    )

    report = build_confidence_report(df, ConfidenceReportConfig(), primary_method="logprobs")
    ranking = report["method_comparison"]

    assert [row["method"] for row in ranking] == ["margin", "logprobs"]
    assert ranking[0]["auroc_error_detection"] == 1.0
    assert ranking[0]["is_primary"] is False
    assert ranking[1]["is_primary"] is True
    assert ranking[0]["description"]


def test_top_level_sections_mirror_the_primary_method(sample_df):
    df = add_method_scores(sample_df, logprobs=[0.99, 0.95, 0.60, 0.55, 0.70, 0.98])

    report = build_confidence_report(df, ConfidenceReportConfig(primary_threshold=0.9))

    assert report["coverage"]["primary"] == report["methods"]["logprobs"]["coverage"]["primary"]
    # The sample is only carried at the top level, not duplicated per method.
    assert report["flagged_sample"]
    assert "flagged_sample" not in report["methods"]["logprobs"]


def test_method_with_partial_availability_reports_its_own_scope(sample_df):
    df = add_method_scores(
        sample_df,
        logprobs=[0.99, 0.95, 0.60, 0.55, 0.70, 0.98],
        verbalized=[0.9, None, None, 0.5, None, None],
    )

    report = build_confidence_report(df, ConfidenceReportConfig(), primary_method="logprobs")

    verbalized = report["methods"]["verbalized"]
    assert verbalized["pairs_scored"] == 2
    assert verbalized["availability"] == pytest.approx(2 / 6)
    assert verbalized["errors_in_scope"] == 1
    assert report["methods"]["logprobs"]["pairs_scored"] == 6


def test_switching_primary_rescores_the_whole_report(sample_df):
    """Changing `primary` and re-running only comparison must recompute, not relabel.

    The `confidence` field in an evaluation report is whichever method was
    primary when the run was evaluated; it is stale the moment the config
    changes, so every headline section has to come from the requested method's
    own column.
    """
    df = add_method_scores(
        sample_df,
        logprobs=[0.99, 0.95, 0.60, 0.55, 0.70, 0.98],
        verbalized=[0.95, 0.93, 0.90, 0.40, 0.45, 0.50],
    )
    df["confidence_source"] = "logprobs"  # what evaluation captured as primary
    config = ConfidenceReportConfig(primary_threshold=0.9)

    as_logprobs = build_confidence_report(df, config, primary_method="logprobs")
    as_verbalized = build_confidence_report(df, config, primary_method="verbalized")

    assert as_verbalized["primary_method"] == "verbalized"
    # verbalized flags all three errors at 0.9; logprobs flags two of them.
    assert as_logprobs["coverage"]["primary"]["coverage_rate"] == pytest.approx(2 / 3)
    assert as_verbalized["coverage"]["primary"]["coverage_rate"] == pytest.approx(1.0)
    assert as_verbalized["coverage"] == as_verbalized["methods"]["verbalized"]["coverage"]
    for key in ("distribution", "calibration", "discrimination", "breakdowns"):
        assert as_logprobs[key] != as_verbalized[key], key
    assert as_verbalized["flagged_sample"][0]["confidence"] == 0.40


def test_low_confidence_csv_follows_the_same_primary(sample_df):
    df = add_method_scores(
        sample_df,
        logprobs=[0.99, 0.95, 0.60, 0.55, 0.70, 0.98],
        verbalized=[0.95, 0.93, 0.90, 0.40, 0.45, 0.50],
    )

    flagged = build_low_confidence_frame(df, 0.9, "verbalized")

    assert list(flagged["confidence"]) == [0.40, 0.45, 0.50]
    assert list(flagged["correct"]) == [False, False, False]


def test_requesting_an_uncaptured_method_warns_instead_of_silently_relabelling(sample_df):
    df = add_method_scores(sample_df, logprobs=[0.99, 0.95, 0.60, 0.55, 0.70, 0.98])
    df["confidence_source"] = "logprobs"

    report = build_confidence_report(
        df, ConfidenceReportConfig(primary_threshold=0.9), primary_method="verbalized"
    )

    source = report["confidence_source"]
    assert source["primary_method_requested"] == "verbalized"
    # The report never claims to be something it is not.
    assert report["primary_method"] == "logprobs"
    assert "was not captured during evaluation" in source["primary_method_warning"]
    assert "Re-run the evaluation step" in source["primary_method_warning"]


def test_pre_multi_method_report_keeps_its_label_without_warning(sample_df):
    # Reports written before per-method scores existed carry `confidence` and a
    # source label, but no conf_* columns. Asking for that same method is not a
    # mismatch and must not warn.
    df = sample_df.copy()
    df["confidence_source"] = "logprobs"

    report = build_confidence_report(df, ConfidenceReportConfig(), primary_method="logprobs")

    assert report["primary_method"] == "logprobs"
    assert report["confidence_source"]["primary_method_warning"] is None


def test_legacy_run_with_no_confidence_does_not_warn_about_the_method():
    df = make_frame(
        [
            {"indicator_id": "1.1", "value": 1.0, "pred_value": 0.0, "confidence": None},
            {"indicator_id": "1.2", "value": 0.0, "pred_value": 0.0, "confidence": None},
        ]
    )

    report = build_confidence_report(df, ConfidenceReportConfig(), primary_method="verbalized")

    # "no confidence data" is already the headline; a method warning on top
    # would only be noise.
    assert report["status"] == STATUS_NO_DATA
    assert report["confidence_source"]["primary_method_warning"] is None


def test_no_warning_when_the_requested_method_is_present(sample_df):
    df = add_method_scores(sample_df, verbalized=[0.95, 0.93, 0.90, 0.40, 0.45, 0.50])

    report = build_confidence_report(df, ConfidenceReportConfig(), primary_method="verbalized")

    assert report["confidence_source"]["primary_method_warning"] is None
    assert report["primary_method"] == "verbalized"


def test_primary_method_inferred_from_confidence_source(sample_df):
    df = add_method_scores(sample_df, margin=[0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
    df["confidence_source"] = "margin"

    report = build_confidence_report(df, ConfidenceReportConfig())

    assert report["primary_method"] == "margin"


def test_report_without_method_columns_still_scores_the_primary(sample_df):
    report = build_confidence_report(sample_df, ConfidenceReportConfig(primary_threshold=0.9))

    assert report["methods"] == {}
    assert report["method_comparison"] == []
    assert report["coverage"]["primary"]["coverage_rate"] == pytest.approx(2 / 3)


def test_report_is_json_serialisable(tmp_path, sample_df):
    report = build_confidence_report(sample_df, ConfidenceReportConfig())

    path = write_confidence_report(report, tmp_path / "confidence_report.json")
    loaded = json.loads(open(path, encoding="utf-8").read())

    assert loaded["coverage"]["primary"]["coverage_rate"] == pytest.approx(2 / 3)
