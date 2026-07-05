import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from quality_eval.v1.criteria import get_indicator_dimension


def parse_report_timestamp(file_path: str, metadata: dict) -> datetime:
    evaluated_at = metadata.get("evaluated_at")
    if evaluated_at:
        parsed = datetime.fromisoformat(evaluated_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    base = os.path.basename(file_path)
    prefix = base.split("-", 1)[0]
    if len(prefix) == 14:
        try:
            return datetime.strptime(prefix, "%d%m%Y%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    return datetime.min.replace(tzinfo=timezone.utc)


def load_report_files(llm_dir_or_file: str) -> list[tuple[datetime, str, dict]]:
    if os.path.isdir(llm_dir_or_file):
        files = [
            os.path.join(llm_dir_or_file, name)
            for name in os.listdir(llm_dir_or_file)
            if name.endswith(".json")
        ]
    else:
        files = [llm_dir_or_file]

    reports: list[tuple[datetime, str, dict]] = []
    for file_path in files:
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            timestamp = parse_report_timestamp(file_path, data)
            reports.append((timestamp, file_path, data))
        except Exception as exc:
            print(f"Loading JSON failed {file_path}: {exc}")
    return reports


def deduplicate_reports(reports: list[tuple[datetime, str, dict]]) -> list[tuple[datetime, str, dict]]:
    latest_by_policy: dict[str, tuple[datetime, str, dict]] = {}
    for timestamp, file_path, data in reports:
        policy_file = data.get("policy_file")
        if not policy_file:
            continue
        current = latest_by_policy.get(policy_file)
        if current is None or timestamp > current[0]:
            latest_by_policy[policy_file] = (timestamp, file_path, data)
    return list(latest_by_policy.values())


def reports_to_dataframe(reports: list[tuple[datetime, str, dict]]) -> pd.DataFrame:
    llm_data = []
    for _, file_path, data in reports:
        filename = data.get("policy_file")
        eval_results = data.get("evaluation_results", {})
        for _, details in eval_results.items():
            llm_data.append(
                {
                    "filename": filename,
                    "indicator_id": str(details.get("id")),
                    "pred_value": 1.0 if details.get("included") == "Yes" else 0.0,
                    "source_report": file_path,
                }
            )
    return pd.DataFrame(llm_data)


def load_llm_results(llm_dir_or_file: str) -> pd.DataFrame:
    reports = load_report_files(llm_dir_or_file)
    deduped = deduplicate_reports(reports)
    if len(reports) != len(deduped):
        print(
            f"Deduplicated {len(reports)} reports to {len(deduped)} "
            f"(latest per policy_file)."
        )
    return reports_to_dataframe(deduped)


def load_golden_dataframe(csv_path: str) -> pd.DataFrame:
    golden_df = pd.read_csv(csv_path, dtype={"indicator_id": str})
    golden_df["filename"] = golden_df["filename"].str.strip()
    golden_df["indicator_id"] = golden_df["indicator_id"].astype(str).str.strip()
    golden_df["value"] = golden_df["value"].astype(float)
    return golden_df


def filter_golden_to_evaluated(golden_df: pd.DataFrame, evaluated_policies: list[str]) -> pd.DataFrame:
    filtered = golden_df[golden_df["filename"].isin(evaluated_policies)].copy()
    if filtered.empty:
        return filtered

    if "year" not in filtered.columns:
        return filtered.drop_duplicates(subset=["filename", "indicator_id"], keep="last")

    latest_year = filtered.groupby("filename")["year"].transform("max")
    return filtered[filtered["year"] == latest_year].drop_duplicates(
        subset=["filename", "indicator_id"], keep="last"
    )


def _value_to_label(value: float) -> str:
    return "Yes" if value == 1.0 else "No"


def _classify_error_type(golden_value: float, pred_value: float) -> str:
    if golden_value == 0.0 and pred_value == 1.0:
        return "false_positive"
    if golden_value == 1.0 and pred_value == 0.0:
        return "false_negative"
    return "match"


def build_error_analysis_df(errors_df: pd.DataFrame) -> pd.DataFrame:
    output = errors_df.copy()
    output["dimension"] = output["indicator_id"].map(get_indicator_dimension)
    output["golden_label"] = output["value"].map(_value_to_label)
    output["pred_label"] = output["pred_value"].map(_value_to_label)
    output["error_type"] = output.apply(
        lambda row: _classify_error_type(row["value"], row["pred_value"]),
        axis=1,
    )
    columns = [
        "fullname",
        "filename",
        "indicator_id",
        "indicator_value",
        "dimension",
        "golden_label",
        "pred_label",
        "value",
        "pred_value",
        "error_type",
    ]
    return output[columns]


def find_unmatched_indicator_pairs(
    golden_filtered: pd.DataFrame,
    llm_df: pd.DataFrame,
) -> pd.DataFrame:
    pair_cols = ["filename", "indicator_id"]
    golden_pairs = golden_filtered[pair_cols].drop_duplicates()
    llm_pairs = llm_df[pair_cols].drop_duplicates()

    outer = golden_pairs.merge(llm_pairs, on=pair_cols, how="outer", indicator=True)
    unmatched = outer[outer["_merge"] != "both"].copy()
    if unmatched.empty:
        return unmatched

    unmatched["match_status"] = unmatched["_merge"].map(
        {"left_only": "golden_only", "right_only": "llm_only"}
    )
    unmatched = unmatched.drop(columns=["_merge"])
    unmatched["dimension"] = unmatched["indicator_id"].map(get_indicator_dimension)

    golden_meta = golden_filtered[
        ["filename", "indicator_id", "fullname", "indicator_value", "value"]
    ].drop_duplicates(subset=pair_cols, keep="last")
    unmatched = unmatched.merge(golden_meta, on=pair_cols, how="left")
    return unmatched[
        [
            "fullname",
            "filename",
            "indicator_id",
            "indicator_value",
            "dimension",
            "match_status",
            "value",
        ]
    ]


def build_error_summary(errors_df: pd.DataFrame) -> dict:
    if errors_df.empty:
        return {
            "total_errors": 0,
            "by_error_type": {},
            "by_dimension": {},
            "by_indicator": {},
        }

    analysis = build_error_analysis_df(errors_df)
    by_error_type = analysis["error_type"].value_counts().to_dict()
    by_dimension = analysis.groupby("dimension")["indicator_id"].count().to_dict()
    by_indicator = (
        analysis.groupby(["indicator_id", "indicator_value", "dimension"])["filename"]
        .count()
        .reset_index(name="error_count")
        .sort_values(["error_count", "indicator_id"], ascending=[False, True])
        .to_dict(orient="records")
    )
    return {
        "total_errors": int(len(analysis)),
        "by_error_type": {str(k): int(v) for k, v in by_error_type.items()},
        "by_dimension": {str(k): int(v) for k, v in by_dimension.items()},
        "by_indicator": by_indicator,
    }


def build_dimension_metrics(merged_df: pd.DataFrame) -> dict:
    merged_df = merged_df.copy()
    merged_df["dimension"] = merged_df["indicator_id"].map(get_indicator_dimension)

    by_dimension = {}
    for dimension, group in merged_df.groupby("dimension", dropna=False):
        label = dimension if pd.notna(dimension) else "unknown"
        correct = (group["value"] == group["pred_value"]).sum()
        total = len(group)
        by_dimension[label] = {
            "accuracy": float(correct / total) if total else 0.0,
            "matched": int(total),
            "correct": int(correct),
        }
    return by_dimension


def calculate_metrics(
    csv_path: str, llm_df: pd.DataFrame
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict | None]:
    golden_df = load_golden_dataframe(csv_path)

    llm_df = llm_df.copy()
    llm_df["filename"] = llm_df["filename"].str.strip()
    llm_df["indicator_id"] = llm_df["indicator_id"].astype(str).str.strip()

    evaluated_policies = sorted(llm_df["filename"].unique())
    golden_total = golden_df["filename"].nunique()
    golden_filtered = filter_golden_to_evaluated(golden_df, evaluated_policies)

    merged_df = pd.merge(
        golden_filtered,
        llm_df,
        on=["filename", "indicator_id"],
        how="inner",
    )

    duplicate_pairs = merged_df.duplicated(["filename", "indicator_id"]).sum()
    if duplicate_pairs:
        print(
            f"Warning: {duplicate_pairs} duplicate (filename, indicator_id) pairs after merge.",
            flush=True,
        )

    print(
        f"Evaluated {len(evaluated_policies)} of {golden_total} golden policies; "
        f"{len(merged_df)} indicator pairs matched."
    )

    if merged_df.empty:
        print("Error: No matching data. Please check the CSV and JSON.")
        return None, None, None

    unmatched_df = find_unmatched_indicator_pairs(golden_filtered, llm_df)
    if not unmatched_df.empty:
        golden_only = int((unmatched_df["match_status"] == "golden_only").sum())
        llm_only = int((unmatched_df["match_status"] == "llm_only").sum())
        print(
            f"Unmatched indicator pairs: {len(unmatched_df)} "
            f"(golden_only={golden_only}, llm_only={llm_only})"
        )

    missing_golden_value = merged_df[merged_df["value"].isna()]
    missing_golden_value_count = int(len(missing_golden_value))
    if missing_golden_value_count:
        pairs = ", ".join(
            f"{row.filename}/{row.indicator_id}"
            for row in missing_golden_value.itertuples()
        )
        print(
            f"Warning: excluding {missing_golden_value_count} matched pair(s) with missing "
            f"golden value (NaN) in {csv_path}: {pairs}"
        )
        merged_df = merged_df[merged_df["value"].notna()].copy()

    if merged_df.empty:
        print("Error: No matching data after excluding missing golden values.")
        return None, None, None

    y_true = merged_df["value"]
    y_pred = merged_df["pred_value"]

    accuracy = (y_true == y_pred).mean()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print("==================================================")
    print("Evaluation Result")
    print("==================================================")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Matched Pairs: {len(merged_df)}")
    print("--------------------------------------------------")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["No (0.0)", "Yes (1.0)"]))
    print("Confusion Matrix:")
    print(f"True Negative: {tn}")
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")
    print(f"True Positive: {tp}")
    print("--------------------------------------------------")

    by_dimension = build_dimension_metrics(merged_df)
    print("Accuracy by dimension:")
    for dimension, stats in sorted(by_dimension.items()):
        print(f"  {dimension}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['matched']})")

    errors = merged_df[merged_df["value"] != merged_df["pred_value"]]
    error_summary = build_error_summary(errors)

    metrics_summary = {
        "overall": {
            "accuracy": float(accuracy),
            "matched_pairs": int(len(merged_df)),
            "evaluated_policies": len(evaluated_policies),
            "golden_policies_total": int(golden_total),
            "evaluated_policy_files": evaluated_policies,
            "value_mismatch_count": int(len(errors)),
            "unmatched_pair_count": int(len(unmatched_df)),
            "missing_golden_value_count": missing_golden_value_count,
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "by_dimension": by_dimension,
        "error_summary": error_summary,
    }

    if not unmatched_df.empty:
        metrics_summary["unmatched_pairs"] = {
            "golden_only": int((unmatched_df["match_status"] == "golden_only").sum()),
            "llm_only": int((unmatched_df["match_status"] == "llm_only").sum()),
        }

    return errors, unmatched_df, metrics_summary


def run_accuracy_evaluation(
    golden_csv: str,
    llm_input: str,
    export_errors: str = "error_analysis.csv",
    export_metrics: str | None = None,
) -> dict | None:
    llm_df = load_llm_results(llm_input)
    if llm_df.empty:
        raise ValueError("No LLM results could be loaded.")

    print("Evaluating...")
    errors_df, unmatched_df, metrics_summary = calculate_metrics(golden_csv, llm_df)
    if metrics_summary is None:
        return None

    metrics_path = export_metrics
    if metrics_path is None:
        metrics_path = str(Path(export_errors).with_suffix(".metrics.json"))

    export_dir = Path(export_errors).parent
    unmatched_path = str(export_dir / "unmatched_indicators.csv")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
    print(f"Metrics summary saved: {metrics_path}")

    if unmatched_df is not None and not unmatched_df.empty:
        unmatched_df.to_csv(unmatched_path, index=False)
        print(f"Unmatched indicators saved: {unmatched_path}")
    elif Path(unmatched_path).exists():
        Path(unmatched_path).unlink()

    if errors_df is not None and not errors_df.empty:
        error_output = build_error_analysis_df(errors_df)
        error_output.to_csv(export_errors, index=False)
        print(f"Error analysis saved: {export_errors}")
    else:
        print("The JSON report is completely correct.")

    return {
        "metrics_summary": metrics_summary,
        "metrics_path": metrics_path,
        "errors_path": export_errors if errors_df is not None and not errors_df.empty else None,
        "unmatched_path": unmatched_path if unmatched_df is not None and not unmatched_df.empty else None,
        "error_count": int(len(errors_df)) if errors_df is not None else 0,
        "unmatched_count": int(len(unmatched_df)) if unmatched_df is not None else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Accuracy Checker Against Golden Dataset."
    )
    parser.add_argument("-g", "--golden_csv", required=True, help="Golden dataset CSV path")
    parser.add_argument(
        "-l",
        "--llm_input",
        required=True,
        help="LLM JSON report, or folder with multiple JSON reports",
    )
    parser.add_argument("-e", "--export_errors", default="error_analysis.csv", help="Export file path")
    parser.add_argument(
        "-m",
        "--export_metrics",
        default=None,
        help="Export metrics summary JSON path (default: alongside error CSV)",
    )
    args = parser.parse_args()

    print("Loading JSON report...")
    try:
        result = run_accuracy_evaluation(
            golden_csv=args.golden_csv,
            llm_input=args.llm_input,
            export_errors=args.export_errors,
            export_metrics=args.export_metrics,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    if result is None:
        return


if __name__ == "__main__":
    main()
