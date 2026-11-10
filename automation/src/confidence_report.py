"""Confidence analysis for the comparison step.

The evaluation step records, for every indicator, the probability the judge put
on the Yes/No answer it gave (probability = exp(logprob)). This module answers
the operational question that follows: if the least-confident predictions were
sent for human review, what share of the actual errors would that catch?

That share -- errors captured / total errors -- is the coverage rate, and it is
the headline number of `confidence_report.json`.
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from automation.src.config_loader import ConfidenceReportConfig
from automation.src.criteria import get_indicator_dimension

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_NO_DATA = "no_confidence_data"

#: Column prefix under which each method's own score is carried through the
#: golden join, e.g. `conf_logprobs`, `conf_margin`, `conf_verbalized`.
CONFIDENCE_METHOD_PREFIX = "conf_"
#: The same methods scored for the coverage answer (see `build_level_report`).
COVERAGE_METHOD_PREFIX = "covconf_"

DEFINITIONS = {
    "probability": "exp(logprob) of the tokens spelling the Yes/No answer",
    "coverage_rate": (
        "errors_captured / total_errors -- the share of wrong answers that fall "
        "below the confidence cutoff"
    ),
    "precision": (
        "errors_captured / flagged -- the share of flagged predictions that are "
        "actually wrong"
    ),
    "lift": "precision / baseline error rate -- how much better than random flagging",
    "auroc_error_detection": (
        "AUROC of (1 - confidence) as a score for predicting that an answer is "
        "wrong; 0.5 means the confidence signal carries no information"
    ),
    "methods": (
        "Each captured confidence method scored independently on the same pairs. "
        "The top-level coverage/calibration/discrimination sections belong to "
        "primary_method"
    ),
}

METHOD_DESCRIPTIONS = {
    "logprobs": "exp(summed logprob) of the tokens spelling the Yes/No answer",
    "margin": (
        "normalised gap between the winning and losing label from the answer "
        "token's top_logprobs; bounded rather than measured when the losing "
        "label falls outside the returned top-k"
    ),
    "verbalized": "the judge's own stated certainty, requested as a response field",
}

LOW_CONFIDENCE_CSV_COLUMNS = [
    "fullname",
    "filename",
    "indicator_id",
    "indicator_value",
    "dimension",
    "confidence",
    "answer_logprob",
    "p_yes",
    "pred_label",
    "golden_label",
    "correct",
    "error_type",
    "rationale",
]


def _label(value: float) -> str:
    return "Yes" if value == 1.0 else "No"


def _error_type(golden_value: float, pred_value: float) -> str:
    if golden_value == pred_value:
        return "match"
    return "false_positive" if pred_value == 1.0 else "false_negative"


def _ratio(numerator: float, denominator: float) -> float | None:
    """Guarded division: a rate over an empty population is unknown, not zero."""
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def prepare_frame(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns the report needs, without touching the input."""
    df = merged_df.copy()
    if "confidence" not in df.columns:
        df["confidence"] = np.nan
    for column in ("answer_logprob", "p_yes", "p_no", "margin"):
        if column not in df.columns:
            df[column] = np.nan
    if "confidence_source" not in df.columns:
        df["confidence_source"] = None

    # Clamped defensively: the API can return a marginally positive logprob for
    # an essentially-certain token, and a confidence above 1.0 would fall
    # outside the last calibration bin.
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").clip(0.0, 1.0)
    df["dimension"] = df["indicator_id"].map(get_indicator_dimension)
    df["correct"] = df["value"] == df["pred_value"]
    df["is_error"] = ~df["correct"]
    df["golden_label"] = df["value"].map(_label)
    df["pred_label"] = df["pred_value"].map(_label)
    df["error_type"] = [
        _error_type(golden, pred) for golden, pred in zip(df["value"], df["pred_value"])
    ]
    return df


def _baseline(df: pd.DataFrame) -> dict:
    value = df["value"]
    pred = df["pred_value"]
    return {
        "accuracy": _ratio(int(df["correct"].sum()), len(df)),
        "pairs": int(len(df)),
        "error_count": int(df["is_error"].sum()),
        "error_rate": _ratio(int(df["is_error"].sum()), len(df)),
        "confusion_matrix": {
            "true_positive": int(((value == 1.0) & (pred == 1.0)).sum()),
            "true_negative": int(((value == 0.0) & (pred == 0.0)).sum()),
            "false_positive": int(((value == 0.0) & (pred == 1.0)).sum()),
            "false_negative": int(((value == 1.0) & (pred == 0.0)).sum()),
        },
    }


def coverage_row(conf_df: pd.DataFrame, threshold: float, kind: str, **extra) -> dict:
    """Coverage and precision for one cutoff.

    A prediction is flagged when its confidence is strictly below `threshold`.
    """
    flagged_mask = conf_df["confidence"] < threshold
    flagged = int(flagged_mask.sum())
    total_errors = int(conf_df["is_error"].sum())
    captured = int((flagged_mask & conf_df["is_error"]).sum())
    unflagged = conf_df[~flagged_mask]
    base_error_rate = _ratio(total_errors, len(conf_df))
    precision = _ratio(captured, flagged)

    row = {
        "kind": kind,
        "threshold": float(threshold),
        "flagged": flagged,
        "flagged_share": _ratio(flagged, len(conf_df)),
        "errors_captured": captured,
        "coverage_rate": _ratio(captured, total_errors),
        "precision": precision,
        "lift": (
            _ratio(precision, base_error_rate)
            if precision is not None and base_error_rate
            else None
        ),
        "remaining_errors": total_errors - captured,
        "accuracy_on_unflagged": _ratio(int(unflagged["correct"].sum()), len(unflagged)),
    }
    row.update(extra)
    return row


def _distribution(conf_df: pd.DataFrame) -> dict:
    confidence = conf_df["confidence"]
    correct = conf_df[conf_df["correct"]]["confidence"]
    incorrect = conf_df[conf_df["is_error"]]["confidence"]
    percentiles = [1, 5, 10, 25, 50, 75, 90]
    return {
        "mean": _finite(confidence.mean()),
        "median": _finite(confidence.median()),
        "std": _finite(confidence.std()),
        "min": _finite(confidence.min()),
        "max": _finite(confidence.max()),
        "percentiles": {
            f"p{p}": _finite(np.percentile(confidence, p)) for p in percentiles
        },
        "mean_by_outcome": {
            "correct": _finite(correct.mean()) if len(correct) else None,
            "incorrect": _finite(incorrect.mean()) if len(incorrect) else None,
        },
    }


def _calibration(conf_df: pd.DataFrame, bins: int) -> dict:
    """Bucketed accuracy against mean confidence, plus ECE / MCE / Brier.

    `confidence` is P(the answer the judge gave) and `correct` says whether that
    answer matched the golden label, so the two are directly comparable.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    confidence = conf_df["confidence"].to_numpy(dtype=float)
    correct = conf_df["correct"].to_numpy(dtype=float)
    total = len(conf_df)

    rows = []
    ece = 0.0
    mce = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": 0,
                    "mean_confidence": None,
                    "accuracy": None,
                    "gap": None,
                }
            )
            continue

        mean_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        gap = abs(accuracy - mean_confidence)
        ece += (count / total) * gap
        mce = max(mce, gap)
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "gap": gap,
            }
        )

    return {
        "bins": rows,
        "expected_calibration_error": float(ece),
        "max_calibration_error": float(mce),
        "brier_score": float(np.mean((confidence - correct) ** 2)),
    }


def _discrimination(conf_df: pd.DataFrame) -> dict:
    errors = conf_df["is_error"].to_numpy(dtype=int)
    confidence = conf_df["confidence"].to_numpy(dtype=float)

    auroc = None
    if errors.min() != errors.max():
        auroc = float(roc_auc_score(errors, 1.0 - confidence))

    # Risk-coverage: keep the most confident predictions first, then track the
    # error rate among the ones kept.
    order = np.argsort(-confidence, kind="stable")
    sorted_errors = errors[order]
    cumulative_errors = np.cumsum(sorted_errors)
    sizes = np.arange(1, len(sorted_errors) + 1)
    selective_risk = cumulative_errors / sizes

    curve = []
    for fraction in np.arange(0.1, 1.0001, 0.1):
        index = max(int(round(fraction * len(sorted_errors))) - 1, 0)
        curve.append(
            {
                "coverage": round(float(fraction), 2),
                "selective_risk": float(selective_risk[index]),
            }
        )

    return {
        "auroc_error_detection": auroc,
        "aurc": float(selective_risk.mean()),
        "risk_coverage_curve": curve,
    }


def _by_dimension(conf_df: pd.DataFrame, threshold: float) -> dict:
    breakdown = {}
    for dimension, group in conf_df.groupby("dimension", dropna=False):
        label = dimension if pd.notna(dimension) else "unknown"
        flagged_mask = group["confidence"] < threshold
        errors = int(group["is_error"].sum())
        captured = int((flagged_mask & group["is_error"]).sum())
        flagged = int(flagged_mask.sum())
        breakdown[str(label)] = {
            "pairs": int(len(group)),
            "errors": errors,
            "mean_confidence": _finite(group["confidence"].mean()),
            "flagged": flagged,
            "errors_captured": captured,
            "coverage_rate": _ratio(captured, errors),
            "precision": _ratio(captured, flagged),
        }
    return breakdown


def _by_error_type(conf_df: pd.DataFrame, threshold: float) -> dict:
    flagged_mask = conf_df["confidence"] < threshold
    breakdown = {}
    for error_type in ("false_positive", "false_negative"):
        mask = conf_df["error_type"] == error_type
        errors = int(mask.sum())
        captured = int((mask & flagged_mask).sum())
        breakdown[error_type] = {
            "errors": errors,
            "errors_captured": captured,
            "coverage_rate": _ratio(captured, errors),
            "mean_confidence": (
                _finite(conf_df.loc[mask, "confidence"].mean()) if errors else None
            ),
        }
    return breakdown


def _flagged_sample(conf_df: pd.DataFrame, size: int) -> list[dict]:
    if size <= 0:
        return []
    lowest = conf_df.nsmallest(size, "confidence")
    return [
        {
            "filename": row.filename,
            "indicator_id": row.indicator_id,
            "dimension": None if pd.isna(row.dimension) else row.dimension,
            "confidence": _finite(row.confidence),
            "p_yes": _finite(row.p_yes),
            "p_no": _finite(row.p_no),
            "pred_label": row.pred_label,
            "golden_label": row.golden_label,
            "correct": bool(row.correct),
        }
        for row in lowest.itertuples()
    ]


def method_columns(df: pd.DataFrame) -> dict[str, str]:
    """Method name -> score column, for every method present in the frame."""
    return {
        column[len(CONFIDENCE_METHOD_PREFIX) :]: column
        for column in df.columns
        if column.startswith(CONFIDENCE_METHOD_PREFIX)
    }


def resolve_primary_column(
    df: pd.DataFrame, requested: str | None
) -> tuple[str, str | None, str | None]:
    """Pick the column the headline sections are computed from.

    The `confidence` field in an evaluation report is whichever method was
    primary *when the run was evaluated*. Comparison must not just relabel it:
    switching `primary` in config and re-running comparison has to re-score
    against the requested method's own column, or the report would claim one
    method while reporting another's numbers.

    Returns (column, resolved_method, warning).
    """
    columns = method_columns(df)

    if requested and requested in columns:
        return columns[requested], requested, None

    # Otherwise fall back to the stored `confidence` field and name it from the
    # source label. Reports written before per-method scores existed carry the
    # label but no columns, so the label alone identifies them.
    labels = {
        str(source).removesuffix("_merged")
        for source in df["confidence_source"].dropna().unique()
    } - {"unavailable", "unaligned"}
    known = labels.intersection(columns) if columns else labels
    stored = known.pop() if len(known) == 1 else None

    warning = None
    has_any_confidence = bool(df["confidence"].notna().any())
    if requested and requested != stored and has_any_confidence:
        warning = (
            f"Requested primary method {requested!r} was not captured during evaluation"
            f" (present: {sorted(columns) or [stored] if stored else 'none'}); reporting"
            f" {stored or 'the stored confidence field'} instead."
            " Re-run the evaluation step with that method enabled."
        )
    return "confidence", stored, warning


def _method_notes(scored: pd.DataFrame, column: str) -> dict | None:
    """Caveats specific to one method's numbers.

    For margin, how much of the signal was measured: when the losing label
    falls outside the returned `top_logprobs`, its probability is bounded
    rather than read, and the margin becomes a monotone function of the answer
    probability — which is why a margin that never observes its counterpart
    tends to rank identically to the logprobs method.
    """
    if column != f"{CONFIDENCE_METHOD_PREFIX}margin":
        return None
    if "margin_counterpart_observed" not in scored.columns or scored.empty:
        return None

    flags = scored["margin_counterpart_observed"].dropna()
    if flags.empty:
        return None
    observed = int(flags.astype(bool).sum())
    return {
        "counterpart_observed": observed,
        "counterpart_bounded": int(len(flags) - observed),
        "counterpart_observed_share": _ratio(observed, len(flags)),
    }


def analyse_scores(
    df: pd.DataFrame,
    column: str,
    config: ConfidenceReportConfig,
    *,
    include_sample: bool = False,
) -> dict:
    """Score one confidence column: coverage, calibration, discrimination.

    `df` must already be through `prepare_frame`. Rows without a score for this
    column sit out; `pairs_scored` says how many took part, which matters when
    comparing methods whose availability differs (margin, for instance, needs
    the answer token to have been aligned).
    """
    scored = df[df[column].notna()].copy()
    block: dict = {
        "pairs_scored": int(len(scored)),
        "availability": _ratio(len(scored), len(df)),
        "errors_in_scope": int(scored["is_error"].sum()) if len(scored) else 0,
        "notes": _method_notes(scored, column),
        "distribution": None,
        "coverage": None,
        "calibration": None,
        "discrimination": None,
        "breakdowns": None,
    }
    if scored.empty:
        return block

    # The analysis helpers all read a column literally named "confidence";
    # aliasing here keeps one implementation for every method.
    scored["confidence"] = scored[column].clip(0.0, 1.0)

    quantile_rows = []
    for quantile in config.quantiles:
        threshold = float(scored["confidence"].quantile(quantile))
        quantile_rows.append(
            coverage_row(scored, threshold, "quantile", quantile=float(quantile))
        )

    block["distribution"] = _distribution(scored)
    block["coverage"] = {
        "primary": coverage_row(scored, config.primary_threshold, "absolute"),
        "by_absolute_threshold": [
            coverage_row(scored, threshold, "absolute")
            for threshold in config.thresholds
        ],
        "by_quantile": quantile_rows,
    }
    block["calibration"] = _calibration(scored, config.calibration_bins)
    block["discrimination"] = _discrimination(scored)
    block["breakdowns"] = {
        "by_dimension": _by_dimension(scored, config.primary_threshold),
        "by_error_type": _by_error_type(scored, config.primary_threshold),
    }
    if include_sample:
        block["flagged_sample"] = _flagged_sample(scored, config.flagged_sample_size)
    return block


def _method_comparison(methods: dict[str, dict], primary: str | None) -> list[dict]:
    """Compact side-by-side ranking, best error-detector first.

    This is the table to read when deciding which method should be `primary`:
    AUROC says how well the score separates wrong answers from right ones,
    independent of where any threshold happens to sit.
    """
    rows = []
    for name, block in methods.items():
        coverage = (block.get("coverage") or {}).get("primary") or {}
        discrimination = block.get("discrimination") or {}
        calibration = block.get("calibration") or {}
        rows.append(
            {
                "method": name,
                "is_primary": name == primary,
                "description": METHOD_DESCRIPTIONS.get(name),
                "pairs_scored": block.get("pairs_scored", 0),
                "availability": block.get("availability"),
                "auroc_error_detection": discrimination.get("auroc_error_detection"),
                "coverage_rate": coverage.get("coverage_rate"),
                "precision": coverage.get("precision"),
                "lift": coverage.get("lift"),
                "expected_calibration_error": calibration.get("expected_calibration_error"),
                "brier_score": calibration.get("brier_score"),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["auroc_error_detection"] is None,
            -(row["auroc_error_detection"] or 0.0),
        ),
    )


def _build_yes_no_report(
    merged_df: pd.DataFrame,
    config: ConfidenceReportConfig | None = None,
    *,
    model: str | None = None,
    temperature: float | None = None,
    evaluation_dir: str | None = None,
    primary_method: str | None = None,
) -> dict:
    """Build the confidence report from the golden-vs-prediction merge.

    Reports on the subset of pairs that carry a confidence score. Runs whose
    evaluation predates this module have none, and get a `no_confidence_data`
    report rather than an error.
    """
    config = config or ConfidenceReportConfig()
    df = prepare_frame(merged_df)

    source_counts = (
        df["confidence_source"].fillna("unavailable").value_counts().to_dict()
    )

    columns = method_columns(df)
    primary_column, resolved_method, warning = resolve_primary_column(df, primary_method)
    if warning:
        logger.warning(warning)
    conf_df = df[df[primary_column].notna()]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": STATUS_OK if len(conf_df) else STATUS_NO_DATA,
        "definitions": DEFINITIONS,
        "primary_method": resolved_method,
        "confidence_source": {
            "formula": "probability = exp(logprob)",
            "model": model,
            "temperature": temperature,
            "evaluation_dir": evaluation_dir,
            "methods_present": sorted(columns),
            "primary_method_requested": primary_method,
            "primary_method_warning": warning,
            "pairs_total": int(len(df)),
            "pairs_with_confidence": int(len(conf_df)),
            "availability": _ratio(len(conf_df), len(df)),
            "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        },
        "baseline": _baseline(df),
        "distribution": None,
        "coverage": None,
        "calibration": None,
        "discrimination": None,
        "breakdowns": None,
        "flagged_sample": [],
        "methods": {},
        "method_comparison": [],
    }

    if conf_df.empty:
        return report

    # The primary method's own analysis is promoted to the top level, so a
    # reader who only cares about the headline never has to know which method
    # produced it.
    primary_block = analyse_scores(df, primary_column, config, include_sample=True)
    for key in ("distribution", "coverage", "calibration", "discrimination", "breakdowns"):
        report[key] = primary_block[key]
    report["flagged_sample"] = primary_block.get("flagged_sample", [])

    report["methods"] = {
        name: analyse_scores(df, column, config) for name, column in sorted(columns.items())
    }
    report["method_comparison"] = _method_comparison(report["methods"], resolved_method)
    return report


LEVEL_NAMES = {0.0: "No", 1.0: "Referenced only", 2.0: "Substantive"}

LEVEL_DEFINITIONS = {
    "scope": (
        "The coverage answer (e.g. Substantive vs Referenced only) is scored on its own, independently of the Yes / No "
        "answer. Only pairs where the golden label AND the prediction are both above No take part: the level of "
        "something that was not found cannot be judged."
    ),
    "confidence": "the judge's confidence in the coverage value it gave, by the same methods as the Yes / No one",
    "correct": "the predicted level equals the golden level",
    "level_over": "predicted a higher level than the golden label",
    "level_under": "predicted a lower level than the golden label",
}


def _level_error_type(golden: float, pred: float) -> str:
    if golden == pred:
        return "correct"
    return "level_over" if pred > golden else "level_under"


def _level_frame(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Pairs on which a level can be judged, shaped so the Yes / No analysis helpers work on them unchanged.

    `value` / `pred_value` become the golden and predicted LEVELS, `confidence` and the `conf_<method>` columns become
    the coverage ones, and the Yes / No scores are dropped so nothing of them can leak in.
    """
    df = merged_df.copy()
    both_positive = (df["pred_value"] == 1.0) & (df["golden_level"] > 0) & df["pred_level"].notna()
    df = df[both_positive].copy()

    df = df.drop(columns=[c for c in df.columns if c.startswith(CONFIDENCE_METHOD_PREFIX)])
    df["value"] = df["golden_level"]
    df["pred_value"] = df["pred_level"]
    df["confidence"] = df.get("coverage_confidence", np.nan)
    df["confidence_source"] = df.get("coverage_confidence_source")
    for column in ("answer_logprob", "p_yes", "p_no", "margin"):
        df[column] = np.nan
    if "coverage_answer_logprob" in df.columns:
        df["answer_logprob"] = df["coverage_answer_logprob"]
    if "coverage_margin" in df.columns:
        df["margin"] = df["coverage_margin"]
    for column in [c for c in df.columns if c.startswith(COVERAGE_METHOD_PREFIX)]:
        df[CONFIDENCE_METHOD_PREFIX + column[len(COVERAGE_METHOD_PREFIX):]] = df[column]

    df = prepare_frame(df)
    df["golden_label"] = df["value"].map(lambda v: LEVEL_NAMES.get(float(v), str(v)))
    df["pred_label"] = df["pred_value"].map(lambda v: LEVEL_NAMES.get(float(v), str(v)))
    df["error_type"] = [_level_error_type(g, p) for g, p in zip(df["value"], df["pred_value"])]
    return df


def build_level_report(
    merged_df: pd.DataFrame,
    config: ConfidenceReportConfig | None = None,
    *,
    primary_method: str | None = None,
) -> dict | None:
    """The confidence report of the coverage answer, independent of the Yes / No one.

    None when the evaluation carries no coverage at all (so a run without `evaluation.coverage` gets exactly the
    report it always had). Without golden levels (no ``golden_level`` column: the golden is Yes / No only) there is
    nothing to be correct about, and the section says so.
    """
    if "pred_level" not in merged_df.columns:
        return None
    config = config or ConfidenceReportConfig()
    report: dict = {"definitions": LEVEL_DEFINITIONS, "status": STATUS_OK, "primary_method": primary_method}

    if "golden_level" not in merged_df.columns:
        report["status"] = "no_level_labels"
        report["note"] = "the golden dataset has no levels above 1, so the level cannot be scored"
        return report

    df = _level_frame(merged_df)
    columns = method_columns(df)
    primary_column, resolved_method, warning = resolve_primary_column(df, primary_method)
    conf_df = df[df[primary_column].notna()]
    report.update(
        primary_method=resolved_method,
        primary_method_warning=warning,
        pairs=int(len(df)),
        pairs_with_confidence=int(len(conf_df)),
        availability=_ratio(len(conf_df), len(df)),
        baseline={
            "accuracy": _ratio(int(df["correct"].sum()), len(df)),
            "pairs": int(len(df)),
            "error_count": int(df["is_error"].sum()),
            "error_rate": _ratio(int(df["is_error"].sum()), len(df)),
            "by_error_type": {str(k): int(v) for k, v in df["error_type"].value_counts().to_dict().items()},
        },
    )
    if df.empty:
        report["status"] = "no_pairs_above_no"
        return report
    if conf_df.empty:
        report["status"] = STATUS_NO_DATA
        return report

    primary_block = analyse_scores(df, primary_column, config, include_sample=True)
    for key in ("distribution", "coverage", "calibration", "discrimination", "breakdowns", "flagged_sample"):
        report[key] = primary_block.get(key)
    report["methods"] = {name: analyse_scores(df, column, config) for name, column in sorted(columns.items())}
    report["method_comparison"] = _method_comparison(report["methods"], resolved_method)
    return report


def build_confidence_report(
    merged_df: pd.DataFrame,
    config: ConfidenceReportConfig | None = None,
    **context,
) -> dict:
    """The confidence report: the Yes / No answer's, plus, when the evaluation carries a coverage answer, its own
    under the key ``level`` (scored independently; see `build_level_report`). Without coverage the report is exactly
    the Yes / No one."""
    report = _build_yes_no_report(merged_df, config, **context)
    level = build_level_report(merged_df, config, primary_method=context.get("primary_method"))
    if level is not None:
        report["level"] = level
    return report


def build_low_confidence_frame(
    merged_df: pd.DataFrame,
    threshold: float,
    primary_method: str | None = None,
) -> pd.DataFrame:
    """Every prediction below `threshold`, with whether it was actually wrong.

    Scored on the same column the report's headline sections use, so the
    worklist and the coverage number it comes from can never disagree.
    """
    df = prepare_frame(merged_df)
    primary_column, _, _ = resolve_primary_column(df, primary_method)
    flagged = df[df[primary_column].notna() & (df[primary_column] < threshold)].copy()
    if flagged.empty:
        return flagged
    flagged["confidence"] = flagged[primary_column]
    flagged = flagged.sort_values("confidence")
    for column in LOW_CONFIDENCE_CSV_COLUMNS:
        if column not in flagged.columns:
            flagged[column] = None
    return flagged[LOW_CONFIDENCE_CSV_COLUMNS]


def write_confidence_report(report: dict, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return str(path)
