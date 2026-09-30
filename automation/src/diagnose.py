"""Discrepancy diagnosis stage: for every (filename, indicator_id) mismatch
recorded by the comparison step, diagnose the root cause using the original
Indonesian text, the golden/predicted labels, the RAG evidence candidates
persisted by the evaluation step (see run_eval.py:_write_candidates_file),
and the translated snippet + rationale the judge produced.

Final stage of the pipeline, run after comparison. Writes one aggregated
JSON report per policy file to
`data/automation/<run_id>/results/diagnosis/<stem>.json`.
A no-op (no report files written) when comparison found zero mismatches.
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Literal

import pandas as pd
from pydantic import BaseModel

from automation.src.concurrency import ConcurrencyLimiter
from automation.src.config_loader import (
    ResolvedPipelineConfig,
    get_run_dir,
    resolve_mid_product_dir,
    resolve_results_dir,
)
from automation.src.diagnose_prompt_builder import build_diagnosis_prompt
from automation.src.evaluate_accuracy import deduplicate_reports, load_report_files
from automation.src.failure_log import clear_failure, record_failure
from automation.src.llm.wrapper import AzureLLMWrapper, format_api_error

logger = logging.getLogger(__name__)

RootCause = Literal[
    "translation_quality", "evaluation_failure", "rag_candidate_issue", "golden_label_issue"
]


class DiscrepancyDiagnosisItem(BaseModel):
    id: str
    root_causes: List[RootCause]
    rationale: str


class DiscrepancyDiagnosisResponse(BaseModel):
    diagnosis_results: List[DiscrepancyDiagnosisItem]


@dataclass
class PolicyDiagnosisResult:
    policy_file: str
    status: str  # "succeeded" | "partial" | "failed"
    report: dict | None = None
    error: str | None = None
    error_type: str | None = None
    error_details: dict | None = None


def _load_error_analysis(comparison_dir: Path) -> "pd.DataFrame | None":
    path = comparison_dir / "error_analysis.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"indicator_id": str})


def _group_by_filename(df: "pd.DataFrame") -> dict[str, list[dict]]:
    return {filename: group.to_dict(orient="records") for filename, group in df.groupby("filename")}


def _load_latest_evaluation_results(evaluation_dir: Path) -> dict[str, dict]:
    reports = load_report_files(str(evaluation_dir))
    deduped = deduplicate_reports(reports)
    return {
        data["policy_file"]: data.get("evaluation_results", {})
        for _, _, data in deduped
        if data.get("policy_file")
    }


def _load_candidates(rag_candidates_dir: Path, policy_filename: str) -> dict[str, list[dict]]:
    path = rag_candidates_dir / f"{Path(policy_filename).stem}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No RAG candidates found at {path}. Re-run the evaluation step (with --force) "
            "for this run to generate mid_product/rag_candidates/ before running "
            "discrepancy_diagnosis."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("candidates", {})


def _merge_diagnosis(
    discrepancy_rows: list[dict],
    evaluation_results: dict,
    candidates_by_indicator: dict[str, list[dict]],
    diagnosis_by_id: dict[str, dict],
    unresolved_indicators: list[str],
) -> tuple[dict, list[str]]:
    diagnoses: dict[str, dict] = {}
    missing_ids: list[str] = []

    for row in discrepancy_rows:
        indicator_id = row["indicator_id"]
        eval_entry = evaluation_results.get(indicator_id, {})
        candidates = candidates_by_indicator.get(indicator_id, [])

        if indicator_id in unresolved_indicators:
            diagnosis_status = "unresolved_indicator"
            llm_item = None
        else:
            llm_item = diagnosis_by_id.get(indicator_id)
            if llm_item is None:
                diagnosis_status = "missing_from_llm_response"
                missing_ids.append(indicator_id)
            else:
                diagnosis_status = "ok"

        diagnoses[indicator_id] = {
            "indicator": row.get("indicator_value") or eval_entry.get("indicator"),
            "dimension": row.get("dimension"),
            "golden_label": row.get("golden_label"),
            "pred_label": row.get("pred_label"),
            "error_type": row.get("error_type"),
            "translated_snippet": eval_entry.get("evidence"),
            "evaluator_rationale": eval_entry.get("rationale"),
            "evidence_candidates": candidates,
            "root_causes": llm_item.get("root_causes", []) if llm_item else [],
            "rationale": llm_item.get("rationale") if llm_item else None,
            "diagnosis_status": diagnosis_status,
        }

    return {"diagnoses": diagnoses}, missing_ids


def _diagnose_policy(
    policy_filename: str,
    discrepancy_rows: list[dict],
    evaluation_results_by_policy: dict[str, dict],
    input_dir: Path,
    rag_candidates_dir: Path,
    template_text: str,
    complete_fn: Callable[[str], dict],
) -> PolicyDiagnosisResult:
    try:
        original_path = input_dir / policy_filename
        if not original_path.exists():
            raise FileNotFoundError(f"Original policy text not found: {original_path}")
        original_text = original_path.read_text(encoding="utf-8")

        evaluation_results = evaluation_results_by_policy.get(policy_filename, {})
        candidates_by_indicator = _load_candidates(rag_candidates_dir, policy_filename)

        unresolved_indicators = [
            row["indicator_id"]
            for row in discrepancy_rows
            if row["indicator_id"] not in candidates_by_indicator
        ]
        diagnosable_rows = [
            row for row in discrepancy_rows if row["indicator_id"] not in unresolved_indicators
        ]

        diagnosis_by_id: dict[str, dict] = {}
        if diagnosable_rows:
            prompt = build_diagnosis_prompt(
                original_text=original_text,
                discrepancy_rows=diagnosable_rows,
                evaluation_results=evaluation_results,
                candidates_by_indicator=candidates_by_indicator,
                template_text=template_text,
            )
            llm_response = complete_fn(prompt)
            diagnosis_by_id = {item["id"]: item for item in llm_response.get("diagnosis_results", [])}

        report, missing_ids = _merge_diagnosis(
            discrepancy_rows,
            evaluation_results,
            candidates_by_indicator,
            diagnosis_by_id,
            unresolved_indicators,
        )
        report["missing_diagnoses"] = missing_ids
        report["unresolved_indicators"] = unresolved_indicators

        status = "succeeded" if not missing_ids and not unresolved_indicators else "partial"
        logger.info("[%s] diagnosis %s (%d discrepancies)", policy_filename, status, len(discrepancy_rows))
        return PolicyDiagnosisResult(policy_file=policy_filename, status=status, report=report)

    except Exception as exc:
        error_type, details = format_api_error(exc)
        logger.error("[%s] diagnosis failed [%s]: %s", policy_filename, error_type, exc)
        return PolicyDiagnosisResult(
            policy_file=policy_filename,
            status="failed",
            error=str(exc),
            error_type=error_type,
            error_details=details,
        )


def _empty_result(output_dir: Path, wrapper: AzureLLMWrapper, elapsed: float, skipped_reason: str | None = None) -> dict:
    result = {
        "counts": {
            "total": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped": 0,
            "saved_reports": 0,
            "discrepancies_total": 0,
        },
        "failed_policies": [],
        "elapsed_s": elapsed,
        "token_usage": dict(wrapper.token_usage),
        "output_dir": str(output_dir),
    }
    if skipped_reason:
        result["skipped_reason"] = skipped_reason
    return result


def run_diagnosis_step(
    run_id: str,
    config: ResolvedPipelineConfig,
    wrapper: AzureLLMWrapper,
    *,
    limiter: ConcurrencyLimiter,
    small_scale: bool = False,
    allow_partial: bool = False,
    force: bool = False,
) -> dict:
    run_dir = get_run_dir(run_id)
    comparison_dir = resolve_results_dir(run_dir, "comparison")
    evaluation_dir = resolve_results_dir(run_dir, "evaluation")
    rag_candidates_dir = resolve_mid_product_dir(run_dir, "rag_candidates")
    output_dir = resolve_results_dir(run_dir, "diagnosis")

    start = time.time()
    df = _load_error_analysis(comparison_dir)
    if df is None:
        elapsed = round(time.time() - start, 2)
        return _empty_result(
            output_dir,
            wrapper,
            elapsed,
            skipped_reason=(
                "no error_analysis.csv (comparison found zero mismatches, "
                "or comparison has not been run)"
            ),
        )

    if small_scale:
        df = df[df["filename"].isin([f"{stem}.txt" for stem in config.paths.small_scale_stems])]

    if df.empty:
        elapsed = round(time.time() - start, 2)
        return _empty_result(output_dir, wrapper, elapsed, skipped_reason="no discrepancies in scope")

    discrepancies_total = int(len(df))
    grouped = _group_by_filename(df)

    output_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    remaining: dict[str, list[dict]] = {}
    for filename, rows in grouped.items():
        report_path = output_dir / f"{Path(filename).stem}.json"
        if not force and report_path.exists():
            skipped += 1
            logger.info("[%s] skipped (diagnosis report exists)", filename)
        else:
            remaining[filename] = rows
    grouped = remaining

    total_candidates = len(grouped) + skipped

    evaluation_results_by_policy = _load_latest_evaluation_results(evaluation_dir)
    template_text = config.discrepancy_diagnosis_template_path.read_text(encoding="utf-8")

    def complete_fn(prompt: str) -> dict:
        return wrapper.complete_structured(prompt, DiscrepancyDiagnosisResponse)

    saved_paths: list[str] = []
    failed_policies: list[str] = []

    if grouped:
        filenames = list(grouped.keys())
        tasks = [
            lambda fn=fn: _diagnose_policy(
                fn,
                grouped[fn],
                evaluation_results_by_policy,
                config.paths.input_dir,
                rag_candidates_dir,
                template_text,
                complete_fn,
            )
            for fn in filenames
        ]
        results = limiter.run_parallel(tasks)

        for filename, result in zip(filenames, results, strict=True):
            if isinstance(result, BaseException):
                error_type, details = format_api_error(result)
                entry = {
                    "policy_file": filename,
                    "error_type": error_type,
                    "message": str(result),
                    "details": details,
                    "attempts": 1,
                }
                record_failure(run_id, "discrepancy_diagnosis", entry)
                logger.error("[%s] diagnosis failed [%s]: %s", filename, error_type, result)
                failed_policies.append(filename)
                continue

            if result.status == "failed":
                entry = {
                    "policy_file": filename,
                    "error_type": result.error_type or "Exception",
                    "message": result.error or "unknown error",
                    "details": result.error_details or {},
                    "attempts": 1,
                }
                record_failure(run_id, "discrepancy_diagnosis", entry)
                failed_policies.append(filename)
                continue

            report = result.report or {}
            if result.status == "partial" and not allow_partial:
                entry = {
                    "policy_file": filename,
                    "error_type": "DiagnosisIncomplete",
                    "message": (
                        f"missing_diagnoses={report.get('missing_diagnoses', [])} "
                        f"unresolved_indicators={report.get('unresolved_indicators', [])}"
                    ),
                    "details": {},
                    "attempts": 1,
                }
                record_failure(run_id, "discrepancy_diagnosis", entry)
                failed_policies.append(filename)
                continue

            missing_count = len(report.get("missing_diagnoses", []))
            unresolved_count = len(report.get("unresolved_indicators", []))
            report.update(
                {
                    "policy_file": filename,
                    "model": wrapper.profile.deployment,
                    "diagnosed_at": datetime.now(timezone.utc).isoformat(),
                    "prompt_version": config.discrepancy_diagnosis_template_path.parent.name,
                    "discrepancy_count": len(grouped[filename]),
                    "diagnosed_count": len(grouped[filename]) - missing_count - unresolved_count,
                }
            )
            output_path = output_dir / f"{Path(filename).stem}.json"
            output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            saved_paths.append(str(output_path))
            clear_failure(run_id, "discrepancy_diagnosis", filename)

    elapsed = round(time.time() - start, 2)
    succeeded = len(grouped) - len(failed_policies)

    return {
        "counts": {
            "total": total_candidates,
            "succeeded": succeeded,
            "failed": len(failed_policies),
            "skipped": skipped,
            "saved_reports": len(saved_paths),
            "discrepancies_total": discrepancies_total,
        },
        "failed_policies": failed_policies,
        "elapsed_s": elapsed,
        "token_usage": dict(wrapper.token_usage),
        "output_dir": str(output_dir),
    }
