"""Sliding-window evaluation: an alternative to the RAG method (see
`run_eval.py`), selected via `evaluation.method: sliding_window` in
pipeline_config.yaml.

Instead of retrieving candidate passages per indicator, the policy is split
into overlapping windows of sentences (`sliding_window/windowing.py`). Every
window is shown in full to the LLM for each dimension, and per-indicator
results are merged across all windows: the final answer is "Yes" if any
window said "Yes", and the final evidence is the union of every window's
resolved evidence sentences.

Reuses the same criteria files, response schema, evidence-citation scheme,
and report-saving/failure-tracking machinery as the RAG method — only the
"how do we get text to show the LLM for this dimension" step differs.
"""

import logging
import time
from pathlib import Path
from typing import Callable

from automation.src.concurrency import ConcurrencyLimiter
from automation.src.config_loader import (
    ResolvedPipelineConfig,
    evaluation_output_names,
    get_output_dir,
    resolve_mid_product_dir,
    resolve_results_dir,
)
from automation.src.criteria import CRITERIA_FILES
from automation.src.failure_log import clear_failure, record_failure
from automation.src.llm.confidence import merge_scores
from automation.src.llm.wrapper import AzureLLMWrapper, format_api_error
from automation.src.policy_files import filter_policy_files
from automation.src.rag.evidence_check import resolve_evidence_citation
from automation.src.rag.prompt_builder import parse_criteria_lines
from automation.src.rag.retriever import RetrievedChunk
from automation.src.run_eval import (
    DimensionResult,
    _failure_entry_from_exc,
    _failure_entry_from_report,
    _has_eval_report,
    _merge_dimension_results,
    _process_policy_result,
    response_model_for,
)
from automation.src.legacy.sliding_window.prompt_builder import (
    build_window_lookup,
    format_criteria_list,
    format_window_sentences,
)
from automation.src.legacy.sliding_window.windowing import split_into_windows

logger = logging.getLogger(__name__)


def _resolve_window_evidence(batch_evals: dict, candidate_lookup: dict[str, str]) -> None:
    """Like `run_eval._resolve_batch_evidence`, but leaves resolved evidence as
    a list of sentences (not yet joined into a string) since it still needs to
    be merged with other windows' evidence before becoming the final report
    field.
    """
    for item in batch_evals.values():
        if item.get("included") != "Yes":
            item["evidence_verified"] = None
            continue

        citations = item.get("evidence")
        resolved_sentences = resolve_evidence_citation(citations, candidate_lookup)
        if resolved_sentences is not None:
            item["evidence"] = resolved_sentences
            item["evidence_verified"] = True
        else:
            item["evidence"] = None
            item["evidence_verified"] = False
            note = "[unresolved evidence citation removed]"
            item["rationale"] = f"{item.get('rationale', '')} {note}".strip()


def _merge_window_confidence(items: list[dict], included: str) -> dict:
    """Fold per-window confidences into one figure for the merged answer.

    A merged "Yes" rests on the windows that voted Yes, so it inherits the
    most confident of those. A merged "No" means every window declined, so it
    is only as strong as its weakest window — hence the min.
    """
    if included == "Yes":
        pool = [it for it in items if it.get("included") == "Yes"]
        reducer = max
    else:
        pool = items
        reducer = min

    scored = [it for it in pool if it.get("confidence") is not None]
    chosen = reducer(scored, key=lambda it: it["confidence"]) if scored else None
    return merge_scores(items, chosen)


def _merge_window_results(ids: list[str], window_results: list[dict[str, dict]]) -> dict[str, dict]:
    """Merge one dimension's per-window results into a single per-indicator
    result: "Yes" if any window says "Yes", evidence is the deduplicated
    union of every contributing window's resolved sentences.
    """
    merged: dict[str, dict] = {}
    for cid in ids:
        items = [wr[cid] for wr in window_results if cid in wr]
        if not items:
            merged[cid] = {
                "id": cid,
                "indicator": "",
                "included": "No",
                "evidence": None,
                "rationale": "No window returned a result for this indicator.",
                "evidence_verified": None,
                **merge_scores([], None),
            }
            continue

        indicator = next((it.get("indicator") for it in items if it.get("indicator")), "")
        yes_items = [it for it in items if it.get("included") == "Yes"]

        if yes_items:
            evidence_sentences: list[str] = []
            for it in yes_items:
                for sentence in it.get("evidence") or []:
                    if sentence not in evidence_sentences:
                        evidence_sentences.append(sentence)

            rationales: list[str] = []
            for it in yes_items:
                rationale = (it.get("rationale") or "").strip()
                if rationale and rationale not in rationales:
                    rationales.append(rationale)

            merged[cid] = {
                "id": cid,
                "indicator": indicator,
                "included": "Yes",
                "evidence": "\n".join(evidence_sentences) if evidence_sentences else None,
                "rationale": " | ".join(rationales),
                "evidence_verified": bool(evidence_sentences),
                **_merge_window_confidence(items, "Yes"),
            }
        else:
            rationale = next((it.get("rationale", "").strip() for it in items if it.get("rationale")), "")
            merged[cid] = {
                "id": cid,
                "indicator": indicator,
                "included": "No",
                "evidence": None,
                "rationale": rationale or "Not addressed in any window.",
                "evidence_verified": None,
                **_merge_window_confidence(items, "No"),
            }

    return merged


def _evaluate_dimension_sliding_window(
    policy_path: Path,
    criteria_file: str,
    criteria_folder: Path,
    template_text: str,
    windows: list[RetrievedChunk],
    complete_fn: Callable[[str], dict],
) -> DimensionResult:
    try:
        criteria_rows = parse_criteria_lines(str(criteria_folder / criteria_file))
        ids = [criterion.id for criterion in criteria_rows]

        window_results: list[dict[str, dict]] = []
        for window in windows:
            final_prompt = template_text.replace(
                "{{WINDOW_SENTENCES}}", format_window_sentences(window)
            ).replace("{{CRITERIA_LIST}}", format_criteria_list(criteria_rows))
            batch_result = complete_fn(final_prompt)
            batch_evals = batch_result.get("evaluation_results", {})
            candidate_lookup = build_window_lookup(window)
            _resolve_window_evidence(batch_evals, candidate_lookup)
            window_results.append(batch_evals)

        merged = _merge_window_results(ids, window_results)

        logger.info(
            "[%s] %s completed (%d windows)", policy_path.name, criteria_file, len(windows)
        )
        return DimensionResult(
            criteria_file=criteria_file,
            batch_evals=merged,
            candidates_by_id={cid: windows for cid in ids},
        )
    except Exception as exc:
        error_type, details = format_api_error(exc)
        logger.error(
            "[%s] %s failed [%s]: %s | status=%s",
            policy_path.name,
            criteria_file,
            error_type,
            str(exc),
            details.get("status_code", "n/a"),
        )
        return DimensionResult(
            criteria_file=criteria_file,
            batch_evals={},
            error=f"{criteria_file}: {exc}",
            error_type=error_type,
            error_details=details,
        )


def _evaluate_policy_serial_sliding_window(
    policy_path: Path,
    criteria_folder: Path,
    template_text: str,
    deployment_name: str,
    prompt_version: str,
    windows: list[RetrievedChunk],
    complete_fn: Callable[[str], dict],
) -> tuple[dict, dict[str, list[RetrievedChunk]]]:
    dimension_results = [
        _evaluate_dimension_sliding_window(
            policy_path, criteria_file, criteria_folder, template_text, windows, complete_fn
        )
        for criteria_file in CRITERIA_FILES
    ]
    return _merge_dimension_results(policy_path, deployment_name, prompt_version, dimension_results)


def _evaluate_policy_parallel_sliding_window(
    policy_path: Path,
    criteria_folder: Path,
    template_text: str,
    deployment_name: str,
    prompt_version: str,
    windows: list[RetrievedChunk],
    complete_fn: Callable[[str], dict],
    limiter: ConcurrencyLimiter,
) -> tuple[dict, dict[str, list[RetrievedChunk]]]:
    logger.info(
        "Starting parallel sliding-window evaluation for %s (%d dimensions, %d windows)",
        policy_path.name,
        len(CRITERIA_FILES),
        len(windows),
    )
    tasks = [
        lambda cf=cf: _evaluate_dimension_sliding_window(
            policy_path, cf, criteria_folder, template_text, windows, complete_fn
        )
        for cf in CRITERIA_FILES
    ]
    dimension_results = limiter.run_parallel(tasks)
    return _merge_dimension_results(policy_path, deployment_name, prompt_version, dimension_results)


def run_sliding_window_evaluation_step(
    config: ResolvedPipelineConfig,
    wrapper: AzureLLMWrapper,
    *,
    limiter: ConcurrencyLimiter,
    small_scale: bool = False,
    allow_partial: bool = False,
    force: bool = False,
) -> dict:
    run_dir = get_output_dir()
    policy_dir = resolve_results_dir(run_dir, "translation")
    output_name, _ = evaluation_output_names("sliding_window")
    output_dir = resolve_results_dir(run_dir, output_name)
    candidates_dir = resolve_mid_product_dir(run_dir, "sliding_window_candidates")
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_files = filter_policy_files(
        policy_dir, small_scale, config.paths.small_scale_stems
    )
    if not policy_files:
        raise FileNotFoundError(f"No policy files to evaluate in {policy_dir}")

    template_text = config.sliding_window_template_path.read_text(encoding="utf-8")

    total_candidates = len(policy_files)
    skipped = 0
    pending_files: list[Path] = []
    for policy_path in policy_files:
        if not force and _has_eval_report(output_dir, policy_path.name):
            skipped += 1
            logger.info("[%s] skipped (eval report exists)", policy_path.name)
        else:
            pending_files.append(policy_path)
    policy_files = pending_files

    response_model = response_model_for(wrapper.confidence)

    def complete_fn(prompt: str) -> dict:
        return wrapper.complete_structured(prompt, response_model)

    deployment_name = wrapper.profile.deployment
    prompt_version = config.sliding_window.prompt_version
    use_serial = not limiter.enabled or limiter.max_workers == 1
    start = time.time()
    saved_paths: list[str] = []
    failed_policies: list[str] = []

    if not policy_files:
        elapsed = round(time.time() - start, 2)
        return {
            "counts": {
                "total": total_candidates,
                "succeeded": 0,
                "failed": 0,
                "skipped": skipped,
                "saved_reports": 0,
            },
            "failed_policies": failed_policies,
            "elapsed_s": elapsed,
            "token_usage": dict(wrapper.token_usage),
            "output_dir": str(output_dir),
        }

    def _windows_for(policy_path: Path) -> list[RetrievedChunk]:
        text = policy_path.read_text(encoding="utf-8")
        return split_into_windows(
            text, config.sliding_window.window_sentences, config.sliding_window.overlap_sentences
        )

    if use_serial:
        for index, policy_path in enumerate(policy_files, 1):
            print("\n" + "=" * 50)
            print(f"Policy [{index}/{len(policy_files)}]: {policy_path.name}")
            print("=" * 50)

            try:
                windows = _windows_for(policy_path)
                final_report, candidates_by_id = _evaluate_policy_serial_sliding_window(
                    policy_path,
                    config.evaluation_criteria_dir,
                    template_text,
                    deployment_name,
                    prompt_version,
                    windows,
                    complete_fn,
                )
            except Exception as exc:
                entry = _failure_entry_from_exc(policy_path.name, exc)
                record_failure("evaluation", entry)
                logger.error(
                    "[%s] evaluation failed [%s]: %s",
                    policy_path.name,
                    entry["error_type"],
                    entry["message"],
                )
                failed_policies.append(policy_path.name)
                continue

            success, output_path, policy_failed = _process_policy_result(
                policy_path,
                final_report,
                output_dir,
                deployment_name,
                allow_partial,
                candidates_by_id,
                candidates_dir,
            )
            if output_path:
                saved_paths.append(output_path)
            if policy_failed:
                failed_policies.append(policy_path.name)
                record_failure("evaluation", _failure_entry_from_report(final_report))
            elif success:
                clear_failure("evaluation", policy_path.name)
    else:
        tasks = [
            lambda p=p: _evaluate_policy_parallel_sliding_window(
                p,
                config.evaluation_criteria_dir,
                template_text,
                deployment_name,
                prompt_version,
                _windows_for(p),
                complete_fn,
                limiter,
            )
            for p in policy_files
        ]
        policy_results = limiter.run_parallel(tasks)

        for policy_path, result in zip(policy_files, policy_results, strict=True):
            if isinstance(result, BaseException):
                entry = _failure_entry_from_exc(policy_path.name, result)
                record_failure("evaluation", entry)
                logger.error(
                    "[%s] evaluation failed [%s]: %s",
                    policy_path.name,
                    entry["error_type"],
                    entry["message"],
                )
                failed_policies.append(policy_path.name)
                continue

            final_report, candidates_by_id = result
            success, output_path, policy_failed = _process_policy_result(
                policy_path,
                final_report,
                output_dir,
                deployment_name,
                allow_partial,
                candidates_by_id,
                candidates_dir,
            )
            if output_path:
                saved_paths.append(output_path)
            if policy_failed:
                failed_policies.append(policy_path.name)
                record_failure("evaluation", _failure_entry_from_report(final_report))
            elif success:
                clear_failure("evaluation", policy_path.name)

    elapsed = round(time.time() - start, 2)
    total = total_candidates
    succeeded = len(policy_files) - len(failed_policies)

    return {
        "counts": {
            "total": total,
            "succeeded": succeeded,
            "failed": len(failed_policies),
            "skipped": skipped,
            "saved_reports": len(saved_paths),
        },
        "failed_policies": failed_policies,
        "elapsed_s": elapsed,
        "token_usage": dict(wrapper.token_usage),
        "output_dir": str(output_dir),
    }
