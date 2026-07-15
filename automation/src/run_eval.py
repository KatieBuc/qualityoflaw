import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from quality_eval.run_eval_new import PolicyEvaluationResponse, finalize_and_save_report
from quality_eval.v1.criteria import CRITERIA_FILES, PROMPT_VERSION

from automation.src.concurrency import ConcurrencyLimiter
from automation.src.config_loader import ResolvedPipelineConfig, RetrievalConfig, get_run_dir
from automation.src.failure_log import clear_failure, record_failure
from automation.src.llm.wrapper import AzureLLMWrapper, LLMCallError, format_api_error
from automation.src.policy_files import filter_policy_files
from automation.src.rag.embedder import AzureEmbedder
from automation.src.rag.evidence_check import resolve_evidence_citation
from automation.src.rag.prompt_builder import (
    build_candidate_lookup,
    format_criteria_with_candidates,
    parse_criteria_lines,
)
from automation.src.rag.retriever import RetrievedChunk, load_policy_store, retrieve_evidence_candidates
from automation.src.rag.store import store_path_for

logger = logging.getLogger(__name__)


@dataclass
class DimensionResult:
    criteria_file: str
    batch_evals: dict
    candidates_by_id: dict[str, list[RetrievedChunk]] | None = None
    error: str | None = None
    error_type: str | None = None
    error_details: dict | None = None


def _has_eval_report(output_dir: Path, policy_name: str) -> bool:
    if not output_dir.is_dir():
        return False
    for json_path in output_dir.glob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("policy_file") == policy_name:
            return True
        basename = policy_name.replace(".txt", ".json")
        if json_path.name.endswith(f"-{basename}"):
            return True
    return False


def _failure_entry_from_exc(
    policy_file: str,
    exc: Exception,
    *,
    failed_dimensions: list[str] | None = None,
    errors: list[str] | None = None,
) -> dict:
    if isinstance(exc, LLMCallError):
        return {
            "policy_file": policy_file,
            "failed_dimensions": failed_dimensions or [],
            "errors": errors or [str(exc)],
            "error_type": exc.error_type,
            "message": str(exc),
            "details": exc.details,
            "attempts": exc.attempts,
        }
    error_type, details = format_api_error(exc)
    return {
        "policy_file": policy_file,
        "failed_dimensions": failed_dimensions or [],
        "errors": errors or [str(exc)],
        "error_type": error_type,
        "message": str(exc),
        "details": details,
        "attempts": 1,
    }


def _failure_entry_from_report(final_report: dict) -> dict:
    errors = final_report.get("errors", [])
    message = errors[0] if errors else "evaluation incomplete"
    return {
        "policy_file": final_report["policy_file"],
        "failed_dimensions": final_report.get("failed_dimensions", []),
        "errors": errors,
        "error_type": "EvaluationError",
        "message": message,
        "details": {},
        "attempts": 1,
    }


def _evaluate_dimension(
    policy_path: Path,
    criteria_file: str,
    criteria_folder: Path,
    template_text: str,
    store_chunks: list[dict],
    embedder: AzureEmbedder | None,
    retrieval_config: RetrievalConfig,
    complete_fn: Callable[[str], dict],
) -> DimensionResult:
    try:
        criteria_rows = parse_criteria_lines(str(criteria_folder / criteria_file))

        if not retrieval_config.enabled:
            criteria_list_str = "\n".join(
                f"- {cid} ({indicator}): {question}" for cid, indicator, question in criteria_rows
            )
            final_prompt = template_text.replace("{{CRITERIA_LIST}}", criteria_list_str).replace(
                "{{POLICY_TEXT}}", policy_path.read_text(encoding="utf-8")
            )
            batch_result = complete_fn(final_prompt)
            batch_evals = batch_result.get("evaluation_results", {})
            for item in batch_evals.values():
                item["evidence_verified"] = None

            logger.info("[%s] %s completed (full-policy mode)", policy_path.name, criteria_file)
            return DimensionResult(criteria_file=criteria_file, batch_evals=batch_evals)

        question_embeddings = embedder.embed_texts([question for _, _, question in criteria_rows])

        candidates_by_id = {
            cid: retrieve_evidence_candidates(
                query=question,
                query_embedding=query_embedding,
                store_chunks=store_chunks,
                config=retrieval_config,
            )
            for (cid, _indicator, question), query_embedding in zip(
                criteria_rows, question_embeddings
            )
        }

        final_prompt = template_text.replace(
            "{{CRITERIA_WITH_CANDIDATES}}",
            format_criteria_with_candidates(criteria_rows, candidates_by_id),
        )
        batch_result = complete_fn(final_prompt)
        batch_evals = batch_result.get("evaluation_results", {})
        candidate_lookup = build_candidate_lookup(candidates_by_id)
        _resolve_batch_evidence(batch_evals, candidate_lookup, retrieval_config, policy_path.name)

        logger.info("[%s] %s completed", policy_path.name, criteria_file)
        return DimensionResult(
            criteria_file=criteria_file, batch_evals=batch_evals, candidates_by_id=candidates_by_id
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


def _resolve_batch_evidence(
    batch_evals: dict,
    candidate_lookup: dict[str, str],
    retrieval_config: RetrievalConfig,
    policy_name: str,
) -> None:
    """Replace each "Yes" item's evidence (an LLM-cited candidate tag, e.g.
    "1.3-4") with the real chunk text copied from `candidate_lookup` — the
    temporary tag -> text table built for this dimension call. Since the
    final evidence is copied verbatim from a retrieved chunk rather than
    transcribed by the LLM, it can never diverge from the real document.
    """
    for cid, item in batch_evals.items():
        if item.get("included") != "Yes":
            item["evidence_verified"] = None
            continue

        if not retrieval_config.evidence_verification_enabled:
            item["evidence_verified"] = None
            continue

        citation = item.get("evidence")
        resolved_text = resolve_evidence_citation(citation, candidate_lookup)
        if resolved_text is not None:
            item["evidence"] = resolved_text
            item["evidence_verified"] = True
        else:
            logger.warning(
                "[%s] %s evidence citation %r did not match any retrieved candidate; nulling.",
                policy_name,
                cid,
                citation,
            )
            item["evidence"] = None
            item["evidence_verified"] = False
            note = "[unresolved evidence citation removed]"
            item["rationale"] = f"{item.get('rationale', '')} {note}".strip()


def _merge_dimension_results(
    policy_path: Path,
    deployment_name: str,
    dimension_results: list[DimensionResult | BaseException],
) -> tuple[dict, dict[str, list[RetrievedChunk]]]:
    evaluated_at = datetime.now(timezone.utc).isoformat()
    final_report = {
        "policy_file": policy_path.name,
        "model": deployment_name,
        "evaluated_at": evaluated_at,
        "prompt_version": PROMPT_VERSION,
        "completed_dimensions": [],
        "failed_dimensions": [],
        "errors": [],
        "evaluation_results": {},
    }
    all_candidates_by_id: dict[str, list[RetrievedChunk]] = {}

    for result in dimension_results:
        if isinstance(result, BaseException):
            error_type, details = format_api_error(result)
            final_report["errors"].append(f"{error_type}: {result}")
            final_report["failed_dimensions"].append("unknown")
            continue

        if result.error:
            final_report["failed_dimensions"].append(result.criteria_file)
            final_report["errors"].append(result.error)
            continue

        for key, item in result.batch_evals.items():
            if key in final_report["evaluation_results"]:
                logger.warning(
                    "[%s] ID '%s' (%s) already exists; overwriting with new outcome.",
                    policy_path.name,
                    key,
                    item.get("indicator", ""),
                )
        final_report["evaluation_results"].update(result.batch_evals)
        final_report["completed_dimensions"].append(result.criteria_file)
        if result.candidates_by_id:
            all_candidates_by_id.update(result.candidates_by_id)

    final_report["indicator_count"] = len(final_report["evaluation_results"])
    final_report["evidence_verification_failures"] = [
        key
        for key, item in final_report["evaluation_results"].items()
        if item.get("evidence_verified") is False
    ]
    return final_report, all_candidates_by_id


def _evaluate_policy_serial(
    policy_path: Path,
    criteria_folder: Path,
    template_text: str,
    deployment_name: str,
    rag_store_dir: Path,
    embedder: AzureEmbedder | None,
    retrieval_config: RetrievalConfig,
    complete_fn: Callable[[str], dict],
) -> tuple[dict, dict[str, list[RetrievedChunk]]]:
    store_chunks = (
        load_policy_store(store_path_for(rag_store_dir, policy_path.name))
        if retrieval_config.enabled
        else []
    )
    dimension_results = [
        _evaluate_dimension(
            policy_path,
            criteria_file,
            criteria_folder,
            template_text,
            store_chunks,
            embedder,
            retrieval_config,
            complete_fn,
        )
        for criteria_file in CRITERIA_FILES
    ]
    return _merge_dimension_results(policy_path, deployment_name, dimension_results)


def _evaluate_policy_parallel(
    policy_path: Path,
    criteria_folder: Path,
    template_text: str,
    deployment_name: str,
    rag_store_dir: Path,
    embedder: AzureEmbedder | None,
    retrieval_config: RetrievalConfig,
    complete_fn: Callable[[str], dict],
    limiter: ConcurrencyLimiter,
) -> tuple[dict, dict[str, list[RetrievedChunk]]]:
    logger.info(
        "Starting parallel evaluation for %s (%d dimensions)", policy_path.name, len(CRITERIA_FILES)
    )
    store_chunks = (
        load_policy_store(store_path_for(rag_store_dir, policy_path.name))
        if retrieval_config.enabled
        else []
    )
    tasks = [
        lambda cf=cf: _evaluate_dimension(
            policy_path,
            cf,
            criteria_folder,
            template_text,
            store_chunks,
            embedder,
            retrieval_config,
            complete_fn,
        )
        for cf in CRITERIA_FILES
    ]
    dimension_results = limiter.run_parallel(tasks)
    return _merge_dimension_results(policy_path, deployment_name, dimension_results)


def _write_candidates_file(
    rag_candidates_dir: Path,
    policy_filename: str,
    candidates_by_id: dict[str, list[RetrievedChunk]],
) -> None:
    """Persist the full RAG candidate list per indicator (not just the one the
    judge cited) to `rag_candidates/<stem>.json`, so the discrepancy_diagnosis
    step can read them later without recomputing retrieval.
    """
    rag_candidates_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_file": policy_filename,
        "candidates": {
            indicator_id: [
                {"chunk_id": c.chunk_id, "text": c.text, "score": c.score} for c in candidates
            ]
            for indicator_id, candidates in candidates_by_id.items()
        },
    }
    path = rag_candidates_dir / f"{Path(policy_filename).stem}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _process_policy_result(
    policy_path: Path,
    final_report: dict,
    output_dir: Path,
    deployment_name: str,
    allow_partial: bool,
    candidates_by_id: dict[str, list[RetrievedChunk]],
    rag_candidates_dir: Path,
) -> tuple[bool, str | None, bool]:
    success, output_path = finalize_and_save_report(
        final_report=final_report,
        policy_path=str(policy_path),
        output_folder=str(output_dir),
        deployment_name=deployment_name,
        allow_partial=allow_partial,
    )
    if output_path:
        _write_candidates_file(rag_candidates_dir, policy_path.name, candidates_by_id)
    policy_failed = not success
    return success, output_path, policy_failed


def run_evaluation_step(
    run_id: str,
    config: ResolvedPipelineConfig,
    wrapper: AzureLLMWrapper,
    embedder: AzureEmbedder | None,
    *,
    limiter: ConcurrencyLimiter,
    small_scale: bool = False,
    allow_partial: bool = False,
    run_missing: bool = False,
) -> dict:
    run_dir = get_run_dir(run_id)
    policy_dir = run_dir / "translation"
    rag_store_dir = run_dir / "rag_store"
    output_dir = run_dir / "evaluation"
    rag_candidates_dir = run_dir / "rag_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_files = filter_policy_files(policy_dir, small_scale)
    if not policy_files:
        raise FileNotFoundError(f"No policy files to evaluate in {policy_dir}")

    template_text = config.evaluation_template_path.read_text(encoding="utf-8")

    total_candidates = len(policy_files)
    skipped = 0
    if run_missing:
        pending_files: list[Path] = []
        for policy_path in policy_files:
            if _has_eval_report(output_dir, policy_path.name):
                skipped += 1
                logger.info("[%s] skipped (eval report exists)", policy_path.name)
            else:
                pending_files.append(policy_path)
        policy_files = pending_files

    def complete_fn(prompt: str) -> dict:
        return wrapper.complete_structured(prompt, PolicyEvaluationResponse)

    deployment_name = wrapper.profile.deployment
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

    if use_serial:
        for index, policy_path in enumerate(policy_files, 1):
            print("\n" + "=" * 50)
            print(f"Policy [{index}/{len(policy_files)}]: {policy_path.name}")
            print("=" * 50)

            try:
                final_report, candidates_by_id = _evaluate_policy_serial(
                    policy_path,
                    config.evaluation_criteria_dir,
                    template_text,
                    deployment_name,
                    rag_store_dir,
                    embedder,
                    config.retrieval,
                    complete_fn,
                )
            except Exception as exc:
                entry = _failure_entry_from_exc(policy_path.name, exc)
                record_failure(run_id, "evaluation", entry)
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
                rag_candidates_dir,
            )
            if output_path:
                saved_paths.append(output_path)
            if policy_failed:
                failed_policies.append(policy_path.name)
                record_failure(run_id, "evaluation", _failure_entry_from_report(final_report))
            elif success:
                clear_failure(run_id, "evaluation", policy_path.name)
    else:
        tasks = [
            lambda p=p: _evaluate_policy_parallel(
                p,
                config.evaluation_criteria_dir,
                template_text,
                deployment_name,
                rag_store_dir,
                embedder,
                config.retrieval,
                complete_fn,
                limiter,
            )
            for p in policy_files
        ]
        policy_results = limiter.run_parallel(tasks)

        for policy_path, result in zip(policy_files, policy_results, strict=True):
            if isinstance(result, BaseException):
                entry = _failure_entry_from_exc(policy_path.name, result)
                record_failure(run_id, "evaluation", entry)
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
                rag_candidates_dir,
            )
            if output_path:
                saved_paths.append(output_path)
            if policy_failed:
                failed_policies.append(policy_path.name)
                record_failure(run_id, "evaluation", _failure_entry_from_report(final_report))
            elif success:
                clear_failure(run_id, "evaluation", policy_path.name)

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
