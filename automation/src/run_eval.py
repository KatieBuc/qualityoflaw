import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from quality_eval.run_eval_new import (
    PolicyEvaluationResponse,
    evaluate_policy,
    finalize_and_save_report,
)
from quality_eval.v1.criteria import CRITERIA_FILES, PROMPT_VERSION
from quality_eval.v1.prompts.prompt_loader import generate_judge_prompt

from automation.src.concurrency import ConcurrencyLimiter
from automation.src.config_loader import ResolvedPipelineConfig, get_run_dir
from automation.src.constants import SMALL_SCALE_FILES
from automation.src.failure_log import clear_failure, record_failure
from automation.src.llm.wrapper import AzureLLMWrapper, LLMCallError, format_api_error

logger = logging.getLogger(__name__)


@dataclass
class DimensionResult:
    criteria_file: str
    batch_evals: dict
    error: str | None = None
    error_type: str | None = None
    error_details: dict | None = None


def _filter_policy_files(policy_dir: Path, small_scale: bool) -> list[Path]:
    files = sorted(policy_dir.glob("*.txt"))
    if small_scale:
        allowed = set(SMALL_SCALE_FILES)
        files = [f for f in files if f.name in allowed]
    return files


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
    template_path: Path,
    complete_fn: Callable[[str], dict],
) -> DimensionResult:
    try:
        final_prompt = generate_judge_prompt(
            criteria_file_path=str(criteria_folder / criteria_file),
            template_file_path=str(template_path),
            policy_file_path=str(policy_path),
        )
        batch_result = complete_fn(final_prompt)
        batch_evals = batch_result.get("evaluation_results", {})
        logger.info("[%s] %s completed", policy_path.name, criteria_file)
        return DimensionResult(criteria_file=criteria_file, batch_evals=batch_evals)
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


def _merge_dimension_results(
    policy_path: Path,
    deployment_name: str,
    dimension_results: list[DimensionResult | BaseException],
) -> dict:
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

    final_report["indicator_count"] = len(final_report["evaluation_results"])
    return final_report


def _evaluate_policy_parallel(
    policy_path: Path,
    criteria_folder: Path,
    template_path: Path,
    deployment_name: str,
    complete_fn: Callable[[str], dict],
    limiter: ConcurrencyLimiter,
) -> dict:
    logger.info("Starting parallel evaluation for %s (%d dimensions)", policy_path.name, len(CRITERIA_FILES))
    tasks = [
        lambda cf=cf: _evaluate_dimension(
            policy_path,
            cf,
            criteria_folder,
            template_path,
            complete_fn,
        )
        for cf in CRITERIA_FILES
    ]
    dimension_results = limiter.run_parallel(tasks)
    return _merge_dimension_results(policy_path, deployment_name, dimension_results)


def _process_policy_result(
    policy_path: Path,
    final_report: dict,
    output_dir: Path,
    deployment_name: str,
    allow_partial: bool,
) -> tuple[bool, str | None, bool]:
    success, output_path = finalize_and_save_report(
        final_report=final_report,
        policy_path=str(policy_path),
        output_folder=str(output_dir),
        deployment_name=deployment_name,
        allow_partial=allow_partial,
    )
    policy_failed = not success
    return success, output_path, policy_failed


def run_evaluation_step(
    run_id: str,
    config: ResolvedPipelineConfig,
    wrapper: AzureLLMWrapper,
    *,
    limiter: ConcurrencyLimiter,
    small_scale: bool = False,
    allow_partial: bool = False,
    run_missing: bool = False,
) -> dict:
    run_dir = get_run_dir(run_id)
    policy_dir = run_dir / "translation"
    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_files = _filter_policy_files(policy_dir, small_scale)
    if not policy_files:
        raise FileNotFoundError(f"No policy files to evaluate in {policy_dir}")

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

            final_report = evaluate_policy(
                policy_path=str(policy_path),
                criteria_folder=str(config.evaluation_criteria_dir),
                template_path=str(config.evaluation_template_path),
                deployment_name=deployment_name,
                complete_fn=complete_fn,
            )
            success, output_path, policy_failed = _process_policy_result(
                policy_path,
                final_report,
                output_dir,
                deployment_name,
                allow_partial,
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
                config.evaluation_template_path,
                deployment_name,
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

            success, output_path, policy_failed = _process_policy_result(
                policy_path,
                result,
                output_dir,
                deployment_name,
                allow_partial,
            )
            if output_path:
                saved_paths.append(output_path)
            if policy_failed:
                failed_policies.append(policy_path.name)
                record_failure(run_id, "evaluation", _failure_entry_from_report(result))
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
