import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from automation.src.compare import run_comparison_step
from automation.src.concurrency import ConcurrencyLimiter
from automation.src.config_loader import ConcurrencyConfig, get_run_dir, load_pipeline_config
from automation.src.constants import ALL_STEPS, DEFAULT_MODEL_CONFIG, DEFAULT_PIPELINE_CONFIG
from automation.src.failure_log import summarize_failures
from automation.src.llm.wrapper import AzureLLMWrapper
from automation.src.metadata import (
    ensure_run_exists,
    generate_run_id,
    init_run_metadata,
    update_metadata,
    validate_run_for_steps,
)
from automation.src.run_eval import run_evaluation_step
from automation.src.translate import run_translation_step


def parse_steps(steps_arg: str | None) -> list[str]:
    if not steps_arg:
        return list(ALL_STEPS)
    steps = [s.strip().lower() for s in steps_arg.split(",") if s.strip()]
    invalid = [s for s in steps if s not in ALL_STEPS]
    if invalid:
        raise ValueError(f"Invalid steps: {', '.join(invalid)}. Valid: {', '.join(ALL_STEPS)}")
    return steps


def requires_run_id(steps: list[str], run_id: str | None) -> bool:
    eval_or_compare_only = (
        any(step in steps for step in ("evaluation", "comparison"))
        and "translation" not in steps
    )
    return run_id is None and eval_or_compare_only


def build_config_summary(config) -> dict:
    return {
        "translation_model": config.translation_model.name,
        "evaluation_model": config.evaluation_model.name,
        "translation_prompt_version": config.translation_prompt_path.parent.name,
        "evaluation_prompt_version": config.evaluation_criteria_dir.name,
        "concurrency": {
            "enabled": config.concurrency.enabled,
            "max_workers": config.concurrency.max_workers,
        },
        "chunking": {
            "enabled": config.chunking.enabled,
            "safe_limit": config.chunking.safe_limit,
        },
    }


def resolve_concurrency_config(
    config_concurrency: ConcurrencyConfig,
    *,
    max_workers_override: int | None,
    no_concurrency: bool,
) -> ConcurrencyConfig:
    if no_concurrency:
        return ConcurrencyConfig(enabled=False, max_workers=1)
    if max_workers_override is not None:
        if max_workers_override < 1:
            raise ValueError("--max-workers must be >= 1")
        return ConcurrencyConfig(
            enabled=config_concurrency.enabled,
            max_workers=max_workers_override,
        )
    return config_concurrency


def build_limiter(concurrency: ConcurrencyConfig) -> ConcurrencyLimiter:
    return ConcurrencyLimiter(
        max_workers=concurrency.max_workers,
        enabled=concurrency.enabled,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run translation, evaluation, and golden-dataset comparison pipeline."
    )
    parser.add_argument(
        "--steps",
        default=None,
        help="Comma-separated steps: translation, evaluation, comparison (default: all).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Existing run ID (required for evaluation/comparison-only on prior translation).",
    )
    parser.add_argument(
        "--small-scale",
        action="store_true",
        help="Process only the 5 benchmark policy files.",
    )
    parser.add_argument("--force", action="store_true", help="Re-translate existing outputs.")
    parser.add_argument(
        "--run-missing",
        action="store_true",
        help="Resume an existing run; each step skips items that already have output (requires --run-id).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Save incomplete LLM evaluation reports.",
    )
    parser.add_argument(
        "--keep-chunk-result",
        action="store_true",
        help=(
            "When translation.chunking is enabled, also save the clean and chunk "
            "results for each policy under data/automation/<run_id>/chunks/."
        ),
    )
    parser.add_argument(
        "--pipeline-config",
        default=str(DEFAULT_PIPELINE_CONFIG),
        help="Path to pipeline_config.yaml.",
    )
    parser.add_argument(
        "--model-config",
        default=str(DEFAULT_MODEL_CONFIG),
        help="Path to model_config.yaml.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override concurrency.max_workers from pipeline config.",
    )
    parser.add_argument(
        "--no-concurrency",
        action="store_true",
        help="Disable parallel API calls (equivalent to max_workers=1).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        steps = parse_steps(args.steps)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.run_missing and not args.run_id:
        print("Error: --run-missing requires --run-id.", file=sys.stderr)
        sys.exit(1)

    if args.run_missing and args.force:
        print("Error: --run-missing and --force are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    if requires_run_id(steps, args.run_id):
        print(
            "Error: --run-id is required when running evaluation or comparison without translation.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        config = load_pipeline_config(
            pipeline_config_path=Path(args.pipeline_config),
            model_config_path=Path(args.model_config),
        )
        concurrency = resolve_concurrency_config(
            config.concurrency,
            max_workers_override=args.max_workers,
            no_concurrency=args.no_concurrency,
        )
        limiter = build_limiter(concurrency)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    run_id = args.run_id or generate_run_id()
    pipeline_start = time.time()
    exit_code = 0

    try:
        run_dir = get_run_dir(run_id)
        if args.run_missing:
            ensure_run_exists(run_id)
            validate_run_for_steps(run_id, steps)
            print(f"Using existing run (run-missing): {run_id}")
        elif not args.run_id:
            init_run_metadata(
                run_id=run_id,
                experiment_name=config.experiment_name,
                small_scale=args.small_scale,
                config_summary=build_config_summary(config),
                pipeline_config_path=config.pipeline_config_path,
                model_config_path=config.model_config_path,
            )
            print(f"Created run: {run_id}")
        elif not run_dir.exists():
            if "translation" in steps:
                init_run_metadata(
                    run_id=run_id,
                    experiment_name=config.experiment_name,
                    small_scale=args.small_scale,
                    config_summary=build_config_summary(config),
                    pipeline_config_path=config.pipeline_config_path,
                    model_config_path=config.model_config_path,
                )
                print(f"Created run: {run_id}")
            else:
                raise FileNotFoundError(
                    f"Run directory not found: {run_dir}. Run translation first or omit --run-id."
                )
        else:
            validate_run_for_steps(run_id, steps)
            print(f"Using existing run: {run_id}")

        if "translation" in steps:
            print("\nStep: translation")
            if args.keep_chunk_result and not config.chunking.enabled:
                print(
                    "Note: --keep-chunk-result has no effect because translation.chunking "
                    "is not enabled in the pipeline config.",
                    file=sys.stderr,
                )
            wrapper = AzureLLMWrapper.from_profile(config.translation_model, limiter=limiter)
            result = run_translation_step(
                run_id=run_id,
                config=config,
                wrapper=wrapper,
                limiter=limiter,
                small_scale=args.small_scale,
                force=args.force,
                run_missing=args.run_missing,
                keep_chunk_result=args.keep_chunk_result,
            )
            update_metadata(
                run_id,
                steps_executed=["translation"],
                file_counts={"translation": result["counts"]},
                timing={"translation_s": result["elapsed_s"]},
                token_usage={"translation": result["token_usage"]},
                failures=summarize_failures(run_id),
            )
            print(
                f"Translation: {result['counts']['succeeded']} succeeded, "
                f"{result['counts']['skipped']} skipped, "
                f"{result['counts']['failed']} failed"
            )
            if result.get("failed_files"):
                for entry in result["failed_files"]:
                    status = entry.get("details", {}).get("status_code", "n/a")
                    print(
                        f"  {entry['filename']}: [{entry['error_type']}] {entry['message']} "
                        f"(status={status})",
                        file=sys.stderr,
                    )
            if result["counts"]["failed"] > 0:
                exit_code = 1

        if "evaluation" in steps:
            print("\nStep: evaluation")
            wrapper = AzureLLMWrapper.from_profile(config.evaluation_model, limiter=limiter)
            result = run_evaluation_step(
                run_id=run_id,
                config=config,
                wrapper=wrapper,
                limiter=limiter,
                small_scale=args.small_scale,
                allow_partial=args.allow_partial,
                run_missing=args.run_missing,
            )
            update_metadata(
                run_id,
                steps_executed=["evaluation"],
                file_counts={"evaluation": result["counts"]},
                timing={"evaluation_s": result["elapsed_s"]},
                token_usage={"evaluation": result["token_usage"]},
                failures=summarize_failures(run_id),
            )
            skipped = result["counts"].get("skipped", 0)
            skipped_str = f", {skipped} skipped" if skipped else ""
            print(
                f"Evaluation: {result['counts']['succeeded']} succeeded, "
                f"{result['counts']['failed']} failed{skipped_str}"
            )
            if result["failed_policies"]:
                print(f"Failed policies: {', '.join(result['failed_policies'])}", file=sys.stderr)
                if not args.allow_partial:
                    exit_code = 1

        if "comparison" in steps:
            print("\nStep: comparison")
            result = run_comparison_step(run_id=run_id, config=config)
            update_metadata(
                run_id,
                steps_executed=["comparison"],
                file_counts={"comparison": result["counts"]},
                timing={"comparison_s": result["elapsed_s"]},
            )
            accuracy = result["counts"].get("accuracy")
            acc_str = f"{accuracy:.2%}" if accuracy is not None else "n/a"
            print(f"Comparison: accuracy {acc_str}, metrics at {result['metrics_path']}")

        total_elapsed = round(time.time() - pipeline_start, 2)
        status = "completed" if exit_code == 0 else "completed_with_errors"
        failure_summary = summarize_failures(run_id)
        update_metadata(run_id, status=status, timing={"total_s": total_elapsed}, failures=failure_summary)

        print(f"\nPipeline finished (run_id={run_id}, status={status})")
        if failure_summary["translation"] or failure_summary["evaluation"]:
            failures_path = get_run_dir(run_id) / "failures.json"
            print(
                f"Failures logged: {failure_summary['translation']} translation, "
                f"{failure_summary['evaluation']} evaluation → {failures_path}",
                file=sys.stderr,
            )

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if run_id:
            try:
                update_metadata(run_id, status="failed")
            except FileNotFoundError:
                pass
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
