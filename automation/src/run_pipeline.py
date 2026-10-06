import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from automation.src.compare import run_comparison_step
from automation.src.concurrency import ConcurrencyLimiter
from automation.src.config_loader import (
    ConcurrencyConfig,
    ResolvedPipelineConfig,
    get_run_dir,
    load_pipeline_config,
    resolve_results_dir,
)
from automation.src.constants import DEFAULT_MODEL_CONFIG, DEFAULT_PIPELINE_CONFIG, DEFAULT_STEPS, VALID_STEPS
from automation.src.diagnose import run_diagnosis_step
from automation.src.failure_log import (
    clear_step_failure,
    load_failures,
    record_step_failure,
    summarize_failures,
)
from automation.src.legacy.translate import run_markdown_step, run_translation_step
from automation.src.legacy.translation_qa import run_translation_qa_step
from automation.src.llm.client import get_cohere_rerank_api_key, get_cohere_rerank_endpoint
from automation.src.llm.wrapper import AzureLLMWrapper
from automation.src.markdown.qa import run_md_translation_qa_step
from automation.src.markdown.to_text import run_md_to_text_step
from automation.src.markdown.translate import run_md_translation_step
from automation.src.metadata import (
    generate_run_id,
    init_run_metadata,
    update_metadata,
    validate_run_for_steps,
)
from automation.src.rag.embedder import AzureEmbedder
from automation.src.rag.store import run_storage_step
from automation.src.run_eval import run_evaluation_step
from automation.src.legacy.run_eval_sliding_window import run_sliding_window_evaluation_step


def parse_steps(steps_arg: str | None) -> list[str]:
    if not steps_arg:
        return list(DEFAULT_STEPS)
    steps = [s.strip().lower() for s in steps_arg.split(",") if s.strip()]
    invalid = [s for s in steps if s not in VALID_STEPS]
    if invalid:
        raise ValueError(f"Invalid steps: {', '.join(invalid)}. Valid: {', '.join(VALID_STEPS)}")
    return steps


def requires_run_id(steps: list[str], run_id: str | None) -> bool:
    eval_or_compare_only = (
        any(
            step in steps
            for step in (
                "translation_qa",
                "translation_qa_md",
                "markdown",
                "md_to_text",
                "storage",
                "evaluation",
                "comparison",
                "discrepancy_diagnosis",
            )
        )
        and "translation" not in steps
        and "translation_md" not in steps
    )
    return run_id is None and eval_or_compare_only


def build_config_summary(config) -> dict:
    return {
        "translation_model": config.translation_model.name,
        "translation_qa_model": config.translation_qa_model.name,
        "evaluation_model": config.evaluation_model.name,
        "discrepancy_diagnosis_model": config.discrepancy_diagnosis_model.name,
        "translation_prompt_version": config.translation_prompt_path.parent.name,
        "translation_qa_prompt_version": config.translation_qa_template_path.parent.name,
        "evaluation_prompt_version": config.evaluation_criteria_dir.name,
        "discrepancy_diagnosis_prompt_version": config.discrepancy_diagnosis_template_path.parent.name,
        "concurrency": {
            "enabled": config.concurrency.enabled,
            "max_workers": config.concurrency.max_workers,
        },
        "chunking": {
            "enabled": config.chunking.enabled,
            "safe_limit": config.chunking.safe_limit,
        },
        "markdown_translation": {
            "prompt_version": config.markdown.prompt_path.parent.name,
            "qa_prompt_version": config.markdown.qa_template_path.parent.name,
            "target_chars": config.markdown.target_chars,
            "safe_limit": config.markdown.safe_limit,
            "qa_max_passes": config.markdown.qa_max_passes,
        },
        "storage": {
            "enabled": config.storage.enabled,
            "batch_size": config.storage.batch_size,
        },
        "retrieval": {
            "enabled": config.retrieval.enabled,
            "top_k": config.retrieval.top_k,
            "hybrid_bm25_enabled": config.retrieval.hybrid_bm25.enabled,
            "reranker_enabled": config.retrieval.reranker.enabled,
            "reranker_model": config.retrieval.reranker.model,
            "reranker_candidate_pool_size": config.retrieval.reranker.candidate_pool_size,
            "reranker_top_k": config.retrieval.reranker.top_k,
            "evidence_verification_enabled": config.retrieval.evidence_verification_enabled,
        },
        "evaluation_method": config.evaluation_method,
        "sliding_window": {
            "window_sentences": config.sliding_window.window_sentences,
            "overlap_sentences": config.sliding_window.overlap_sentences,
            "prompt_version": config.sliding_window.prompt_version,
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


def _print_step_failure(step: str, message: str) -> None:
    print(f"Step '{step}' failed: {message}", file=sys.stderr)


def _print_policy_failures(run_id: str, step: str, failed_policies: list[str]) -> None:
    messages = {
        entry.get("policy_file"): entry.get("message", "unknown error")
        for entry in load_failures(run_id).get(step, [])
    }
    for name in failed_policies:
        print(f"  {name}: {messages.get(name, 'unknown error')}", file=sys.stderr)


def _print_overall_alignment_rate(run_id: str) -> None:
    """Aggregate `evidence_original_aligned_rate` across every evaluation report
    in the run and print one overall figure: how often the original-language
    evidence was paired at exact sentence granularity rather than falling back
    to the whole chunk (see rag/sentence_align.py). Silent when no report
    carries the stat (e.g. evaluation never ran, or ran in non-RAG mode)."""
    eval_dir = resolve_results_dir(get_run_dir(run_id), "evaluation")
    if not eval_dir.is_dir():
        return

    sentence_aligned = resolved = 0
    for report_path in sorted(eval_dir.glob("*.json")):
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        stats = data.get("evidence_original_aligned_rate")
        if isinstance(stats, dict):
            sentence_aligned += stats.get("sentence_aligned") or 0
            resolved += stats.get("resolved") or 0

    if not resolved:
        return
    print(
        f"Evidence original-language alignment: {sentence_aligned}/{resolved} "
        f"locally aligned ({sentence_aligned / resolved:.1%})"
    )


def build_limiter(concurrency: ConcurrencyConfig) -> ConcurrencyLimiter:
    return ConcurrencyLimiter(
        max_workers=concurrency.max_workers,
        enabled=concurrency.enabled,
    )


@dataclass
class StepContext:
    run_id: str
    config: ResolvedPipelineConfig
    limiter: ConcurrencyLimiter
    args: argparse.Namespace


@dataclass(frozen=True)
class Step:
    """One pipeline step.

    `run` returns the step's result dict; `report` prints a summary and returns
    True when the step should make the pipeline exit non-zero. The flags say
    which optional sections of the result are recorded in metadata.json.
    """

    name: str
    run: Callable[[StepContext], dict]
    report: Callable[[StepContext, dict], bool]
    has_counts: bool = True
    has_tokens: bool = False
    records_failures: bool = False
    extra_scope: Callable[[dict], dict] = lambda result: {}
    header: Callable[[StepContext], str] | None = None
    skip: Callable[[StepContext], str | None] | None = None


def _llm_runner(get_step_fn, model_attr: str, *, allow_partial: bool = False):
    # `get_step_fn` is looked up lazily so the step functions imported into this
    # module stay patchable (tests replace them on `automation.src.run_pipeline`).
    def run(ctx: StepContext) -> dict:
        wrapper = AzureLLMWrapper.from_profile(getattr(ctx.config, model_attr), limiter=ctx.limiter)
        kwargs = {}
        if allow_partial:
            kwargs["allow_partial"] = ctx.args.allow_partial
        return get_step_fn()(
            run_id=ctx.run_id,
            config=ctx.config,
            wrapper=wrapper,
            limiter=ctx.limiter,
            small_scale=ctx.args.small_scale,
            force=ctx.args.force,
            **kwargs,
        )

    return run


def _run_without_llm(get_step_fn):
    def run(ctx: StepContext) -> dict:
        return get_step_fn()(
            run_id=ctx.run_id,
            config=ctx.config,
            small_scale=ctx.args.small_scale,
            force=ctx.args.force,
        )

    return run


def _run_storage(ctx: StepContext) -> dict:
    embedder = AzureEmbedder.from_env(batch_size=ctx.config.storage.batch_size)
    return run_storage_step(
        run_id=ctx.run_id,
        config=ctx.config,
        embedder=embedder,
        limiter=ctx.limiter,
        small_scale=ctx.args.small_scale,
        force=ctx.args.force,
    )


def _run_evaluation(ctx: StepContext) -> dict:
    config = ctx.config
    wrapper = AzureLLMWrapper.from_profile(
        config.evaluation_model,
        limiter=ctx.limiter,
        confidence=config.confidence,
    )
    if config.evaluation_method == "sliding_window":
        return run_sliding_window_evaluation_step(
            run_id=ctx.run_id,
            config=config,
            wrapper=wrapper,
            limiter=ctx.limiter,
            small_scale=ctx.args.small_scale,
            allow_partial=ctx.args.allow_partial,
            force=ctx.args.force,
        )
    embedder = (
        AzureEmbedder.from_env(batch_size=config.storage.batch_size)
        if config.retrieval.enabled
        else None
    )
    if config.retrieval.reranker.enabled:
        # Fail fast on missing credentials, rather than every
        # worker thread independently hitting the same error.
        get_cohere_rerank_endpoint()
        get_cohere_rerank_api_key()
    return run_evaluation_step(
        run_id=ctx.run_id,
        config=config,
        wrapper=wrapper,
        embedder=embedder,
        limiter=ctx.limiter,
        small_scale=ctx.args.small_scale,
        allow_partial=ctx.args.allow_partial,
        force=ctx.args.force,
    )


def _run_comparison(ctx: StepContext) -> dict:
    return run_comparison_step(run_id=ctx.run_id, config=ctx.config)


def _print_failed_files(result: dict, *, style: str) -> None:
    """style: "status" (with HTTP status), "plain", or "artifact"."""
    for entry in result.get("failed_files") or []:
        if style == "status":
            status = entry.get("details", {}).get("status_code", "n/a")
            line = (
                f"  {entry['filename']}: [{entry['error_type']}] {entry['message']} "
                f"(status={status})"
            )
        elif style == "artifact":
            line = f"  {entry['filename']} ({entry['artifact']}): {entry['message']}"
        else:
            line = f"  {entry['filename']}: [{entry['error_type']}] {entry['message']}"
        print(line, file=sys.stderr)


def _report_translation_md(ctx: StepContext, result: dict) -> bool:
    print(
        f"Translation (markdown): {result['counts']['succeeded']} succeeded, "
        f"{result['counts']['skipped']} skipped, "
        f"{result['counts']['failed']} failed"
    )
    _print_failed_files(result, style="status")
    return result["counts"]["failed"] > 0


def _report_translation_qa_md(ctx: StepContext, result: dict) -> bool:
    print(
        f"Translation QA (markdown): {result['counts']['succeeded']} succeeded "
        f"({result['counts']['corrected']} corrected over "
        f"{result['counts']['qa_passes']} passes, "
        f"{result['counts']['not_converged']} chunks unconverged, "
        f"{result['counts']['incomplete']} incomplete-response), "
        f"{result['counts']['skipped']} skipped, "
        f"{result['counts']['failed']} failed, "
        f"{result['counts']['clauses_lost']} with LOST CLAUSES"
    )
    _print_failed_files(result, style="plain")
    return result["counts"]["failed"] > 0


def _report_md_to_text(ctx: StepContext, result: dict) -> bool:
    print(
        f"Markdown to text: {result['counts']['succeeded']} succeeded, "
        f"{result['counts']['skipped']} skipped, "
        f"{result['counts']['failed']} failed, "
        f"{result['counts']['repaired']} structure-repaired, "
        f"{result['counts'].get('source_unaligned', 0)} with unpaired source, "
        f"{result['counts']['clauses_lost']} with LOST CLAUSES"
    )
    _print_failed_files(result, style="artifact")
    return result["counts"]["failed"] > 0


def _report_translation(ctx: StepContext, result: dict) -> bool:
    print(
        f"Translation: {result['counts']['succeeded']} succeeded, "
        f"{result['counts']['skipped']} skipped, "
        f"{result['counts']['failed']} failed"
    )
    _print_failed_files(result, style="status")
    return result["counts"]["failed"] > 0


def _report_translation_qa(ctx: StepContext, result: dict) -> bool:
    print(
        f"Translation QA: {result['counts']['succeeded']} succeeded "
        f"({result['counts']['corrected']} corrected, "
        f"{result['counts']['incomplete']} incomplete-response), "
        f"{result['counts']['skipped']} skipped, "
        f"{result['counts']['failed']} failed"
    )
    _print_failed_files(result, style="plain")
    return result["counts"]["failed"] > 0


def _report_markdown(ctx: StepContext, result: dict) -> bool:
    print(
        f"Markdown: {result['counts']['succeeded']} succeeded, "
        f"{result['counts']['skipped']} skipped, "
        f"{result['counts']['failed']} failed"
    )
    _print_failed_files(result, style="artifact")
    return result["counts"]["failed"] > 0


def _report_storage(ctx: StepContext, result: dict) -> bool:
    print(
        f"Storage: {result['counts']['succeeded']} succeeded, "
        f"{result['counts']['skipped']} skipped, "
        f"{result['counts']['failed']} failed"
    )
    _print_failed_files(result, style="plain")
    return result["counts"]["failed"] > 0


def _report_evaluation(ctx: StepContext, result: dict) -> bool:
    skipped = result["counts"].get("skipped", 0)
    skipped_str = f", {skipped} skipped" if skipped else ""
    print(
        f"Evaluation: {result['counts']['succeeded']} succeeded, "
        f"{result['counts']['failed']} failed{skipped_str}"
    )
    if result["failed_policies"]:
        print(f"Failed policies: {', '.join(result['failed_policies'])}", file=sys.stderr)
        _print_policy_failures(ctx.run_id, "evaluation", result["failed_policies"])
        return not ctx.args.allow_partial
    return False


def _report_comparison(ctx: StepContext, result: dict) -> bool:
    accuracy = result["counts"].get("accuracy")
    acc_str = f"{accuracy:.2%}" if accuracy is not None else "n/a"
    print(f"Comparison: accuracy {acc_str}, metrics at {result['metrics_path']}")
    return False


def _report_diagnosis(ctx: StepContext, result: dict) -> bool:
    if result["counts"]["discrepancies_total"] == 0:
        reason = result.get("skipped_reason", "no mismatches found by comparison")
        print(f"Discrepancy diagnosis: nothing to diagnose ({reason})")
        return False
    skipped = result["counts"].get("skipped", 0)
    skipped_str = f", {skipped} skipped" if skipped else ""
    print(
        f"Discrepancy diagnosis: {result['counts']['succeeded']} succeeded, "
        f"{result['counts']['failed']} failed{skipped_str} "
        f"({result['counts']['discrepancies_total']} discrepancies)"
    )
    if result["failed_policies"]:
        print(f"Failed policies: {', '.join(result['failed_policies'])}", file=sys.stderr)
        _print_policy_failures(ctx.run_id, "discrepancy_diagnosis", result["failed_policies"])
        return not ctx.args.allow_partial
    return False


def _storage_skip(ctx: StepContext) -> str | None:
    if not ctx.config.storage.enabled:
        return "\nStep: storage (skipped — evaluation.rag.enabled is false)"
    return None


# Steps run in this order, regardless of the order given to --steps.
STEPS: tuple[Step, ...] = (
    # Markdown path (default chain).
    Step(
        "translation_md",
        _llm_runner(lambda: run_md_translation_step, "translation_model"),
        _report_translation_md,
        has_tokens=True,
        records_failures=True,
    ),
    Step(
        "translation_qa_md",
        _llm_runner(lambda: run_md_translation_qa_step, "translation_qa_model"),
        _report_translation_qa_md,
        has_tokens=True,
        records_failures=True,
    ),
    Step(
        "md_to_text",
        _run_without_llm(lambda: run_md_to_text_step),
        _report_md_to_text,
        records_failures=True,
    ),
    # Legacy raw-text path (automation.src.legacy).
    Step(
        "translation",
        _llm_runner(lambda: run_translation_step, "translation_model"),
        _report_translation,
        has_tokens=True,
        records_failures=True,
    ),
    Step(
        "translation_qa",
        _llm_runner(lambda: run_translation_qa_step, "translation_qa_model"),
        _report_translation_qa,
        has_tokens=True,
        records_failures=True,
    ),
    Step("markdown", _run_without_llm(lambda: run_markdown_step), _report_markdown),
    # Shared downstream steps.
    Step(
        "storage",
        _run_storage,
        _report_storage,
        records_failures=True,
        skip=_storage_skip,
    ),
    Step(
        "evaluation",
        _run_evaluation,
        _report_evaluation,
        has_tokens=True,
        records_failures=True,
        header=lambda ctx: f"\nStep: evaluation (method={ctx.config.evaluation_method})",
    ),
    Step(
        "comparison",
        _run_comparison,
        _report_comparison,
        has_counts=False,
        extra_scope=lambda result: {"evaluated_policy_files": result["evaluated_policy_files"]},
    ),
    Step(
        "discrepancy_diagnosis",
        _llm_runner(
            lambda: run_diagnosis_step, "discrepancy_diagnosis_model", allow_partial=True
        ),
        _report_diagnosis,
        has_tokens=True,
        records_failures=True,
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run translation, evaluation, and golden-dataset comparison pipeline."
    )
    parser.add_argument(
        "--steps",
        default=None,
        help=(
            "Comma-separated steps. Markdown path (default): translation_md, "
            "md_to_text, storage, evaluation, comparison. translation_qa_md and "
            "discrepancy_diagnosis are part of the Markdown path but must be requested "
            "explicitly (not run by default). The raw-OCR-text path (translation, "
            "translation_qa, markdown) stays available for comparison but is not run by "
            "default either. The evaluation/comparison steps' behavior "
            "(RAG vs sliding window, and which results/ subfolder they use) is controlled by "
            "evaluation.method in pipeline_config.yaml, not by --steps."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Existing run ID (required for evaluation/comparison-only on prior translation).",
    )
    parser.add_argument(
        "--small-scale",
        action="store_true",
        help="Process only the small-scale benchmark policies (paths.small_scale_stems).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run a step even if its output already exists (steps are idempotent by default).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Save incomplete LLM evaluation or discrepancy diagnosis reports.",
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
        if not args.run_id:
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
            if "translation" in steps or "translation_md" in steps:
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
            validate_run_for_steps(
                run_id,
                steps,
                retrieval_enabled=config.retrieval.enabled,
                evaluation_method=config.evaluation_method,
                translation_markdown_dir=config.paths.translation_markdown_dir,
                processed_translation_markdown_dir=config.paths.processed_translation_markdown_dir,
            )
            print(f"Using existing run: {run_id}")

        stopped = False
        ctx = StepContext(run_id=run_id, config=config, limiter=limiter, args=args)

        for step in STEPS:
            if stopped or step.name not in steps:
                continue
            skip_message = step.skip(ctx) if step.skip else None
            if skip_message:
                print(skip_message)
                continue
            print(step.header(ctx) if step.header else f"\nStep: {step.name}")
            try:
                result = step.run(ctx)
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                exit_code = 1
                stopped = True
                record_step_failure(run_id, step.name, str(exc))
                _print_step_failure(step.name, str(exc))
            else:
                clear_step_failure(run_id, step.name)
                updates: dict = {
                    "execution_scope": {"steps_executed": [step.name], **step.extra_scope(result)}
                }
                if step.has_counts:
                    updates["file_counts"] = {step.name: result["counts"]}
                updates["timing_seconds"] = {step.name: result["elapsed_s"]}
                if step.has_tokens:
                    updates["token_usage"] = {step.name: result["token_usage"]}
                if step.records_failures:
                    updates["failures"] = summarize_failures(run_id)
                update_metadata(run_id, **updates)
                if step.report(ctx, result):
                    exit_code = 1

        total_elapsed = round(time.time() - pipeline_start, 2)
        status = "completed" if exit_code == 0 else "completed_with_errors"
        failure_summary = summarize_failures(run_id)
        update_metadata(
            run_id,
            status=status,
            timing_seconds={"total": total_elapsed},
            failures=failure_summary,
        )

        print(f"\nPipeline finished (run_id={run_id}, status={status})")
        _print_overall_alignment_rate(run_id)
        if (
            failure_summary["translation"]
            or failure_summary["translation_qa"]
            or failure_summary["storage"]
            or failure_summary["evaluation"]
            or failure_summary["discrepancy_diagnosis"]
            or failure_summary["step_failures"]
        ):
            failures_path = get_run_dir(run_id) / "failures.json"
            print(
                f"Failures logged: {failure_summary['translation']} translation, "
                f"{failure_summary['translation_qa']} translation_qa, "
                f"{failure_summary['storage']} storage, "
                f"{failure_summary['evaluation']} evaluation, "
                f"{failure_summary['discrepancy_diagnosis']} discrepancy_diagnosis, "
                f"{failure_summary['step_failures']} step-level → {failures_path}",
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
