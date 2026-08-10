import time

from automation.src.evaluate_accuracy import run_accuracy_evaluation

from automation.src.config_loader import (
    ResolvedPipelineConfig,
    evaluation_output_names,
    get_run_dir,
    resolve_results_dir,
)
from automation.src.constants import DEFAULT_MANUAL_OVERWRITES


def run_comparison_step(
    run_id: str,
    config: ResolvedPipelineConfig,
) -> dict:
    run_dir = get_run_dir(run_id)
    eval_name, comparison_name = evaluation_output_names(config.evaluation_method)
    evaluation_dir = resolve_results_dir(run_dir, eval_name)
    comparison_dir = resolve_results_dir(run_dir, comparison_name)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    export_errors = str(comparison_dir / "error_analysis.csv")
    export_metrics = str(comparison_dir / "metrics.csv")

    start = time.time()
    result = run_accuracy_evaluation(
        golden_csv=str(config.paths.golden_csv),
        llm_input=str(evaluation_dir),
        export_errors=export_errors,
        export_metrics=export_metrics,
        manual_overwrites_path=str(DEFAULT_MANUAL_OVERWRITES),
    )
    elapsed = round(time.time() - start, 2)

    if result is None:
        raise RuntimeError("Accuracy comparison produced no results.")

    metrics = result["metrics_summary"]
    overall = metrics.get("overall", {})

    return {
        "counts": {
            "matched_pairs": overall.get("matched_pairs", 0),
            "evaluated_policies": overall.get("evaluated_policies", 0),
            "accuracy": overall.get("accuracy"),
        },
        "evaluated_policy_files": overall.get("evaluated_policy_files", []),
        "elapsed_s": elapsed,
        "metrics_path": result["metrics_path"],
        "errors_path": result.get("errors_path"),
        "output_dir": str(comparison_dir),
    }
