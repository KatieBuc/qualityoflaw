import time

from automation.src.evaluate_accuracy import run_accuracy_evaluation

from automation.src.config_loader import (
    ResolvedPipelineConfig,
    evaluation_output_names,
    get_run_dir,
    resolve_results_dir,
)

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
        manual_overwrites_path=str(config.paths.manual_overwrites),
        confidence_config=config.comparison_confidence,
        confidence_context={
            "run_id": run_id,
            "model": config.evaluation_model.deployment,
            "temperature": config.evaluation_model.temperature,
            "evaluation_dir": str(evaluation_dir),
            "primary_method": config.confidence.primary,
        },
    )
    elapsed = round(time.time() - start, 2)

    if result is None:
        raise RuntimeError("Accuracy comparison produced no results.")

    metrics = result["metrics_summary"]
    overall = metrics.get("overall", {})
    confidence = result.get("confidence_report") or {}
    primary_coverage = (confidence.get("coverage") or {}).get("primary") or {}

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
        "confidence_report_path": result.get("confidence_report_path"),
        "low_confidence_path": result.get("low_confidence_path"),
        "confidence": {
            "status": confidence.get("status"),
            "availability": (confidence.get("confidence_source") or {}).get("availability"),
            "primary_threshold": primary_coverage.get("threshold"),
            "coverage_rate": primary_coverage.get("coverage_rate"),
            "flagged": primary_coverage.get("flagged"),
            "errors_captured": primary_coverage.get("errors_captured"),
        },
        "output_dir": str(comparison_dir),
    }
