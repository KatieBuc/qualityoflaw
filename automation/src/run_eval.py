import time
from pathlib import Path

from quality_eval.run_eval_new import (
    PolicyEvaluationResponse,
    evaluate_policy,
    finalize_and_save_report,
)

from automation.src.config_loader import ResolvedPipelineConfig, get_run_dir
from automation.src.constants import SMALL_SCALE_FILES
from automation.src.llm.wrapper import AzureLLMWrapper


def _filter_policy_files(policy_dir: Path, small_scale: bool) -> list[Path]:
    files = sorted(policy_dir.glob("*.txt"))
    if small_scale:
        allowed = set(SMALL_SCALE_FILES)
        files = [f for f in files if f.name in allowed]
    return files


def run_evaluation_step(
    run_id: str,
    config: ResolvedPipelineConfig,
    wrapper: AzureLLMWrapper,
    *,
    small_scale: bool = False,
    allow_partial: bool = False,
) -> dict:
    run_dir = get_run_dir(run_id)
    policy_dir = run_dir / "translation"
    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_files = _filter_policy_files(policy_dir, small_scale)
    if not policy_files:
        raise FileNotFoundError(f"No policy files to evaluate in {policy_dir}")

    def complete_fn(prompt: str) -> dict:
        return wrapper.complete_structured(prompt, PolicyEvaluationResponse)

    start = time.time()
    saved_paths: list[str] = []
    failed_policies: list[str] = []

    for index, policy_path in enumerate(policy_files, 1):
        print("\n" + "=" * 50)
        print(f"Policy [{index}/{len(policy_files)}]: {policy_path.name}")
        print("=" * 50)

        final_report = evaluate_policy(
            policy_path=str(policy_path),
            criteria_folder=str(config.evaluation_criteria_dir),
            template_path=str(config.evaluation_template_path),
            deployment_name=wrapper.profile.deployment,
            complete_fn=complete_fn,
        )

        success, output_path = finalize_and_save_report(
            final_report=final_report,
            policy_path=str(policy_path),
            output_folder=str(output_dir),
            deployment_name=wrapper.profile.deployment,
            allow_partial=allow_partial,
        )
        if success and output_path:
            saved_paths.append(output_path)
        else:
            failed_policies.append(policy_path.name)
            if output_path:
                saved_paths.append(output_path)

    elapsed = round(time.time() - start, 2)
    total = len(policy_files)
    succeeded = total - len(failed_policies)

    return {
        "counts": {
            "total": total,
            "succeeded": succeeded,
            "failed": len(failed_policies),
            "saved_reports": len(saved_paths),
        },
        "failed_policies": failed_policies,
        "elapsed_s": elapsed,
        "output_dir": str(output_dir),
    }
