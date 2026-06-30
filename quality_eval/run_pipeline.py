import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from quality_eval.evaluate_accuracy import run_accuracy_evaluation
from quality_eval.run_eval_new import run_evaluations
from quality_eval.v1.criteria import PROMPTS_DIR, V1_DIR

load_dotenv()

PROJECT_ROOT = V1_DIR.parent.parent
DEFAULT_POLICY_DIR = V1_DIR / "translated_policy"
DEFAULT_GOLDEN_CSV = PROJECT_ROOT / "data" / "processed" / "long_policy_encoding.csv"
DEFAULT_TEMPLATE = PROMPTS_DIR / "prompt_template.txt"


def default_output_dir(model: str) -> Path:
    return V1_DIR / "result" / model


def default_error_export(model: str) -> Path:
    return V1_DIR / f"error_analysis_{model}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM policy evaluation and accuracy reporting in one command."
    )
    parser.add_argument(
        "-p",
        "--policy",
        default=str(DEFAULT_POLICY_DIR),
        help="Policy .txt file or folder of policy files (default: v1/translated_policy).",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        default=None,
        help="Folder for LLM JSON reports (default: v1/result/<model>/).",
    )
    parser.add_argument(
        "-c",
        "--criteria_folder",
        default=str(PROMPTS_DIR),
        help="Criteria prompts folder (default: v1/prompts).",
    )
    parser.add_argument(
        "-t",
        "--template",
        default=str(DEFAULT_TEMPLATE),
        help="Prompt template file (default: v1/prompts/prompt_template.txt).",
    )
    parser.add_argument(
        "-g",
        "--golden_csv",
        default=str(DEFAULT_GOLDEN_CSV),
        help="Golden dataset CSV path.",
    )
    parser.add_argument(
        "-e",
        "--export_errors",
        default=None,
        help="Error analysis CSV path (default: v1/error_analysis_<model>.csv).",
    )
    parser.add_argument(
        "-m",
        "--export_metrics",
        default=None,
        help="Metrics summary JSON path (default: alongside error CSV).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Save incomplete LLM reports when some dimensions fail.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip LLM evaluation and only run accuracy against existing JSON reports.",
    )
    parser.add_argument(
        "--skip-accuracy",
        action="store_true",
        help="Run LLM evaluation only; skip accuracy comparison.",
    )
    args = parser.parse_args()

    model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
    output_folder = args.output_folder or str(default_output_dir(model))
    export_errors = args.export_errors or str(default_error_export(model))

    if args.skip_eval and args.skip_accuracy:
        print("Error: Cannot skip both evaluation and accuracy.", file=sys.stderr)
        sys.exit(1)

    failed_policies: list[str] = []

    if not args.skip_eval:
        print("Step 1/2: Running LLM evaluation...")
        try:
            _, failed_policies = run_evaluations(
                policy_input=args.policy,
                output_folder=output_folder,
                criteria_folder=args.criteria_folder,
                template_path=args.template,
                allow_partial=args.allow_partial,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        if failed_policies:
            print(
                f"LLM evaluation finished with failures: {', '.join(failed_policies)}",
                file=sys.stderr,
            )
            if not args.skip_accuracy and not args.allow_partial:
                sys.exit(1)

    if args.skip_accuracy:
        if failed_policies:
            sys.exit(1)
        print("Skipping accuracy comparison (--skip-accuracy).")
        sys.exit(0)

    print("\nStep 2/2: Running accuracy comparison...")
    try:
        result = run_accuracy_evaluation(
            golden_csv=args.golden_csv,
            llm_input=output_folder,
            export_errors=export_errors,
            export_metrics=args.export_metrics,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if result is None:
        sys.exit(1)

    print("\nPipeline completed.")
    print(f"  LLM reports folder: {output_folder}")
    print(f"  Metrics summary:    {result['metrics_path']}")
    if result["errors_path"]:
        print(f"  Error analysis:     {result['errors_path']}")
    else:
        print("  Error analysis:     none (100% match)")

    sys.exit(1 if failed_policies else 0)


if __name__ == "__main__":
    main()
