import os
import json
import sys
import argparse
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

from quality_eval.v1.prompts.prompt_loader import generate_judge_prompt
from quality_eval.v1.criteria import (
    CRITERIA_FILES,
    EXPECTED_INDICATOR_COUNT,
    EXPECTED_INDICATOR_IDS,
    PROMPT_VERSION,
)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0


class CriterionResult(BaseModel):
    id: str = Field(description="The ID of the indicator being evaluated. Only contain numbers and a single dot for seperation.")
    indicator: str = Field(description="The name of the indicator being evaluated.")
    included: str = Field(description="Must be 'Yes' or 'No'.")
    evidence: Optional[str] = Field(default=None, description="Exact quote from the text if included is 'Yes', otherwise null.")
    rationale: str = Field(description="Explanation of how the text addresses or fails to address this indicator.")


class PolicyEvaluationResponse(BaseModel):
    evaluation_results: List[CriterionResult]


load_dotenv()


def resolve_policy_paths(policy_input: str) -> list[str]:
    path = Path(policy_input)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find path: {policy_input}")

    if path.is_file():
        if path.suffix.lower() != ".txt":
            raise ValueError(f"Policy file must be a .txt file: {policy_input}")
        return [str(path)]

    if path.is_dir():
        policy_files = sorted(
            str(item)
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() == ".txt"
        )
        if not policy_files:
            raise ValueError(f"No .txt policy files found in folder: {policy_input}")
        return policy_files

    raise ValueError(f"Policy input must be a file or folder: {policy_input}")


def run_policy_evaluation(client: OpenAI, deployment_name: str, prompt: str) -> dict:
    response = client.beta.chat.completions.parse(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt}],
        response_format=PolicyEvaluationResponse,
        temperature=0.1,
    )

    parsed_output = response.choices[0].message.parsed
    if not parsed_output:
        raise ValueError("LLM return structure error.")

    output_dict = parsed_output.model_dump()
    output_dict["evaluation_results"] = {
        item["id"]: item for item in output_dict["evaluation_results"]
    }
    return output_dict


def call_with_retry(client: OpenAI, deployment_name: str, prompt: str) -> dict:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return run_policy_evaluation(client, deployment_name, prompt)
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY ** (attempt + 1)
            print(
                f"Retry {attempt + 1}/{MAX_RETRIES} after error: {exc}",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise last_error  # pragma: no cover


def validate_completeness(evaluation_results: dict) -> tuple[bool, list[str]]:
    found_ids = set(evaluation_results.keys())
    missing = sorted(EXPECTED_INDICATOR_IDS - found_ids)
    extra = sorted(found_ids - set(EXPECTED_INDICATOR_IDS))
    issues: list[str] = []
    if missing:
        issues.append(f"Missing indicators ({len(missing)}): {', '.join(missing)}")
    if extra:
        issues.append(f"Unexpected indicators ({len(extra)}): {', '.join(extra)}")
    if len(found_ids) != EXPECTED_INDICATOR_COUNT:
        issues.append(
            f"Expected {EXPECTED_INDICATOR_COUNT} indicators, found {len(found_ids)}"
        )
    return len(issues) == 0, issues


def evaluate_policy(
    policy_path: str,
    criteria_folder: str,
    template_path: str,
    client: OpenAI,
    deployment_name: str,
) -> dict:
    evaluated_at = datetime.now(timezone.utc).isoformat()
    final_report = {
        "policy_file": os.path.basename(policy_path),
        "model": deployment_name,
        "evaluated_at": evaluated_at,
        "prompt_version": PROMPT_VERSION,
        "completed_dimensions": [],
        "failed_dimensions": [],
        "errors": [],
        "evaluation_results": {},
    }

    print(f"Starting evaluation for {len(CRITERIA_FILES)} criteria dimensions...\n")

    for idx, criteria_file in enumerate(CRITERIA_FILES, 1):
        print("--------------------------------------------------")
        print(f"[{idx}/{len(CRITERIA_FILES)}] Processing: {criteria_file}")

        try:
            final_prompt = generate_judge_prompt(
                criteria_file_path=os.path.join(criteria_folder, criteria_file),
                template_file_path=template_path,
                policy_file_path=policy_path,
            )

            print("Requesting...")
            batch_result = call_with_retry(client, deployment_name, final_prompt)
            batch_evals = batch_result.get("evaluation_results", {})

            for key, item in batch_evals.items():
                if key in final_report["evaluation_results"]:
                    print(
                        f"Warning: ID '{key}' ({item['indicator']}) already exists; "
                        "overwriting with new outcome."
                    )

            final_report["evaluation_results"].update(batch_evals)
            final_report["completed_dimensions"].append(criteria_file)
            print(f"{criteria_file} completed")

        except Exception as exc:
            error_msg = f"{criteria_file}: {exc}"
            final_report["failed_dimensions"].append(criteria_file)
            final_report["errors"].append(error_msg)
            print(f"Error when processing {criteria_file}: {exc}", file=sys.stderr)
            traceback.print_exc()

    final_report["indicator_count"] = len(final_report["evaluation_results"])
    return final_report


def save_report(report: dict, output_folder: str, deployment_name: str, policy_path: str) -> str:
    os.makedirs(output_folder, exist_ok=True)
    dt = datetime.now().strftime("%d%m%Y%H%M%S")
    basename = os.path.basename(policy_path).replace(".txt", ".json")
    output_name = f"{dt}-{deployment_name}-{basename}"
    output_path = os.path.join(output_folder, output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return output_path


def finalize_and_save_report(
    final_report: dict,
    policy_path: str,
    output_folder: str,
    deployment_name: str,
    allow_partial: bool,
) -> tuple[bool, str | None]:
    is_complete, completeness_issues = validate_completeness(
        final_report["evaluation_results"]
    )
    has_failures = bool(final_report["failed_dimensions"])

    if not is_complete:
        for issue in completeness_issues:
            print(f"Completeness issue: {issue}", file=sys.stderr)

    if is_complete and not has_failures:
        output_path = save_report(
            final_report, output_folder, deployment_name, policy_path
        )
        print(f"Evaluation completed successfully: {output_path}")
        return True, output_path

    if allow_partial:
        output_path = save_report(
            final_report, output_folder, deployment_name, policy_path
        )
        print(f"Partial evaluation saved: {output_path}", file=sys.stderr)
        return False, output_path

    print(
        "Evaluation incomplete; report not saved. Use --allow-partial to save anyway.",
        file=sys.stderr,
    )
    return False, None


def run_evaluations(
    policy_input: str,
    output_folder: str,
    criteria_folder: str,
    template_path: str,
    allow_partial: bool = False,
    client: OpenAI | None = None,
    deployment_name: str | None = None,
) -> tuple[list[str], list[str]]:
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        raise RuntimeError("API key not found.")

    policy_paths = resolve_policy_paths(policy_input)
    deployment_name = deployment_name or os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
    client = client or OpenAI(
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    saved_paths: list[str] = []
    failed_policies: list[str] = []

    for index, policy_path in enumerate(policy_paths, 1):
        print("\n" + "=" * 50)
        print(f"Policy [{index}/{len(policy_paths)}]: {os.path.basename(policy_path)}")
        print("=" * 50)

        final_report = evaluate_policy(
            policy_path=policy_path,
            criteria_folder=criteria_folder,
            template_path=template_path,
            client=client,
            deployment_name=deployment_name,
        )

        success, output_path = finalize_and_save_report(
            final_report=final_report,
            policy_path=policy_path,
            output_folder=output_folder,
            deployment_name=deployment_name,
            allow_partial=allow_partial,
        )
        if success and output_path:
            saved_paths.append(output_path)
        else:
            failed_policies.append(os.path.basename(policy_path))
            if output_path:
                saved_paths.append(output_path)

    return saved_paths, failed_policies


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy quality evaluation.")
    parser.add_argument(
        "-p",
        "--policy",
        required=True,
        help="Filepath of a policy document, or a folder containing .txt policy files.",
    )
    parser.add_argument("-o", "--output_folder", required=True, help="Output folder for the evaluation.")
    parser.add_argument("-c", "--criteria_folder", required=True, help="Base folder of the criteria text files.")
    parser.add_argument("-t", "--template", required=True, help="Path of the prompt template text file.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Save incomplete reports when some dimensions fail (still exits non-zero).",
    )
    args = parser.parse_args()

    try:
        saved_paths, failed_policies = run_evaluations(
            policy_input=args.policy,
            output_folder=args.output_folder,
            criteria_folder=args.criteria_folder,
            template_path=args.template,
            allow_partial=args.allow_partial,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 50)
    print(f"Batch summary: {len(saved_paths)} report(s) saved, {len(failed_policies)} failed.")
    if failed_policies:
        print("Failed policies:", ", ".join(failed_policies), file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
