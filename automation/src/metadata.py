import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from automation.src.config_loader import get_run_dir, snapshot_configs
from automation.src.constants import DEFAULT_DATA_ROOT


def generate_run_id() -> str:
    base = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = base
    suffix = 2
    while get_run_dir(run_id).exists():
        run_id = f"{base}_{suffix}"
        suffix += 1
    return run_id


def init_run_metadata(
    run_id: str,
    experiment_name: str,
    small_scale: bool,
    config_summary: dict[str, Any],
    pipeline_config_path: Path,
    model_config_path: Path,
) -> Path:
    run_dir = get_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_configs(run_dir, pipeline_config_path, model_config_path)

    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "status": "running",
        "timestamps": {
            "created_at": now,
            "updated_at": now,
        },
        "config": config_summary,
        "execution_scope": {
            "steps_executed": [],
            "small_scale": small_scale,
            "evaluated_policy_files": [],
        },
        "file_counts": {},
        "failures": {},
        "token_usage": {},
        "timing_seconds": {},
    }

    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def load_metadata(run_id: str) -> dict[str, Any]:
    metadata_path = get_run_dir(run_id) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found for run_id '{run_id}': {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _merge_execution_scope(existing: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    for key, v in value.items():
        if key == "steps_executed" and isinstance(v, list):
            executed = list(existing.get("steps_executed", []))
            for step in v:
                if step not in executed:
                    executed.append(step)
            existing["steps_executed"] = executed
        else:
            existing[key] = v
    return existing


def update_metadata(run_id: str, **updates: Any) -> dict[str, Any]:
    metadata = load_metadata(run_id)
    metadata.setdefault("timestamps", {})["updated_at"] = datetime.now(timezone.utc).isoformat()

    for key, value in updates.items():
        if key == "execution_scope" and isinstance(value, dict):
            existing = metadata.setdefault("execution_scope", {})
            metadata["execution_scope"] = _merge_execution_scope(existing, value)
        elif key in ("file_counts", "timing_seconds", "token_usage", "failures") and isinstance(value, dict):
            existing = metadata.get(key, {})
            existing.update(value)
            metadata[key] = existing
        elif key == "config" and isinstance(value, dict):
            metadata["config"] = value
        else:
            metadata[key] = value

    metadata_path = get_run_dir(run_id) / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def ensure_run_exists(run_id: str) -> Path:
    run_dir = get_run_dir(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}. Create a run with translation first."
        )
    return run_dir


def validate_run_for_steps(run_id: str, steps: list[str]) -> Path:
    run_dir = ensure_run_exists(run_id)

    if "storage" in steps:
        translation_dir = run_dir / "translation"
        if not translation_dir.is_dir():
            raise FileNotFoundError(f"Translation output required for storage: {translation_dir}")
        txt_files = list(translation_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No translated .txt files in {translation_dir}")

    if "evaluation" in steps:
        translation_dir = run_dir / "translation"
        if not translation_dir.is_dir():
            raise FileNotFoundError(
                f"Translation output required for evaluation: {translation_dir}"
            )
        txt_files = list(translation_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No translated .txt files in {translation_dir}")

        rag_store_dir = run_dir / "rag_store"
        if not rag_store_dir.is_dir():
            raise FileNotFoundError(
                f"RAG store required for evaluation: {rag_store_dir}. Run the storage step first."
            )
        store_files = list(rag_store_dir.glob("*.json"))
        if not store_files:
            raise FileNotFoundError(f"No RAG store files in {rag_store_dir}")

    if "comparison" in steps:
        evaluation_dir = run_dir / "evaluation"
        if not evaluation_dir.is_dir():
            raise FileNotFoundError(
                f"Evaluation output required for comparison: {evaluation_dir}"
            )
        json_files = list(evaluation_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No evaluation .json files in {evaluation_dir}")

    return run_dir
