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
        "created_at": now,
        "updated_at": now,
        "status": "running",
        "steps_executed": [],
        "small_scale": small_scale,
        "file_counts": {},
        "timing": {},
        "token_usage": {},
        "config": config_summary,
    }

    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def load_metadata(run_id: str) -> dict[str, Any]:
    metadata_path = get_run_dir(run_id) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found for run_id '{run_id}': {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def update_metadata(run_id: str, **updates: Any) -> dict[str, Any]:
    metadata = load_metadata(run_id)
    metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
    for key, value in updates.items():
        if key in ("file_counts", "timing", "token_usage", "config") and isinstance(value, dict):
            existing = metadata.get(key, {})
            existing.update(value)
            metadata[key] = existing
        elif key == "steps_executed" and isinstance(value, list):
            executed = list(metadata.get("steps_executed", []))
            for step in value:
                if step not in executed:
                    executed.append(step)
            metadata["steps_executed"] = executed
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

    if "evaluation" in steps:
        translation_dir = run_dir / "translation"
        if not translation_dir.is_dir():
            raise FileNotFoundError(
                f"Translation output required for evaluation: {translation_dir}"
            )
        txt_files = list(translation_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No translated .txt files in {translation_dir}")

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
