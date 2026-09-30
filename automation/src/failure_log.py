import json
from datetime import datetime, timezone
from typing import Any

from automation.src.config_loader import get_run_dir

STEPS = ("translation", "translation_qa", "storage", "evaluation", "discrepancy_diagnosis")


def _failures_path(run_id: str):
    return get_run_dir(run_id) / "failures.json"


def _empty_log() -> dict[str, Any]:
    return {
        "updated_at": None,
        "translation": [],
        "translation_qa": [],
        "storage": [],
        "evaluation": [],
        "discrepancy_diagnosis": [],
        "step_failures": [],
    }


def load_failures(run_id: str) -> dict[str, Any]:
    path = _failures_path(run_id)
    if not path.exists():
        return _empty_log()
    return json.loads(path.read_text(encoding="utf-8"))


def _write_failures(run_id: str, log: dict[str, Any]) -> None:
    log["updated_at"] = datetime.now(timezone.utc).isoformat()
    path = _failures_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def _item_key(step: str, entry: dict[str, Any]) -> str:
    if step in ("translation", "translation_qa", "storage"):
        return entry["filename"]
    if step in ("evaluation", "discrepancy_diagnosis"):
        return entry["policy_file"]
    raise ValueError(f"Unknown failure step: {step}")


def record_failure(run_id: str, step: str, entry: dict[str, Any]) -> None:
    if step not in STEPS:
        raise ValueError(f"Unknown failure step: {step}")

    log = load_failures(run_id)
    now = datetime.now(timezone.utc).isoformat()
    entry = {**entry, "at": entry.get("at", now)}

    items: list[dict[str, Any]] = log.get(step, [])
    key = _item_key(step, entry)
    items = [item for item in items if _item_key(step, item) != key]
    items.append(entry)
    log[step] = items
    _write_failures(run_id, log)


def clear_failure(run_id: str, step: str, item_key: str) -> None:
    if step not in STEPS:
        raise ValueError(f"Unknown failure step: {step}")

    path = _failures_path(run_id)
    if not path.exists():
        return

    log = load_failures(run_id)
    items: list[dict[str, Any]] = log.get(step, [])
    filtered = [item for item in items if _item_key(step, item) != item_key]
    if len(filtered) == len(items):
        return

    log[step] = filtered
    _write_or_delete(run_id, log)


def record_step_failure(run_id: str, step: str, message: str) -> None:
    """Record a fatal, whole-step failure (e.g. the step raised before
    producing any per-item results) rather than a per-item failure.
    """
    log = load_failures(run_id)
    step_failures: list[dict[str, Any]] = log.get("step_failures", [])
    step_failures = [item for item in step_failures if item.get("step") != step]
    step_failures.append(
        {"step": step, "message": message, "at": datetime.now(timezone.utc).isoformat()}
    )
    log["step_failures"] = step_failures
    _write_failures(run_id, log)


def clear_step_failure(run_id: str, step: str) -> None:
    path = _failures_path(run_id)
    if not path.exists():
        return

    log = load_failures(run_id)
    step_failures: list[dict[str, Any]] = log.get("step_failures", [])
    filtered = [item for item in step_failures if item.get("step") != step]
    if len(filtered) == len(step_failures):
        return

    log["step_failures"] = filtered
    _write_or_delete(run_id, log)


def _write_or_delete(run_id: str, log: dict[str, Any]) -> None:
    path = _failures_path(run_id)
    if (
        not log.get("translation")
        and not log.get("translation_qa")
        and not log.get("storage")
        and not log.get("evaluation")
        and not log.get("discrepancy_diagnosis")
        and not log.get("step_failures")
    ):
        path.unlink(missing_ok=True)
        return
    _write_failures(run_id, log)


def summarize_failures(run_id: str) -> dict[str, int]:
    log = load_failures(run_id)
    return {
        "translation": len(log.get("translation", [])),
        "translation_qa": len(log.get("translation_qa", [])),
        "storage": len(log.get("storage", [])),
        "evaluation": len(log.get("evaluation", [])),
        "discrepancy_diagnosis": len(log.get("discrepancy_diagnosis", [])),
        "step_failures": len(log.get("step_failures", [])),
    }
