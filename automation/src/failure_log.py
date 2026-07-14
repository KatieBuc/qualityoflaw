import json
from datetime import datetime, timezone
from typing import Any

from automation.src.config_loader import get_run_dir

STEPS = ("translation", "storage", "evaluation", "discrepancy_diagnosis")


def _failures_path(run_id: str):
    return get_run_dir(run_id) / "failures.json"


def _empty_log() -> dict[str, Any]:
    return {
        "updated_at": None,
        "translation": [],
        "storage": [],
        "evaluation": [],
        "discrepancy_diagnosis": [],
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
    if step in ("translation", "storage"):
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
    if (
        not log.get("translation")
        and not log.get("storage")
        and not log.get("evaluation")
        and not log.get("discrepancy_diagnosis")
    ):
        path.unlink(missing_ok=True)
        return
    _write_failures(run_id, log)


def summarize_failures(run_id: str) -> dict[str, int]:
    log = load_failures(run_id)
    return {
        "translation": len(log.get("translation", [])),
        "storage": len(log.get("storage", [])),
        "evaluation": len(log.get("evaluation", [])),
        "discrepancy_diagnosis": len(log.get("discrepancy_diagnosis", [])),
    }
