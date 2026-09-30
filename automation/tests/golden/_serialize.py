"""Deterministic JSON-able view of config/result objects, for golden tests."""

import dataclasses
import enum
import json
from pathlib import Path

from automation.src.constants import PROJECT_ROOT

GOLDEN_DIR = Path(__file__).parent


def to_jsonable(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Path):
        try:
            return obj.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            return obj.as_posix()
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = [to_jsonable(v) for v in obj]
        return sorted(items, key=repr) if isinstance(obj, (set, frozenset)) else items
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return {"__repr__": repr(obj)}


def assert_matches_golden(name: str, actual, update_env: str = "UPDATE_GOLDEN"):
    """Compare `actual` (jsonable) with tests/golden/<name>.json.

    Set UPDATE_GOLDEN=1 to (re)write the baseline. A missing baseline fails,
    so a forgotten file can never turn the test into a no-op.
    """
    import os

    path = GOLDEN_DIR / f"{name}.json"
    text = json.dumps(actual, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if os.environ.get(update_env) == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        return
    assert path.exists(), f"missing golden file {path}; run with {update_env}=1 to create it"
    expected = path.read_text(encoding="utf-8")
    assert text == expected, f"output differs from golden {path.name}"
