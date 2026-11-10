import re
from pathlib import Path
from typing import Dict, FrozenSet, List, Set

import yaml

from automation.src.constants import DEFAULT_PIPELINE_CONFIG, DEFAULT_PROJECT, PROMPTS_ROOT
from automation.src.paths import project_dirs

DEFAULT_INDEX_SCHEMA_PATH = project_dirs(DEFAULT_PROJECT).index_schema

#: A criteria file is numbered: ``01_scope_of_violence.txt``. ``prompt_template.txt`` is not one.
CRITERIA_FILE_RE = re.compile(r"^\d{2}_.+\.txt$")
#: An indicator id: ``1.1``, ``3.10``.
INDICATOR_ID_RE = re.compile(r"^\d+(?:\.\d+)*$")


def list_criteria_files(criteria_dir: Path) -> List[str]:
    """The criteria files of a prompt version, in order: every ``NN_<dimension>.txt`` in the folder.

    The files themselves say which dimensions a prompt version has; nothing here names them.
    """
    criteria_dir = Path(criteria_dir)
    if not criteria_dir.is_dir():
        return []
    return sorted(p.name for p in criteria_dir.iterdir() if p.is_file() and CRITERIA_FILE_RE.match(p.name))


def dimension_name(criteria_file: str) -> str:
    """``02_Protection_and_Justice.txt`` -> ``protection_and_justice``."""
    return re.sub(r"^\d+_", "", Path(criteria_file).stem).lower()


def _criterion_lines(path: Path):
    """``(id, indicator, field_count)`` of every line that starts like a criterion (``<id> | ...``)."""
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) >= 2 and INDICATOR_ID_RE.match(parts[0].strip()):
                yield parts[0].strip(), parts[1].strip(), len(parts)


#: A line that starts like a numbered criterion -- ``5. Marital rape / spousal exemption | Does ...`` -- but is not in the
#: ``<id> | <indicator> | <question>`` form. The evaluation reads such a line as a coding rule of the criterion above it, while
#: the judge, seeing a numbered item, may answer it as a criterion of its own (with a made-up id) and spoil the whole answer.
_SUSPECT_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+[^|\s][^|]*\|")


def _suspect_lines(path: Path):
    """``(line number, text)`` of every line that looks like a criterion but does not start with ``<id> |``."""
    with path.open(encoding="utf-8") as f:
        for number, raw in enumerate(f, start=1):
            line = raw.strip()
            if _SUSPECT_RE.match(line):
                yield number, line


def extract_schema_from_prompts(criteria_dir: Path, files: List[str] | None = None) -> dict:
    """The index schema (``{dimension: {indicator id: {"indicator": name}}}``), read straight from the criteria files.

    Used for a project that has no ``mapping/index_schema.yaml``: the prompt is then the only definition of its
    indicators. A line is a criterion only in the form the evaluation reads, ``id | indicator | question``.
    """
    criteria_dir = Path(criteria_dir)
    schema: dict = {}
    for name in files if files is not None else list_criteria_files(criteria_dir):
        path = criteria_dir / name
        if not path.exists():
            continue
        entries = {
            indicator_id: {"indicator": indicator}
            for indicator_id, indicator, n_fields in _criterion_lines(path)
            if n_fields == 3
        }
        schema[dimension_name(name)] = entries
    return schema


def active_settings(pipeline_config: Path | None = None) -> tuple[str, str | None]:
    """``(project, evaluation prompt version)`` from pipeline_config.yaml, read without building the whole config."""
    try:
        with Path(pipeline_config or DEFAULT_PIPELINE_CONFIG).open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        data = {}
    project = str(data.get("project") or DEFAULT_PROJECT)
    version = (data.get("evaluation") or {}).get("prompt_version")
    return project, (str(version) if version else None)


def load_index_schema(schema_path: Path | None = None) -> dict:
    path = schema_path or DEFAULT_INDEX_SCHEMA_PATH
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_criteria_maps(schema: dict) -> tuple[FrozenSet[str], Dict[str, str], Dict[str, List[str]]]:
    indicator_to_dimension: Dict[str, str] = {}
    dimension_indicators: Dict[str, List[str]] = {}

    for dimension, indicators in schema.items():
        ids = list(indicators.keys())
        dimension_indicators[dimension] = ids
        for indicator_id in ids:
            indicator_to_dimension[indicator_id] = dimension

    expected_ids = frozenset(indicator_to_dimension.keys())
    return expected_ids, indicator_to_dimension, dimension_indicators


def resolve_schema(project: str, criteria_dir: Path | None) -> dict:
    """The schema of the active project: its ``mapping/index_schema.yaml`` when it has one, else the prompt's own."""
    schema_path = project_dirs(project).index_schema
    if schema_path.exists():
        return load_index_schema(schema_path)
    if criteria_dir is not None and list_criteria_files(criteria_dir):
        return extract_schema_from_prompts(criteria_dir)
    return load_index_schema()


# What the rest of the code imports. They follow the ACTIVE project and evaluation prompt version of
# pipeline_config.yaml, so switching `project:` / `evaluation.prompt_version` switches the files, the dimensions and the
# indicator list together. (Code that is handed a criteria folder uses `list_criteria_files` / `extract_schema_from_prompts`
# on that folder instead, and does not depend on these.)
_PROJECT, _PROMPT_VERSION = active_settings()
_CRITERIA_DIR = PROMPTS_ROOT / "quality_eval" / _PROMPT_VERSION if _PROMPT_VERSION else None

CRITERIA_FILES: List[str] = list_criteria_files(_CRITERIA_DIR) if _CRITERIA_DIR else []
DIMENSION_ORDER: List[str] = [dimension_name(name) for name in CRITERIA_FILES]
DIMENSION_TO_CRITERIA_FILE: Dict[str, str] = dict(zip(DIMENSION_ORDER, CRITERIA_FILES))

_SCHEMA = resolve_schema(_PROJECT, _CRITERIA_DIR)
EXPECTED_INDICATOR_IDS, INDICATOR_TO_DIMENSION, DIMENSION_INDICATORS = build_criteria_maps(_SCHEMA)
EXPECTED_INDICATOR_COUNT = len(EXPECTED_INDICATOR_IDS)


def get_indicator_dimension(indicator_id: str) -> str | None:
    return INDICATOR_TO_DIMENSION.get(str(indicator_id).strip())


def load_criteria_file_ids(criteria_file_path: Path) -> Set[str]:
    """The indicator ids a criteria file defines (only the ``id | indicator | question`` lines; rubric lines are not ids)."""
    return {indicator_id for indicator_id, _, n_fields in _criterion_lines(Path(criteria_file_path)) if n_fields == 3}


def validate_criteria_files(
    criteria_dir: Path,
    expected_ids: FrozenSet[str] | Set[str] | None = None,
    files: List[str] | None = None,
) -> tuple[bool, List[str]]:
    """Check a prompt version's criteria files.

    Reports a missing file, a line that starts like a criterion but not in the ``id | indicator | question`` form the
    evaluation reads (it would be silently skipped), and -- when `expected_ids` is given -- ids missing from or unknown
    to that list. `files` defaults to the numbered files found in the folder.
    """
    criteria_dir = Path(criteria_dir)
    names = files if files is not None else list_criteria_files(criteria_dir)
    issues: List[str] = []

    file_ids: Set[str] = set()
    for criteria_file in names:
        path = criteria_dir / criteria_file
        if not path.exists():
            issues.append(f"Missing criteria file: {path}")
            continue
        for number, line in _suspect_lines(path):
            issues.append(
                f"{criteria_file}: line {number} looks like a criterion but is not 'id | indicator | question' "
                f"(it would be read as a coding rule, and the judge may answer it under a made-up id): {line[:70]!r}"
            )
        for indicator_id, _, n_fields in _criterion_lines(path):
            if n_fields != 3:
                issues.append(
                    f"{criteria_file}: criterion {indicator_id} has {n_fields} field(s) on its line; "
                    "expected 'id | indicator | question' on ONE line"
                )
        file_ids.update(load_criteria_file_ids(path))

    if not names:
        issues.append(f"No numbered criteria files (NN_*.txt) in {criteria_dir}")
    if expected_ids is not None:
        missing_in_files = set(expected_ids) - file_ids
        extra_in_files = file_ids - set(expected_ids)
        if missing_in_files:
            issues.append(f"Criteria files missing IDs from schema: {sorted(missing_in_files)}")
        if extra_in_files:
            issues.append(f"Criteria files contain unknown IDs: {sorted(extra_in_files)}")

    return len(issues) == 0, issues
