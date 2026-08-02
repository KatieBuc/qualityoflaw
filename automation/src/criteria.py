from pathlib import Path
from typing import Dict, FrozenSet, List, Set

import yaml

from automation.src.constants import PROJECT_ROOT

DEFAULT_INDEX_SCHEMA_PATH = PROJECT_ROOT / "data" / "mapping" / "index_schema.yaml"

CRITERIA_FILES: List[str] = [
    "01_scope_of_violence.txt",
    "02_institutional_mechanism.txt",
    "03_specialised_support_services.txt",
    "04_primary_prevention.txt",
    "05_stakeholder_engagement.txt",
    "06_intersectional_approach.txt",
    "07_budget_funding_sources.txt",
]

DIMENSION_ORDER: List[str] = [
    "scope_of_violence",
    "institutional_mechanism",
    "specialised_support_services",
    "primary_prevention",
    "stakeholder_engagement",
    "intersectional_approach",
    "budget_funding_sources",
]

DIMENSION_TO_CRITERIA_FILE: Dict[str, str] = dict(
    zip(DIMENSION_ORDER, CRITERIA_FILES)
)


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


_SCHEMA = load_index_schema()
EXPECTED_INDICATOR_IDS, INDICATOR_TO_DIMENSION, DIMENSION_INDICATORS = build_criteria_maps(_SCHEMA)
EXPECTED_INDICATOR_COUNT = len(EXPECTED_INDICATOR_IDS)


def get_indicator_dimension(indicator_id: str) -> str | None:
    return INDICATOR_TO_DIMENSION.get(str(indicator_id).strip())


def load_criteria_file_ids(criteria_file_path: Path) -> Set[str]:
    ids: Set[str] = set()
    with criteria_file_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) >= 1:
                ids.add(parts[0].strip())
    return ids


def validate_criteria_files(criteria_dir: Path) -> tuple[bool, List[str]]:
    issues: List[str] = []

    file_ids: Set[str] = set()
    for criteria_file in CRITERIA_FILES:
        path = criteria_dir / criteria_file
        if not path.exists():
            issues.append(f"Missing criteria file: {path}")
            continue
        file_ids.update(load_criteria_file_ids(path))

    missing_in_files = EXPECTED_INDICATOR_IDS - file_ids
    extra_in_files = file_ids - set(EXPECTED_INDICATOR_IDS)

    if missing_in_files:
        issues.append(f"Criteria files missing IDs from schema: {sorted(missing_in_files)}")
    if extra_in_files:
        issues.append(f"Criteria files contain unknown IDs: {sorted(extra_in_files)}")

    return len(issues) == 0, issues
