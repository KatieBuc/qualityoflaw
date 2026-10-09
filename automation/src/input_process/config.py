"""Stage 1 configuration.

Read straight from ``pipeline_config.yaml`` (the ``project`` key and the
``input_process`` block) instead of through ``load_pipeline_config``: stage 1
needs no model profiles or prompts, and must run before any of them matter.
Folder locations default to ``data/<project>/raw/...`` (see ``paths.py``) and can
each be overridden in the config file.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from automation.src.constants import DEFAULT_PIPELINE_CONFIG, DEFAULT_PROJECT, PROJECT_ROOT
from automation.src.paths import project_dirs

DEFAULT_LANGUAGE = "Indonesian"


@dataclass(frozen=True)
class InputProcessConfig:
    project: str
    language: str
    raw_files_dir: Path
    extracted_txt_dir: Path
    cleaned_txt_dir: Path
    report_path: Path
    thresholds: dict = field(default_factory=dict)
    small_scale_stems: tuple[str, ...] = ()

    @property
    def raw_dir(self) -> Path:
        """Folder that held the originals before ``migrate-raw`` (``data/<project>/raw``)."""
        return project_dirs(self.project).raw


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_input_process_config(data: dict, *, project: str | None = None) -> InputProcessConfig:
    project = project or str(data.get("project") or DEFAULT_PROJECT)
    block = data.get("input_process") or {}
    dirs = project_dirs(project)

    thresholds = block.get("thresholds") or {}
    if not isinstance(thresholds, dict):
        raise ValueError("input_process.thresholds must be a mapping of threshold name -> value")

    stems = (data.get("paths") or {}).get("small_scale_stems") or ()
    return InputProcessConfig(
        project=project,
        language=str(block.get("language") or DEFAULT_LANGUAGE),
        raw_files_dir=_resolve(block.get("raw_files_dir") or dirs.raw_files),
        extracted_txt_dir=_resolve(block.get("extracted_txt_dir") or dirs.extracted_txt),
        cleaned_txt_dir=_resolve(block.get("cleaned_txt_dir") or dirs.cleaned_txt),
        report_path=_resolve(block.get("report_path") or dirs.input_report),
        thresholds=dict(thresholds),
        small_scale_stems=tuple(str(s) for s in stems),
    )


def load_input_process_config(
    pipeline_config_path: Path | None = None, *, project: str | None = None
) -> InputProcessConfig:
    path = pipeline_config_path or DEFAULT_PIPELINE_CONFIG
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return parse_input_process_config(data, project=project)
