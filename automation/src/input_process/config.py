"""Stage 1 configuration.

Read straight from ``pipeline_config.yaml`` (the ``project`` key and the
``input_process`` block) instead of through ``load_pipeline_config``: stage 1
needs no model profiles or prompts, and must run before any of them matter.
Folder locations default to ``data/<project>/raw/...`` (see ``paths.py``) and can
each be overridden in the config file.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from automation.src.constants import DEFAULT_PIPELINE_CONFIG, DEFAULT_PROJECT, PROJECT_ROOT
from automation.src.paths import project_dirs

DEFAULT_LANGUAGE = "Indonesian"
MARKDOWN_SOURCES = ("cleaned_txt", "gold")


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
    #: Regex with one group that picks the language out of a policy name
    #: (e.g. ``DomViol - ([A-Za-z]+)``). Names it does not match use `language`.
    language_pattern: str = ""
    #: Hand-cleaned texts, an alternative input for stage 1.2 (TXT -> Markdown).
    gold_dir: Path | None = None
    #: Which folder stage 1.2 converts: "cleaned_txt" (this stage's output) or "gold".
    markdown_source: str = "cleaned_txt"

    @property
    def markdown_source_dir(self) -> Path:
        """The folder stage 1.2 reads, as chosen by `markdown_source`."""
        return self.gold_dir if self.markdown_source == "gold" else self.cleaned_txt_dir

    @property
    def language_regex(self) -> "re.Pattern | None":
        return re.compile(self.language_pattern) if self.language_pattern else None

    def language_for(self, stem: str) -> str:
        """Language of one policy: from its name when `language_pattern` matches, else `language`."""
        from automation.src.input_process.quality.metrics import language_of

        return language_of(stem, self.language_regex, self.language)

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

    pattern = str(block.get("language_pattern") or "")
    if pattern:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"input_process.language_pattern is not a valid regex: {exc}") from exc
        if compiled.groups < 1:
            raise ValueError("input_process.language_pattern needs one group around the language name")

    source = str(block.get("markdown_source") or "cleaned_txt")
    if source not in MARKDOWN_SOURCES:
        raise ValueError(
            f"input_process.markdown_source must be one of {', '.join(MARKDOWN_SOURCES)}, got '{source}'"
        )

    stems = (data.get("paths") or {}).get("small_scale_stems") or ()
    return InputProcessConfig(
        project=project,
        language=str(block.get("language") or DEFAULT_LANGUAGE),
        raw_files_dir=_resolve(block.get("raw_files_dir") or dirs.raw_files),
        extracted_txt_dir=_resolve(block.get("extracted_txt_dir") or dirs.extracted_txt),
        cleaned_txt_dir=_resolve(block.get("cleaned_txt_dir") or dirs.cleaned_txt),
        report_path=_resolve(block.get("report_path") or dirs.input_report),
        thresholds=dict(thresholds),
        language_pattern=pattern,
        gold_dir=_resolve(block.get("gold_dir") or dirs.gold),
        markdown_source=source,
        small_scale_stems=tuple(str(s) for s in stems),
    )


def load_input_process_config(
    pipeline_config_path: Path | None = None, *, project: str | None = None
) -> InputProcessConfig:
    path = pipeline_config_path or DEFAULT_PIPELINE_CONFIG
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return parse_input_process_config(data, project=project)
