"""Single place that turns a project name into its on-disk layout.

    data/<project>/
        raw/
            raw_files/            original PDF / HTML / TXT
            extracted_txt/        <stem>.txt waiting for OCR / a fix (stage 1.1); only unresolved files stay
            gold/                 <stem>.txt hand-cleaned texts (optional source for TXT -> Markdown)
            input_process.log     what stage 1 did to each file; WARNING = needs your attention
        preprocessed/             stage 1-2 output, open to human edits (stage 3)
            cleaned_txt/          <stem>.txt that passed the quality check (kept across runs)
            cleaned_markdown/
            translation_markdown/
            header_check.log      pairs whose headings differ (written by the `validate` step)
        processed/                pairs YOU moved here after checking them; the input to evaluation (stage 4)
            cleaned_markdown/
            translation_markdown/
        automation/                  evaluation outputs (one folder; a rerun cleans it)
        mapping/index_schema.yaml
        corrections/manual_overwrites.yaml
        long_policy_encoding.csv  golden indicator labels (comparison only)

Folder names live in `constants.py`; which project is active is a `config`
choice (`project:` in pipeline_config.yaml). Nothing else should build a
`data/...` path by hand.
"""

from dataclasses import dataclass
from pathlib import Path

from automation.src.constants import (
    AUTOMATION_DIRNAME,
    CLEANED_MARKDOWN_DIRNAME,
    CLEANED_TXT_DIRNAME,
    CORRECTIONS_DIRNAME,
    DATA_ROOT,
    EXTRACTED_TXT_DIRNAME,
    GOLD_DIRNAME,
    GOLDEN_CSV_FILENAME,
    INDEX_SCHEMA_RELPATH,
    INPUT_LOG_FILENAME,
    HEADER_CHECK_LOG_FILENAME,
    MANUAL_OVERWRITES_FILENAME,
    PREPROCESSED_DIRNAME,
    PROCESSED_DIRNAME,
    RAW_DIRNAME,
    RAW_FILES_DIRNAME,
    TRANSLATION_MARKDOWN_DIRNAME,
)


@dataclass(frozen=True)
class ProjectDirs:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / RAW_DIRNAME

    @property
    def raw_files(self) -> Path:
        return self.raw / RAW_FILES_DIRNAME

    @property
    def extracted_txt(self) -> Path:
        return self.raw / EXTRACTED_TXT_DIRNAME

    @property
    def cleaned_txt(self) -> Path:
        return self.preprocessed / CLEANED_TXT_DIRNAME

    @property
    def gold(self) -> Path:
        return self.raw / GOLD_DIRNAME

    @property
    def input_log(self) -> Path:
        return self.raw / INPUT_LOG_FILENAME

    @property
    def header_check_log(self) -> Path:
        return self.preprocessed / HEADER_CHECK_LOG_FILENAME

    @property
    def preprocessed(self) -> Path:
        return self.root / PREPROCESSED_DIRNAME

    @property
    def processed(self) -> Path:
        return self.root / PROCESSED_DIRNAME

    @property
    def preprocessed_cleaned_markdown(self) -> Path:
        return self.preprocessed / CLEANED_MARKDOWN_DIRNAME

    @property
    def preprocessed_translation_markdown(self) -> Path:
        return self.preprocessed / TRANSLATION_MARKDOWN_DIRNAME

    @property
    def processed_cleaned_markdown(self) -> Path:
        return self.processed / CLEANED_MARKDOWN_DIRNAME

    @property
    def processed_translation_markdown(self) -> Path:
        return self.processed / TRANSLATION_MARKDOWN_DIRNAME

    @property
    def automation(self) -> Path:
        return self.root / AUTOMATION_DIRNAME

    @property
    def golden_csv(self) -> Path:
        return self.root / GOLDEN_CSV_FILENAME

    @property
    def index_schema(self) -> Path:
        return self.root / INDEX_SCHEMA_RELPATH

    @property
    def manual_overwrites(self) -> Path:
        return self.root / CORRECTIONS_DIRNAME / MANUAL_OVERWRITES_FILENAME


def project_dirs(project: str) -> ProjectDirs:
    return ProjectDirs(DATA_ROOT / project)
