"""Stage 1.2 - TXT -> Markdown (placeholder).

Stage 1.1 (PDF/HTML/TXT -> quality-checked text) is implemented in
``automation.src.input_process``. It leaves the texts that passed in
``data/<project>/raw/cleaned_txt/<POLICY>.txt``; hand-cleaned texts can live in
``data/<project>/raw/gold/``. Which of the two this stage converts is
``input_process.markdown_source`` (``cleaned_txt`` by default); the folder is
``InputProcessConfig.markdown_source_dir``.

This stage will turn them into ``data/<project>/preprocessed/cleaned_markdown/
<POLICY>.cleaned.md``, the input of the ``translation_md`` step. The scripts for
it will be dropped in later; until then the pipeline starts at stage 2 from an
existing ``cleaned_markdown`` folder (see `paths.ProjectDirs`).
"""

from pathlib import Path


class IngestNotImplementedError(NotImplementedError):
    pass


def run_ingest_step(source_dir: Path, cleaned_markdown_dir: Path) -> dict:
    """`source_dir` is ``InputProcessConfig.markdown_source_dir``."""
    raise IngestNotImplementedError(
        "Stage 1.2 (TXT -> Markdown) is not implemented yet. Place its scripts in "
        f"automation/src/ingest/, or put cleaned Markdown in {cleaned_markdown_dir} "
        f"(text to convert: {source_dir}) and run from translation_md."
    )
