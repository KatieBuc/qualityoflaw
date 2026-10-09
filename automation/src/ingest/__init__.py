"""Stage 1.2 - TXT -> Markdown (placeholder).

Stage 1.1 (PDF/HTML/TXT -> quality-checked text) is implemented in
``automation.src.input_process``; it leaves the files that passed in
``data/<project>/raw/cleaned_txt/<POLICY>.txt``.

This stage will turn those into ``data/<project>/preprocessed/cleaned_markdown/
<POLICY>.cleaned.md``, the input of the ``translation_md`` step. The scripts for
it will be dropped in later; until then the pipeline starts at stage 2 from an
existing ``cleaned_markdown`` folder (see `paths.ProjectDirs`).
"""

from pathlib import Path


class IngestNotImplementedError(NotImplementedError):
    pass


def run_ingest_step(cleaned_txt_dir: Path, cleaned_markdown_dir: Path) -> dict:
    raise IngestNotImplementedError(
        "Stage 1.2 (TXT -> Markdown) is not implemented yet. Place its scripts in "
        f"automation/src/ingest/, or put cleaned Markdown in {cleaned_markdown_dir} "
        f"(stage 1.1 output: {cleaned_txt_dir}) and run from translation_md."
    )
