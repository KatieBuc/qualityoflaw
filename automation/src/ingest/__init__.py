"""Stage 1 - input processing (placeholder).

Contract for the scripts that will fill this package:

1.1  PDF / HTML -> TXT. Assess TXT quality, apply preprocessing, and filter
     out low-quality files early. Reads ``data/<project>/raw/``.
1.2  TXT -> Markdown. Writes ``data/<project>/preprocessed/cleaned_markdown/
     <POLICY>.cleaned.md`` (the input to the `translation_md` step).

The scripts for this stage already exist and will be dropped in later; until
then the pipeline starts at stage 2 from an existing ``cleaned_markdown``
folder (see `paths.ProjectDirs`).
"""

from pathlib import Path


class IngestNotImplementedError(NotImplementedError):
    pass


def run_ingest_step(project_raw_dir: Path, cleaned_markdown_dir: Path) -> dict:
    raise IngestNotImplementedError(
        "Stage 1 (input processing) is not implemented yet. Place the PDF/HTML->TXT->Markdown "
        f"scripts in automation/src/ingest/, or put cleaned Markdown in {cleaned_markdown_dir} "
        f"(raw inputs: {project_raw_dir}) and run from translation_md."
    )
