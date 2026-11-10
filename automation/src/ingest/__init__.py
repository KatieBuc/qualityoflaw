"""Stage 1.2 - TXT -> Markdown.

Stage 1.1 (``automation.src.input_process``) leaves the texts that passed in
``data/<project>/preprocessed/cleaned_txt/<POLICY>.txt``; hand-cleaned texts can live in
``data/<project>/raw/gold/``. Which of the two is converted is
``input_process.markdown_source`` (``InputProcessConfig.markdown_source_dir``).

This stage writes ``data/<project>/preprocessed/cleaned_markdown/<POLICY>.cleaned.md``, the
input of the ``translation_md`` step. The converter is ``input_process.txt_to_md_v7``; its
vocabulary is ``input_process.heading_keywords`` and the heading depths of a language that has a
profile there (Indonesian) follow that language's convention.

    python -m automation.src.input_process markdown [--force] [--small-scale]
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from automation.src.input_process import heading_keywords as keywords
from automation.src.input_process import txt_to_md_v7 as converter
from automation.src.input_process.runlog import run_log

OUTPUT_SUFFIX = ".cleaned.md"
NO_STRUCTURE = "no heading structure detected"
NON_STATUTORY = "listed as non-statutory"


class IngestNotImplementedError(NotImplementedError):
    """Kept for callers of the former stub."""


def convert_text(text: str, stem: str, *, language: str | None = None, depth: str = "relative") -> tuple[str | None, str]:
    """Convert one text. Returns ``(markdown, hierarchy)``; ``markdown`` is None when the text has no
    heading structure (``hierarchy`` is then empty)."""
    profile = keywords.profile_for(language)
    prepared = converter.preprocess(converter._strip_toc_block(text))
    info = converter.analyse(prepared, profile)
    if not info["active"] and not info["bare_enabled"]:
        return None, ""
    chain = " > ".join(info["active"]) or "BARE-numbered"
    xref_pre = next((k for tag, k in keywords.XREF_PREFIX_FILENAME_TAGS.items() if tag in stem), None)
    md = converter.convert(prepared, stem, info, depth=depth, xref_pre=xref_pre, profile=profile)
    return md, chain


def run_ingest_step(
    source_dir: Path,
    cleaned_markdown_dir: Path,
    *,
    language_for: Callable[[str], str | None] = lambda stem: None,
    stems: set[str] | None = None,
    force: bool = False,
    log_path: Path | None = None,
    suffix: str = OUTPUT_SUFFIX,
    done_dirs: tuple[Path, ...] = (),
) -> dict:
    """Convert every ``*.txt`` of `source_dir` into `cleaned_markdown_dir`/<stem>.cleaned.md.

    An existing output is kept unless `force`. `stems` limits the run to those policies.
    Texts without a recognisable heading structure are not written and are listed under
    ``skipped``; with `log_path` every file is also logged (a WARNING for a text that was not converted). A policy that already has its
    Markdown in one of `done_dirs` (the ``processed/`` copy) is not converted again.
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Markdown source folder not found: {source_dir}")
    out_dir = Path(cleaned_markdown_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    converted: list[dict] = []
    kept: list[str] = []
    skipped: list[dict] = []
    for path in sorted(source_dir.glob("*.txt")):
        stem = path.stem
        if stems is not None and stem not in stems:
            continue
        target = out_dir / f"{stem}{suffix}"
        if (target.exists() or any((d / f"{stem}{suffix}").exists() for d in done_dirs)) and not force:
            kept.append(stem)
            continue
        if any(sub in path.name for sub in converter.SKIP_SUBSTRINGS):
            skipped.append({"stem": stem, "reason": NON_STATUTORY})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        md, chain = convert_text(text, stem, language=language_for(stem))
        if md is None:
            skipped.append({"stem": stem, "reason": NO_STRUCTURE})
            continue
        target.write_text(md, encoding="utf-8")
        converted.append({"stem": stem, "hierarchy": chain})

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "output_dir": str(out_dir),
        "summary": {"converted": len(converted), "kept": len(kept), "skipped": len(skipped)},
        "converted": converted,
        "skipped": skipped,
    }
    if log_path is not None:
        with run_log(Path(log_path)) as log:
            log.info("MARKDOWN CONVERSION started (%s -> %s)", source_dir, out_dir)
            for item in converted:
                log.info("MARKDOWN CONVERTED %s (%s)", item["stem"], item["hierarchy"])
            for item in skipped:
                log.warning("MARKDOWN SKIPPED %s: %s - write its Markdown by hand into %s to include it", item["stem"], item["reason"], out_dir)
            log.info(
                "MARKDOWN CONVERSION finished: %d converted, %d already there, %d skipped",
                len(converted), len(kept), len(skipped),
            )
    return result
