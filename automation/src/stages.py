"""Pipeline steps for stage 1 (input), stage 3 (human adjustment) and the start of stage 4.

    extract         originals -> extracted_txt/<name>.txt (policies in neither extracted_txt nor cleaned_txt)
    quality_check   extracted_txt -> quality check and fix -> preprocessed/cleaned_txt   (breakpoint: OCR / eliminated files)
    txt_to_md       cleaned_txt (or gold) -> preprocessed/cleaned_markdown
    human_review    optional pause so a person can edit cleaned/translation markdown (breakpoint, off by default)
    validate        header check of every pair in preprocessed/ -> header_check.log (breakpoint: nothing in processed/ yet)

Nothing is ever moved to ``processed/`` by the pipeline: you move the pairs you accept. Each step is idempotent. A
breakpoint is raised as `PipelineBreakpoint`; `run_pipeline` prints it and stops, and running the same command
again continues.
"""

import time
from pathlib import Path

import yaml

from automation.src.constants import HEADER_CHECK_LOG_FILENAME
from automation.src.breakpoints import DOCS_INPUT_PROCESS, DOCS_PIPELINE, PipelineBreakpoint
from automation.src.config_loader import ResolvedPipelineConfig
from automation.src.ingest import run_ingest_step
from automation.src.input_process.config import InputProcessConfig, load_input_process_config
from automation.src.input_process.extract import extract_all
from automation.src.input_process.pipeline import check_all
from automation.src.markdown.policy_files import policy_stem
from automation.src.input_process.runlog import run_log
from automation.src.markdown.validate import validate_directories

SKIP_BLOCKED_HINT = "re-run with --skip-blocked to continue with the files that are ready"
MAX_LISTED = 8


def _stems(config: ResolvedPipelineConfig, small_scale: bool) -> tuple[str, ...] | None:
    return tuple(config.paths.small_scale_stems) if small_scale else None


def _listed(items: list[str]) -> list[str]:
    shown = items[:MAX_LISTED]
    return shown + ([f"... and {len(items) - MAX_LISTED} more"] if len(items) > MAX_LISTED else [])


def _skipped(reason: str, started: float) -> dict:
    return {"counts": {"total": 0, "skipped": 1}, "elapsed_s": round(time.time() - started, 2), "skipped_reason": reason}


def _input_config(config: ResolvedPipelineConfig) -> InputProcessConfig:
    return load_input_process_config(config.pipeline_config_path)


# --------------------------------------------------------------------------- #
# Stage 1.1                                                                   #
# --------------------------------------------------------------------------- #
def run_extract_step(config: ResolvedPipelineConfig, *, small_scale: bool = False) -> dict:
    """PDF / HTML / TXT originals -> ``extracted_txt/<name>.txt``, for the policies in neither
    ``cleaned_txt/`` nor ``extracted_txt/``. Judges nothing: a scanned PDF gives an empty file."""
    start = time.time()
    ip = _input_config(config)
    if not ip.raw_files_dir.is_dir():
        return _skipped(f"no originals folder ({ip.raw_files_dir})", start)

    results = extract_all(ip, small_scale=small_scale)
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    return {
        "counts": {
            "total": len(results),
            "succeeded": by_status.get("extracted", 0),
            "skipped": sum(n for status, n in by_status.items() if status.startswith("skipped")),
            "failed": by_status.get("failed", 0),
            "unsupported": by_status.get("unsupported", 0),
        },
        "elapsed_s": round(time.time() - start, 2),
        "failures": [(r.source, r.detail) for r in results if r.status in ("failed", "unsupported")],
        "log_path": str(ip.log_path),
    }


def run_quality_check_step(config: ResolvedPipelineConfig, *, small_scale: bool = False, skip_blocked: bool = False) -> dict:
    """Quality-check every text waiting in ``extracted_txt/``; passed texts go to ``cleaned_txt/``."""
    start = time.time()
    ip = _input_config(config)
    if not ip.extracted_txt_dir.is_dir():
        return _skipped(f"nothing to check ({ip.extracted_txt_dir} does not exist; run the extract step first)", start)

    report = check_all(ip, write_cleaned=True, small_scale=small_scale)
    summary = report["summary"]
    needs_ocr, eliminated = report["needs_ocr"], report["eliminated"]
    result = {
        "counts": {
            "total": summary["files"],
            "succeeded": summary["passed"],
            "needs_ocr": summary["needs_ocr"],
            "eliminated": summary["eliminated"],
            "skipped": 0,
            "failed": 0,
        },
        "elapsed_s": round(time.time() - start, 2),
        "log_path": str(ip.log_path),
    }

    unresolved = len(needs_ocr) + len(eliminated)
    if unresolved and not skip_blocked:
        details = [
            f"{summary['passed']} of {summary['files']} policies passed the quality check and are in {ip.cleaned_txt_dir}.",
            f"{len(needs_ocr)} need OCR and {len(eliminated)} were eliminated; they stay in {ip.extracted_txt_dir}.",
        ]
        details += [f"needs OCR: {e['pdf'] or e['stem']}" for e in needs_ocr[:MAX_LISTED]]
        details += [f"eliminated: {e['stem']}" for e in eliminated[:MAX_LISTED]]
        raise PipelineBreakpoint(
            step="quality_check",
            title="some files need your attention",
            details=details,
            next_steps=[
                f"Open the log {ip.log_path}: the WARNING lines name every file with the quality indicators that failed.",
                f"For a PDF that needs OCR: OCR it and replace {ip.extracted_txt_dir / '<name>.txt'} with the text.",
                f"For an eliminated file: fix {ip.extracted_txt_dir / '<name>.txt'} (it is checked again), or move it "
                f"into {ip.cleaned_txt_dir} if you accept it as it is.",
                "Files you do not want to include can stay where they are.",
            ],
            docs=[DOCS_INPUT_PROCESS, DOCS_PIPELINE],
            override=SKIP_BLOCKED_HINT,
        )
    return result


# --------------------------------------------------------------------------- #
# Stage 1.2                                                                   #
# --------------------------------------------------------------------------- #
def run_txt_to_md_step(config: ResolvedPipelineConfig, *, small_scale: bool = False) -> dict:
    start = time.time()
    ip = _input_config(config)
    source = ip.markdown_source_dir
    if not source.is_dir():
        return _skipped(f"no text folder to convert ({source})", start)

    suffix = config.paths.markdown_input_suffix
    stems = _stems(config, small_scale)
    result = run_ingest_step(
        source,
        config.paths.markdown_input_dir,
        language_for=ip.language_for,
        stems=set(stems) if stems else None,
        suffix=suffix,
        done_dirs=(config.paths.processed_cleaned_markdown_dir,),
        log_path=ip.log_path,
    )
    summary = result["summary"]
    return {
        "counts": {
            "total": summary["converted"] + summary["kept"] + summary["skipped"],
            "succeeded": summary["converted"],
            "skipped": summary["kept"],
            "failed": 0,
            "no_structure": summary["skipped"],
        },
        "elapsed_s": round(time.time() - start, 2),
        "unconverted": [item["stem"] for item in result["skipped"]],
        "source": str(source),
    }


# --------------------------------------------------------------------------- #
# Stage 3 and the start of stage 4                                            #
# --------------------------------------------------------------------------- #
def human_review_enabled(config: ResolvedPipelineConfig) -> bool:
    """`human_review.pause: true` in pipeline_config.yaml pauses the run after translation."""
    with Path(config.pipeline_config_path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return bool((data.get("human_review") or {}).get("pause", False))


def _pending_review(config: ResolvedPipelineConfig, stems: tuple[str, ...] | None) -> list[str]:
    folder = config.paths.translation_markdown_dir
    found = sorted(policy_stem(p) for p in folder.glob("*.md")) if folder.is_dir() else []
    return [s for s in found if stems is None or s in stems]


def run_human_review_step(config: ResolvedPipelineConfig, *, small_scale: bool = False, reviewed: bool = False) -> dict:
    start = time.time()
    if not human_review_enabled(config):
        return _skipped("human_review.pause is off", start)
    pending = _pending_review(config, _stems(config, small_scale))
    if not pending:
        return _skipped("nothing waiting for review", start)
    if reviewed:
        return {"counts": {"total": len(pending), "succeeded": len(pending)}, "elapsed_s": round(time.time() - start, 2)}
    raise PipelineBreakpoint(
        step="human_review",
        title="translations are ready for your review",
        details=[
            f"{len(pending)} translated file(s) wait in {config.paths.translation_markdown_dir}.",
            f"Their source Markdown is in {config.paths.markdown_input_dir}.",
        ],
        next_steps=[
            "Read and edit the cleaned and translated Markdown if you want to (both folders are open to edits).",
            "Keep the headings of each pair identical: same number of headings and the same level at every position. "
            "Text under the headings is free to change.",
            "Run the same command with --reviewed. The `validate` step then checks every pair; you move the accepted ones to processed/.",
        ],
        docs=[DOCS_PIPELINE],
        override="pass --reviewed (or set human_review.pause to false in pipeline_config.yaml)",
    )


def run_validate_step(config: ResolvedPipelineConfig, *, small_scale: bool = False) -> dict:
    """First step of stage 4: check the headings of every pair in ``preprocessed/`` and log the mismatches.

    Moves nothing. Evaluation reads ``processed/``, which you fill by moving the files of each pair you
    accept (the cleaned Markdown and its translation, same file names). The run pauses while ``processed/``
    holds no translation yet; otherwise it goes on with what is there.
    """
    start = time.time()
    paths = config.paths
    source_dir, translated_dir = paths.markdown_input_dir, paths.translation_markdown_dir
    log_path = source_dir.parent / HEADER_CHECK_LOG_FILENAME
    stems = _stems(config, small_scale)

    if source_dir.is_dir() and translated_dir.is_dir():
        checks = validate_directories(source_dir, translated_dir, paths.markdown_input_suffix, stems).checks
    else:
        checks = []
    mismatches = [c for c in checks if not c.ok]
    in_processed = sorted(
        policy_stem(p)
        for p in paths.processed_translation_markdown_dir.glob("*.md")
        if stems is None or policy_stem(p) in stems
    ) if paths.processed_translation_markdown_dir.is_dir() else []

    with run_log(log_path) as log:
        log.info("HEADER CHECK started (%s vs %s)", source_dir, translated_dir)
        for c in checks:
            if c.ok:
                log.info("OK %s (%d headings)", c.stem, c.source_headings)
            else:
                log.warning("MISMATCH %s: %s", c.stem, c.problem)
        log.info(
            "HEADER CHECK finished: %d pair(s) in preprocessed/ match, %d mismatch; %d policy(ies) in processed/",
            len(checks) - len(mismatches), len(mismatches), len(in_processed),
        )

    result = {
        "counts": {
            "total": len(checks) + len(in_processed),
            "succeeded": len(checks) - len(mismatches),
            "skipped": len(in_processed),
            "failed": len(mismatches),
        },
        "elapsed_s": round(time.time() - start, 2),
        "mismatches": [(c.stem, c.problem) for c in mismatches],
        "log_path": str(log_path),
        "in_processed": len(in_processed),
    }
    if not in_processed:
        matching = len(checks) - len(mismatches)
        raise PipelineBreakpoint(
            step="validate",
            title="move the pairs you accept to processed/",
            details=[
                f"{matching} pair(s) in {source_dir.parent} have matching headings; {len(mismatches)} do not (log: {log_path}).",
                f"Evaluation reads {paths.processed_translation_markdown_dir.parent}, which is empty. Nothing is moved for you.",
            ]
            + _listed([f"mismatch {stem}: {problem}" for stem, problem in result["mismatches"]]),
            next_steps=[
                "Fix each mismatch above: make the headings of the cleaned and the translated file match "
                "(same number of headings, same level at every position).",
                f"Move each pair you accept from {source_dir.parent} to {paths.processed_translation_markdown_dir.parent}: "
                "the cleaned_markdown file into processed/cleaned_markdown and the translation_markdown file into "
                "processed/translation_markdown (same file names).",
                "Policies left in preprocessed/ are not evaluated.",
            ],
            docs=[DOCS_PIPELINE],
        )
    return result
