"""Pipeline steps for stage 1 (input), stage 3 (human adjustment) and the start of stage 4.

    input_process   PDF/HTML/TXT -> extracted text -> quality check -> cleaned_txt   (breakpoint: OCR / eliminated files)
    txt_to_md       cleaned_txt (or gold) -> preprocessed/cleaned_markdown
    human_review    optional pause so a person can edit cleaned/translation markdown (breakpoint, off by default)
    promote         header check, then preprocessed/ -> processed/                   (breakpoint: mismatching headings)

Each step is idempotent. A breakpoint is raised as `PipelineBreakpoint`; `run_pipeline` prints it and
stops, and running the same command again continues.
"""

import time
from pathlib import Path

import yaml

from automation.src.breakpoints import DOCS_INPUT_PROCESS, DOCS_PIPELINE, PipelineBreakpoint
from automation.src.config_loader import ResolvedPipelineConfig
from automation.src.ingest import run_ingest_step
from automation.src.input_process.config import InputProcessConfig, load_input_process_config
from automation.src.input_process.extract import extract_all
from automation.src.input_process.pipeline import check_all
from automation.src.markdown.policy_files import policy_stem
from automation.src.markdown.validate import promote_ready

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
def run_input_process_step(config: ResolvedPipelineConfig, *, small_scale: bool = False, skip_blocked: bool = False) -> dict:
    start = time.time()
    ip = _input_config(config)
    if not ip.raw_files_dir.is_dir():
        return _skipped(f"no originals folder ({ip.raw_files_dir})", start)

    extraction = extract_all(ip, small_scale=small_scale)
    failed_extractions = [r for r in extraction if r.status == "failed"]
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
            "failed": len(failed_extractions),
        },
        "elapsed_s": round(time.time() - start, 2),
        "report_path": str(ip.report_path),
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
            step="input_process",
            title="some files need your attention",
            details=details,
            next_steps=[
                f"Open the report {ip.report_path}: it lists every file with the quality indicators that failed.",
                f"For a PDF that needs OCR: OCR it and save the text as {ip.extracted_txt_dir / '<name>__ocr.txt'}.",
                f"For an eliminated file: fix the text and save it as {ip.extracted_txt_dir / '<name>__txt.txt'} "
                "(it is checked again), or move it straight into "
                f"{ip.cleaned_txt_dir} if you accept it as it is.",
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
        report_path=ip.report_path.with_name("markdown_report.json"),
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
            "Run the same command with --reviewed. The pair is checked and moved to processed/ at the start of stage 4.",
        ],
        docs=[DOCS_PIPELINE],
        override="set human_review.pause to false in pipeline_config.yaml, or pass --reviewed",
    )


def run_promote_step(config: ResolvedPipelineConfig, *, small_scale: bool = False, skip_blocked: bool = False) -> dict:
    """First step of stage 4: check the headings of each pair, then move the pair to processed/."""
    start = time.time()
    paths = config.paths
    source_dir, translated_dir = paths.markdown_input_dir, paths.translation_markdown_dir
    stems = _stems(config, small_scale)
    if not source_dir.is_dir() or not translated_dir.is_dir():
        pending: list[str] = []
        blocked = []
        promoted: list[str] = []
    else:
        promoted, blocked = promote_ready(
            source_dir,
            translated_dir,
            paths.processed_cleaned_markdown_dir,
            paths.processed_translation_markdown_dir,
            source_suffix=paths.markdown_input_suffix,
            stems=stems,
        )
    already = [
        policy_stem(p)
        for p in paths.processed_translation_markdown_dir.glob("*.md")
        if stems is None or policy_stem(p) in stems
    ] if paths.processed_translation_markdown_dir.is_dir() else []
    result = {
        "counts": {
            "total": len(promoted) + len(blocked) + len(set(already) - set(promoted)),
            "succeeded": len(promoted),
            "skipped": len(set(already) - set(promoted)),
            "failed": len(blocked),
        },
        "elapsed_s": round(time.time() - start, 2),
        "blocked": [(b.stem, b.problem) for b in blocked],
    }
    if blocked and not skip_blocked:
        raise PipelineBreakpoint(
            step="promote",
            title="some pairs cannot move to processed/ yet",
            details=[f"{len(promoted)} pair(s) moved to {paths.processed_translation_markdown_dir.parent}; {len(blocked)} blocked:"]
            + _listed([f"{b.stem}: {b.problem}" for b in blocked]),
            next_steps=[
                f"For each file above, make the headings of {source_dir} and {translated_dir} match "
                "(same number of headings, same level at every position), or add the missing file.",
                "A blocked policy is left out of the evaluation until its pair passes.",
            ],
            docs=[DOCS_PIPELINE],
            override=SKIP_BLOCKED_HINT,
        )
    if not (promoted or already):
        raise FileNotFoundError(
            f"Nothing to evaluate: no validated pair in {paths.processed_translation_markdown_dir} "
            f"and nothing to promote from {translated_dir}"
        )
    return result
