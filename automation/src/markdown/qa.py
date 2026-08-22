"""The `translation_qa_md` step: audit and repair the Markdown translation.

Counterpart to `automation.src.translation_qa`, which audits the raw-text
path and is left untouched. Same contract -- a second model checks each
translation unit against its Indonesian source, and either leaves it alone or
returns a full corrected version -- with four differences:

- Each chunk is audited **repeatedly**, up to `qa_max_passes` (default 5),
  until a pass reports nothing left to fix. One rewrite rarely finishes the
  job on this corpus; see `qa_chunk`.
- The unit of audit is the *packed* translation chunk from
  ``mid_product/translation_chunks/``, not the per-heading retrieval chunk.
  Those are rebuilt later by `md_to_text` from whatever this step leaves
  behind, so corrections made here reach retrieval automatically.
- The QA prompt (``translation_qa/v2``) adds Markdown fidelity as its first
  and most important check, since heading structure is what downstream
  retrieval splits on.
- Corrections are written back to ``results/translation_markdown/<POLICY>.md``
  and to the chunk artifact, both atomically per file: a real API failure on
  any chunk leaves the file's previous state entirely untouched.

`_qa_unit`'s retry-then-degrade behaviour is imported rather than
reimplemented, so both paths handle a model that flags a problem but returns
no correction identically.

Nothing here treats a heading or list-item *count* as a defect. The curated
corpus carries OCR damage that a good translation repairs, and every repair
moves those counts, so the only structural failure reported is a lost
numbered clause -- see `markdown.chunking.check_structure`.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from automation.src.chunking import combine_translations
from automation.src.concurrency import ConcurrencyLimiter
from automation.src.config_loader import (
    ResolvedPipelineConfig,
    get_run_dir,
    resolve_mid_product_dir,
    resolve_results_dir,
)
from automation.src.failure_log import clear_failure, record_failure
from automation.src.llm.wrapper import AzureLLMWrapper, format_api_error
from automation.src.markdown.chunking import MarkdownChunk, check_structure
from automation.src.markdown.policy_files import markdown_policy_files, policy_stem
from automation.src.markdown.translate import translation_chunks_path
from automation.src.translation_qa import (
    QaFileResult,
    TranslationQaResult,
    _load_chunk_records,
    _qa_unit,
)

logger = logging.getLogger(__name__)

FAILURE_STEP = "translation_qa"


@dataclass
class ChunkQaOutcome:
    """What repeated QA passes settled on for one chunk."""

    text: str
    issues: list[str]
    passes: int
    converged: bool
    corrected: bool
    response_incomplete: bool


def qa_chunk(
    original_text: str,
    translated_text: str,
    template_text: str,
    complete_fn: Callable[[str], dict],
    *,
    context: str | None = None,
    max_passes: int = 5,
) -> ChunkQaOutcome:
    """Re-check a chunk until the auditor stops finding problems, up to a cap.

    A single pass only guarantees the model *reported* what was wrong; a
    correction can introduce or leave a further problem, and the source
    chunks here carry heavy OCR damage (detached list markers, sentences
    split across a spurious heading), so one rewrite often does not finish
    the job. Each pass re-audits the *previous pass's output* against the
    unchanged Indonesian source, so the loop converges on a translation the
    auditor signs off rather than on whatever the first rewrite produced.

    Stops as soon as a pass reports no action required (`converged`), and
    otherwise after `max_passes`. Not converging is worth a human's
    attention, but the last correction is still kept -- it is the most
    audited version available, and discarding it would throw away several
    passes of genuine repair.

    Whether to keep going is the *auditor's* verdict, never a structural
    count: heading totals legitimately move when a repair lands (see
    `markdown.chunking.check_structure`), so looping on those would chase
    successful repairs forever.
    """
    current = translated_text
    all_issues: list[str] = []
    incomplete = False
    passes = 0

    for _ in range(max(1, max_passes)):
        passes += 1
        action_required, issues, corrected_text, response_incomplete = _qa_unit(
            original_text, current, template_text, complete_fn, context=context
        )
        all_issues += issues
        incomplete = incomplete or response_incomplete

        if response_incomplete:
            # _qa_unit already retried and degraded this unit to "no action";
            # another pass would just repeat that, so stop and let the report
            # carry response_incomplete for review.
            break
        if not action_required:
            return ChunkQaOutcome(
                text=current,
                issues=all_issues,
                passes=passes,
                converged=True,
                corrected=current != translated_text,
                response_incomplete=incomplete,
            )
        current = corrected_text

    return ChunkQaOutcome(
        text=current,
        issues=all_issues,
        passes=passes,
        converged=False,
        corrected=current != translated_text,
        response_incomplete=incomplete,
    )


def _qa_policy_file(
    stem: str,
    markdown_dir: Path,
    chunks_dir: Path,
    template_text: str,
    complete_fn: Callable[[str], dict],
    *,
    max_passes: int = 5,
) -> QaFileResult:
    """Audit one policy's packed chunks and rewrite both of its artifacts.

    Nothing is written until every chunk's QA call has returned, so a
    mid-file API failure can't leave the Markdown and the chunk artifact
    disagreeing.
    """
    filename = f"{stem}.md"
    try:
        translated_path = markdown_dir / filename
        if not translated_path.exists():
            return QaFileResult(filename=filename, status="skipped")

        chunks_path = translation_chunks_path(chunks_dir, stem)
        records = _load_chunk_records(chunks_path)
        if records is None:
            raise FileNotFoundError(
                f"Translation chunks not found or unusable: {chunks_path}"
            )

        items: list[dict] = []
        corrected_translations: list[str] = []
        chunks: list[MarkdownChunk] = []
        any_corrected = False
        any_incomplete = False

        for record in records:
            outcome = qa_chunk(
                record["text"],
                record["translated_text"],
                template_text,
                complete_fn,
                context=record.get("context"),
                max_passes=max_passes,
            )
            any_corrected = any_corrected or outcome.corrected
            any_incomplete = any_incomplete or outcome.response_incomplete
            corrected_translations.append(outcome.text)
            chunks.append(
                MarkdownChunk(
                    text=record["text"],
                    type=record["type"],
                    context=record.get("context"),
                    section_id=record["section_id"],
                    chunk_index=record["chunk_index"],
                    heading_path=record.get("heading_path") or [],
                )
            )
            items.append(
                {
                    "chunk_index": record["chunk_index"],
                    "section_id": record["section_id"],
                    "action_required": outcome.corrected,
                    "issues": outcome.issues,
                    "passes": outcome.passes,
                    "converged": outcome.converged,
                    "response_incomplete": outcome.response_incomplete,
                }
            )
            if not outcome.converged and not outcome.response_incomplete:
                logger.warning(
                    "[%s] chunk %d still reported problems after %d QA pass(es); "
                    "keeping the last correction",
                    filename,
                    record["section_id"],
                    outcome.passes,
                )

        updated_records = [
            {
                **record,
                "translated_text": text,
                "qa_action_required": item["action_required"],
                "qa_issues": item["issues"],
                "qa_passes": item["passes"],
                "qa_converged": item["converged"],
                "qa_response_incomplete": item["response_incomplete"],
            }
            for record, text, item in zip(records, corrected_translations, items)
        ]
        chunks_path.write_text(
            json.dumps(updated_records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        combined = combine_translations(chunks, corrected_translations)
        translated_path.write_text(combined, encoding="utf-8")

        # Only content loss is reported. Heading and list-item totals move
        # whenever a repair lands -- the corpus promotes cross-references to
        # headings, splits numbers as `Pasal 3 1`, and leaves fragments like
        # `##### 1  AN` -- so a count is not a defect, and treating one as
        # such buries the findings that are.
        source_md = "\n\n".join(record["text"] for record in records)
        diff = check_structure(source_md, combined)
        lost_clauses = diff.lost_clauses if diff else []
        if lost_clauses:
            logger.error(
                "[%s] QA output is missing %d numbered clause(s): %s",
                filename,
                len(lost_clauses),
                ", ".join(lost_clauses[:5]),
            )

        not_converged = sum(
            1 for item in items if not item["converged"] and not item["response_incomplete"]
        )
        report = {
            "granularity": "chunked",
            "chunks_checked": len(records),
            "chunks_corrected": sum(1 for item in items if item["action_required"]),
            "chunks_response_incomplete": sum(
                1 for item in items if item["response_incomplete"]
            ),
            "chunks_not_converged": not_converged,
            "qa_passes_total": sum(item["passes"] for item in items),
            "max_passes": max_passes,
            "lost_clauses": lost_clauses,
            "items": items,
        }
        logger.info(
            "[%s] translation_qa_md done | %d chunks checked | %d corrected | "
            "%d incomplete | %d unconverged | %d passes total",
            filename,
            report["chunks_checked"],
            report["chunks_corrected"],
            report["chunks_response_incomplete"],
            not_converged,
            report["qa_passes_total"],
        )
        return QaFileResult(
            filename=filename,
            status="succeeded",
            corrected=any_corrected,
            response_incomplete=any_incomplete,
            report=report,
        )

    except Exception as exc:
        error_type, details = format_api_error(exc)
        logger.error("[%s] translation_qa_md failed [%s]: %s", filename, error_type, exc)
        return QaFileResult(
            filename=filename,
            status="failed",
            error=str(exc),
            error_type=error_type,
            error_details=details,
        )


def run_md_translation_qa_step(
    run_id: str,
    config: ResolvedPipelineConfig,
    wrapper: AzureLLMWrapper,
    *,
    limiter: ConcurrencyLimiter,
    small_scale: bool = False,
    force: bool = False,
) -> dict:
    """QA every translated Markdown policy in the run, one task per file."""
    run_dir = get_run_dir(run_id)
    markdown_dir = resolve_results_dir(run_dir, "translation_markdown")
    chunks_dir = resolve_mid_product_dir(run_dir, "translation_chunks")
    output_dir = resolve_results_dir(run_dir, "translation_qa")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not markdown_dir.is_dir():
        raise FileNotFoundError(f"No translated markdown to QA in {markdown_dir}")

    start = time.time()
    template_text = config.markdown.qa_template_path.read_text(encoding="utf-8")

    def complete_fn(prompt: str) -> dict:
        return wrapper.complete_structured(prompt, TranslationQaResult)

    counts = {
        "total": 0,
        "succeeded": 0,
        "corrected": 0,
        "incomplete": 0,
        # Chunks the auditor still flagged after max_passes. Their last
        # correction is kept; the count is what asks for a human look.
        "not_converged": 0,
        # Files whose QA output is missing a numbered clause from the source.
        "clauses_lost": 0,
        "qa_passes": 0,
        "skipped": 0,
        "failed": 0,
    }
    failed_files: list[dict] = []

    pending: list[str] = []
    for translated_path in markdown_policy_files(markdown_dir, small_scale, suffix=".md"):
        counts["total"] += 1
        stem = policy_stem(translated_path)
        if (output_dir / f"{stem}.json").exists() and not force:
            counts["skipped"] += 1
            continue
        pending.append(stem)

    if pending:
        tasks = [
            lambda s=s: _qa_policy_file(
                s,
                markdown_dir,
                chunks_dir,
                template_text,
                complete_fn,
                max_passes=config.markdown.qa_max_passes,
            )
            for s in pending
        ]
        results = limiter.run_parallel(tasks)

        for stem, result in zip(pending, results, strict=True):
            filename = f"{stem}.md"
            if isinstance(result, BaseException):
                error_type, details = format_api_error(result)
                entry = {
                    "filename": filename,
                    "error_type": error_type,
                    "message": str(result),
                    "details": details,
                    "attempts": 1,
                }
                record_failure(run_id, FAILURE_STEP, entry)
                logger.error("[%s] translation_qa_md failed [%s]: %s", filename, error_type, result)
                failed_files.append(entry)
                counts["failed"] += 1
                continue

            if result.status == "failed":
                entry = {
                    "filename": filename,
                    "error_type": result.error_type or "Exception",
                    "message": result.error or "unknown error",
                    "details": result.error_details or {},
                    "attempts": 1,
                }
                record_failure(run_id, FAILURE_STEP, entry)
                failed_files.append(entry)
                counts["failed"] += 1
                continue

            if result.status == "skipped":
                counts["skipped"] += 1
                continue

            report = result.report or {}
            report.update(
                {
                    "policy_file": filename,
                    "model": wrapper.profile.deployment,
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                    "prompt_version": config.markdown.qa_template_path.parent.name,
                }
            )
            (output_dir / f"{stem}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            clear_failure(run_id, FAILURE_STEP, filename)
            counts["succeeded"] += 1
            counts["not_converged"] += report.get("chunks_not_converged", 0)
            counts["qa_passes"] += report.get("qa_passes_total", 0)
            if report.get("lost_clauses"):
                counts["clauses_lost"] += 1
            if result.corrected:
                counts["corrected"] += 1
            if result.response_incomplete:
                counts["incomplete"] += 1

    elapsed = round(time.time() - start, 2)
    return {
        "counts": counts,
        "failed_files": failed_files,
        "elapsed_s": elapsed,
        "token_usage": dict(wrapper.token_usage),
        "output_dir": str(output_dir),
    }
