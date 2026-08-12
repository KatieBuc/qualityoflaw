"""Translation QA stage: LLM-checks each translated chunk (or whole file)
against its source Indonesian text and either leaves the translation alone or
replaces it with a corrected version.

Runs right after the `translation` step. Two problems this catches that a
regex-only pass (`llm.wrapper.clean_translation_response`) or the chunking
heuristics (`chunking.py`) can't: leftover LLM notes/meaning drift, and
numbered/lettered/roman list markers scrambled by OCR layout errors that need
semantic understanding to re-associate correctly.

Per-file granularity mirrors `translate.py`'s chunked vs non-chunked split:
- When `mid_product/chunks/<stem>.chunks.json` exists and is usable, QA runs
  per chunk (reusing the stored {text, translated_text} pairs) and rewrites
  BOTH chunks.json (in place) and the regenerated
  `results/translation/<stem>.txt` -- chunks.json is what RAG storage
  actually reads (`rag/store.py:_load_translation_chunks`), so corrections
  must land there, not just in the merged file. If any chunk's QA call
  fails, nothing is written for that file (atomic per policy) and the
  original translation/chunks.json are left untouched.
- Otherwise (chunking was disabled for this run), QA runs once on the whole
  file.

Writes one audit report per policy to
`data/automation/<run_id>/results/translation_qa/<stem>.json` -- metadata
only (action_required/issues per unit), not the corrected text itself, which
already lives in `results/translation/` and `mid_product/chunks/`.
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from automation.src.chunking import Chunk, combine_translations
from automation.src.concurrency import ConcurrencyLimiter
from automation.src.config_loader import (
    ResolvedPipelineConfig,
    get_run_dir,
    resolve_mid_product_dir,
    resolve_results_dir,
)
from automation.src.failure_log import clear_failure, record_failure
from automation.src.llm.wrapper import AzureLLMWrapper, format_api_error
from automation.src.translate import chunks_artifact_path, resolve_input_files
from automation.src.translation_qa_prompt_builder import build_translation_qa_prompt

logger = logging.getLogger(__name__)


class TranslationQaResult(BaseModel):
    action_required: bool
    issues: list[str]
    corrected_text: str | None


@dataclass
class QaFileResult:
    filename: str
    status: str  # "succeeded" | "skipped" | "failed"
    corrected: bool = False
    response_incomplete: bool = False
    report: dict | None = None
    error: str | None = None
    error_type: str | None = None
    error_details: dict | None = None


def _load_chunk_records(chunks_path: Path) -> list[dict] | None:
    """Load the translation step's chunks.json, if present and usable.

    Same validity checks as `rag/store.py:_load_translation_chunks` (exists,
    valid JSON, non-empty, every record has a translated_text) but returns
    the raw stored records unchanged -- chunk_index/section_id/type/context/
    text/translated_text -- instead of store.py's renumbered/flattened shape,
    since this step needs to write corrections back into the same file.
    """
    if not chunks_path.exists():
        return None
    try:
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, list) or not data:
        return None
    if not all(isinstance(c, dict) and c.get("translated_text") for c in data):
        return None
    return data


def _qa_unit(
    original_text: str,
    translated_text: str,
    template_text: str,
    complete_fn: Callable[[str], dict],
    context: str | None = None,
) -> tuple[bool, list[str], str | None, bool]:
    """Returns (action_required, issues, corrected_text, response_incomplete).

    Occasionally the LLM sets action_required=true but doesn't actually
    supply corrected_text -- a content-quality slip, not an API failure. A
    single policy file can be split into dozens or hundreds of chunks, each
    QA'd with its own LLM call (see _qa_policy_file's atomic per-file
    behavior below); treating this as a hard failure would mean even a small
    per-chunk chance of it happening compounds into most files failing
    outright and discarding every other chunk's already-good result. So:
    retry once, and if it's still missing, degrade this one unit to "no
    action" (keep its original text) rather than raising -- response_incomplete
    is set so the caller can flag it in the report for a human to review.
    """
    prompt = build_translation_qa_prompt(original_text, translated_text, template_text, context=context)

    response: dict = {}
    for attempt in range(2):
        response = complete_fn(prompt)
        action_required = bool(response.get("action_required"))
        corrected_text = response.get("corrected_text")
        if not action_required or corrected_text:
            issues = list(response.get("issues") or [])
            return action_required, issues, (corrected_text if action_required else None), False
        logger.warning(
            "LLM reported action_required=true but returned no corrected_text (attempt %d/2)%s",
            attempt + 1,
            "; retrying" if attempt == 0 else "; keeping original text for this unit",
        )

    issues = list(response.get("issues") or []) + [
        "LLM flagged this text for correction but did not return corrected text after retry; "
        "original text kept unchanged."
    ]
    return False, issues, None, True


def _qa_policy_file(
    filename: str,
    input_dir: Path,
    translation_dir: Path,
    chunks_dir: Path,
    template_text: str,
    complete_fn: Callable[[str], dict],
) -> QaFileResult:
    try:
        original_path = input_dir / filename
        translated_path = translation_dir / filename
        if not translated_path.exists():
            return QaFileResult(filename=filename, status="skipped")
        if not original_path.exists():
            raise FileNotFoundError(f"Original policy text not found: {original_path}")

        chunks_path = chunks_artifact_path(chunks_dir, filename)
        records = _load_chunk_records(chunks_path)

        if records is not None:
            items: list[dict] = []
            corrected_translations: list[str] = []
            chunks: list[Chunk] = []
            any_corrected = False

            any_incomplete = False
            for record in records:
                action_required, issues, corrected_text, response_incomplete = _qa_unit(
                    record["text"],
                    record["translated_text"],
                    template_text,
                    complete_fn,
                    context=record.get("context"),
                )
                final_text = corrected_text if action_required else record["translated_text"]
                any_corrected = any_corrected or action_required
                any_incomplete = any_incomplete or response_incomplete
                corrected_translations.append(final_text)
                chunks.append(
                    Chunk(
                        text=record["text"],
                        type=record["type"],
                        context=record.get("context"),
                        section_id=record["section_id"],
                        chunk_index=record["chunk_index"],
                    )
                )
                items.append(
                    {
                        "chunk_index": record["chunk_index"],
                        "section_id": record["section_id"],
                        "action_required": action_required,
                        "issues": issues,
                        "response_incomplete": response_incomplete,
                    }
                )

            # Only reached if every chunk's QA call above returned (raised
            # exceptions -- real API/LLM-call failures, not the malformed
            # content case _qa_unit already retries/degrades -- skip
            # straight to the except block below, so chunks.json/translation
            # are never partially updated).
            updated_records = [
                {
                    **record,
                    "translated_text": text,
                    "qa_action_required": item["action_required"],
                    "qa_issues": item["issues"],
                    "qa_response_incomplete": item["response_incomplete"],
                }
                for record, text, item in zip(records, corrected_translations, items)
            ]
            chunks_path.write_text(
                json.dumps(updated_records, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            combined = combine_translations(chunks, corrected_translations)
            translated_path.write_text(combined, encoding="utf-8")

            report = {
                "granularity": "chunked",
                "chunks_checked": len(records),
                "chunks_corrected": sum(1 for item in items if item["action_required"]),
                "chunks_response_incomplete": sum(1 for item in items if item["response_incomplete"]),
                "items": items,
            }
            logger.info(
                "[%s] translation_qa done (chunked) | %d chunks checked | %d corrected | %d incomplete",
                filename,
                report["chunks_checked"],
                report["chunks_corrected"],
                report["chunks_response_incomplete"],
            )
            return QaFileResult(
                filename=filename,
                status="succeeded",
                corrected=any_corrected,
                response_incomplete=any_incomplete,
                report=report,
            )

        original_text = original_path.read_text(encoding="utf-8")
        translated_text = translated_path.read_text(encoding="utf-8")
        action_required, issues, corrected_text, response_incomplete = _qa_unit(
            original_text, translated_text, template_text, complete_fn
        )
        if action_required:
            translated_path.write_text(corrected_text, encoding="utf-8")

        report = {
            "granularity": "whole_file",
            "chunks_checked": 1,
            "chunks_corrected": 1 if action_required else 0,
            "chunks_response_incomplete": 1 if response_incomplete else 0,
            "items": [
                {
                    "chunk_index": 0,
                    "section_id": 0,
                    "action_required": action_required,
                    "issues": issues,
                    "response_incomplete": response_incomplete,
                }
            ],
        }
        logger.info(
            "[%s] translation_qa done (whole_file) | corrected=%s | incomplete=%s",
            filename,
            action_required,
            response_incomplete,
        )
        return QaFileResult(
            filename=filename,
            status="succeeded",
            corrected=action_required,
            response_incomplete=response_incomplete,
            report=report,
        )

    except Exception as exc:
        error_type, details = format_api_error(exc)
        logger.error("[%s] translation_qa failed [%s]: %s", filename, error_type, exc)
        return QaFileResult(
            filename=filename,
            status="failed",
            error=str(exc),
            error_type=error_type,
            error_details=details,
        )


def run_translation_qa_step(
    run_id: str,
    config: ResolvedPipelineConfig,
    wrapper: AzureLLMWrapper,
    *,
    limiter: ConcurrencyLimiter,
    small_scale: bool = False,
    force: bool = False,
) -> dict:
    run_dir = get_run_dir(run_id)
    translation_dir = resolve_results_dir(run_dir, "translation")
    chunks_dir = resolve_mid_product_dir(run_dir, "chunks")
    output_dir = resolve_results_dir(run_dir, "translation_qa")
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    template_text = config.translation_qa_template_path.read_text(encoding="utf-8")

    def complete_fn(prompt: str) -> dict:
        return wrapper.complete_structured(prompt, TranslationQaResult)

    counts = {"total": 0, "succeeded": 0, "corrected": 0, "incomplete": 0, "skipped": 0, "failed": 0}
    failed_files: list[dict] = []

    pending: list[str] = []
    for input_path in resolve_input_files(config.paths.input_dir, small_scale):
        counts["total"] += 1
        filename = input_path.name
        report_path = output_dir / f"{input_path.stem}.json"
        if report_path.exists() and not force:
            counts["skipped"] += 1
            continue
        pending.append(filename)

    if pending:
        tasks = [
            lambda fn=fn: _qa_policy_file(
                fn, config.paths.input_dir, translation_dir, chunks_dir, template_text, complete_fn
            )
            for fn in pending
        ]
        results = limiter.run_parallel(tasks)

        for filename, result in zip(pending, results, strict=True):
            if isinstance(result, BaseException):
                error_type, details = format_api_error(result)
                entry = {
                    "filename": filename,
                    "error_type": error_type,
                    "message": str(result),
                    "details": details,
                    "attempts": 1,
                }
                record_failure(run_id, "translation_qa", entry)
                logger.error("[%s] translation_qa failed [%s]: %s", filename, error_type, result)
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
                record_failure(run_id, "translation_qa", entry)
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
                    "prompt_version": config.translation_qa_template_path.parent.name,
                }
            )
            report_path = output_dir / f"{Path(filename).stem}.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            clear_failure(run_id, "translation_qa", filename)
            counts["succeeded"] += 1
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
