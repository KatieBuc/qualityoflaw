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
`data/<project>/automation/results/translation_qa/<stem>.json` -- metadata
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
from automation.src.common import (
    QaFileResult,
    TranslationQaResult,
    build_translation_qa_prompt,
    chunks_artifact_path,
    load_chunk_records,
    qa_unit,
)
from automation.src.concurrency import ConcurrencyLimiter
from automation.src.config_loader import (
    ResolvedPipelineConfig,
    get_output_dir,
    resolve_mid_product_dir,
    resolve_results_dir,
)
from automation.src.failure_log import clear_failure, record_failure
from automation.src.llm.wrapper import AzureLLMWrapper, format_api_error
from automation.src.legacy.translate import resolve_input_files

logger = logging.getLogger(__name__)


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
        records = load_chunk_records(chunks_path)

        if records is not None:
            items: list[dict] = []
            corrected_translations: list[str] = []
            chunks: list[Chunk] = []
            any_corrected = False

            any_incomplete = False
            for record in records:
                action_required, issues, corrected_text, response_incomplete = qa_unit(
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
            # content case qa_unit already retries/degrades -- skip
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
        action_required, issues, corrected_text, response_incomplete = qa_unit(
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
    config: ResolvedPipelineConfig,
    wrapper: AzureLLMWrapper,
    *,
    limiter: ConcurrencyLimiter,
    small_scale: bool = False,
    force: bool = False,
) -> dict:
    run_dir = get_output_dir()
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
                record_failure("translation_qa", entry)
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
                record_failure("translation_qa", entry)
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
            clear_failure("translation_qa", filename)
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
