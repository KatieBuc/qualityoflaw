import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from automation.src.chunking import (
    Chunk,
    CleanedLine,
    check_fallback_output,
    chunk_policy_text,
    chunk_policy_text_with_debug,
    combine_translations,
)
from automation.src.concurrency import ConcurrencyLimiter
from automation.src.config_loader import ChunkingConfig, ResolvedPipelineConfig, get_run_dir
from automation.src.constants import SMALL_SCALE_FILES, STRUCTURE_HINT
from automation.src.failure_log import clear_failure, record_failure
from automation.src.llm.wrapper import AzureLLMWrapper, LLMCallError, format_api_error

logger = logging.getLogger(__name__)


def _build_translation_prompt(template: str, source_text: str) -> str:
    prompt = template.replace("{text}", source_text)
    return f"{prompt}\n\n{STRUCTURE_HINT}"


def _build_fallback_prompt(template: str, chunk: Chunk) -> str:
    return template.replace("{context}", chunk.context or "").replace("{text}", chunk.text)


def _translate_chunk(
    chunk: Chunk,
    *,
    structural_template: str,
    fallback_template: str,
    wrapper: AzureLLMWrapper,
    filename: str,
) -> tuple[str, dict[str, int]]:
    if chunk.type == "structural":
        prompt = _build_translation_prompt(structural_template, chunk.text)
    else:
        prompt = _build_fallback_prompt(fallback_template, chunk)

    translated, usage = wrapper.complete_text(prompt)

    if chunk.type == "fallback" and check_fallback_output(chunk.text, translated):
        logger.warning(
            "[%s] fallback chunk (section %d, chunk %d) output looks suspiciously long "
            "(%d chars vs %d source chars) — possible echoed context; flagging for manual review.",
            filename,
            chunk.section_id,
            chunk.chunk_index,
            len(translated),
            len(chunk.text),
        )

    return translated, usage


@dataclass
class TranslateResult:
    filename: str
    status: str
    token_usage: dict[str, int]
    source_chars: int = 0
    error: str | None = None
    error_type: str | None = None
    error_details: dict | None = None
    attempts: int | None = None


def resolve_input_files(
    input_dir: Path,
    small_scale: bool,
    output_dir: Path | None = None,
) -> list[Path]:
    if small_scale:
        files = [input_dir / name for name in SMALL_SCALE_FILES]
    else:
        files = sorted(p for p in input_dir.glob("*.txt") if p.is_file())
        skipped = sorted(
            p.name for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() != ".txt"
        )
        if skipped:
            logger.info(
                "Skipping %d non-.txt file(s) in %s (only .txt is processed): %s",
                len(skipped),
                input_dir,
                ", ".join(skipped),
            )

    if output_dir is not None and small_scale:
        existing = {p.name for p in output_dir.glob("*.txt")}
        files = [p for p in files if p.name in SMALL_SCALE_FILES or p.name in existing]

    return files


def _failure_entry_from_exc(filename: str, exc: Exception) -> dict:
    if isinstance(exc, LLMCallError):
        return {
            "filename": filename,
            "error_type": exc.error_type,
            "message": str(exc),
            "details": exc.details,
            "attempts": exc.attempts,
        }
    error_type, details = format_api_error(exc)
    return {
        "filename": filename,
        "error_type": error_type,
        "message": str(exc),
        "details": details,
        "attempts": 1,
    }


def _translate_one(
    input_path: Path,
    output_path: Path,
    prompt_template: str,
    wrapper: AzureLLMWrapper,
) -> TranslateResult:
    filename = input_path.name
    if not input_path.exists():
        return TranslateResult(filename=filename, status="failed", token_usage={}, error="not found")

    try:
        source_text = input_path.read_text(encoding="utf-8")
        prompt = _build_translation_prompt(prompt_template, source_text)
        translated, usage = wrapper.complete_text(prompt)
        output_path.write_text(translated, encoding="utf-8")
        logger.info(
            "[%s] done | %d chars | tokens: %s",
            filename,
            len(source_text),
            usage.get("total_tokens", "n/a"),
        )
        return TranslateResult(
            filename=filename,
            status="succeeded",
            token_usage=usage,
            source_chars=len(source_text),
        )
    except Exception as exc:
        entry = _failure_entry_from_exc(filename, exc)
        logger.error(
            "[%s] failed [%s]: %s | status=%s",
            filename,
            entry["error_type"],
            entry["message"],
            entry["details"].get("status_code", "n/a"),
        )
        return TranslateResult(
            filename=filename,
            status="failed",
            token_usage={},
            error=entry["message"],
            error_type=entry["error_type"],
            error_details=entry["details"],
            attempts=entry["attempts"],
        )


def _write_chunk_debug_artifacts(
    chunks_dir: Path,
    filename: str,
    cleaned_lines: list[CleanedLine],
    chunks: list[Chunk],
) -> None:
    """Persist the clean (section 1) and chunk (section 2) results for inspection.

    Writes ``<stem>.cleaned.txt`` (the reflowed text after OCR-noise removal,
    one cleaned line per output line) and ``<stem>.chunks.json`` (chunk
    metadata: text, type, context, section_id, chunk_index) into
    ``data/automation/<run_id>/chunks/``.
    """
    chunks_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem

    cleaned_text = "\n".join(line.text for line in cleaned_lines)
    (chunks_dir / f"{stem}.cleaned.txt").write_text(cleaned_text, encoding="utf-8")

    chunks_data = [
        {
            "chunk_index": chunk.chunk_index,
            "section_id": chunk.section_id,
            "type": chunk.type,
            "context": chunk.context,
            "text": chunk.text,
        }
        for chunk in chunks
    ]
    (chunks_dir / f"{stem}.chunks.json").write_text(
        json.dumps(chunks_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _translate_one_chunked(
    input_path: Path,
    output_path: Path,
    structural_template: str,
    fallback_template: str,
    wrapper: AzureLLMWrapper,
    safe_limit: int,
    chunks_dir: Path | None = None,
) -> TranslateResult:
    """Clean -> chunk -> translate each chunk -> combine, per the chunking spec.

    Structural chunks (one per BAB/Pasal/Bagian/Paragraf/title boundary) are
    translated directly. Any section still over safe_limit after structural
    splitting is further split into fallback sub-chunks, each translated with
    the previous sub-chunk's tail as non-translated context.

    When `chunks_dir` is given (--keep-chunk-result), the clean and chunk
    results are also written there for inspection/debugging.
    """
    filename = input_path.name
    if not input_path.exists():
        return TranslateResult(filename=filename, status="failed", token_usage={}, error="not found")

    try:
        source_text = input_path.read_text(encoding="utf-8")
        if chunks_dir is not None:
            cleaned_lines, chunks = chunk_policy_text_with_debug(source_text, safe_limit=safe_limit)
            _write_chunk_debug_artifacts(chunks_dir, filename, cleaned_lines, chunks)
        else:
            chunks = chunk_policy_text(source_text, safe_limit=safe_limit)

        translations: list[str] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for chunk in chunks:
            translated, usage = _translate_chunk(
                chunk,
                structural_template=structural_template,
                fallback_template=fallback_template,
                wrapper=wrapper,
                filename=filename,
            )
            translations.append(translated)
            for key in total_usage:
                total_usage[key] += usage.get(key, 0)

        combined = combine_translations(chunks, translations)
        output_path.write_text(combined, encoding="utf-8")
        logger.info(
            "[%s] done (chunked) | %d chars | %d chunks (%d fallback) | tokens: %s",
            filename,
            len(source_text),
            len(chunks),
            sum(1 for c in chunks if c.type == "fallback"),
            total_usage.get("total_tokens", "n/a"),
        )
        return TranslateResult(
            filename=filename,
            status="succeeded",
            token_usage=total_usage,
            source_chars=len(source_text),
        )
    except Exception as exc:
        entry = _failure_entry_from_exc(filename, exc)
        logger.error(
            "[%s] failed (chunked) [%s]: %s | status=%s",
            filename,
            entry["error_type"],
            entry["message"],
            entry["details"].get("status_code", "n/a"),
        )
        return TranslateResult(
            filename=filename,
            status="failed",
            token_usage={},
            error=entry["message"],
            error_type=entry["error_type"],
            error_details=entry["details"],
            attempts=entry["attempts"],
        )


def run_translation_step(
    run_id: str,
    config: ResolvedPipelineConfig,
    wrapper: AzureLLMWrapper,
    *,
    limiter: ConcurrencyLimiter,
    small_scale: bool = False,
    force: bool = False,
    run_missing: bool = False,
    keep_chunk_result: bool = False,
) -> dict:
    run_dir = get_run_dir(run_id)
    output_dir = run_dir / "translation"
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = config.translation_prompt_path.read_text(encoding="utf-8")
    chunking: ChunkingConfig = config.chunking
    fallback_template = (
        chunking.fallback_prompt_path.read_text(encoding="utf-8") if chunking.enabled else None
    )
    chunks_dir = run_dir / "chunks" if (chunking.enabled and keep_chunk_result) else None
    input_files = resolve_input_files(config.paths.input_dir, small_scale)

    counts = {"total": len(input_files), "succeeded": 0, "skipped": 0, "failed": 0}
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    failed_files: list[dict] = []
    start = time.time()

    pending: list[tuple[Path, Path]] = []
    for input_path in input_files:
        output_path = output_dir / input_path.name
        if output_path.exists() and not force:
            counts["skipped"] += 1
            logger.info("[%s] skipped (output exists)", input_path.name)
            continue
        pending.append((input_path, output_path))

    if pending:
        if chunking.enabled:
            tasks = [
                lambda inp=inp, out=out: _translate_one_chunked(
                    inp,
                    out,
                    prompt_template,
                    fallback_template,
                    wrapper,
                    chunking.safe_limit,
                    chunks_dir=chunks_dir,
                )
                for inp, out in pending
            ]
        else:
            tasks = [
                lambda inp=inp, out=out: _translate_one(inp, out, prompt_template, wrapper)
                for inp, out in pending
            ]
        results = limiter.run_parallel(tasks)
        for result in results:
            if isinstance(result, BaseException):
                counts["failed"] += 1
                error_type, details = format_api_error(result)
                entry = {
                    "filename": "unknown",
                    "error_type": error_type,
                    "message": str(result),
                    "details": details,
                    "attempts": 1,
                }
                failed_files.append(entry)
                record_failure(run_id, "translation", entry)
                logger.error("Translation task failed [%s]: %s", error_type, result)
                continue
            if result.status == "succeeded":
                counts["succeeded"] += 1
                clear_failure(run_id, "translation", result.filename)
                for key in token_usage:
                    token_usage[key] += result.token_usage.get(key, 0)
            else:
                counts["failed"] += 1
                entry = {
                    "filename": result.filename,
                    "error_type": result.error_type or "Exception",
                    "message": result.error or "unknown error",
                    "details": result.error_details or {},
                    "attempts": result.attempts or 1,
                }
                failed_files.append(entry)
                record_failure(run_id, "translation", entry)

    elapsed = round(time.time() - start, 2)
    return {
        "counts": counts,
        "failed_files": failed_files,
        "elapsed_s": elapsed,
        "token_usage": token_usage,
        "output_dir": str(output_dir),
    }
