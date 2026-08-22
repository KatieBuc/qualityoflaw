"""The `translation_md` step: Markdown in, Markdown out.

Counterpart to `automation.src.translate`, which translates raw OCR text and
is left untouched. The differences that matter:

- Input is the curated Markdown corpus, so structure is read from headings
  rather than guessed at -- see `automation.src.markdown.chunking`.
- Chunks are *packed*: adjacent heading sections merge up to `target_chars`
  (8,000), which takes a full-corpus run from ~15,300 LLM calls to ~1,100. A
  packed chunk carries its heading breadcrumb into the prompt so the model can
  still resolve cross-references when it starts mid-chapter.
- Output is Markdown, at ``results/translation_markdown/<POLICY>.md``. The
  plain-text artifact every downstream stage reads is produced afterwards by
  the `md_to_text` step, which also rebuilds the per-heading retrieval chunks.
- The input is copied to ``results/source_markdown/<POLICY>.md`` so a run
  records exactly what it translated, even if the curated corpus moves on.

Because both sides are Markdown, a translation can be checked against its
source exactly (`chunking.check_structure`) instead of by the length-ratio
heuristics the raw-text path has to rely on.
"""

import json
import logging
import shutil
import time
from pathlib import Path

from automation.src.concurrency import ConcurrencyLimiter
from automation.src.config_loader import (
    ResolvedPipelineConfig,
    get_run_dir,
    resolve_mid_product_dir,
    resolve_results_dir,
)
from automation.src.constants import MARKDOWN_STRUCTURE_HINT
from automation.src.failure_log import clear_failure, record_failure
from automation.src.llm.wrapper import AzureLLMWrapper, format_api_error
from automation.src.markdown.chunking import MarkdownChunk, check_structure, chunk_markdown
from automation.src.markdown.policy_files import markdown_policy_files, policy_stem
from automation.src.translate import TranslateResult, _failure_entry_from_exc
from automation.src.chunking import combine_translations

logger = logging.getLogger(__name__)

# Reported as its own step to the user, but logged into failures.json under
# "translation" -- failure_log's step vocabulary is shared with the raw-text
# path, and a failed translation is a failed translation either way.
FAILURE_STEP = "translation"


def build_prompt(template: str, chunk: MarkdownChunk) -> str:
    """Render the prompt for one chunk.

    A structural chunk gets the plain Markdown template; a fallback sub-chunk
    (a single section too large for `safe_limit`, sentence-split) gets the
    template with its reference-only context and heading breadcrumb filled in.
    """
    breadcrumb = " > ".join(chunk.heading_path)
    if chunk.type == "structural":
        prompt = template.replace("{text}", chunk.text)
        if breadcrumb:
            prompt = f"{prompt}\n\nThis excerpt sits under: {breadcrumb}"
        return f"{prompt}\n\n{MARKDOWN_STRUCTURE_HINT}"
    return (
        template.replace("{heading_path}", breadcrumb or "(top level)")
        .replace("{context}", chunk.context or "")
        .replace("{text}", chunk.text)
    )


def translation_chunks_path(chunks_dir: Path, stem: str) -> Path:
    return chunks_dir / f"{stem}.json"


def _write_chunks_artifact(
    chunks_dir: Path, stem: str, chunks: list[MarkdownChunk], translations: list[str]
) -> None:
    """Persist the packed translation units and their translations.

    Deliberately *not* written to ``mid_product/chunks/`` -- that path holds
    the per-heading retrieval units RAG storage reads, which `md_to_text`
    rebuilds from the translated Markdown at a much finer granularity. These
    packed units exist for the `translation_qa_md` step and for debugging.
    """
    chunks_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "chunk_index": chunk.chunk_index,
            "section_id": chunk.section_id,
            "type": chunk.type,
            "context": chunk.context,
            "heading_path": chunk.heading_path,
            "text": chunk.text,
            "translated_text": translated,
        }
        for chunk, translated in zip(chunks, translations)
    ]
    translation_chunks_path(chunks_dir, stem).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def translate_one(
    input_path: Path,
    output_path: Path,
    source_copy_path: Path,
    chunks_dir: Path,
    *,
    prompt_template: str,
    fallback_template: str,
    wrapper: AzureLLMWrapper,
    target_chars: int,
    safe_limit: int,
) -> TranslateResult:
    """Parse -> pack -> translate each chunk -> combine, for one policy."""
    filename = input_path.name
    if not input_path.exists():
        return TranslateResult(filename=filename, status="failed", token_usage={}, error="not found")

    stem = policy_stem(input_path)
    try:
        source_md = input_path.read_text(encoding="utf-8")
        chunks = chunk_markdown(source_md, target_chars=target_chars, safe_limit=safe_limit)
        if not chunks:
            raise ValueError("no chunks produced")

        translations: list[str] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for chunk in chunks:
            template = prompt_template if chunk.type == "structural" else fallback_template
            translated, usage = wrapper.complete_text(build_prompt(template, chunk))
            translations.append(translated)
            for key in total_usage:
                total_usage[key] += usage.get(key, 0)

        _write_chunks_artifact(chunks_dir, stem, chunks, translations)

        combined = combine_translations(chunks, translations)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(combined, encoding="utf-8")

        source_copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_path, source_copy_path)

        diff = check_structure(source_md, combined)
        if diff is not None:
            logger.warning(
                "[%s] translation structure drift: %s -- flagged for review",
                filename,
                diff.describe(),
            )

        logger.info(
            "[%s] done | %d chars | %d chunks (%d fallback) | tokens: %s",
            filename,
            len(source_md),
            len(chunks),
            sum(1 for c in chunks if c.type == "fallback"),
            total_usage.get("total_tokens", "n/a"),
        )
        return TranslateResult(
            filename=filename,
            status="succeeded",
            token_usage=total_usage,
            source_chars=len(source_md),
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


def run_md_translation_step(
    run_id: str,
    config: ResolvedPipelineConfig,
    wrapper: AzureLLMWrapper,
    *,
    limiter: ConcurrencyLimiter,
    small_scale: bool = False,
    force: bool = False,
) -> dict:
    """Translate every Markdown policy in the corpus, one task per file.

    Idempotent in the same way as the raw-text step: a policy whose
    ``.md`` output already exists is skipped unless `--force`. Parallelism is
    across files; the chunks of one file are translated in order so a
    packed chunk's predecessor is always available for context.
    """
    run_dir = get_run_dir(run_id)
    output_dir = resolve_results_dir(run_dir, "translation_markdown")
    source_dir = resolve_results_dir(run_dir, "source_markdown")
    chunks_dir = resolve_mid_product_dir(run_dir, "translation_chunks")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = config.paths.markdown_input_dir
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Markdown input directory not found: {input_dir}")

    prompt_template = config.markdown.prompt_path.read_text(encoding="utf-8")
    fallback_template = config.markdown.fallback_prompt_path.read_text(encoding="utf-8")
    input_files = markdown_policy_files(input_dir, small_scale)

    counts = {"total": len(input_files), "succeeded": 0, "skipped": 0, "failed": 0}
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    failed_files: list[dict] = []
    start = time.time()

    pending: list[tuple[Path, Path, Path]] = []
    for input_path in input_files:
        stem = policy_stem(input_path)
        output_path = output_dir / f"{stem}.md"
        if output_path.exists() and not force:
            counts["skipped"] += 1
            logger.info("[%s] skipped (output exists)", input_path.name)
            continue
        pending.append((input_path, output_path, source_dir / f"{stem}.md"))

    if pending:
        tasks = [
            lambda inp=inp, out=out, src=src: translate_one(
                inp,
                out,
                src,
                chunks_dir,
                prompt_template=prompt_template,
                fallback_template=fallback_template,
                wrapper=wrapper,
                target_chars=config.markdown.target_chars,
                safe_limit=config.markdown.safe_limit,
            )
            for inp, out, src in pending
        ]
        for result in limiter.run_parallel(tasks):
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
                record_failure(run_id, FAILURE_STEP, entry)
                logger.error("Markdown translation task failed [%s]: %s", error_type, result)
                continue
            if result.status == "succeeded":
                counts["succeeded"] += 1
                clear_failure(run_id, FAILURE_STEP, result.filename)
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
                record_failure(run_id, FAILURE_STEP, entry)

    elapsed = round(time.time() - start, 2)
    return {
        "counts": counts,
        "failed_files": failed_files,
        "elapsed_s": elapsed,
        "token_usage": token_usage,
        "output_dir": str(output_dir),
    }
