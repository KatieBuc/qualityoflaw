"""The `translation_md` step: Markdown in, Markdown out.

Counterpart to `automation.src.legacy.translate`, which translates raw OCR text and
is left untouched. The differences that matter:

- Input is the curated Markdown corpus, so structure is read from headings
  rather than guessed at -- see `automation.src.markdown.chunking`.
- Chunks are *packed*: adjacent heading sections merge up to `target_chars`
  (8,000), which takes a full-corpus run from ~15,300 LLM calls to ~1,100. A
  packed chunk carries its heading breadcrumb into the prompt so the model can
  still resolve cross-references when it starts mid-chapter.
- Output is Markdown, at ``<project>/preprocessed/translation_markdown/
  <POLICY>.md`` (``paths.translation_markdown_dir``), next to the input in
  ``preprocessed/cleaned_markdown``. A human may edit either folder before the
  pair is promoted to ``processed/``. Chunks are translated and recombined in
  memory; nothing about them is saved.

Because both sides are Markdown, a translation can be checked against its
source exactly (`chunking.check_structure`) instead of by the length-ratio
heuristics the raw-text path has to rely on.
"""

import logging
import time
from pathlib import Path

from automation.src.concurrency import ConcurrencyLimiter
from automation.src.config_loader import ResolvedPipelineConfig
from automation.src.constants import MARKDOWN_STRUCTURE_HINT
from automation.src.failure_log import clear_failure, record_failure
from automation.src.llm.wrapper import AzureLLMWrapper, format_api_error
from automation.src.markdown.chunking import MarkdownChunk, check_structure, chunk_markdown
from automation.src.markdown.policy_files import markdown_policy_files, policy_stem
from automation.src.common import TranslateResult, failure_entry_from_exc
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


def translate_one(
    input_path: Path,
    output_path: Path,
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

        combined = combine_translations(chunks, translations)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(combined, encoding="utf-8")

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
        entry = failure_entry_from_exc(filename, exc)
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
    output_dir = config.paths.translation_markdown_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = config.paths.markdown_input_dir
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Markdown input directory not found: {input_dir}")

    prompt_template = config.markdown.prompt_path.read_text(encoding="utf-8")
    fallback_template = config.markdown.fallback_prompt_path.read_text(encoding="utf-8")
    input_files = markdown_policy_files(
        input_dir,
        small_scale,
        suffix=config.paths.markdown_input_suffix,
        stems=config.paths.small_scale_stems,
    )

    counts = {"total": len(input_files), "succeeded": 0, "skipped": 0, "failed": 0}
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    failed_files: list[dict] = []
    start = time.time()

    pending: list[tuple[Path, Path]] = []
    for input_path in input_files:
        stem = policy_stem(input_path)
        output_path = output_dir / f"{stem}.md"
        if output_path.exists() and not force:
            counts["skipped"] += 1
            logger.info("[%s] skipped (output exists)", input_path.name)
            continue
        pending.append((input_path, output_path))

    if pending:
        tasks = [
            lambda inp=inp, out=out: translate_one(
                inp,
                out,
                prompt_template=prompt_template,
                fallback_template=fallback_template,
                wrapper=wrapper,
                target_chars=config.markdown.target_chars,
                safe_limit=config.markdown.safe_limit,
            )
            for inp, out in pending
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
