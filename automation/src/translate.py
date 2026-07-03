import logging
import time
from dataclasses import dataclass
from pathlib import Path

from automation.src.concurrency import ConcurrencyLimiter
from automation.src.config_loader import ResolvedPipelineConfig, get_run_dir
from automation.src.constants import SMALL_SCALE_FILES, STRUCTURE_HINT
from automation.src.llm.wrapper import AzureLLMWrapper

logger = logging.getLogger(__name__)


def _build_translation_prompt(template: str, source_text: str) -> str:
    prompt = template.replace("{text}", source_text)
    return f"{prompt}\n\n{STRUCTURE_HINT}"


@dataclass
class TranslateResult:
    filename: str
    status: str
    token_usage: dict[str, int]
    source_chars: int = 0
    error: str | None = None


def resolve_input_files(
    input_dir: Path,
    small_scale: bool,
    output_dir: Path | None = None,
) -> list[Path]:
    if small_scale:
        files = [input_dir / name for name in SMALL_SCALE_FILES]
    else:
        files = sorted(input_dir.glob("*.txt"))

    if output_dir is not None and small_scale:
        existing = {p.name for p in output_dir.glob("*.txt")}
        files = [p for p in files if p.name in SMALL_SCALE_FILES or p.name in existing]

    return files


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
        logger.error("[%s] failed: %s", filename, exc)
        return TranslateResult(
            filename=filename,
            status="failed",
            token_usage={},
            error=str(exc),
        )


def run_translation_step(
    run_id: str,
    config: ResolvedPipelineConfig,
    wrapper: AzureLLMWrapper,
    *,
    limiter: ConcurrencyLimiter,
    small_scale: bool = False,
    force: bool = False,
) -> dict:
    run_dir = get_run_dir(run_id)
    output_dir = run_dir / "translation"
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = config.translation_prompt_path.read_text(encoding="utf-8")
    input_files = resolve_input_files(config.paths.input_dir, small_scale)

    counts = {"total": len(input_files), "succeeded": 0, "skipped": 0, "failed": 0}
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
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
        tasks = [
            lambda inp=inp, out=out: _translate_one(inp, out, prompt_template, wrapper)
            for inp, out in pending
        ]
        results = limiter.run_parallel(tasks)
        for result in results:
            if isinstance(result, BaseException):
                counts["failed"] += 1
                logger.error("Translation task failed: %s", result)
                continue
            if result.status == "succeeded":
                counts["succeeded"] += 1
                for key in token_usage:
                    token_usage[key] += result.token_usage.get(key, 0)
            else:
                counts["failed"] += 1

    elapsed = round(time.time() - start, 2)
    return {
        "counts": counts,
        "elapsed_s": elapsed,
        "token_usage": token_usage,
        "output_dir": str(output_dir),
    }
