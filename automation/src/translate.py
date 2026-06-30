import logging
import time
from pathlib import Path

from automation.src.config_loader import ResolvedPipelineConfig, get_run_dir
from automation.src.constants import SMALL_SCALE_FILES, STRUCTURE_HINT
from automation.src.llm.wrapper import AzureLLMWrapper

logger = logging.getLogger(__name__)


def _build_translation_prompt(template: str, source_text: str) -> str:
    prompt = template.replace("{text}", source_text)
    return f"{prompt}\n\n{STRUCTURE_HINT}"


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


def run_translation_step(
    run_id: str,
    config: ResolvedPipelineConfig,
    wrapper: AzureLLMWrapper,
    *,
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

    for index, input_path in enumerate(input_files, 1):
        output_path = output_dir / input_path.name
        logger.info("[%d/%d] %s", index, len(input_files), input_path.name)

        if not input_path.exists():
            logger.error("Input file not found: %s", input_path)
            counts["failed"] += 1
            continue

        if output_path.exists() and not force:
            counts["skipped"] += 1
            logger.info("  skipped (output exists)")
            continue

        try:
            source_text = input_path.read_text(encoding="utf-8")
            prompt = _build_translation_prompt(prompt_template, source_text)
            translated, usage = wrapper.complete_text(prompt)
            output_path.write_text(translated, encoding="utf-8")

            for key in token_usage:
                token_usage[key] += usage.get(key, 0)
            counts["succeeded"] += 1
            logger.info(
                "  done | %d chars | tokens: %s",
                len(source_text),
                usage.get("total_tokens", "n/a"),
            )
        except Exception as exc:
            logger.error("  failed: %s", exc)
            counts["failed"] += 1

    elapsed = round(time.time() - start, 2)
    return {
        "counts": counts,
        "elapsed_s": elapsed,
        "token_usage": token_usage,
        "output_dir": str(output_dir),
    }
