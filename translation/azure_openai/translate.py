import argparse
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from translation.llm.azure_client import get_azure_client, get_azure_model, get_project_root
from translation.llm.translator import PolicyLLMTranslator

SMALL_SCALE_FILES = [
    "ACEH_BIREUEN.txt",
    "LAMPUNG_LAMPUNG_TIMUR.txt",
    "NUSA_TENGGARA_TIMUR_TIMOR_TENGAH_UTARA.txt",
    "SUMATERA_BARAT_PADANG_PARIAMAN.txt",
    "JAWA_TENGAH_SEMARANG.txt",
]

DEFAULT_INPUT_DIR = get_project_root() / "data" / "raw" / "localpolicies"
DEFAULT_PROMPT = get_project_root() / "translation" / "small_scale_test" / "v1" / "prompt.txt"


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-") or "unknown_model"


def get_default_output_dir(model: str) -> Path:
    return get_project_root() / "translation" / "azure_openai" / sanitize_model_name(model)


def resolve_input_files(input_dir: Path, small_scale: bool) -> list[Path]:
    if small_scale:
        return [input_dir / name for name in SMALL_SCALE_FILES]
    return sorted(input_dir.glob("*.txt"))


def run_batch(
    translator: PolicyLLMTranslator,
    input_files: list[Path],
    output_dir: Path,
    force: bool,
) -> dict[str, int]:
    counts = {"succeeded": 0, "skipped": 0, "failed": 0}
    total = len(input_files)

    for index, input_path in enumerate(input_files, 1):
        output_path = output_dir / input_path.name
        logging.info("[%d/%d] %s", index, total, input_path.name)

        if not input_path.exists():
            logging.error("Input file not found: %s", input_path)
            counts["failed"] += 1
            continue

        try:
            result = translator.translate_file(input_path, output_path, force=force)
            status = result["status"]
            counts[status] += 1

            if status == "succeeded":
                usage = result.get("usage", {})
                logging.info(
                    "  done in %.1fs | %d chars | tokens: %s",
                    result["elapsed_s"],
                    result["chars"],
                    usage.get("total_tokens", "n/a"),
                )
            else:
                logging.info("  skipped (output exists)")
        except Exception as exc:
            logging.error("  failed: %s", exc)
            counts["failed"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate policy files using Azure OpenAI."
    )
    parser.add_argument("-i", "--input", type=str, help="Input file path")
    parser.add_argument("-o", "--output", type=str, help="Output file path")
    parser.add_argument(
        "--small-scale",
        action="store_true",
        help="Translate only the 5 benchmark files",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Input directory for batch mode",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for batch mode (defaults to translation/azure_openai/<model>)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=str(DEFAULT_PROMPT),
        help="Path to translation prompt template",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Azure OpenAI deployment name (default: AZURE_OPENAI_MODEL env var)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-translate files even if output already exists",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    client = get_azure_client()
    model = args.model or get_azure_model()
    translator = PolicyLLMTranslator(
        client=client,
        model=model,
        prompt_path=Path(args.prompt),
        temperature=args.temperature,
    )

    if args.input and args.output:
        result = translator.translate_file(
            Path(args.input), Path(args.output), force=args.force
        )
        logging.info("%s -> %s (%s)", args.input, args.output, result["status"])
        return

    if args.input or args.output:
        parser.error("Both -i/--input and -o/--output are required for single-file mode.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else get_default_output_dir(model)

    input_files = resolve_input_files(input_dir, args.small_scale)
    mode = "small-scale" if args.small_scale else "full corpus"
    logging.info(
        "Batch mode: %s | %d files | model: %s | input: %s | output: %s",
        mode,
        len(input_files),
        model,
        input_dir,
        output_dir,
    )

    counts = run_batch(translator, input_files, output_dir, args.force)
    logging.info(
        "Summary: %d succeeded, %d skipped, %d failed",
        counts["succeeded"],
        counts["skipped"],
        counts["failed"],
    )


if __name__ == "__main__":
    main()
