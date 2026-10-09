"""Command line for stage 1.1.

    python -m automation.src.input_process migrate-raw [--dry-run]
    python -m automation.src.input_process extract [--small-scale]
"""

import argparse
import sys
from pathlib import Path

from automation.src.constants import DEFAULT_PIPELINE_CONFIG
from automation.src.input_process.config import load_input_process_config
from automation.src.input_process.extract import extract_all
from automation.src.input_process.migrate import migrate_raw


def _cmd_migrate_raw(args: argparse.Namespace, config) -> int:
    result = migrate_raw(config, dry_run=args.dry_run)
    verb = "Would move" if args.dry_run else "Moved"
    print(f"{verb} {len(result.moved)} file(s) from {config.raw_dir} to {config.raw_files_dir}")
    if result.skipped_existing:
        print(f"Skipped {len(result.skipped_existing)} already present in the target:")
        for name in result.skipped_existing:
            print(f"  {name}")
    if result.ignored:
        print(f"Left {len(result.ignored)} non-source file(s) in place: {', '.join(result.ignored)}")
    return 0


def _cmd_extract(args: argparse.Namespace, config) -> int:
    results = extract_all(config, small_scale=args.small_scale)
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print(f"Extracted from {config.raw_files_dir} into {config.extracted_txt_dir}")
    print(", ".join(f"{n} {status}" for status, n in sorted(counts.items())) or "no files")
    for r in results:
        if r.status in ("failed", "unsupported"):
            print(f"  {r.status}: {r.source} ({r.detail})", file=sys.stderr)
    return 1 if counts.get("failed") else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m automation.src.input_process",
        description="Stage 1.1: extract, quality-check and fix the raw files.",
    )
    parser.add_argument("--pipeline-config", default=str(DEFAULT_PIPELINE_CONFIG))
    parser.add_argument("--project", default=None, help="Override the project in the config.")
    sub = parser.add_subparsers(dest="command", required=True)

    migrate = sub.add_parser(
        "migrate-raw", help="Move originals from raw/ into raw/raw_files/ (one-time)."
    )
    migrate.add_argument("--dry-run", action="store_true")
    migrate.set_defaults(func=_cmd_migrate_raw)

    extract = sub.add_parser(
        "extract", help="PDF/HTML -> txt, TXT -> copy, into extracted_txt/ (existing outputs are kept)."
    )
    extract.add_argument("--small-scale", action="store_true", help="Only paths.small_scale_stems.")
    extract.set_defaults(func=_cmd_extract)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_input_process_config(Path(args.pipeline_config), project=args.project)
    try:
        return args.func(args, config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
