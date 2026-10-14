"""Command line for stage 1.1.

    python -m automation.src.input_process migrate-raw [--dry-run]
    python -m automation.src.input_process migrate-layout [--dry-run]   # one-off: move to the current folder layout
    python -m automation.src.input_process extract [--small-scale]
    python -m automation.src.input_process check [--report-only] [--small-scale]
    python -m automation.src.input_process run [--small-scale]      # extract + check
    python -m automation.src.input_process tidy [--dry-run]         # one-off clean-up of extracted_txt
    python -m automation.src.input_process markdown [--force] [--small-scale]   # stage 1.2: TXT -> Markdown
"""

import argparse
import sys
from pathlib import Path

from automation.src.constants import DEFAULT_PIPELINE_CONFIG
from automation.src.input_process.config import load_input_process_config
from automation.src.input_process.extract import extract_all
from automation.src.input_process.migrate import migrate_layout, migrate_raw
from automation.src.input_process.pipeline import check_all, tidy_extracted


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


def _cmd_migrate_layout(args: argparse.Namespace, config) -> int:
    r = migrate_layout(config, dry_run=args.dry_run)
    verb = "Would" if args.dry_run else "Did"
    print(f"{verb}: move {len(r.moved_cleaned)} file(s) to {config.cleaned_txt_dir}, rename {len(r.renamed)}, "
          f"remove {len(r.removed_superseded)} superseded extraction(s) and {len(r.removed_reports)} old report(s)")
    for item in r.conflicts:
        print(f"  left alone (target exists): {item}")
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


def _print_report_summary(report: dict, config) -> None:
    s = report["summary"]
    print(
        f"Quality check: {s['files']} policy(ies) -> {s['passed']} in cleaned_txt, "
        f"{s['needs_ocr']} need OCR, {s['eliminated']} eliminated"
    )
    for name, items in report.get("warnings", {}).items():
        print(f"  warning: {len(items)} {name.replace('_', ' ')} (see the log)")
    for entry in report["needs_ocr"]:
        print(f"  needs OCR: {entry['pdf'] or entry['stem']} -> replace "
              f"{config.extracted_txt_dir / entry['expected_ocr_file']} with the OCR text")
    print(f"Log: {config.log_path}")


def _cmd_tidy(args: argparse.Namespace, config) -> int:
    removed = tidy_extracted(config, dry_run=args.dry_run)
    verb = "Would remove" if args.dry_run else "Removed"
    print(f"{verb} {len(removed)} extracted file(s) of policies already in {config.cleaned_txt_dir}")
    return 0


def _cmd_check(args: argparse.Namespace, config) -> int:
    report = check_all(config, write_cleaned=not args.report_only, small_scale=args.small_scale)
    _print_report_summary(report, config)
    if args.report_only:
        print("(log only: cleaned_txt/ was not touched)")
    else:
        print(f"Passed files are in {config.cleaned_txt_dir}")
    return 0


def _cmd_markdown(args: argparse.Namespace, config) -> int:
    from automation.src.ingest import run_ingest_step
    from automation.src.paths import project_dirs

    out_dir = project_dirs(config.project).preprocessed_cleaned_markdown
    result = run_ingest_step(
        config.markdown_source_dir,
        out_dir,
        language_for=config.language_for,
        stems=set(config.small_scale_stems) if args.small_scale else None,
        force=args.force,
        log_path=config.log_path,
    )
    summary = result["summary"]
    print(
        f"Markdown ({config.markdown_source} -> {out_dir}): {summary['converted']} converted, "
        f"{summary['kept']} already there, {summary['skipped']} skipped"
    )
    for item in result["skipped"]:
        print(f"  skipped {item['stem']}: {item['reason']}")
    return 0


def _cmd_run(args: argparse.Namespace, config) -> int:
    code = _cmd_extract(args, config)
    check_args = argparse.Namespace(report_only=False, small_scale=args.small_scale)
    return max(code, _cmd_check(check_args, config))


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

    layout = sub.add_parser(
        "migrate-layout",
        help="One-off: move data made by the earlier layout (markers in file names, raw/cleaned_txt, JSON reports).",
    )
    layout.add_argument("--dry-run", action="store_true")
    layout.set_defaults(func=_cmd_migrate_layout)

    extract = sub.add_parser(
        "extract", help="PDF/HTML -> txt, TXT -> copy, into extracted_txt/ (existing outputs are kept)."
    )
    extract.add_argument("--small-scale", action="store_true", help="Only paths.small_scale_stems.")
    extract.set_defaults(func=_cmd_extract)

    check = sub.add_parser(
        "check",
        help="Quality-check, fix and re-check extracted_txt/; passed texts go to cleaned_txt/; everything is logged.",
    )
    check.add_argument("--report-only", action="store_true", help="Only log the result; move and delete nothing.")
    check.add_argument("--small-scale", action="store_true", help="Only paths.small_scale_stems.")
    check.set_defaults(func=_cmd_check)

    tidy = sub.add_parser(
        "tidy",
        help="Remove extracted files of policies already in cleaned_txt (one-off, for data from before).",
    )
    tidy.add_argument("--dry-run", action="store_true")
    tidy.set_defaults(func=_cmd_tidy)

    markdown = sub.add_parser(
        "markdown",
        help="Stage 1.2: convert markdown_source (cleaned_txt or gold) to preprocessed/cleaned_markdown/.",
    )
    markdown.add_argument("--force", action="store_true", help="Overwrite existing .cleaned.md files.")
    markdown.add_argument("--small-scale", action="store_true", help="Only paths.small_scale_stems.")
    markdown.set_defaults(func=_cmd_markdown)

    run = sub.add_parser("run", help="extract, then check.")
    run.add_argument("--small-scale", action="store_true", help="Only paths.small_scale_stems.")
    run.set_defaults(func=_cmd_run)
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
