"""Header validation between a source Markdown file and its translation.

Stage 3 (human adjustment) lets people edit ``cleaned_markdown`` and
``translation_markdown`` freely, with one rule: the headings of corresponding
files must still match, because the evaluation stage pairs sections by
position. This module checks that rule -- strictly -- and `promote` uses it as
the gate for moving the pair from ``preprocessed/`` to ``processed/``.

Unlike `chunking.check_structure`, which tolerates OCR repairs and only logs,
a mismatch here is a failure: same number of headings, same level at every
position (so the same hierarchy).

CLI:  python -m automation.src.markdown.validate [--project NAME] [--promote]
"""

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from automation.src.constants import DEFAULT_PROJECT
from automation.src.markdown.chunking import heading_signature
from automation.src.markdown.policy_files import CLEANED_MD_SUFFIX, policy_stem
from automation.src.paths import project_dirs


@dataclass
class HeaderCheck:
    stem: str
    ok: bool
    source_headings: int = 0
    translated_headings: int = 0
    # Human-readable reason when not ok.
    problem: str = ""


@dataclass
class ValidationReport:
    checks: list[HeaderCheck] = field(default_factory=list)

    @property
    def failed(self) -> list[HeaderCheck]:
        return [c for c in self.checks if not c.ok]

    @property
    def ok(self) -> bool:
        return bool(self.checks) and not self.failed


def validate_headers(stem: str, source_md: str, translated_md: str) -> HeaderCheck:
    """Compare heading count and level sequence of one source/translation pair."""
    src = heading_signature(source_md)
    tr = heading_signature(translated_md)
    check = HeaderCheck(stem, True, len(src), len(tr))
    if len(src) != len(tr):
        check.ok = False
        check.problem = f"heading count differs: source {len(src)}, translation {len(tr)}"
        return check
    for position, (a, b) in enumerate(zip(src, tr), start=1):
        if a != b:
            check.ok = False
            check.problem = (
                f"heading hierarchy differs at heading {position}: "
                f"source level {a}, translation level {b}"
            )
            break
    return check


def validate_directories(
    source_dir: Path,
    translated_dir: Path,
    source_suffix: str = CLEANED_MD_SUFFIX,
    stems: tuple[str, ...] | None = None,
) -> ValidationReport:
    """Validate every policy found on either side; a missing counterpart fails."""
    sources = {policy_stem(p): p for p in source_dir.glob(f"*{source_suffix}") if p.is_file()}
    translations = {policy_stem(p): p for p in translated_dir.glob("*.md") if p.is_file()}
    names = sorted(set(sources) | set(translations))
    if stems is not None:
        names = [n for n in names if n in set(stems)]

    report = ValidationReport()
    for stem in names:
        if stem not in sources:
            report.checks.append(HeaderCheck(stem, False, problem="no source file"))
        elif stem not in translations:
            report.checks.append(HeaderCheck(stem, False, problem="no translation file"))
        else:
            report.checks.append(
                validate_headers(
                    stem,
                    sources[stem].read_text(encoding="utf-8"),
                    translations[stem].read_text(encoding="utf-8"),
                )
            )
    return report


def promote(
    project: str, *, source_suffix: str = CLEANED_MD_SUFFIX, force: bool = False
) -> ValidationReport:
    """Validate ``preprocessed/`` and, only if every pair passes, move both
    folders' files to ``processed/``. Returns the report either way; nothing is
    moved on failure. Existing files in ``processed/`` are never overwritten
    unless `force`."""
    dirs = project_dirs(project)
    pairs = [
        (dirs.preprocessed_cleaned_markdown, dirs.processed_cleaned_markdown),
        (dirs.preprocessed_translation_markdown, dirs.processed_translation_markdown),
    ]
    for source, _ in pairs:
        if not source.is_dir():
            raise FileNotFoundError(f"Missing folder: {source}")

    report = validate_directories(
        dirs.preprocessed_cleaned_markdown,
        dirs.preprocessed_translation_markdown,
        source_suffix,
    )
    if not report.ok:
        return report

    moves = [
        (f, dest_dir / f.name)
        for src_dir, dest_dir in pairs
        for f in sorted(src_dir.iterdir())
        if f.is_file()
    ]
    clashes = [dest for _, dest in moves if dest.exists()]
    if clashes and not force:
        raise FileExistsError(
            f"{len(clashes)} file(s) already in processed/ (e.g. {clashes[0]}); "
            "use force to overwrite"
        )
    for _, dest_dir in pairs:
        dest_dir.mkdir(parents=True, exist_ok=True)
    for src_file, dest in moves:
        shutil.move(str(src_file), str(dest))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate (and optionally promote) markdown pairs.")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--suffix", default=CLEANED_MD_SUFFIX, help="Source file suffix.")
    parser.add_argument("--promote", action="store_true", help="Move preprocessed -> processed if valid.")
    parser.add_argument("--force", action="store_true", help="Overwrite files already in processed/.")
    args = parser.parse_args(argv)

    if args.promote:
        report = promote(args.project, source_suffix=args.suffix, force=args.force)
    else:
        dirs = project_dirs(args.project)
        report = validate_directories(
            dirs.preprocessed_cleaned_markdown, dirs.preprocessed_translation_markdown, args.suffix
        )

    for check in report.failed:
        print(f"FAILED {check.stem}: {check.problem}")
    print(f"{len(report.checks) - len(report.failed)}/{len(report.checks)} pairs valid")
    if not report.checks:
        print("No files found to validate.")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
