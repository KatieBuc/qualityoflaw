"""One-time move of originals from ``raw/`` into ``raw/raw_files/``.

Before the stage 1 refactor, the PDF / HTML / TXT originals sat directly in
``data/<project>/raw/``. They now live in ``raw/raw_files/``. This moves
them across, never overwriting a file that is already there.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from automation.src.input_process.config import InputProcessConfig

#: Extensions treated as originals (lower case; matching is case-insensitive).
SOURCE_SUFFIXES = (".pdf", ".html", ".htm", ".txt")


@dataclass
class MigrationResult:
    moved: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)


def migrate_raw(config: InputProcessConfig, *, dry_run: bool = False) -> MigrationResult:
    source_dir = config.raw_dir
    target_dir = config.raw_files_dir
    result = MigrationResult()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Raw folder not found: {source_dir}")
    if source_dir.resolve() == target_dir.resolve():
        raise ValueError("raw_files_dir is the raw folder itself; nothing to migrate")

    for path in sorted(p for p in source_dir.iterdir() if p.is_file()):
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            result.ignored.append(path.name)
            continue
        destination = target_dir / path.name
        if destination.exists():
            result.skipped_existing.append(path.name)
            continue
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
        result.moved.append(path.name)
    return result


# --------------------------------------------------------------------------- layout migration
#: Markers the earlier layout put in file names, outranking one another in this order.
OLD_MARKERS = ("ocr", "txt", "html", "pdf")
OLD_REPORTS = ("input_report.json", "markdown_report.json")


@dataclass
class LayoutMigration:
    moved_cleaned: list[str] = field(default_factory=list)
    renamed: list[str] = field(default_factory=list)
    removed_superseded: list[str] = field(default_factory=list)
    removed_reports: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


def _split_old_marker(name: str) -> tuple[str, str] | None:
    if not name.lower().endswith(".txt"):
        return None
    stem, sep, fmt = name[: -len(".txt")].rpartition("__")
    return (stem, fmt) if sep and stem and fmt in OLD_MARKERS else None


def migrate_layout(config: InputProcessConfig, *, dry_run: bool = False) -> LayoutMigration:
    """Bring data made by the earlier stage 1 layout to the current one. Idempotent; never overwrites.

    1. ``raw/cleaned_txt/`` -> ``preprocessed/cleaned_txt/`` (and a ``<stem>__txt.txt`` in it loses its marker).
    2. ``extracted_txt/<stem>__pdf|html|txt|ocr.txt`` -> ``<stem>.txt``. When a policy has several, your
       ``__ocr`` text wins, then txt, html, pdf; the others are our own extractions and are removed. The
       extractions we keep are recorded in ``raw/extraction_manifest.json`` (``__ocr`` is yours: no entry).
    3. The JSON reports of the earlier layout are removed (the log replaced them).
    """
    from automation.src.input_process.extract import ExtractionManifest

    result = LayoutMigration()

    old_cleaned = config.raw_dir / "cleaned_txt"
    new_cleaned = config.cleaned_txt_dir
    if old_cleaned.is_dir() and old_cleaned.resolve() != new_cleaned.resolve():
        for path in sorted(p for p in old_cleaned.iterdir() if p.is_file()):
            target = new_cleaned / path.name
            if target.exists():
                result.conflicts.append(f"cleaned_txt/{path.name}")
                continue
            result.moved_cleaned.append(path.name)
            if not dry_run:
                new_cleaned.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(target))
        if not dry_run and not any(old_cleaned.iterdir()):
            old_cleaned.rmdir()

    if new_cleaned.is_dir():
        for path in sorted(new_cleaned.glob("*.txt")):
            parsed = _split_old_marker(path.name)
            if parsed is None:
                continue
            target = path.with_name(f"{parsed[0]}.txt")
            if target.exists():
                result.conflicts.append(f"cleaned_txt/{path.name}")
                continue
            result.renamed.append(f"{path.name} -> {target.name}")
            if not dry_run:
                path.rename(target)

    if config.extracted_txt_dir.is_dir():
        groups: dict[str, dict[str, Path]] = {}
        for path in sorted(config.extracted_txt_dir.glob("*.txt")):
            parsed = _split_old_marker(path.name)
            if parsed:
                groups.setdefault(parsed[0], {})[parsed[1]] = path
        manifest = ExtractionManifest(config.manifest_path)
        for stem, formats in sorted(groups.items()):
            fmt = next(f for f in OLD_MARKERS if f in formats)
            target = config.extracted_txt_dir / f"{stem}.txt"
            if target.exists():
                result.conflicts.append(f"extracted_txt/{formats[fmt].name}")
                continue
            result.renamed.append(f"{formats[fmt].name} -> {target.name}")
            for other_fmt, other in formats.items():
                if other_fmt != fmt:
                    result.removed_superseded.append(other.name)
            if dry_run:
                continue
            formats[fmt].rename(target)
            if fmt != "ocr":
                manifest.record(stem, fmt, target)
            for other_fmt, other in formats.items():
                if other_fmt != fmt:
                    other.unlink(missing_ok=True)
        if not dry_run:
            manifest.save()

    for name in OLD_REPORTS:
        report = config.raw_dir / name
        if report.exists():
            result.removed_reports.append(name)
            if not dry_run:
                report.unlink()
    return result
