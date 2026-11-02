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
#: The folder that held the unresolved texts before everything was kept in cleaned_txt.
OLD_EXTRACTED_DIRNAME = "extracted_txt"


@dataclass
class LayoutMigration:
    moved_cleaned: list[str] = field(default_factory=list)
    moved_extracted: list[str] = field(default_factory=list)
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
    """Bring data made by an earlier stage 1 layout to the current one. Idempotent; never overwrites.

    1. ``raw/cleaned_txt/`` -> ``preprocessed/cleaned_txt/`` (and a ``<stem>__txt.txt`` in it loses its marker).
    2. ``raw/extracted_txt/`` -> ``preprocessed/cleaned_txt/``: the unresolved texts join the passed ones. A name
       with a marker (``<stem>__pdf|html|txt|ocr.txt``) loses it; when a policy has several, your ``__ocr``
       text wins, then txt, html, pdf, and the others (our own extractions) are removed. A policy that already
       has its file in ``cleaned_txt/`` keeps that one; the other is left where it is and reported.
       The extractions moved are recorded in ``raw/extraction_manifest.json`` (``__ocr`` is yours: no entry).
    3. The JSON reports of the earlier layout are removed (the log replaced them).
    """
    from automation.src.input_process.extract import ExtractionManifest

    result = LayoutMigration()
    target_dir = config.cleaned_txt_dir

    old_cleaned = config.raw_dir / "cleaned_txt"
    if old_cleaned.is_dir() and old_cleaned.resolve() != target_dir.resolve():
        for path in sorted(p for p in old_cleaned.iterdir() if p.is_file()):
            target = target_dir / path.name
            if target.exists():
                result.conflicts.append(f"raw/cleaned_txt/{path.name}")
                continue
            result.moved_cleaned.append(path.name)
            if not dry_run:
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(target))
        if not dry_run and not any(old_cleaned.iterdir()):
            old_cleaned.rmdir()

    if target_dir.is_dir():
        for path in sorted(target_dir.glob("*.txt")):
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

    old_extracted = config.raw_dir / OLD_EXTRACTED_DIRNAME
    if old_extracted.is_dir():
        groups: dict[str, dict[str, Path]] = {}
        for path in sorted(old_extracted.glob("*.txt")):
            parsed = _split_old_marker(path.name)
            stem, fmt = parsed if parsed else (path.stem, None)
            groups.setdefault(stem, {})[fmt or "plain"] = path
        manifest = ExtractionManifest(config.manifest_path)
        for stem, formats in sorted(groups.items()):
            fmt = next((f for f in ("plain",) + OLD_MARKERS if f in formats), None)
            chosen = formats[fmt]
            target = target_dir / f"{stem}.txt"
            if target.exists():
                result.conflicts.append(f"extracted_txt/{chosen.name}")
                continue
            result.moved_extracted.append(f"{chosen.name} -> {target.name}")
            for other_fmt, other in formats.items():
                if other_fmt != fmt:
                    result.removed_superseded.append(other.name)
            if dry_run:
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(chosen), str(target))
            if fmt not in ("plain", "ocr"):
                manifest.record(stem, fmt, target)
            for other_fmt, other in formats.items():
                if other_fmt != fmt:
                    other.unlink(missing_ok=True)
        if not dry_run:
            manifest.save()
            if not any(old_extracted.iterdir()):
                old_extracted.rmdir()

    for name in OLD_REPORTS:
        report = config.raw_dir / name
        if report.exists():
            result.removed_reports.append(name)
            if not dry_run:
                report.unlink()
    return result


# --------------------------------------------------------------------------- manifest rebuild
@dataclass
class ManifestRebuild:
    registered: list[str] = field(default_factory=list)      # no entry before: now a record of our extraction
    source_added: list[str] = field(default_factory=list)    # our extraction whose original had not been recorded
    up_to_date: list[str] = field(default_factory=list)
    edited: list[str] = field(default_factory=list)          # has an entry, but the text is no longer what we wrote
    no_original: list[str] = field(default_factory=list)     # no original in raw_files: left unregistered


def rebuild_manifest(config: InputProcessConfig, *, dry_run: bool = False) -> ManifestRebuild:
    """Register every ``cleaned_txt/<stem>.txt`` in ``raw/extraction_manifest.json`` as our extraction of its original.

    The manifest only had entries for the files that were still waiting under the two-folder layout; the texts that
    had passed were dropped from it. Without an entry a text counts as yours, so a PDF text is judged by the OCR
    rules instead of being able to "need OCR", and a change of its original is never noticed. This gives each such
    text an entry: its format and hash, and the hash of the original it is taken to come from (``txt`` before
    ``html`` before ``pdf``, as `extract` chooses).

    Every text without an entry is taken to be our extraction. Do not run it if you saved OCR text or hand
    corrections over files that have no entry: they would be judged as extracted text. A text that has an entry but
    no longer matches it is yours and is left alone. Idempotent.
    """
    from automation.src.input_process.extract import (
        ExtractionManifest,
        choose_originals,
        cleaned_stems,
        list_sources,
    )

    result = ManifestRebuild()
    texts = cleaned_stems(config.cleaned_txt_dir)
    originals, _ = choose_originals(list_sources(config)) if config.raw_files_dir.is_dir() else ({}, [])
    manifest = ExtractionManifest(config.manifest_path)

    for stem, text_path in sorted(texts.items()):
        entry = manifest.data.get(stem)
        original = originals.get(stem)
        if entry is None:
            if original is None:
                result.no_original.append(stem)
                continue
            result.registered.append(stem)
            if not dry_run:
                manifest.record(stem, original[0], text_path, source=original[1])
        elif manifest.machine_format(stem, text_path) is None:
            result.edited.append(stem)
        elif "source_sha256" not in entry and original is not None:
            result.source_added.append(stem)
            if not dry_run:
                manifest.remember_source(stem, original[1])
        else:
            result.up_to_date.append(stem)
    if not dry_run:
        manifest.save()
    return result
