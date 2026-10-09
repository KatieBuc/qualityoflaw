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
