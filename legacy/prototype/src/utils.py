import difflib
from pathlib import Path


def print_file_diff(file1: Path, file2: Path) -> None:
    """Print line-by-line differences between two text files."""

    lines1 = file1.read_text(encoding="utf-8").splitlines()
    lines2 = file2.read_text(encoding="utf-8").splitlines()

    for line in difflib.ndiff(lines1, lines2):
        if line.startswith(("- ", "+ ")):
            print(line)
