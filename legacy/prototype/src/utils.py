import difflib
import re
from pathlib import Path


def print_file_diff(file1: Path, file2: Path) -> None:
    """Print line-by-line differences between two text files."""

    lines1 = file1.read_text(encoding="utf-8").splitlines()
    lines2 = file2.read_text(encoding="utf-8").splitlines()

    for line in difflib.ndiff(lines1, lines2):
        if line.startswith(("- ", "+ ")):
            print(line)


def clean_markdown_folder(folder, dry_run: bool = False) -> int:
    """
    Clean up markdown formatting in every .md file within `folder` (recursively).

    Rules applied:
    1. "word :" -> "word:" (removes space(s) before a colon that follows
       a non-space character).
    2. Lines ending in ';' get exactly two trailing spaces added, so the
       following newline renders as a markdown hard line break.
    3. Lines ending in '.' get the same two-space treatment.
    4. Bullet dashes ("- " at the start of a line) are stripped out.

    Args:
        folder: Path to the folder of .md files to clean.
        dry_run: If True, report what would change without writing anything.

    Returns:
        The number of files that were changed (or would be, in dry-run mode).
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    changed = 0
    for path in sorted(folder.rglob("*.md")):
        original = path.read_text(encoding="utf-8")
        cleaned = _clean_markdown_text(original)
        if cleaned != original:
            changed += 1
            if dry_run:
                print(f"[dry-run] Would update: {path}")
            else:
                path.write_text(cleaned, encoding="utf-8")
                print(f"Updated: {path}")
        else:
            print(f"No changes: {path}")

    return changed


def _clean_markdown_text(text: str) -> str:
    """Apply the markdown cleanup rules to a single string of text."""
    # # Remove "\n" + any number of spaces + "-" + any number of spaces
    # text = re.sub(r"\n *- *", "", text)

    # # Rules 2 & 3: lines ending in ';' or '.' get exactly two trailing spaces
    # lines = text.split("\n")
    # for i, line in enumerate(lines):
    #     trimmed = line.rstrip(" ")
    #     if trimmed.endswith(";") or trimmed.endswith("."):
    #         lines[i] = trimmed + "  "
    # text = "\n".join(lines)

    # # Rule 1: "word :" -> "word:"
    # text = re.sub(r"(\S) +:", r"\1:", text)

    # # Rule 4: strip bullet dashes at the start of a line
    # text = re.sub(r"(?m)^- ", "", text)

    # #New rule: ";  X." -> ";  \nX." (single letter followed by a period)
    # text = re.sub(r";  ([A-Za-z])\.", r";  \n\1.", text)

    #New rule: "; X." -> ";  \nX." (single letter followed by a period)
    # text = re.sub(r"; ([A-Za-z])\.", r";  \n\1.", text)
    # text = re.sub(r": ([A-Za-z])\.", r":  \n\1.", text)
    
    # text = re.sub(r": \n([A-Za-z])\.", r":  \n\1.", text)
    # text = re.sub(r"; \n([A-Za-z])\.", r";  \n\1.", text)

    # text = re.sub(r":\n([A-Za-z])\.", r":  \n\1.", text)
    # text = re.sub(r";\n([A-Za-z])\.", r";  \n\1.", text)

    #text = re.sub(r", ([A-Za-z])\.", r",  \n\1.", text)
    #text = re.sub(r",\n([A-Za-z])\.", r",  \n\1.", text)
    text = re.sub(r"; dan([A-Za-z])\.", r"; dan  \n\1.", text)

    # "Menimbang: " / "Mengingat: " -> "Menimbang:  \n" / "Mengingat:  \n"
    text = re.sub(r"(Menimbang|Mengingat): ", r"\1:  \n", text)

    return text

"""
from src.utils import clean_markdown_folder
clean_markdown_folder("data/processed/localpolicies/cleaned_markdown")
"""