import os
import re
from pathlib import Path


def clean_content(text: str) -> str:
    """Removes split page markers, image placeholders, and trailing whitespace."""

    # 1. Remove Page Split placeholders (case-insensitive)
    text = re.sub(r"<---*\s*Page Split\s*---*>", "", text, flags=re.IGNORECASE)

    # 2. Remove Markdown image tags like ![](images/...) or ![alt](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # 3. Remove HTML/XML-like image placeholders if any exist (e.g., <img ...>)
    text = re.sub(r"<img\s+[^>]*>", "", text, flags=re.IGNORECASE)

    # 4. Clean up multiple empty lines left behind by tag removal (keep max 2 newlines)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # 5. Strip trailing spaces from each line
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned_text = "\n".join(lines).strip() + "\n"

    return cleaned_text


def process_mmd_folder(input_dir: str, output_dir: str):
    """Recursively processes .mmd files from input_dir and saves output files as .cleaned.md directly into output_dir.

    Ignores files ending with _det.mmd.
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()

    if not input_path.exists():
        print(f"Error: Input directory '{input_path}' does not exist.")
        return

    # Create target directory if it doesn't exist yet
    output_path.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped_count = 0

    # Walk recursively through all files in input_path
    for root, _, files in os.walk(input_path):
        for file in files:
            file_lower = file.lower()

            # Skip files ending with _det.mmd
            if file_lower.endswith("_det.mmd"):
                skipped_count += 1
                continue

            if file_lower.endswith(".mmd"):
                source_file_path = Path(root) / file

                # Change extension from .mmd to .cleaned.md
                stem_name = source_file_path.stem
                target_file_path = output_path / f"{stem_name}.cleaned.md"

                # Read source .mmd file
                with open(source_file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Clean the file content
                cleaned_text = clean_content(content)

                # Write cleaned content directly to the .cleaned.md target file
                with open(target_file_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)

                print(f"Saved: {target_file_path.name}")
                processed_count += 1

    print(
        f"\nDone! Successfully created {processed_count} .cleaned.md file(s) in '{output_path}' "
        f"(skipped {skipped_count} '_det.mmd' file(s))."
    )


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from automation.src.constants import DEFAULT_PROJECT
    from automation.src.paths import project_dirs

    parser = argparse.ArgumentParser(
        description="Clean raw .mmd extractions into .cleaned.md files for the pipeline."
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Project under data/.")
    parser.add_argument("--input", default=None, help="Folder of .mmd files (default: <project>/preprocessed/QOL_reextract).")
    parser.add_argument(
        "--output",
        default=None,
        help="Folder that receives the .cleaned.md files (default: <project>/preprocessed/cleaned_markdown).",
    )
    args = parser.parse_args()
    dirs = project_dirs(args.project)
    args.input = args.input or str(dirs.preprocessed / "QOL_reextract")
    args.output = args.output or str(dirs.preprocessed_cleaned_markdown)

    process_mmd_folder(args.input, args.output)
