"""Locating and naming policies in the Markdown path.

The curated corpus names files ``<POLICY>.cleaned.md``, but every artifact
downstream of translation is keyed on the bare ``<POLICY>`` -- the golden
dataset's `filename` column joins on ``<POLICY>.txt``
(`evaluate_accuracy.run_accuracy_evaluation`), and `diagnose` looks the
original up by that same name. So the ``.cleaned`` suffix is stripped once,
here, at the point of entry, and never appears in a run folder.

Counterpart to `automation.src.policy_files`, which globs ``*.txt`` for the
raw-text path and is left untouched.
"""

from pathlib import Path

from automation.src.constants import SMALL_SCALE_FILES

# The curated corpus's naming convention.
CLEANED_MD_SUFFIX = ".cleaned.md"

# The same four benchmark policies `--small-scale` selects for the raw-text
# path, as bare stems. Derived from the single source of truth rather than
# duplicated, so the two paths can never drift apart.
SMALL_SCALE_STEMS = tuple(Path(name).stem for name in SMALL_SCALE_FILES)


def policy_stem(path: Path) -> str:
    """``ACEH_BIREUEN.cleaned.md`` -> ``ACEH_BIREUEN`` (and ``.md`` -> stem)."""
    return path.name.removesuffix(CLEANED_MD_SUFFIX).removesuffix(".md")


def source_markdown_path(input_dir: Path, stem: str) -> Path:
    return input_dir / f"{stem}{CLEANED_MD_SUFFIX}"


def markdown_policy_files(
    policy_dir: Path, small_scale: bool, suffix: str = CLEANED_MD_SUFFIX
) -> list[Path]:
    """Every Markdown policy in `policy_dir`, sorted, optionally the benchmark four.

    `suffix` distinguishes the curated input (``.cleaned.md``) from the
    run's own translated output (``.md``); both resolve to the same stems.
    """
    files = sorted(p for p in policy_dir.glob(f"*{suffix}") if p.is_file())
    if small_scale:
        allowed = set(SMALL_SCALE_STEMS)
        files = [p for p in files if policy_stem(p) in allowed]
    return files
