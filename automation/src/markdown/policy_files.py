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

from automation.src.constants import DEFAULT_SMALL_SCALE_STEMS
from automation.src.policy_files import select_small_scale

# The curated corpus's naming convention.
CLEANED_MD_SUFFIX = ".cleaned.md"

def policy_stem(path: Path) -> str:
    """``ACEH_BIREUEN.cleaned.md`` -> ``ACEH_BIREUEN`` (and ``.md`` -> stem)."""
    return path.name.removesuffix(CLEANED_MD_SUFFIX).removesuffix(".md")


def source_markdown_path(input_dir: Path, stem: str, suffix: str = CLEANED_MD_SUFFIX) -> Path:
    return input_dir / f"{stem}{suffix}"


def markdown_policy_files(
    policy_dir: Path,
    small_scale: bool,
    suffix: str = CLEANED_MD_SUFFIX,
    stems: tuple[str, ...] = DEFAULT_SMALL_SCALE_STEMS,
) -> list[Path]:
    """Every Markdown policy in `policy_dir`, sorted, optionally the benchmark set.

    `stems` is `config.paths.small_scale_stems` -- the benchmark set belongs to
    the corpus, not to this module. Matching is on `policy_stem`, so it works
    whichever suffix the corpus uses.

    `suffix` distinguishes the curated input (``.cleaned.md``) from the
    run's own translated output (``.md``); both resolve to the same stems.
    """
    files = sorted(p for p in policy_dir.glob(f"*{suffix}") if p.is_file())
    if small_scale:
        files = select_small_scale(files, set(stems), policy_dir, key=policy_stem)
    return files
