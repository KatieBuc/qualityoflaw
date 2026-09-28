import logging
from pathlib import Path
from typing import Callable

from automation.src.constants import DEFAULT_SMALL_SCALE_STEMS

logger = logging.getLogger(__name__)


def select_small_scale(
    files: list[Path],
    allowed: set[str],
    policy_dir: Path,
    key: Callable[[Path], str] = lambda path: path.name,
) -> list[Path]:
    """Restrict `files` to the `--small-scale` benchmark set.

    A file is kept when `key(file)` is in `allowed` (its filename by default;
    the Markdown path keys on the bare stem so the suffix doesn't matter). Selecting
    nothing from a non-empty directory is almost always a corpus/config
    mismatch (`paths.small_scale_stems` naming documents that aren't in this
    corpus), and continuing would make every step run on zero files without
    saying why -- so that case raises instead of returning an empty list.
    A partial match is only warned about.
    """
    selected = [f for f in files if key(f) in allowed]
    if files and not selected:
        raise FileNotFoundError(
            f"--small-scale matched none of the {len(files)} file(s) in {policy_dir}. "
            f"Expected: {', '.join(sorted(allowed))}. Check "
            "`paths.small_scale_stems` in pipeline_config.yaml against this corpus."
        )
    missing = sorted(allowed - {key(f) for f in selected})
    if selected and missing:
        logger.warning(
            "--small-scale: %d expected file(s) not found in %s: %s",
            len(missing),
            policy_dir,
            ", ".join(missing),
        )
    return selected


def filter_policy_files(
    policy_dir: Path,
    small_scale: bool,
    stems: tuple[str, ...] = DEFAULT_SMALL_SCALE_STEMS,
) -> list[Path]:
    """Every plain-text policy in `policy_dir`, optionally the benchmark set.

    `stems` is `config.paths.small_scale_stems`; the plain-text artifacts are
    named `<stem>.txt`.
    """
    files = sorted(policy_dir.glob("*.txt"))
    if small_scale:
        files = select_small_scale(files, {f"{s}.txt" for s in stems}, policy_dir)
    return files
