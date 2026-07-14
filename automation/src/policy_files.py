from pathlib import Path

from automation.src.constants import SMALL_SCALE_FILES


def filter_policy_files(policy_dir: Path, small_scale: bool) -> list[Path]:
    files = sorted(policy_dir.glob("*.txt"))
    if small_scale:
        allowed = set(SMALL_SCALE_FILES)
        files = [f for f in files if f.name in allowed]
    return files
