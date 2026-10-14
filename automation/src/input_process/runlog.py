"""The stage 1 log file: what was done to each file and how it went.

One line per event, appended to ``raw/input_process.log``:

    2026-10-14 10:21:05,412 INFO    PASS ACEH_BIREUEN (txt) -> cleaned_txt/ACEH_BIREUEN.txt
    2026-10-14 10:21:06,090 WARNING NEEDS_OCR BALI_X: C1_garbage=0.31 (limit 0.05) - replace extracted_txt/BALI_X.txt ...

WARNING marks a file that failed the quality check (or something else that needs a person); everything
else is INFO. The log is appended to, never rewritten, so the history of every run stays.
"""

import logging
from contextlib import contextmanager
from pathlib import Path

LOGGER_NAME = "qol.input_process"
FORMAT = "%(asctime)s %(levelname)-7s %(message)s"


@contextmanager
def run_log(path: Path, extra: Path | None = None):
    """Yield a logger that appends to `path` (and to `extra`, when given); the handlers are removed on exit."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False        # the console summary is printed by the caller
    handlers = []
    for target in (path, extra):
        if target is None:
            continue
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(target, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter(FORMAT))
        logger.addHandler(handler)
        handlers.append(handler)
    try:
        yield logger
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
