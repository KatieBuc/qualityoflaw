"""Fixers for the problems the quality check can repair in place.

Every fixer is a pure ``str -> str`` and idempotent (running it twice equals
running it once). The pipeline picks them from the findings that fired
(``criteria.fix_plan``), always runs ``hygiene``, and re-measures afterwards; a
result that breaks the safety limits (``criteria.fix_acceptable``) is thrown
away. Unlike the per-file scripts of the GlobalLaws pipeline, nothing here
knows a filename.
"""

import html
import re
import unicodedata

from automation.src.input_process.quality.metrics import (
    CHROME_RE,
    HTML_RESIDUE_RE,
    URL_RE,
)

# A line this short that contains chrome / a URL is navigation, not statute text.
MAX_BOILERPLATE_LINE_WORDS = 8
# Running headers / footers: identical, repeated, and neither prose-length nor a heading.
MIN_REPEAT_COUNT = 3
MIN_REPEAT_CHARS, MAX_REPEAT_CHARS = 15, 100

PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:[-–—]\s*\d{1,4}\s*[-–—]|(?:halaman|hal\.?|page)\s*\d{1,4}"
    r"(?:\s*(?:dari|/|of)\s*\d{1,4})?)\s*$",
    re.IGNORECASE,
)
# Lines that must survive even when they repeat: headings and list items.
STRUCTURAL_START_RE = re.compile(
    r"^\s*(?:pasal|bab|bagian|paragraf|ayat|article|section|chapter|§|\(?\d+[.)]|\(?[a-z][.)])",
    re.IGNORECASE,
)
SPACED_RUN_RE = re.compile(r"(?<![^\W\d_])(?:[^\W\d_] ){3,}[^\W\d_](?![^\W\d_])")
TAG_RE = re.compile(HTML_RESIDUE_RE.pattern.split("|&")[0], re.IGNORECASE)  # tags only, not entities


def hygiene(text: str) -> str:
    """Lossless normalization: BOM, line endings, NFC, HTML entities, trailing
    spaces, runs of blank lines (capped at one)."""
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", html.unescape(text))
    out: list[str] = []
    blank = 0
    for line in (l.rstrip(" \t") for l in text.split("\n")):
        if not line.strip():
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(line)
    return "\n".join(out).strip() + "\n" if "".join(out).strip() else ""


def strip_html_residue(text: str) -> str:
    """Remove leftover tags (``<div>``, ``</p>``, bare pseudo-tags). Entities are
    decoded by ``hygiene``."""
    return TAG_RE.sub("", text)


def _is_boilerplate_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped.split()) > MAX_BOILERPLATE_LINE_WORDS:
        return False
    return bool(CHROME_RE.search(stripped) or URL_RE.search(stripped))


def strip_boilerplate(text: str) -> str:
    """Drop web / print furniture, one line at a time, without touching prose:

    * short lines holding web chrome or a URL / e-mail address;
    * bare page numbers ("- 3 -", "Halaman 3 dari 20");
    * running headers / footers: identical lines repeated 3+ times that are
      neither prose-length nor a heading / list item.
    """
    lines = text.split("\n")
    counts: dict[str, int] = {}
    for line in lines:
        key = line.strip()
        if MIN_REPEAT_CHARS <= len(key) <= MAX_REPEAT_CHARS:
            counts[key] = counts.get(key, 0) + 1

    kept = []
    for line in lines:
        key = line.strip()
        if _is_boilerplate_line(line) or PAGE_NUMBER_RE.match(line):
            continue
        if counts.get(key, 0) >= MIN_REPEAT_COUNT and not STRUCTURAL_START_RE.match(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def _join_run(match: re.Match) -> str:
    return match.group(0).replace(" ", "")


def join_spaced_letters(text: str) -> str:
    """Rejoin letter-spaced words ("p e r l i n d u n g a n" -> "perlindungan").

    Where the source kept a double space between words, those breaks survive:
    "a n a k  d a n" -> "anak dan".
    """
    out = []
    for line in text.split("\n"):
        if SPACED_RUN_RE.search(line):
            parts = re.split(r"(\s{2,})", line)
            # A 2+ space gap is a word break in the original: one space now.
            line = "".join(
                " " if part.isspace() else SPACED_RUN_RE.sub(_join_run, part) for part in parts
            )
        out.append(line)
    return "\n".join(out)


FIXERS = {
    "strip_html_residue": strip_html_residue,
    "strip_boilerplate": strip_boilerplate,
    "join_spaced_letters": join_spaced_letters,
}


def apply_fixes(text: str, plan: list[str]) -> tuple[str, list[str]]:
    """hygiene -> each planned fixer -> hygiene. Returns the text and the names
    of the fixers that actually changed it (``hygiene`` included when it did)."""
    applied = []
    current = hygiene(text)
    if current != text:
        applied.append("hygiene")
    for name in plan:
        changed = FIXERS[name](current)
        if changed != current:
            applied.append(name)
            current = changed
    final = hygiene(current)
    return final, applied
