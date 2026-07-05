"""Clean -> chunk -> combine pipeline for oversized legal policy text files.

Implements the spec: OCR noise removal, structural-marker detection
(BAB / Pasal / Bagian / Paragraf / all-caps titles), paragraph reflow,
structural section splitting (no overlap), and fallback sentence-boundary
sub-chunking (with trailing-context carryover) for any section that still
exceeds the safe character limit.

This module is pure text processing: no I/O, no LLM calls. It is used by
`automation.src.translate` when `translation.chunking.enabled` is set in
`pipeline_config.yaml`.
"""

import logging
import re
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# Below the observed ~50,000 char timeout ceiling, with safety margin.
SAFE_LIMIT_DEFAULT = 32000

# Trailing characters of the previous sub-chunk carried forward as context
# for the next fallback sub-chunk.
CONTEXT_TAIL_CHARS = 300

# Section 1.1 — short, meaningless OCR residue lines (page numbers, scan noise).
NOISE_LINE_RE = re.compile(r"^[\dA-Za-z]{1,3}$")

# Section 1.1 — a line that is nothing but a single punctuation mark (e.g. a
# stray "." left behind after the paragraph it belonged to already ended on
# the previous line). Also OCR residue; dropped before it can reach a chunk.
LONE_PUNCTUATION_RE = re.compile(r"^[.;,]$")

# Section 1.2 — structural marker lines (BAB/Pasal/Bagian/Paragraf + a token).
STRUCTURE_MARKER_RE = re.compile(r"^(BAB|Pasal|Bagian|Paragraf)\s+\S+")

# Section 1.2 — all-caps titles shorter than 60 characters (e.g. section headers).
UPPER_TITLE_MAX_LEN = 60

# Section 1.3 — paragraph-ending punctuation that triggers a flush.
END_SENTENCE_RE = re.compile(r"[;.]\s*$")

# Section 1.3 — list item start ("1." / "1)" / "a." / "a)") that forces the
# previous buffer to flush before this line starts a new one.
LIST_START_RE = re.compile(r"^(\d+[.)]|[a-z][.)])")

# Section 2 — fallback sentence boundary ('.' / ';' followed by whitespace).
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+")

ChunkType = Literal["structural", "fallback"]


@dataclass
class CleanedLine:
    """A single line after cleaning: either a structural marker or a paragraph."""

    text: str
    is_structure: bool


@dataclass
class Chunk:
    text: str
    type: ChunkType
    context: str | None
    section_id: int
    chunk_index: int = 0  # position within the section (0-based); debugging aid only


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    return bool(NOISE_LINE_RE.match(stripped) or LONE_PUNCTUATION_RE.match(stripped))


def _is_all_upper_title(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) >= UPPER_TITLE_MAX_LEN:
        return False
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    return all(c.isupper() for c in letters)


def is_structure_marker(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if STRUCTURE_MARKER_RE.match(stripped):
        return True
    return _is_all_upper_title(stripped)


def clean_text(raw_text: str) -> list[CleanedLine]:
    """Section 1: strip OCR noise, isolate structural markers, reflow paragraphs.

    Returns a list of CleanedLine entries where each entry is either a
    structural marker line (kept standalone) or a fully reflowed paragraph.
    """
    text = raw_text.lstrip("﻿")
    lines = text.splitlines()

    result: list[CleanedLine] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            paragraph = " ".join(buffer).strip()
            if paragraph:
                result.append(CleanedLine(text=paragraph, is_structure=False))
            buffer.clear()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if is_noise_line(line):
            continue
        if is_structure_marker(line):
            flush()
            result.append(CleanedLine(text=line, is_structure=True))
            continue
        if LIST_START_RE.match(line) and buffer:
            flush()
        buffer.append(line)
        if END_SENTENCE_RE.search(line):
            flush()
    flush()

    return result


def split_into_sections(cleaned_lines: list[CleanedLine]) -> list[str]:
    """Section 2, layer 1: split at structural markers (no overlap).

    Consecutive structural marker lines (e.g. "BAB II" / "MAKSUD DAN TUJUAN" /
    "Pasal 2" appearing back-to-back with no body text between them) are
    merged into the *same* section rather than each starting a new one — a
    new section only begins once a structural marker follows actual body
    content. This keeps a chapter's title stack together with its content
    instead of fragmenting into a run of one-line title-only sections.
    """
    sections: list[list[str]] = []
    current: list[str] = []
    current_has_body = False

    for cleaned in cleaned_lines:
        if cleaned.is_structure:
            if current and current_has_body:
                sections.append(current)
                current = [cleaned.text]
                current_has_body = False
            else:
                current.append(cleaned.text)
        else:
            current.append(cleaned.text)
            current_has_body = True

    if current:
        sections.append(current)

    return ["\n".join(section) for section in sections]


def _split_sentences(text: str) -> list[str]:
    return [part for part in SENTENCE_SPLIT_RE.split(text) if part]


def _pack_sentences(sentences: list[str], safe_limit: int) -> list[str]:
    """Greedily pack sentences into sub-chunks <= safe_limit chars.

    A single sentence longer than safe_limit (e.g. an unpunctuated table row
    in a Lampiran) is hard-split by character count as a last resort; this
    case isn't called out explicitly in the spec but is needed so fallback
    splitting can never produce a chunk larger than safe_limit.
    """
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > safe_limit:
            if current:
                chunks.append(current)
                current = ""
            logger.warning(
                "Fallback chunking: single sentence (%d chars) exceeds safe_limit (%d); "
                "hard-splitting by character count.",
                len(sentence),
                safe_limit,
            )
            for i in range(0, len(sentence), safe_limit):
                chunks.append(sentence[i : i + safe_limit])
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > safe_limit and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def fallback_split(
    section_text: str,
    safe_limit: int = SAFE_LIMIT_DEFAULT,
    context_chars: int = CONTEXT_TAIL_CHARS,
) -> list[tuple[str, str | None]]:
    """Section 2, layer 2: split an oversized section into (text, context) pairs.

    Only invoked when len(section_text) > safe_limit. Each sub-chunk after
    the first carries the trailing `context_chars` of the *previous*
    sub_chunk's own (untranslated) text as context.
    """
    sentences = _split_sentences(section_text)
    sub_chunks = _pack_sentences(sentences, safe_limit)

    result: list[tuple[str, str | None]] = []
    for i, sub_chunk in enumerate(sub_chunks):
        if i == 0:
            result.append((sub_chunk, None))
        else:
            context = sub_chunks[i - 1][-context_chars:]
            result.append((sub_chunk, context))
    return result


def clean_and_split_sections(raw_text: str) -> tuple[list[CleanedLine], list[str]]:
    """Sections 1-2 layer 1: clean the raw text and split it into sections."""
    cleaned_lines = clean_text(raw_text)
    sections = split_into_sections(cleaned_lines)
    return cleaned_lines, sections


def _chunks_from_sections(sections: list[str], safe_limit: int) -> list[Chunk]:
    """Section 2 layer 2: turn sections into structural/fallback chunks."""
    chunks: list[Chunk] = []
    for section_id, section_text in enumerate(sections):
        if len(section_text) <= safe_limit:
            chunks.append(
                Chunk(text=section_text, type="structural", context=None, section_id=section_id)
            )
        else:
            for chunk_index, (text, context) in enumerate(fallback_split(section_text, safe_limit)):
                chunks.append(
                    Chunk(
                        text=text,
                        type="fallback",
                        context=context,
                        section_id=section_id,
                        chunk_index=chunk_index,
                    )
                )
    return chunks


def chunk_policy_text(raw_text: str, safe_limit: int = SAFE_LIMIT_DEFAULT) -> list[Chunk]:
    """Run the full clean -> chunk pipeline (sections 1-2) and return chunk metadata."""
    _, sections = clean_and_split_sections(raw_text)
    return _chunks_from_sections(sections, safe_limit)


def chunk_policy_text_with_debug(
    raw_text: str, safe_limit: int = SAFE_LIMIT_DEFAULT
) -> tuple[list[CleanedLine], list[Chunk]]:
    """Same as chunk_policy_text, but also returns the cleaned lines (section 1
    output) so callers can persist both the clean and chunk results, e.g. for
    the --keep-chunk-result debugging option."""
    cleaned_lines, sections = clean_and_split_sections(raw_text)
    chunks = _chunks_from_sections(sections, safe_limit)
    return cleaned_lines, chunks


def combine_translations(chunks: list[Chunk], translations: list[str]) -> str:
    """Section 4: combine translated chunks back into the final document text.

    Sub-chunks belonging to the same section are concatenated directly (no
    added whitespace, since they were split mid-flow). Sections are then
    joined with a blank line, matching regulatory document spacing.
    """
    if len(chunks) != len(translations):
        raise ValueError("chunks and translations must be the same length")

    sections: dict[int, list[str]] = {}
    order: list[int] = []
    for chunk, translated in zip(chunks, translations):
        if chunk.section_id not in sections:
            sections[chunk.section_id] = []
            order.append(chunk.section_id)
        sections[chunk.section_id].append(translated)

    section_texts = ["".join(sections[section_id]) for section_id in order]
    return "\n\n".join(section_texts)


# Section 3.4 — rough sanity check: a fallback translation whose output is
# disproportionately long relative to its own target text may have echoed
# back a translation of the context. This is advisory only (log for manual
# review); the spec explicitly says not to auto-correct.
FALLBACK_LENGTH_RATIO_MAX = 1.6


def check_fallback_output(target_text: str, translated_text: str) -> bool:
    """Return True if the translated output looks suspiciously long.

    A True result means the output may contain a duplicated translation of
    the context that was supposed to be excluded; callers should log this
    for human review rather than attempt to auto-fix it.
    """
    if not target_text:
        return False
    return len(translated_text) > len(target_text) * FALLBACK_LENGTH_RATIO_MAX
