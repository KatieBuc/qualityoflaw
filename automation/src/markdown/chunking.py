"""Parse -> pack -> verify pipeline for the curated Markdown policy corpus.

Counterpart to `automation.src.chunking`, which cleans and chunks *raw OCR
text*. Here the input is already clean Markdown whose structure is explicit
(`# BAB`, `## Bagian`, `### Paragraf`, `#### Pasal`), so there is no cleaning
stage at all -- headings are read directly instead of being guessed at with
`STRUCTURE_MARKER_RE` and the all-caps-title heuristic. Running
`chunking.clean_text` over this corpus would in fact *destroy* the structure:
`#### Pasal 5` matches neither the marker regex nor the all-caps rule, so it
would be folded into the preceding paragraph.

Two layers, mirroring the raw-text module's shape:

1. `parse_sections` splits at headings, at the same boundary rule as
   `chunking.split_into_sections` (a heading only starts a new section once
   the current one has body content, so a title stack stays with its content).
2. `pack_sections` merges *adjacent* sections up to `target_chars`. This is
   the one deliberate difference from the raw-text path: heading sections have
   a median size of ~200 chars, so one LLM call per section means ~15,300
   calls per full corpus run. Packing to 8,000 chars brings that to ~1,100.

Anything still over `safe_limit` falls through to `chunking.fallback_split`
(sentence packing with trailing-context carryover) -- reused, not
reimplemented. On the current corpus that path is unreachable (largest single
section: 16,583 chars), exactly as it is for the raw-text path.

Pure text processing: no I/O, no LLM calls.
"""

import difflib
import logging
import re
from dataclasses import dataclass, field

from automation.src.chunking import HEADING_KEYWORD_LEVELS, ChunkType, fallback_split

logger = logging.getLogger(__name__)

# Target size for a packed translation unit. Sized from the corpus (see the
# module docstring), not from any model limit -- `safe_limit` is what guards
# the model's context.
TARGET_CHARS_DEFAULT = 8000

# ATX heading. The corpus uses no Setext headings, no front matter and no
# fenced code, so this single pattern is the whole structural grammar.
MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

# Markdown list items as `render_markdown` emits them and as the curated
# corpus contains them: "1. ", "1) ", "(1) ", "- a. ", "   - i. ".
MD_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+)?(?:\(\d+\)|\d+[.)]|[A-Za-z]+[.)])\s+\S")

# Sections are joined by a blank line when packed, matching the block
# separation Markdown already uses.
SECTION_JOIN = "\n\n"


@dataclass
class MarkdownSection:
    """One heading section: its heading line(s), its body, and its ancestry.

    `text` is the raw Markdown, heading lines included, so a section can be
    handed to the model verbatim. `heading_path` holds the *ancestor*
    headings above this section's first heading (not the heading itself) --
    a chunk that starts mid-chapter carries them into the prompt so the model
    can resolve cross-references without the chapter text being present.
    """

    text: str
    level: int
    heading: str | None
    heading_path: list[str] = field(default_factory=list)


@dataclass
class MarkdownChunk:
    """A translation unit. Field-compatible with `chunking.Chunk` (which only
    reads `.section_id`), so `chunking.combine_translations` works on these
    unchanged; `heading_path` is the one addition."""

    text: str
    type: ChunkType
    context: str | None
    section_id: int
    chunk_index: int = 0
    heading_path: list[str] = field(default_factory=list)


def heading_outline(md_text: str) -> list[tuple[int, str]]:
    """Every heading as `(level, text)`, in document order."""
    return [
        (len(match.group(1)), match.group(2).strip())
        for match in (MD_HEADING_RE.match(line) for line in md_text.splitlines())
        if match
    ]


def heading_signature(md_text: str) -> list[int]:
    """The sequence of heading levels in a document, e.g. `[1, 2, 4, 4, 1]`.

    The structural fingerprint a faithful translation must reproduce exactly:
    same headings, same order, same nesting depth. Used both to verify a
    translation (`check_structure`) and to decide whether source and
    translation sections can be paired up positionally (see `to_text`).
    """
    return [level for level, _text in heading_outline(md_text)]


def count_list_items(md_text: str) -> int:
    return sum(1 for line in md_text.splitlines() if MD_LIST_ITEM_RE.match(line))


# A numbered clause heading: the keyword plus a digit or Roman numeral, and
# nothing else. "Pasal 34" / "Article 34" and "BAB IV" / "CHAPTER IV" qualify;
# "Bagian Kesatu" / "Part One" does not (the ordinal is spelled out, so the
# token is language-specific), and neither does anything longer.
CLAUSE_TOKEN_RE = re.compile(r"^(?:\d+|[ivxlcdm]+)$", re.IGNORECASE)


def _clause_identity(heading_text: str) -> tuple[int, str] | None:
    """The numbered-clause identity of one heading, language-independently.

    `HEADING_KEYWORD_LEVELS` maps both languages onto one canonical level
    (bab/chapter -> 1, bagian/part/section -> 2, paragraf/paragraph -> 3,
    pasal/article -> 4), and a numbered token is the same on both sides, so
    `Pasal 34` and `Article 34` both yield `(4, "34")`.

    Returns None for a heading that carries no such identity -- OCR fragments
    (`1  AN`), all-caps titles, spelled-out ordinals (`Bagian Kesatu`), and
    cross-reference text the corpus wrongly promoted to a heading
    (`Pasal 28 ayat (1) huruf f, meliputi:` -- more than two words).
    """
    words = heading_text.split()
    if len(words) != 2:
        return None
    keyword_level = HEADING_KEYWORD_LEVELS.get(words[0].strip(".:;,").lower())
    token = words[1].strip(".:;,")
    if keyword_level is None or not CLAUSE_TOKEN_RE.match(token):
        return None
    return (keyword_level, token.upper())


def clause_ids(md_text: str) -> list[tuple[int, str]]:
    """The document's numbered clause headings, language-independently.

    A `(canonical_level, token)` sequence directly comparable between a source
    and its translation -- which raw heading counts are not. See
    `_clause_identity` for which headings qualify.
    """
    ids: list[tuple[int, str]] = []
    for _level, text in heading_outline(md_text):
        cid = _clause_identity(text)
        if cid is not None:
            ids.append(cid)
    return ids


def section_clause_key(section: MarkdownSection) -> tuple[int, str] | None:
    """The most specific numbered clause anywhere in `section.text`, as a
    language-independent `(canonical_level, token)` pair, or None.

    `parse_sections` keeps a run of consecutive headings (`# BAB III` /
    `#### Pasal 4`) together in one section, so its `.heading` is only the
    first, coarsest one. For pairing a translated section to its source
    counterpart the finest anchor is what matters, so this returns the
    deepest-level clause the section contains -- `Pasal 4` over `BAB III`.
    """
    best: tuple[int, str] | None = None
    for _level, text in heading_outline(section.text):
        cid = _clause_identity(text)
        if cid is not None and (best is None or cid[0] > best[0]):
            best = cid
    return best


def align_sections(
    source: list[MarkdownSection], translated: list[MarkdownSection]
) -> list[int | None]:
    """Pair each translated section to a source section by structural identity.

    Returns a list the length of `translated`; entry `j` is the index of the
    source section paired with `translated[j]`, or None when the translation
    has a section with no sound source counterpart (structural drift the
    translation introduced -- e.g. an OCR repair that split one block in two).

    Matches with `difflib.SequenceMatcher` over one key per section: its
    numbered-clause identity when it has one (`section_clause_key`, a hard
    cross-language anchor -- `Pasal 4` == `Article 4`), otherwise its heading
    nesting level. Within an `equal` run, and within an equal-length `replace`
    run (a heading relabelled or re-levelled but the same underlying clause),
    sections pair positionally; unequal-length `replace` and translation-only
    `insert` runs leave those translated sections unpaired. `autojunk=False`:
    with a few hundred sections the many identical level-only keys would
    otherwise be treated as junk and stop matching.
    """

    def key(section: MarkdownSection) -> tuple:
        cid = section_clause_key(section)
        return ("C", *cid) if cid is not None else ("L", section.level)

    source_keys = [key(s) for s in source]
    translated_keys = [key(s) for s in translated]
    pairing: list[int | None] = [None] * len(translated)

    matcher = difflib.SequenceMatcher(a=source_keys, b=translated_keys, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for offset in range(j2 - j1):
                pairing[j1 + offset] = i1 + offset
    return pairing


def _format_clause(clause: tuple[int, str]) -> str:
    names = {1: "Chapter", 2: "Part", 3: "Paragraph", 4: "Article"}
    level, token = clause
    return f"{names.get(level, f'level {level}')} {token}"


def parse_sections(md_text: str) -> list[MarkdownSection]:
    """Split Markdown into heading sections, ancestry attached.

    A heading only starts a new section once the current one has body
    content, so a run of consecutive headings (`# BAB III` / `## RUANG
    LINGKUP` / `#### Pasal 4`) stays together with the content that follows
    it rather than fragmenting into title-only sections. This is the same
    rule as `chunking.split_into_sections`, and it is what keeps the section
    count here (15,291) in line with the raw-text path's chunk count (15,313).

    Text before the first heading becomes a level-0, heading-less section.
    """
    sections: list[MarkdownSection] = []
    ancestors: dict[int, str] = {}

    current: list[str] = []
    current_level = 0
    current_heading: str | None = None
    current_path: list[str] = []
    has_body = False

    def flush() -> None:
        text = "\n".join(current).strip()
        if text:
            sections.append(
                MarkdownSection(
                    text=text,
                    level=current_level,
                    heading=current_heading,
                    heading_path=list(current_path),
                )
            )
        current.clear()

    for line in md_text.splitlines():
        match = MD_HEADING_RE.match(line)
        if not match:
            current.append(line)
            if line.strip():
                has_body = True
            continue

        level, heading = len(match.group(1)), match.group(2).strip()
        path = [ancestors[lvl] for lvl in sorted(ancestors) if lvl < level]

        if current and has_body:
            flush()
            current_level, current_heading, current_path = level, heading, path
            has_body = False
        elif not current:
            current_level, current_heading, current_path = level, heading, path

        current.append(line)
        ancestors = {lvl: text for lvl, text in ancestors.items() if lvl < level}
        ancestors[level] = heading

    flush()
    return sections


def pack_sections(
    sections: list[MarkdownSection],
    target_chars: int = TARGET_CHARS_DEFAULT,
    safe_limit: int | None = None,
) -> list[MarkdownChunk]:
    """Greedily merge adjacent sections into translation units.

    A section that is on its own larger than `safe_limit` is flushed as its
    own unit and sentence-split via `chunking.fallback_split`, each sub-chunk
    carrying the previous one's tail as reference-only context. A section
    between `target_chars` and `safe_limit` passes through whole -- splitting
    it would gain nothing and cost structural coherence.

    `section_id` numbers the packed units (not the source sections), so
    `chunking.combine_translations` reassembles them in order.
    """
    if safe_limit is None:
        safe_limit = max(target_chars, 32000)

    chunks: list[MarkdownChunk] = []
    buffer: list[str] = []
    buffer_path: list[str] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        chunks.append(
            MarkdownChunk(
                text=SECTION_JOIN.join(buffer),
                type="structural",
                context=None,
                section_id=len(chunks),
                heading_path=list(buffer_path),
            )
        )
        buffer.clear()

    for section in sections:
        if len(section.text) > safe_limit:
            flush_buffer()
            section_id = len(chunks)
            logger.warning(
                "Section %r (%d chars) exceeds safe_limit (%d); sentence-splitting.",
                section.heading or "<untitled>",
                len(section.text),
                safe_limit,
            )
            for chunk_index, (text, context) in enumerate(
                fallback_split(section.text, safe_limit)
            ):
                chunks.append(
                    MarkdownChunk(
                        text=text,
                        type="fallback",
                        context=context,
                        section_id=section_id,
                        chunk_index=chunk_index,
                        heading_path=list(section.heading_path),
                    )
                )
            continue

        packed_len = sum(len(part) for part in buffer) + len(SECTION_JOIN) * len(buffer)
        if buffer and packed_len + len(section.text) > target_chars:
            flush_buffer()
        if not buffer:
            buffer_path = list(section.heading_path)
        buffer.append(section.text)

    flush_buffer()
    return chunks


def chunk_markdown(
    md_text: str,
    target_chars: int = TARGET_CHARS_DEFAULT,
    safe_limit: int | None = None,
) -> list[MarkdownChunk]:
    """Full parse -> pack pipeline for one document."""
    return pack_sections(parse_sections(md_text), target_chars, safe_limit)


@dataclass
class StructureDiff:
    """How a translation's heading structure departs from its source's.

    Not every difference is a defect. The curated corpus carries OCR damage
    that a good translation repairs -- a cross-reference promoted to a
    heading, a fragment like `##### 1  AN`, a Pasal number split as
    `Pasal 3 1` -- and each repair changes the heading counts. What must
    never change is the set of *numbered clauses* (`lost_clauses`): losing an
    Article loses content and a retrieval boundary, and that is the only
    condition callers treat as a failure.
    """

    source_outline: list[tuple[int, str]]
    translated_outline: list[tuple[int, str]]
    source_clauses: list[tuple[int, str]]
    translated_clauses: list[tuple[int, str]]
    source_list_items: int
    translated_list_items: int

    @property
    def lost_clauses(self) -> list[str]:
        """Numbered clauses present in the source but missing from the
        translation. A real defect: content and a retrieval boundary gone."""
        remaining = list(self.translated_clauses)
        lost = []
        for clause in self.source_clauses:
            if clause in remaining:
                remaining.remove(clause)
            else:
                lost.append(_format_clause(clause))
        return lost

    @property
    def gained_clauses(self) -> list[str]:
        """Numbered clauses the translation has and the source does not.

        Usually a repair rather than an invention -- the source's own heading
        was too mangled to parse as a clause (`#### Pasal 3 1`). Worth
        reporting, never a failure: nothing is lost by it.
        """
        remaining = list(self.source_clauses)
        gained = []
        for clause in self.translated_clauses:
            if clause in remaining:
                remaining.remove(clause)
            else:
                gained.append(_format_clause(clause))
        return gained

    @property
    def source_headings(self) -> list[int]:
        return [level for level, _ in self.source_outline]

    @property
    def translated_headings(self) -> list[int]:
        return [level for level, _ in self.translated_outline]

    @property
    def headings_match(self) -> bool:
        return self.source_headings == self.translated_headings

    def level_changes(self) -> dict[int, tuple[int, int]]:
        """Per-heading-level `{level: (source_count, translated_count)}`, for
        the levels whose counts differ.

        This, rather than naming individual headings, is what distinguishes
        the findings that matter. `#### Pasal` is level 4 and `#####` is the
        all-caps-fragment level, so "level 4: 84 -> 83" means an Article was
        lost (serious) while "level 5: 3 -> 2" means the model dropped an OCR
        fragment such as `##### P  P` (benign, and arguably correct).

        Individual headings are deliberately *not* named: source and
        translation are in different languages, so the only thing to align on
        is the level sequence, and that cannot tell two same-level headings
        apart -- it would confidently name the wrong Pasal.
        """
        levels = {lvl for lvl, _ in self.source_outline} | {
            lvl for lvl, _ in self.translated_outline
        }
        changes = {}
        for level in sorted(levels):
            before = sum(1 for lvl, _ in self.source_outline if lvl == level)
            after = sum(1 for lvl, _ in self.translated_outline if lvl == level)
            if before != after:
                changes[level] = (before, after)
        return changes

    def describe(self, max_examples: int = 5) -> str:
        parts = []
        lost, gained = self.lost_clauses, self.gained_clauses
        if lost:
            shown = ", ".join(lost[:max_examples])
            more = f" (+{len(lost) - max_examples} more)" if len(lost) > max_examples else ""
            parts.append(f"LOST {len(lost)} numbered clause(s): {shown}{more}")
        if gained:
            shown = ", ".join(gained[:max_examples])
            more = f" (+{len(gained) - max_examples} more)" if len(gained) > max_examples else ""
            parts.append(f"gained {len(gained)}: {shown}{more}")
        parts.append(f"headings {len(self.source_outline)} -> {len(self.translated_outline)}")
        for level, (before, after) in self.level_changes().items():
            parts.append(f"level {level}: {before} -> {after}")
        # Context only, never a verdict -- see check_structure.
        parts.append(f"list items {self.source_list_items} -> {self.translated_list_items}")
        return "; ".join(parts)


def check_structure(source_md: str, translated_md: str) -> StructureDiff | None:
    """Return a `StructureDiff` when the translation's structure differs.

    Because both sides are Markdown, this compares real structure rather than
    the length-ratio guesswork the raw-text path falls back on
    (`chunking.check_structural_output`). Advisory, same philosophy: report,
    never auto-correct. Returns None when nothing differs.

    A returned diff is **not** by itself a defect -- read `lost_clauses` for
    that. Two kinds of difference are expected and benign, because the
    curated corpus carries OCR damage that a good translation repairs:

    - Heading counts change when the model demotes a heading the corpus
      invented (`#### Pasal 28 ayat (1) huruf f, meliputi:` is a
      cross-reference, not a clause) or drops a fragment (`##### 1  AN`), and
      they change the other way when it recovers one the corpus broke
      (`#### Pasal 3 1` -> `#### Article 31`).
    - List-item counts change because repairing markers that OCR detached
      from their content is exactly `translation_qa_md`'s job (its prompt's
      problem 3). Observed: 68 source items becoming 262 correctly marked
      ones.

    So both are reported as context, and neither is the verdict. Losing a
    numbered clause is.
    """
    diff = StructureDiff(
        source_outline=heading_outline(source_md),
        translated_outline=heading_outline(translated_md),
        source_clauses=clause_ids(source_md),
        translated_clauses=clause_ids(translated_md),
        source_list_items=count_list_items(source_md),
        translated_list_items=count_list_items(translated_md),
    )
    if diff.headings_match and not diff.lost_clauses and not diff.gained_clauses:
        return None
    return diff
