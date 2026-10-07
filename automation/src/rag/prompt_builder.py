"""Build the {{CRITERIA_WITH_CANDIDATES}} block for the v3 RAG evaluation prompt.

Candidate chunks are split into sentences (see `sentence_split.py`) and each
sentence gets its own citation tag, so the LLM can cite individual supporting
sentences instead of an entire chunk — the retrieved evidence unit that ends
up in the final report is a handful of sentences, not a multi-KB passage.
"""

import logging
import re
from dataclasses import dataclass, field

from automation.src.rag.retriever import RetrievedChunk
from automation.src.rag.evidence_align import align_chunk_evidence
from automation.src.rag.sentence_split import split_sentences

logger = logging.getLogger(__name__)

#: A criterion line starts with an indicator id like `4.1`. Requiring the shape
#: keeps a rubric line that happens to contain two pipes from being read as a
#: new criterion.
INDICATOR_ID_RE = re.compile(r"^\d+(?:\.\d+)*$")


@dataclass
class Criterion:
    """One indicator: the question to answer, plus the rules for answering it.

    `rubric` holds the "Code Yes if … / Code No if …" lines that follow the
    criterion in the criteria file. From v3 onward these carry the actual
    coding guidance — the distinctions between, say, education *about violence
    against women* and education about women's rights generally — so they are
    what makes a borderline call decidable.
    """

    id: str
    indicator: str
    question: str
    rubric: list[str] = field(default_factory=list)

    @property
    def heading(self) -> str:
        return f"- {self.id} ({self.indicator}): {self.question}"


def parse_criteria_lines(criteria_file_path: str) -> list[Criterion]:
    """Parse a criteria file into criteria plus their coding rubric.

    A line of `id | indicator | question` starts a criterion; every following
    line belongs to it as rubric, until the next criterion. Earlier versions of
    this function kept only the three-field lines, which silently discarded 395
    of the 451 lines in the v3 criteria — the entire rubric never reached the
    judge. Anything that cannot be attributed is now logged rather than
    dropped in silence.
    """
    criteria: list[Criterion] = []
    orphans: list[str] = []

    with open(criteria_file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("|")
            if len(parts) == 3 and INDICATOR_ID_RE.match(parts[0].strip()):
                criteria.append(
                    Criterion(
                        id=parts[0].strip(),
                        indicator=parts[1].strip(),
                        question=parts[2].strip(),
                    )
                )
            elif criteria:
                criteria[-1].rubric.append(line)
            else:
                orphans.append(line)

    if orphans:
        logger.warning(
            "%s: %d line(s) before the first criterion were ignored (first: %r)",
            criteria_file_path,
            len(orphans),
            orphans[0][:80],
        )
    return criteria


def _sentence_tag(cid: str, candidate: RetrievedChunk, sentence_idx: int) -> str:
    return f"{cid}-{candidate.chunk_id}.{sentence_idx}"


def format_rubric_block(criterion: Criterion, indent: str = "  ") -> list[str]:
    """Render a criterion's coding rules, labelled so the judge knows what they
    are. Returns an empty list for criteria files that carry no rubric (v1/v2),
    leaving those prompts byte-identical."""
    if not criterion.rubric:
        return []
    return [f"{indent}Coding rules:"] + [f"{indent}  {line}" for line in criterion.rubric]


def format_criteria_list(criteria_rows: list[Criterion]) -> str:
    """Criteria plus rubric, without retrieved candidates.

    Used by the sliding-window method (one window covers every criterion, so
    candidates are listed once elsewhere) and by the non-RAG full-policy mode.
    """
    lines: list[str] = []
    for criterion in criteria_rows:
        lines.append(criterion.heading)
        lines.extend(format_rubric_block(criterion))
    return "\n".join(lines)


def format_criteria_with_candidates(
    criteria_rows: list[Criterion],
    candidates_by_id: dict[str, list[RetrievedChunk]],
) -> str:
    blocks: list[str] = []
    for criterion in criteria_rows:
        blocks.append(criterion.heading)
        # Rules first, then evidence: the judge should know how the call is
        # decided before reading the sentences it decides on.
        blocks.extend(format_rubric_block(criterion))
        candidates = candidates_by_id.get(criterion.id, [])
        if not candidates:
            blocks.append("  Candidate passages: (none retrieved)")
        else:
            for candidate in candidates:
                for sentence_idx, sentence in enumerate(split_sentences(candidate.text)):
                    blocks.append(
                        f"  [{_sentence_tag(criterion.id, candidate, sentence_idx)}] {sentence}"
                    )
        blocks.append("")
    return "\n".join(blocks)


def build_candidate_lookup(candidates_by_id: dict[str, list[RetrievedChunk]]) -> dict[str, str]:
    """Build the temporary tag -> real-sentence-text lookup for one dimension call.

    The LLM is asked to cite one or more sentence tags (e.g. "1.3-4.2") rather
    than transcribe their text; the caller resolves those citations against
    this table afterwards and copies the real sentence text over as the final
    evidence, so the stored evidence can never diverge from what was actually
    retrieved.
    """
    return {
        _sentence_tag(cid, candidate, sentence_idx): sentence
        for cid, candidates in candidates_by_id.items()
        for candidate in candidates
        for sentence_idx, sentence in enumerate(split_sentences(candidate.text))
    }


def build_source_text_lookup(
    candidates_by_id: dict[str, list[RetrievedChunk]], method: str = "vecalign"
) -> dict[str, str]:
    """Build tag -> original-language text, for showing alongside the
    verified translated evidence sentence a tag resolves to.

    Resolved to the matching original-language sentence when the chunk's
    translated and original text produce the same sentence count (strict
    1:1, see `align_chunk_sentences`), otherwise the whole chunk. Chunks
    without source text (translation couldn't be aligned to source sections)
    are simply absent from the lookup.
    """
    return {
        _sentence_tag(cid, candidate, flat_idx): source_text
        for cid, candidates in candidates_by_id.items()
        for candidate in candidates
        for flat_idx, (source_text, _granularity) in align_chunk_evidence(candidate, method).items()
    }


def build_source_alignment_lookup(
    candidates_by_id: dict[str, list[RetrievedChunk]], method: str = "vecalign"
) -> dict[str, str]:
    """Tag -> the granularity its `build_source_text_lookup` entry was
    resolved at (one of `sentence_align.GRANULARITY_*`), so the caller can
    tell an exact per-sentence pairing from a best-effort whole-chunk
    fallback (see evidence_check.resolve_evidence_original_alignment).
    """
    return {
        _sentence_tag(cid, candidate, flat_idx): granularity
        for cid, candidates in candidates_by_id.items()
        for candidate in candidates
        for flat_idx, (_text, granularity) in align_chunk_evidence(candidate, method).items()
    }
