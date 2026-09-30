"""Resolve an LLM's evidence citations against the temporary candidate lookup
built for one dimension call (see `prompt_builder.build_candidate_lookup`).

The LLM is asked to cite one or more sentence tags rather than transcribe
their text, so resolving those citations and copying the real sentence text
over is what guarantees the final evidence is never hallucinated or subtly
mistyped — it's the same strings that came straight from the retrieved
chunks.
"""

from automation.src.rag.sentence_align import GRANULARITY_ITEM, GRANULARITY_SENTENCE

#: Granularities tight enough to show as a genuine side-by-side pairing: an
#: exact sentence, or the one list item the cited sentence sits in. A
#: whole-chunk fallback is not one of these.
_LOCALLY_ALIGNED = frozenset({GRANULARITY_SENTENCE, GRANULARITY_ITEM})


def resolve_evidence_citation(
    citations: list[str] | None, candidate_lookup: dict[str, str]
) -> list[str] | None:
    """Resolve every cited tag to its real sentence text, in citation order.

    If any tag fails to resolve (unknown, or the LLM transcribed text instead
    of a tag), the whole citation is rejected rather than partially accepted,
    so evidence can never mix a verified sentence with a hallucinated one.
    """
    if not citations:
        return None

    resolved: list[str] = []
    for citation in citations:
        text = candidate_lookup.get((citation or "").strip().strip("[]").strip())
        if text is None:
            return None
        resolved.append(text)
    return resolved or None


def resolve_evidence_source_text(
    citations: list[str] | None, source_lookup: dict[str, str]
) -> str | None:
    """Resolve cited tags to their original-language chunk text, deduplicated
    and in citation order, for side-by-side display next to the verified
    translated evidence.

    Unlike `resolve_evidence_citation`, a citation whose chunk has no
    original-language text (source/translation alignment couldn't be
    verified for that section) is skipped rather than failing the whole
    citation — this is a supplementary comparison aid, not a correctness
    guarantee the way verified translated evidence is.
    """
    if not citations:
        return None

    seen: list[str] = []
    for citation in citations:
        text = source_lookup.get((citation or "").strip().strip("[]").strip())
        if text and text not in seen:
            seen.append(text)
    return "\n\n".join(seen) if seen else None


def resolve_evidence_original_alignment(
    citations: list[str] | None, alignment_lookup: dict[str, str]
) -> bool | None:
    """Whether every citation's `evidence_original` text came from a tight local
    pairing -- an exact sentence or its single list item
    (`sentence_align._LOCALLY_ALIGNED`) -- rather than a whole-chunk fallback.

    None when no citation resolved to any granularity at all, mirroring
    `resolve_evidence_source_text` returning None in that same case -- there
    is nothing to report confidence about.
    """
    if not citations:
        return None

    granularities = [
        alignment_lookup.get((citation or "").strip().strip("[]").strip()) for citation in citations
    ]
    resolved = [g for g in granularities if g is not None]
    if not resolved:
        return None
    return all(g in _LOCALLY_ALIGNED for g in resolved)
