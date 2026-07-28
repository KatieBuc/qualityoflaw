"""Resolve an LLM's evidence citations against the temporary candidate lookup
built for one dimension call (see `prompt_builder.build_candidate_lookup`).

The LLM is asked to cite one or more sentence tags rather than transcribe
their text, so resolving those citations and copying the real sentence text
over is what guarantees the final evidence is never hallucinated or subtly
mistyped — it's the same strings that came straight from the retrieved
chunks.
"""


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
