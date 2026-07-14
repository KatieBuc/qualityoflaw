"""Resolve an LLM's evidence citation against the temporary candidate lookup
built for one dimension call (see `prompt_builder.build_candidate_lookup`).

The LLM is asked to cite a candidate's tag rather than transcribe its text, so
resolving that citation and copying the real chunk text over is what
guarantees the final evidence is never hallucinated or subtly mistyped — it's
the same string object that came straight from the retrieved chunk.
"""


def resolve_evidence_citation(citation: str | None, candidate_lookup: dict[str, str]) -> str | None:
    if not citation:
        return None
    return candidate_lookup.get(citation.strip().strip("[]").strip())
