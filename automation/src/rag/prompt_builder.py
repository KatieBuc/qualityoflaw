"""Build the {{CRITERIA_WITH_CANDIDATES}} block for the v2 RAG evaluation prompt."""

from automation.src.rag.retriever import RetrievedChunk


def parse_criteria_lines(criteria_file_path: str) -> list[tuple[str, str, str]]:
    """Parse `id | indicator | question` rows from a criteria file.

    Same format/parsing as `quality_eval.v1.prompts.prompt_loader.generate_judge_prompt`.
    """
    rows: list[tuple[str, str, str]] = []
    with open(criteria_file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) == 3:
                rows.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return rows


def _candidate_tag(cid: str, candidate: RetrievedChunk) -> str:
    return f"{cid}-{candidate.chunk_id}"


def format_criteria_with_candidates(
    criteria_rows: list[tuple[str, str, str]],
    candidates_by_id: dict[str, list[RetrievedChunk]],
) -> str:
    blocks: list[str] = []
    for cid, indicator, question in criteria_rows:
        blocks.append(f"- {cid} ({indicator}): {question}")
        candidates = candidates_by_id.get(cid, [])
        if not candidates:
            blocks.append("  Candidate passages: (none retrieved)")
        else:
            for candidate in candidates:
                blocks.append(f"  [{_candidate_tag(cid, candidate)}] {candidate.text}")
        blocks.append("")
    return "\n".join(blocks)


def build_candidate_lookup(candidates_by_id: dict[str, list[RetrievedChunk]]) -> dict[str, str]:
    """Build the temporary tag -> real-chunk-text lookup for one dimension call.

    The LLM is asked to cite a candidate's tag (e.g. "1.3-4") rather than
    transcribe its text; the caller resolves that citation against this table
    afterwards and copies the real text over as the final evidence, so the
    stored evidence can never diverge from what was actually retrieved.
    """
    return {
        _candidate_tag(cid, candidate): candidate.text
        for cid, candidates in candidates_by_id.items()
        for candidate in candidates
    }
