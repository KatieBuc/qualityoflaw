"""Build the {{CRITERIA_WITH_CANDIDATES}} block for the v3 RAG evaluation prompt.

Candidate chunks are split into sentences (see `sentence_split.py`) and each
sentence gets its own citation tag, so the LLM can cite individual supporting
sentences instead of an entire chunk — the retrieved evidence unit that ends
up in the final report is a handful of sentences, not a multi-KB passage.
"""

from automation.src.rag.retriever import RetrievedChunk
from automation.src.rag.sentence_split import split_sentences


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


def _sentence_tag(cid: str, candidate: RetrievedChunk, sentence_idx: int) -> str:
    return f"{cid}-{candidate.chunk_id}.{sentence_idx}"


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
                for sentence_idx, sentence in enumerate(split_sentences(candidate.text)):
                    blocks.append(f"  [{_sentence_tag(cid, candidate, sentence_idx)}] {sentence}")
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
