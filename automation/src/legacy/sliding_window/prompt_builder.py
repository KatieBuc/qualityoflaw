"""Build the {{WINDOW_SENTENCES}} / {{CRITERIA_LIST}} blocks for the
sliding-window evaluation prompt.

Unlike the RAG method's per-indicator retrieved candidates, one window
applies to every indicator in a dimension at once — so its sentences must be
listed exactly once per LLM call, not once per criterion. Reusing
`rag/prompt_builder.py`'s `format_criteria_with_candidates` here would repeat
the same window's sentences under every criterion (a `candidates_by_id` with
the same window for every id), bloating the prompt N-fold for an
N-indicator dimension — hence this separate, window-shaped formatter.
"""

from automation.src.rag.prompt_builder import format_criteria_list
from automation.src.rag.retriever import RetrievedChunk
from automation.src.rag.sentence_split import split_sentences

# Criteria formatting is shared with the RAG method — the criteria and their
# coding rubric are identical, only the evidence layout differs — so it is
# re-exported here rather than duplicated.
__all__ = [
    "build_window_lookup",
    "format_criteria_list",
    "format_window_sentences",
    "window_sentence_tag",
]


def window_sentence_tag(window: RetrievedChunk, sentence_idx: int) -> str:
    return f"{window.chunk_id}.{sentence_idx}"


def format_window_sentences(window: RetrievedChunk) -> str:
    sentences = split_sentences(window.text)
    if not sentences:
        return "(this window contains no sentences)"
    return "\n".join(
        f"[{window_sentence_tag(window, idx)}] {sentence}" for idx, sentence in enumerate(sentences)
    )


def build_window_lookup(window: RetrievedChunk) -> dict[str, str]:
    """Tag -> real-sentence-text lookup for one window, used to resolve the
    LLM's cited tags back to verbatim text (see `evidence_check.py`)."""
    return {
        window_sentence_tag(window, idx): sentence
        for idx, sentence in enumerate(split_sentences(window.text))
    }
