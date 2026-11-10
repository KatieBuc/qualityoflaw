"""Split a policy's full text into overlapping, sentence-aligned windows for
the sliding-window evaluation method.

Windows are built from sentences (not characters) so the same citation
scheme used by the RAG method (`{criterion_id}-{unit_id}.{sentence_idx}`,
see `rag/prompt_builder.py`) applies unchanged — a window is just a
`RetrievedChunk` whose `chunk_id` is its window index.
"""

from __future__ import annotations

from automation.src.rag.retriever import RetrievedChunk
from automation.src.rag.sentence_split import split_sentences


def split_into_windows(
    text: str, window_sentences: int, overlap_sentences: int
) -> list[RetrievedChunk]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    stride = window_sentences - overlap_sentences
    windows: list[RetrievedChunk] = []
    window_id = 0
    start = 0
    while True:
        window_sents = sentences[start : start + window_sentences]
        windows.append(
            RetrievedChunk(chunk_id=window_id, text=" ".join(window_sents), score=0.0)
        )
        window_id += 1
        if start + window_sentences >= len(sentences):
            break
        start += stride

    return windows
