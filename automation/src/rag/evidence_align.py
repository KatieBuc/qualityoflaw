"""Step 4.6: map each translated evidence sentence to its source sentence(s).

Same contract as `sentence_align.align_chunk_sentences_detailed` (so the
report-building code is unchanged): ``{translated sentence index: (source text,
granularity)}``.

1. Equal sentence counts -> pair by index. Benchmarked on the Indonesian gold
   set: exactly right on 57 of 60 count-matched units, and 97% of real chunks
   are count-matched.
2. Otherwise -> a LaBSE aligner (`alignment.method`: ``vecalign`` or
   ``bertalign``). A bead's source side is joined into one text; a 1-1 bead is
   `GRANULARITY_SENTENCE`, a many-to-many bead `GRANULARITY_ITEM` (still local).
   A translated sentence in a bead with no source sentence is left out, i.e.
   "no original text".
3. If the aligner is unavailable or fails -> the earlier list-marker heuristic
   (`sentence_align`), so a missing optional dependency never breaks a run.
"""

import logging
from functools import lru_cache

from automation.src.rag.embedding_align import AlignerUnavailable, align_sentences
from automation.src.rag.retriever import RetrievedChunk
from automation.src.rag.sentence_align import (
    GRANULARITY_ITEM,
    GRANULARITY_SENTENCE,
    align_chunk_sentences_detailed as heuristic_align,
)
from automation.src.rag.sentence_split import split_sentences

logger = logging.getLogger(__name__)

HEURISTIC = "heuristic"


@lru_cache(maxsize=4096)
def _embedding_beads(src: tuple[str, ...], tgt: tuple[str, ...], method: str):
    return tuple(
        (tuple(s), tuple(t)) for s, t in align_sentences(list(src), list(tgt), method)
    )


def align_chunk_evidence(
    candidate: RetrievedChunk, method: str = "vecalign"
) -> dict[int, tuple[str, str]]:
    if not candidate.source_text:
        return {}

    tgt = split_sentences(candidate.text)
    if not tgt:
        return {}
    src = split_sentences(candidate.source_text, protect_abbreviations=True)

    if len(src) == len(tgt):
        return {i: (src[i], GRANULARITY_SENTENCE) for i in range(len(tgt))}

    if method == HEURISTIC or not src:
        return heuristic_align(candidate)

    try:
        beads = _embedding_beads(tuple(src), tuple(tgt), method)
    except (AlignerUnavailable, ValueError, RuntimeError) as exc:
        logger.warning("alignment method %r failed (%s); using the heuristic aligner", method, exc)
        return heuristic_align(candidate)

    resolved: dict[int, tuple[str, str]] = {}
    for src_idx, tgt_idx in beads:
        if not src_idx:
            continue
        text = " ".join(src[i] for i in src_idx)
        granularity = GRANULARITY_SENTENCE if len(src_idx) == 1 and len(tgt_idx) == 1 else GRANULARITY_ITEM
        for i in tgt_idx:
            resolved[i] = (text, granularity)
    return resolved
