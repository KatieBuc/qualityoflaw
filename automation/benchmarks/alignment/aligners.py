"""Aligners under test. Each is a callable ``Unit -> list[bead]`` where a bead
is ``(src_indices, tgt_indices)``.

* ``index``     -- pair by position when sentence counts match, otherwise
                   return nothing. This is the shortcut step 4.6 takes first.
* ``heuristic`` -- the pipeline's current list-marker aligner
                   (`rag.sentence_align`), expressed as beads.

The embedding-based aligners (Bertalign / Vecalign with LaBSE) live in
`embedding_aligners.py` because they need optional dependencies.
"""

from collections.abc import Callable

from automation.benchmarks.alignment.data import Unit
from automation.benchmarks.alignment.metrics import Bead, normalize
from automation.src.rag.retriever import RetrievedChunk
from automation.src.rag.sentence_align import align_chunk_sentences_detailed

Aligner = Callable[[Unit], set[Bead]]


def align_by_index(unit: Unit) -> set[Bead]:
    if len(unit.src) != len(unit.tgt):
        return set()
    return normalize(([i], [i]) for i in range(len(unit.src)))


def _squash(text: str) -> str:
    return " ".join(text.split())


def align_with_heuristic(unit: Unit) -> set[Bead]:
    """Run `align_chunk_sentences_detailed` and read its output back as beads.

    That function returns, per translated sentence, the *source text* it chose
    (a sentence, a list item, or the whole chunk). The source sentences
    contained in that text become the bead's source side; translated sentences
    that resolved to the same source text share a bead.
    """
    resolved = align_chunk_sentences_detailed(
        RetrievedChunk(chunk_id=0, text=unit.tgt_text, score=0.0, source_text=unit.src_text)
    )
    if not resolved:
        return set()

    squashed_src = [_squash(s) for s in unit.src]
    by_text: dict[str, list[int]] = {}
    for tgt_index in sorted(resolved):
        text, _granularity = resolved[tgt_index]
        by_text.setdefault(_squash(text), []).append(tgt_index)

    beads = []
    for text, tgt_indices in by_text.items():
        src_indices = [i for i, s in enumerate(squashed_src) if s and s in text]
        beads.append((src_indices, tgt_indices))
    return normalize(beads)


BASELINE_ALIGNERS: dict[str, Aligner] = {
    "index": align_by_index,
    "heuristic": align_with_heuristic,
}
