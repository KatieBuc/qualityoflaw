"""Bertalign and Vecalign with LaBSE embeddings, as benchmark aligners.

Thin wrappers over `automation.src.rag.embedding_align`, so the benchmark
scores exactly the code the pipeline runs. Both embed the same pre-split
sentences the other aligners see (the pipeline's `split_sentences`), so the
comparison isolates the alignment algorithm rather than the sentence splitter.

Beads are monotonic and may be many-to-many: at most `MAX_ALIGN` sentences
across both sides of a bead, for both aligners.
"""

from automation.benchmarks.alignment.data import Unit
from automation.benchmarks.alignment.metrics import Bead, normalize
from automation.src.rag.embedding_align import align_sentences


def make_aligner(method: str):
    def align(unit: Unit) -> set[Bead]:
        return normalize(align_sentences(unit.src, unit.tgt, method))

    return align


def get_aligner(name: str):
    if name in ("bertalign", "vecalign"):
        return make_aligner(name)
    raise ValueError(f"Unknown aligner '{name}'")
