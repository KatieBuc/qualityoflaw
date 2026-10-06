"""Bertalign and Vecalign with LaBSE embeddings, as benchmark aligners.

Both need optional dependencies (see ``requirements-align.txt``) and download
the LaBSE weights on first use, so this module is imported only when one of
them is requested. Both aligners embed the *same* pre-split sentences the
other aligners see (the pipeline's `split_sentences`), so the comparison
isolates the alignment algorithm rather than the sentence splitter.

Beads are monotonic and may be many-to-many: up to `max_align` sentences
across both sides of a bead for Bertalign, and `N-M with N+M <= max_align`
for Vecalign (the same limit, so the two are comparable).
"""

from functools import lru_cache

import numpy as np

from automation.benchmarks.alignment.data import Unit
from automation.benchmarks.alignment.metrics import Bead, normalize

LABSE = "sentence-transformers/LaBSE"
DEFAULT_MAX_ALIGN = 5


def _one_line(sentences: list[str]) -> list[str]:
    """Aligners read one sentence per line, so a sentence must not contain a newline."""
    return [" ".join(s.split()) or "." for s in sentences]


@lru_cache(maxsize=1)
def _labse():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(LABSE)


def make_bertalign(*, max_align: int = DEFAULT_MAX_ALIGN, **kwargs):
    def align(unit: Unit) -> set[Bead]:
        from bertalign import Bertalign
        from bertalign.encoder import get_encoder

        src, tgt = _one_line(unit.src), _one_line(unit.tgt)
        aligner = Bertalign(
            "\n".join(src),
            "\n".join(tgt),
            is_split=True,
            max_align=max_align,
            model=get_encoder("LaBSE"),
            **kwargs,
        )
        if (aligner.src_num, aligner.tgt_num) != (len(src), len(tgt)):
            raise ValueError(
                f"[{unit.id}] bertalign re-segmented the input "
                f"({aligner.src_num}/{aligner.tgt_num} vs {len(src)}/{len(tgt)})"
            )
        return normalize(aligner.align_sents())

    return align


def _embed_with_overlaps(lines: list[str], max_overlap: int, model) -> tuple[dict, np.ndarray]:
    """Embeddings for every line and every overlap (k consecutive lines joined),
    in the form Vecalign's `make_doc_embedding` looks them up."""
    from vecalign.dp_utils import layer, preprocess_line

    lines = [preprocess_line(line) for line in lines]
    texts: list[str] = []
    for overlap in range(1, max_overlap + 1):
        texts.extend(layer(lines, overlap))
    unique = list(dict.fromkeys(t[:10000].strip() for t in texts))
    vectors = model.encode(unique, normalize_embeddings=True, show_progress_bar=False)
    return {t: i for i, t in enumerate(unique)}, np.asarray(vectors, dtype=np.float32)


def make_vecalign(*, max_align: int = DEFAULT_MAX_ALIGN, del_percentile_frac: float = 0.2):
    def align(unit: Unit) -> set[Bead]:
        from vecalign.dp_utils import make_alignment_types, make_doc_embedding, vecalign

        src, tgt = _one_line(unit.src), _one_line(unit.tgt)
        model = _labse()
        # Vecalign limits N+M <= max_align, so one side overlaps at most max_align-1 lines.
        n_overlap = max(1, max_align - 1)
        s2l_src, emb_src = _embed_with_overlaps(src, n_overlap, model)
        s2l_tgt, emb_tgt = _embed_with_overlaps(tgt, n_overlap, model)
        vecs0 = make_doc_embedding(s2l_src, emb_src, src, n_overlap)
        vecs1 = make_doc_embedding(s2l_tgt, emb_tgt, tgt, n_overlap)

        stack = vecalign(
            vecs0=vecs0,
            vecs1=vecs1,
            final_alignment_types=make_alignment_types(max_align),
            del_percentile_frac=del_percentile_frac,
            width_over2=max(3, max_align),
            max_size_full_dp=300,
            costs_sample_size=20000,
            num_samps_for_norm=100,
        )
        return normalize(stack[0]["final_alignments"])

    return align


def get_aligner(name: str):
    if name == "bertalign":
        return make_bertalign()
    if name == "vecalign":
        return make_vecalign()
    raise ValueError(f"Unknown aligner '{name}'")
