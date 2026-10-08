"""Sentence alignment with LaBSE embeddings (Vecalign or Bertalign).

Used by `evidence_align` when a chunk's translated and source sentence counts
differ. Both aligners are monotonic and may return many-to-many beads. They
were benchmarked on Indonesian gold data (`automation/benchmarks/alignment`);
Vecalign was marginally ahead and is the default.

LaBSE is loaded once per process (`_labse`) and shared by both aligners.

Dependencies are optional and imported lazily: `bertalign`, `vecalign` and
`sentence-transformers` (see `automation/requirements-align.txt`). LaBSE
weights are downloaded on first use.
"""

from functools import lru_cache

import numpy as np

LABSE = "sentence-transformers/LaBSE"
MAX_ALIGN = 5  # max sentences across both sides of one bead

#: ``(source indices, target indices)``; either side may be empty.
Bead = tuple[list[int], list[int]]

METHODS = ("vecalign", "bertalign")


class AlignerUnavailable(RuntimeError):
    """An optional dependency is missing or the model could not be loaded."""


def _one_line(sentences: list[str]) -> list[str]:
    return [" ".join(s.split()) or "." for s in sentences]


@lru_cache(maxsize=1)
def _labse():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(LABSE)
    except Exception as exc:  # ImportError, offline model download, ...
        raise AlignerUnavailable(f"LaBSE unavailable: {exc}") from exc


@lru_cache(maxsize=1)
def _bertalign_encoder():
    """Bertalign's `Encoder` wrapped around the one shared LaBSE model.

    `bertalign.encoder.get_encoder` would load a second copy of LaBSE; its
    `Encoder` only needs `.model` and `.model_name`, so this subclass skips the
    loading constructor and reuses `_labse()`.
    """
    from bertalign.encoder import Encoder

    class SharedEncoder(Encoder):
        def __init__(self, model) -> None:  # noqa: super().__init__ would reload the model
            self.model = model
            self.model_name = LABSE

    return SharedEncoder(_labse())


def _embed_with_overlaps(lines: list[str], max_overlap: int) -> tuple[dict, np.ndarray]:
    from vecalign.dp_utils import layer, preprocess_line

    lines = [preprocess_line(line) for line in lines]
    texts: list[str] = []
    for overlap in range(1, max_overlap + 1):
        texts.extend(layer(lines, overlap))
    unique = list(dict.fromkeys(t[:10000].strip() for t in texts))
    vectors = _labse().encode(unique, normalize_embeddings=True, show_progress_bar=False)
    return {t: i for i, t in enumerate(unique)}, np.asarray(vectors, dtype=np.float32)


def align_vecalign(
    src: list[str], tgt: list[str], *, max_align: int = MAX_ALIGN, del_percentile_frac: float = 0.2
) -> list[Bead]:
    try:
        from vecalign.dp_utils import make_alignment_types, make_doc_embedding, vecalign
    except ImportError as exc:
        raise AlignerUnavailable(f"vecalign not installed: {exc}") from exc

    src, tgt = _one_line(src), _one_line(tgt)
    n_overlap = max(1, max_align - 1)
    s2l_src, emb_src = _embed_with_overlaps(src, n_overlap)
    s2l_tgt, emb_tgt = _embed_with_overlaps(tgt, n_overlap)
    stack = vecalign(
        vecs0=make_doc_embedding(s2l_src, emb_src, src, n_overlap),
        vecs1=make_doc_embedding(s2l_tgt, emb_tgt, tgt, n_overlap),
        final_alignment_types=make_alignment_types(max_align),
        del_percentile_frac=del_percentile_frac,
        width_over2=max(3, max_align),
        max_size_full_dp=300,
        costs_sample_size=20000,
        num_samps_for_norm=100,
    )
    return [([int(i) for i in s], [int(i) for i in t]) for s, t in stack[0]["final_alignments"]]


def align_bertalign(src: list[str], tgt: list[str], *, max_align: int = MAX_ALIGN) -> list[Bead]:
    try:
        from bertalign import Bertalign
    except ImportError as exc:
        raise AlignerUnavailable(f"bertalign not installed: {exc}") from exc

    src, tgt = _one_line(src), _one_line(tgt)
    aligner = Bertalign(
        "\n".join(src),
        "\n".join(tgt),
        is_split=True,
        max_align=max_align,
        model=_bertalign_encoder(),
    )
    if (aligner.src_num, aligner.tgt_num) != (len(src), len(tgt)):
        raise ValueError("bertalign re-segmented the input")
    return [([int(i) for i in s], [int(i) for i in t]) for s, t in aligner.align_sents()]


def align_sentences(src: list[str], tgt: list[str], method: str = "vecalign") -> list[Bead]:
    if method == "vecalign":
        return align_vecalign(src, tgt)
    if method == "bertalign":
        return align_bertalign(src, tgt)
    raise ValueError(f"Unknown alignment method '{method}'; expected one of {METHODS}")
