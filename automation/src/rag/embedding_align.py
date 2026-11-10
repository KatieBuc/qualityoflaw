"""Sentence alignment with LaBSE embeddings (Vecalign or Bertalign).

Used by `evidence_align` when a chunk's translated and source sentence counts
differ. Both aligners are monotonic and may return many-to-many beads. They
were benchmarked on Indonesian gold data (`automation/benchmarks/alignment`);
Vecalign was marginally ahead and is the default.

LaBSE is loaded once per process (`_labse`) and shared by both aligners. Loading is guarded by a lock:
when evaluation threads ask for the model at the same time, one loads it and the others wait for that
instance (a bare `lru_cache` would let every thread that misses the cache load its own copy). A failed
load is remembered, so later calls fail at once instead of retrying the download in every thread.
`warm_up` loads everything before the worker threads start.

Dependencies are optional and imported lazily: `bertalign`, `vecalign` and
`sentence-transformers` (see `automation/requirements-align.txt`). LaBSE
weights are downloaded on first use.
"""

import logging
import threading

import numpy as np

#: At most this many threads encode with LaBSE at once; the others wait. Encoding on the CPU allocates a lot of
#: temporary memory per call (up to 512 tokens per sentence, many sentences per call). With dozens of evaluation
#: threads encoding together the process grew to ~43 GB of virtual memory and PyTorch crashed (c10.dll access
#: violation). Everything else in the evaluation still runs in parallel.
ENCODE_CONCURRENCY = 1
#: Threads PyTorch uses inside one encoding call.
TORCH_THREADS = 4
#: Sentences per forward pass: smaller batches mean a lower peak memory.
ENCODE_BATCH_SIZE = 16

LABSE = "sentence-transformers/LaBSE"
MAX_ALIGN = 5  # max sentences across both sides of one bead

#: ``(source indices, target indices)``; either side may be empty.
Bead = tuple[list[int], list[int]]

METHODS = ("vecalign", "bertalign")


class AlignerUnavailable(RuntimeError):
    """An optional dependency is missing or the model could not be loaded."""


def _one_line(sentences: list[str]) -> list[str]:
    return [" ".join(s.split()) or "." for s in sentences]


_LOAD_LOCK = threading.RLock()
_LOADED: dict[str, object] = {}
_FAILED: dict[str, AlignerUnavailable] = {}


def _load_once(key: str, loader):
    """Run `loader` at most once per process, however many threads ask; share its result (or its failure)."""
    if key in _LOADED:
        return _LOADED[key]
    with _LOAD_LOCK:
        if key in _LOADED:
            return _LOADED[key]
        if key in _FAILED:
            raise _FAILED[key]
        try:
            _LOADED[key] = loader()
        except AlignerUnavailable as exc:
            _FAILED[key] = exc
            raise
        except Exception as exc:  # ImportError, offline model download, ...
            _FAILED[key] = AlignerUnavailable(f"{key} unavailable: {exc}")
            raise _FAILED[key] from exc
        return _LOADED[key]


_ENCODE_SLOTS = threading.BoundedSemaphore(ENCODE_CONCURRENCY)


class _GuardedModel:
    """The shared LaBSE model with `encode` limited to `ENCODE_CONCURRENCY` threads at a time.

    Both aligners reach the model through this object (Bertalign's encoder wraps it too), so one
    guard covers every encoding call. Any other attribute is passed through to the real model.
    """

    def __init__(self, model) -> None:
        self._model = model

    def encode(self, *args, **kwargs):
        kwargs.setdefault("batch_size", ENCODE_BATCH_SIZE)
        with _ENCODE_SLOTS:
            return self._model.encode(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


def _load_labse():
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(TORCH_THREADS)
    return _GuardedModel(SentenceTransformer(LABSE))


def _labse():
    return _load_once("LaBSE", _load_labse)


def _make_bertalign_encoder():
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


def _bertalign_encoder():
    return _load_once("bertalign encoder", _make_bertalign_encoder)


def warm_up(method: str) -> None:
    """Load what `method` needs, once, before worker threads start. Raises `AlignerUnavailable` if it cannot."""
    if method not in METHODS:
        return
    _labse()
    if method == "bertalign":
        _bertalign_encoder()


def _embed_with_overlaps(lines: list[str], max_overlap: int) -> tuple[dict, np.ndarray]:
    from vecalign.dp_utils import layer, preprocess_line

    lines = [preprocess_line(line) for line in lines]
    texts: list[str] = []
    for overlap in range(1, max_overlap + 1):
        texts.extend(layer(lines, overlap))
    unique = list(dict.fromkeys(t[:10000].strip() for t in texts))
    vectors = _labse().encode(unique, normalize_embeddings=True, show_progress_bar=False)
    return {t: i for i, t in enumerate(unique)}, np.asarray(vectors, dtype=np.float32)


def quiet_vecalign_timings() -> None:
    """Vecalign logs six timing lines at INFO for every alignment; hide them (warnings and errors still show)."""
    logging.getLogger("vecalign").setLevel(logging.WARNING)


def align_vecalign(
    src: list[str], tgt: list[str], *, max_align: int = MAX_ALIGN, del_percentile_frac: float = 0.2
) -> list[Bead]:
    try:
        from vecalign.dp_utils import make_alignment_types, make_doc_embedding, vecalign
    except ImportError as exc:
        raise AlignerUnavailable(f"vecalign not installed: {exc}") from exc

    quiet_vecalign_timings()
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
