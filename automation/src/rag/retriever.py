"""Retrieval stage: vector (+ optional BM25 hybrid) search over a single
policy's stored chunk embeddings, capped to top_k, with a pluggable
(currently stub) reranker slot.

Every function here operates on the `store_chunks` of exactly one policy's
store — callers must never merge chunks from multiple policies into a single
call, since evidence for one policy file must never be drawn from another.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from automation.src.config_loader import RetrievalConfig


@dataclass
class RetrievedChunk:
    chunk_id: int
    text: str
    score: float


def load_policy_store(store_path: Path) -> list[dict]:
    if not store_path.exists():
        raise FileNotFoundError(
            f"No RAG store found at {store_path}. Run the storage step for this policy first."
        )
    data = json.loads(store_path.read_text(encoding="utf-8"))
    return data["chunks"]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def vector_search(
    query_embedding: list[float], chunks: list[dict], top_k: int
) -> list[RetrievedChunk]:
    scored = [
        RetrievedChunk(
            chunk_id=c["chunk_id"],
            text=c["text"],
            score=_cosine_similarity(query_embedding, c["embedding"]),
        )
        for c in chunks
    ]
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]


def _bm25_search(query: str, chunks: list[dict], top_k: int) -> list[RetrievedChunk]:
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [
        RetrievedChunk(chunk_id=c["chunk_id"], text=c["text"], score=float(s))
        for c, s in ranked[:top_k]
    ]


def _reciprocal_rank_fusion(
    *ranked_lists: list[RetrievedChunk], k: int = 60
) -> list[RetrievedChunk]:
    fused_scores: dict[int, float] = {}
    chunk_by_id: dict[int, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            fused_scores[item.chunk_id] = fused_scores.get(item.chunk_id, 0.0) + 1.0 / (
                k + rank + 1
            )
            chunk_by_id[item.chunk_id] = item
    ordered_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    return [
        RetrievedChunk(chunk_id=cid, text=chunk_by_id[cid].text, score=fused_scores[cid])
        for cid in ordered_ids
    ]


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Stub: passthrough truncation to top_k.

    `retrieval.reranker.enabled` exists as a config toggle but is not wired to
    an actual scoring model yet (no cross-encoder/LLM reranker implemented) —
    enabling it today has no effect beyond this truncation.
    """
    return candidates[:top_k]


def retrieve_evidence_candidates(
    query: str,
    query_embedding: list[float],
    store_chunks: list[dict],
    config: "RetrievalConfig",
) -> list[RetrievedChunk]:
    pool_size = config.top_k if not config.reranker.enabled else config.top_k * 3
    vector_hits = vector_search(query_embedding, store_chunks, pool_size)

    if config.hybrid_bm25.enabled:
        bm25_hits = _bm25_search(query, store_chunks, pool_size)
        candidates = _reciprocal_rank_fusion(vector_hits, bm25_hits, k=config.hybrid_bm25.rrf_k)[
            :pool_size
        ]
    else:
        candidates = vector_hits

    if config.reranker.enabled:
        return rerank(query, candidates, config.top_k)
    return candidates[: config.top_k]
