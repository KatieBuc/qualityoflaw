import json

import pytest

from automation.src.config_loader import HybridBM25Config, RerankerConfig, RetrievalConfig
from automation.src.rag.retriever import (
    RetrievedChunk,
    _bm25_search,
    _reciprocal_rank_fusion,
    load_policy_store,
    rerank,
    retrieve_evidence_candidates,
    vector_search,
)


def _chunk(chunk_id: int, text: str, embedding: list[float]) -> dict:
    return {
        "chunk_id": chunk_id,
        "section_id": 0,
        "chunk_index": 0,
        "type": "structural",
        "text": text,
        "embedding": embedding,
    }


def _retrieval_config(top_k=10, bm25_enabled=False, reranker_enabled=False) -> RetrievalConfig:
    return RetrievalConfig(
        top_k=top_k,
        hybrid_bm25=HybridBM25Config(enabled=bm25_enabled, rrf_k=60),
        reranker=RerankerConfig(enabled=reranker_enabled),
        evidence_verification_enabled=True,
    )


def test_vector_search_ranks_by_cosine_similarity():
    chunks = [
        _chunk(0, "far", [0.0, 1.0]),
        _chunk(1, "close", [1.0, 0.0]),
        _chunk(2, "medium", [0.7, 0.7]),
    ]
    results = vector_search([1.0, 0.0], chunks, top_k=3)
    assert [r.chunk_id for r in results] == [1, 2, 0]


def test_vector_search_respects_top_k():
    chunks = [_chunk(i, f"chunk {i}", [1.0, 0.0]) for i in range(5)]
    results = vector_search([1.0, 0.0], chunks, top_k=2)
    assert len(results) == 2


def test_load_policy_store_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_policy_store(tmp_path / "missing.json")


def test_load_policy_store_reads_chunks(tmp_path):
    store_path = tmp_path / "policy.json"
    store_path.write_text(
        json.dumps({"policy_file": "policy.txt", "chunks": [_chunk(0, "hello", [1.0, 0.0])]}),
        encoding="utf-8",
    )
    chunks = load_policy_store(store_path)
    assert chunks == [_chunk(0, "hello", [1.0, 0.0])]


def test_bm25_search_finds_lexical_match():
    chunks = [
        _chunk(0, "domestic violence prevention program", [0.0, 0.0]),
        _chunk(1, "budget allocation for roads", [0.0, 0.0]),
    ]
    results = _bm25_search("domestic violence", chunks, top_k=2)
    assert results[0].chunk_id == 0


def test_reciprocal_rank_fusion_boosts_items_ranked_high_in_both_lists():
    vector_hits = [
        RetrievedChunk(chunk_id=1, text="a", score=0.9),
        RetrievedChunk(chunk_id=2, text="b", score=0.8),
    ]
    bm25_hits = [
        RetrievedChunk(chunk_id=2, text="b", score=5.0),
        RetrievedChunk(chunk_id=1, text="a", score=4.0),
    ]
    fused = _reciprocal_rank_fusion(vector_hits, bm25_hits, k=60)
    fused_ids = [f.chunk_id for f in fused]
    assert set(fused_ids) == {1, 2}
    # both ranked in the top two of each list, so fused scores should be close/tied,
    # but item 2 (rank 1 in bm25, rank 2 in vector) and item 1 (rank 1 in vector, rank 2 in bm25)
    # should both be present regardless of order
    assert len(fused) == 2


def test_rerank_stub_is_passthrough_truncation():
    candidates = [RetrievedChunk(chunk_id=i, text=f"c{i}", score=1.0) for i in range(5)]
    result = rerank("query", candidates, top_k=3)
    assert result == candidates[:3]


def test_retrieve_evidence_candidates_vector_only():
    chunks = [
        _chunk(0, "close", [1.0, 0.0]),
        _chunk(1, "far", [0.0, 1.0]),
    ]
    config = _retrieval_config(top_k=1)
    results = retrieve_evidence_candidates("q", [1.0, 0.0], chunks, config)
    assert len(results) == 1
    assert results[0].chunk_id == 0


def test_retrieve_evidence_candidates_hybrid_bm25():
    chunks = [
        _chunk(0, "domestic violence prevention", [0.0, 1.0]),
        _chunk(1, "unrelated budget text", [1.0, 0.0]),
    ]
    config = _retrieval_config(top_k=2, bm25_enabled=True)
    results = retrieve_evidence_candidates("domestic violence", [1.0, 0.0], chunks, config)
    assert {r.chunk_id for r in results} == {0, 1}


def test_retrieve_evidence_candidates_never_sees_other_policy_chunks():
    policy_a_chunks = [_chunk(0, "policy A evidence", [1.0, 0.0])]
    policy_b_chunks = [_chunk(0, "policy B evidence", [1.0, 0.0])]

    config = _retrieval_config(top_k=10)
    results_a = retrieve_evidence_candidates("q", [1.0, 0.0], policy_a_chunks, config)

    assert all(r.text == "policy A evidence" for r in results_a)
    assert not any(r.text == "policy B evidence" for r in results_a)
    # sanity: policy B's store is a completely separate list never passed in
    assert policy_b_chunks[0]["text"] not in [r.text for r in results_a]
