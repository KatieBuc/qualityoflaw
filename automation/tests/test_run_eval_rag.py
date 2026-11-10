import json
from pathlib import Path
from unittest.mock import MagicMock

from automation.src.config_loader import HybridBM25Config, RerankerConfig, RetrievalConfig
from automation.src.run_eval import _evaluate_dimension


def _write_store(store_path: Path, texts: list[str]) -> list[dict]:
    chunks = [
        {
            "chunk_id": i,
            "section_id": 0,
            "chunk_index": 0,
            "type": "structural",
            "text": text,
            "embedding": [1.0, 0.0],
        }
        for i, text in enumerate(texts)
    ]
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps({"policy_file": "policy.txt", "chunks": chunks}), encoding="utf-8")
    return chunks


def _embedder():
    embedder = MagicMock()
    embedder.embed_texts.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    return embedder


def _retrieval_config(verification_enabled=True) -> RetrievalConfig:
    return RetrievalConfig(
        top_k=10,
        hybrid_bm25=HybridBM25Config(enabled=False, rrf_k=60),
        reranker=RerankerConfig(enabled=False),
        evidence_verification_enabled=verification_enabled,
        # Hermetic: the heuristic aligner needs no model download.
        alignment_method="heuristic",
    )


def test_cited_candidate_gets_copied_as_final_evidence(tmp_path):
    store_chunks = [
        {
            "chunk_id": 0,
            "section_id": 0,
            "chunk_index": 0,
            "type": "structural",
            "text": "The policy explicitly covers domestic violence and sexual violence.",
            "embedding": [1.0, 0.0],
        }
    ]
    criteria_dir = tmp_path / "criteria"
    criteria_dir.mkdir()
    (criteria_dir / "01_scope.txt").write_text("1.1 | Domestic violence | does it cover DV?\n", encoding="utf-8")

    def complete_fn(prompt: str) -> dict:
        # The candidate for "1.1" is chunk_id 0, sentence 0, so its tag is "1.1-0.0".
        assert "[1.1-0.0]" in prompt
        return {
            "evaluation_results": {
                "1.1": {
                    "id": "1.1",
                    "indicator": "Domestic violence",
                    "included": "Yes",
                    "evidence": ["1.1-0.0"],
                    "rationale": "stated directly",
                }
            }
        }

    result = _evaluate_dimension(
        policy_path=Path("policy.txt"),
        criteria_file="01_scope.txt",
        criteria_folder=criteria_dir,
        template_text="{{CRITERIA_WITH_CANDIDATES}}",
        store_chunks=store_chunks,
        embedder=_embedder(),
        retrieval_config=_retrieval_config(),
        complete_fn=complete_fn,
    )

    assert result.error is None
    item = result.batch_evals["1.1"]
    assert item["evidence_verified"] is True
    # Final evidence is the real chunk text, copied verbatim — not whatever
    # string the LLM happened to produce.
    assert item["evidence"] == "The policy explicitly covers domestic violence and sexual violence."
    # The full candidate list (not just the cited one) is preserved for the
    # discrepancy_diagnosis step to persist later.
    assert list(result.candidates_by_id.keys()) == ["1.1"]
    assert result.candidates_by_id["1.1"][0].text == (
        "The policy explicitly covers domestic violence and sexual violence."
    )


def test_cited_candidate_pairs_original_language_sentence_not_whole_chunk(tmp_path):
    store_chunks = [
        {
            "chunk_id": 0,
            "section_id": 0,
            "chunk_index": 0,
            "type": "structural",
            "text": (
                "The policy explicitly covers domestic violence. "
                "The policy also covers sexual violence."
            ),
            "source_text": (
                "Kebijakan ini secara eksplisit mencakup kekerasan dalam rumah tangga. "
                "Kebijakan ini juga mencakup kekerasan seksual."
            ),
            "embedding": [1.0, 0.0],
        }
    ]
    criteria_dir = tmp_path / "criteria"
    criteria_dir.mkdir()
    (criteria_dir / "01_scope.txt").write_text("1.1 | Domestic violence | does it cover DV?\n", encoding="utf-8")

    def complete_fn(prompt: str) -> dict:
        # Cite only the second sentence.
        return {
            "evaluation_results": {
                "1.1": {
                    "id": "1.1",
                    "indicator": "Domestic violence",
                    "included": "Yes",
                    "evidence": ["1.1-0.1"],
                    "rationale": "stated directly",
                }
            }
        }

    result = _evaluate_dimension(
        policy_path=Path("policy.txt"),
        criteria_file="01_scope.txt",
        criteria_folder=criteria_dir,
        template_text="{{CRITERIA_WITH_CANDIDATES}}",
        store_chunks=store_chunks,
        embedder=_embedder(),
        retrieval_config=_retrieval_config(),
        complete_fn=complete_fn,
    )

    item = result.batch_evals["1.1"]
    assert item["evidence"] == "The policy also covers sexual violence."
    # Only the matching Indonesian sentence, not the whole chunk's source text.
    assert item["evidence_original"] == "Kebijakan ini juga mencakup kekerasan seksual."
    # Exact per-sentence pairing (both sides have 1 line, 2 sentences each).
    assert item["evidence_original_aligned"] is True


def test_chunk_level_fallback_reports_not_aligned(tmp_path):
    store_chunks = [
        {
            "chunk_id": 0,
            "section_id": 0,
            "chunk_index": 0,
            "type": "structural",
            "text": "Article 1\nThe Regent shall establish a committee. It shall report annually.",
            # The Indonesian merges two translated sentences into one, so the
            # whole-chunk sentence counts disagree (2 vs 3) and every sentence
            # falls back to the whole chunk's original text (see sentence_align.py).
            "source_text": "Pasal 1\nBupati harus membentuk komite yang melapor setiap tahun.",
            "embedding": [1.0, 0.0],
        }
    ]
    criteria_dir = tmp_path / "criteria"
    criteria_dir.mkdir()
    (criteria_dir / "01_scope.txt").write_text("1.1 | Domestic violence | does it cover DV?\n", encoding="utf-8")

    def complete_fn(prompt: str) -> dict:
        # Cite the second sentence (flat index 1 = "The Regent shall...").
        return {
            "evaluation_results": {
                "1.1": {
                    "id": "1.1",
                    "indicator": "Domestic violence",
                    "included": "Yes",
                    "evidence": ["1.1-0.1"],
                    "rationale": "stated directly",
                }
            }
        }

    result = _evaluate_dimension(
        policy_path=Path("policy.txt"),
        criteria_file="01_scope.txt",
        criteria_folder=criteria_dir,
        template_text="{{CRITERIA_WITH_CANDIDATES}}",
        store_chunks=store_chunks,
        embedder=_embedder(),
        retrieval_config=_retrieval_config(),
        complete_fn=complete_fn,
    )

    item = result.batch_evals["1.1"]
    assert item["evidence"] == "The Regent shall establish a committee."
    # Best-effort pairing still produced (the whole chunk's original text),
    # but flagged as not an exact per-sentence match.
    assert item["evidence_original"] == "Pasal 1\nBupati harus membentuk komite yang melapor setiap tahun."
    assert item["evidence_original_aligned"] is False


def test_cited_candidate_without_source_text_leaves_original_null(tmp_path):
    store_chunks = [
        {
            "chunk_id": 0,
            "section_id": 0,
            "chunk_index": 0,
            "type": "structural",
            "text": "The policy explicitly covers domestic violence.",
            # No "source_text" key — mirrors an unaligned translation, or the
            # direct-chunking fallback in rag/store.py.
            "embedding": [1.0, 0.0],
        }
    ]
    criteria_dir = tmp_path / "criteria"
    criteria_dir.mkdir()
    (criteria_dir / "01_scope.txt").write_text("1.1 | Domestic violence | does it cover DV?\n", encoding="utf-8")

    def complete_fn(prompt: str) -> dict:
        return {
            "evaluation_results": {
                "1.1": {
                    "id": "1.1",
                    "indicator": "Domestic violence",
                    "included": "Yes",
                    "evidence": ["1.1-0.0"],
                    "rationale": "stated directly",
                }
            }
        }

    result = _evaluate_dimension(
        policy_path=Path("policy.txt"),
        criteria_file="01_scope.txt",
        criteria_folder=criteria_dir,
        template_text="{{CRITERIA_WITH_CANDIDATES}}",
        store_chunks=store_chunks,
        embedder=_embedder(),
        retrieval_config=_retrieval_config(),
        complete_fn=complete_fn,
    )

    item = result.batch_evals["1.1"]
    assert item["evidence_verified"] is True
    assert item["evidence_original"] is None
    assert item["evidence_original_aligned"] is None


def test_cited_candidate_tag_may_include_brackets(tmp_path):
    store_chunks = [
        {
            "chunk_id": 0,
            "section_id": 0,
            "chunk_index": 0,
            "type": "structural",
            "text": "The policy explicitly covers domestic violence.",
            "embedding": [1.0, 0.0],
        }
    ]
    criteria_dir = tmp_path / "criteria"
    criteria_dir.mkdir()
    (criteria_dir / "01_scope.txt").write_text("1.1 | Domestic violence | does it cover DV?\n", encoding="utf-8")

    def complete_fn(prompt: str) -> dict:
        return {
            "evaluation_results": {
                "1.1": {
                    "id": "1.1",
                    "indicator": "Domestic violence",
                    "included": "Yes",
                    "evidence": ["[1.1-0.0]"],
                    "rationale": "stated directly",
                }
            }
        }

    result = _evaluate_dimension(
        policy_path=Path("policy.txt"),
        criteria_file="01_scope.txt",
        criteria_folder=criteria_dir,
        template_text="{{CRITERIA_WITH_CANDIDATES}}",
        store_chunks=store_chunks,
        embedder=_embedder(),
        retrieval_config=_retrieval_config(),
        complete_fn=complete_fn,
    )

    item = result.batch_evals["1.1"]
    assert item["evidence_verified"] is True
    assert item["evidence"] == "The policy explicitly covers domestic violence."


def test_unresolvable_citation_gets_nulled_and_flagged(tmp_path):
    store_chunks = [
        {
            "chunk_id": 0,
            "section_id": 0,
            "chunk_index": 0,
            "type": "structural",
            "text": "The policy discusses budget allocation for roads.",
            "embedding": [1.0, 0.0],
        }
    ]
    criteria_dir = tmp_path / "criteria"
    criteria_dir.mkdir()
    (criteria_dir / "01_scope.txt").write_text("1.1 | Domestic violence | does it cover DV?\n", encoding="utf-8")

    def complete_fn(prompt: str) -> dict:
        # LLM ignores instructions and transcribes text instead of citing a tag,
        # or cites a tag that doesn't exist — either way it fails to resolve.
        return {
            "evaluation_results": {
                "1.1": {
                    "id": "1.1",
                    "indicator": "Domestic violence",
                    "included": "Yes",
                    "evidence": ["This sentence about domestic violence was never actually in the document."],
                    "rationale": "stated directly",
                }
            }
        }

    result = _evaluate_dimension(
        policy_path=Path("policy.txt"),
        criteria_file="01_scope.txt",
        criteria_folder=criteria_dir,
        template_text="{{CRITERIA_WITH_CANDIDATES}}",
        store_chunks=store_chunks,
        embedder=_embedder(),
        retrieval_config=_retrieval_config(),
        complete_fn=complete_fn,
    )

    item = result.batch_evals["1.1"]
    assert item["evidence_verified"] is False
    assert item["evidence"] is None
    assert item["evidence_original"] is None
    assert item["evidence_original_aligned"] is None
    assert "unresolved evidence citation removed" in item["rationale"]


def test_no_answer_skips_resolution(tmp_path):
    store_chunks = [
        {
            "chunk_id": 0,
            "section_id": 0,
            "chunk_index": 0,
            "type": "structural",
            "text": "Unrelated text.",
            "embedding": [1.0, 0.0],
        }
    ]
    criteria_dir = tmp_path / "criteria"
    criteria_dir.mkdir()
    (criteria_dir / "01_scope.txt").write_text("1.1 | Domestic violence | does it cover DV?\n", encoding="utf-8")

    def complete_fn(prompt: str) -> dict:
        return {
            "evaluation_results": {
                "1.1": {
                    "id": "1.1",
                    "indicator": "Domestic violence",
                    "included": "No",
                    "evidence": None,
                    "rationale": "not addressed",
                }
            }
        }

    result = _evaluate_dimension(
        policy_path=Path("policy.txt"),
        criteria_file="01_scope.txt",
        criteria_folder=criteria_dir,
        template_text="{{CRITERIA_WITH_CANDIDATES}}",
        store_chunks=store_chunks,
        embedder=_embedder(),
        retrieval_config=_retrieval_config(),
        complete_fn=complete_fn,
    )

    item = result.batch_evals["1.1"]
    assert item["evidence_verified"] is None
    assert item["evidence_original"] is None
    assert item["evidence_original_aligned"] is None


def test_verification_disabled_skips_resolution_entirely(tmp_path):
    store_chunks = [
        {
            "chunk_id": 0,
            "section_id": 0,
            "chunk_index": 0,
            "type": "structural",
            "text": "The policy explicitly covers domestic violence.",
            "embedding": [1.0, 0.0],
        }
    ]
    criteria_dir = tmp_path / "criteria"
    criteria_dir.mkdir()
    (criteria_dir / "01_scope.txt").write_text("1.1 | Domestic violence | does it cover DV?\n", encoding="utf-8")

    def complete_fn(prompt: str) -> dict:
        return {
            "evaluation_results": {
                "1.1": {
                    "id": "1.1",
                    "indicator": "Domestic violence",
                    "included": "Yes",
                    "evidence": ["1.1-0.0"],
                    "rationale": "stated directly",
                }
            }
        }

    result = _evaluate_dimension(
        policy_path=Path("policy.txt"),
        criteria_file="01_scope.txt",
        criteria_folder=criteria_dir,
        template_text="{{CRITERIA_WITH_CANDIDATES}}",
        store_chunks=store_chunks,
        embedder=_embedder(),
        retrieval_config=_retrieval_config(verification_enabled=False),
        complete_fn=complete_fn,
    )

    item = result.batch_evals["1.1"]
    # Resolution is skipped entirely when disabled: the raw LLM output passes
    # through untouched (legacy trust-the-LLM behavior).
    assert item["evidence"] == ["1.1-0.0"]
    assert item["evidence_verified"] is None
    assert item["evidence_original"] is None
    assert item["evidence_original_aligned"] is None
