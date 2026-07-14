import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from automation.src.concurrency import ConcurrencyLimiter
from automation.src.config_loader import (
    ChunkingConfig,
    ConcurrencyConfig,
    HybridBM25Config,
    PipelinePaths,
    RerankerConfig,
    ResolvedPipelineConfig,
    RetrievalConfig,
    StorageConfig,
)
from automation.src.constants import AUTOMATION_ROOT, CHUNKING_FALLBACK_PROMPT, PROJECT_ROOT
from automation.src.llm.model_profile import ModelProfile
from automation.src.rag.store import _load_translation_chunks, run_storage_step, store_path_for


@pytest.fixture
def pipeline_config():
    translate_model = ModelProfile(name="test-translate", deployment="gpt-5.2", temperature=0.2)
    eval_model = ModelProfile(name="test-eval", deployment="gpt-4o", temperature=0.1)
    return ResolvedPipelineConfig(
        experiment_name="test",
        translation_model=translate_model,
        evaluation_model=eval_model,
        translation_prompt_path=AUTOMATION_ROOT / "prompts" / "translation" / "v1" / "prompt.txt",
        evaluation_criteria_dir=AUTOMATION_ROOT / "prompts" / "quality_eval" / "v2",
        evaluation_template_path=AUTOMATION_ROOT / "prompts" / "quality_eval" / "v2" / "prompt_template.txt",
        paths=PipelinePaths(
            input_dir=PROJECT_ROOT / "data" / "raw" / "localpolicies",
            golden_csv=PROJECT_ROOT / "data" / "processed" / "long_policy_encoding.csv",
            index_schema=PROJECT_ROOT / "data" / "mapping" / "index_schema.yaml",
        ),
        concurrency=ConcurrencyConfig(enabled=True, max_workers=2),
        chunking=ChunkingConfig(
            enabled=False, safe_limit=32000, fallback_prompt_path=CHUNKING_FALLBACK_PROMPT
        ),
        storage=StorageConfig(enabled=True, batch_size=16),
        retrieval=RetrievalConfig(
            top_k=10,
            hybrid_bm25=HybridBM25Config(enabled=False, rrf_k=60),
            reranker=RerankerConfig(enabled=False),
            evidence_verification_enabled=True,
        ),
        pipeline_config_path=AUTOMATION_ROOT / "config" / "pipeline_config.yaml",
        model_config_path=AUTOMATION_ROOT / "config" / "model_config.yaml",
    )


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.metadata.DEFAULT_DATA_ROOT", tmp_path)
    monkeypatch.setattr("automation.src.config_loader.DEFAULT_DATA_ROOT", tmp_path)
    return tmp_path


def _fake_embedder():
    embedder = MagicMock()
    embedder.embed_texts.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    return embedder


def test_run_storage_step_writes_one_json_per_policy(pipeline_config, data_root):
    run_id = "storage_run"
    translation_dir = data_root / run_id / "translation"
    translation_dir.mkdir(parents=True)
    (translation_dir / "A.txt").write_text(
        "Article 1\nSome policy text about domestic violence.", encoding="utf-8"
    )
    (translation_dir / "B.txt").write_text(
        "Article 1\nSome other policy text about budgets.", encoding="utf-8"
    )

    embedder = _fake_embedder()
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    result = run_storage_step(run_id, pipeline_config, embedder, limiter=limiter)

    assert result["counts"]["succeeded"] == 2
    assert result["counts"]["failed"] == 0

    store_dir = data_root / run_id / "rag_store"
    a_store = json.loads((store_dir / "A.json").read_text(encoding="utf-8"))
    b_store = json.loads((store_dir / "B.json").read_text(encoding="utf-8"))

    assert a_store["policy_file"] == "A.txt"
    assert all("embedding" in c for c in a_store["chunks"])
    assert b_store["policy_file"] == "B.txt"
    # each policy's chunk text only ever comes from its own source file
    assert all("domestic violence" not in c["text"] for c in b_store["chunks"])
    assert all("budgets" not in c["text"] for c in a_store["chunks"])


def test_run_storage_step_falls_back_to_english_marker_chunking_without_chunks_artifact(
    pipeline_config, data_root
):
    """No data/automation/<run_id>/chunks/<stem>.chunks.json exists (e.g.
    translation.chunking was disabled), so the translated .txt is chunked
    directly using the English structural marker regex."""
    run_id = "storage_fallback"
    translation_dir = data_root / run_id / "translation"
    translation_dir.mkdir(parents=True)
    (translation_dir / "A.txt").write_text(
        "CHAPTER I\nGENERAL PROVISIONS\nArticle 1\nContent of article one.\n"
        "CHAPTER II\nPURPOSE\nArticle 2\nContent of article two.",
        encoding="utf-8",
    )

    embedder = _fake_embedder()
    limiter = ConcurrencyLimiter(max_workers=1, enabled=False)

    result = run_storage_step(run_id, pipeline_config, embedder, limiter=limiter)
    assert result["counts"]["succeeded"] == 1

    store = json.loads((data_root / run_id / "rag_store" / "A.json").read_text(encoding="utf-8"))
    assert len(store["chunks"]) == 2
    assert store["chunks"][0]["text"].startswith("CHAPTER I\nGENERAL PROVISIONS")
    assert store["chunks"][1]["text"].startswith("CHAPTER II\nPURPOSE")


def test_run_storage_step_reuses_translation_chunks_when_available(pipeline_config, data_root):
    """When the translation step already chunked the source doc and persisted
    per-chunk English translations, RAG storage reuses those exact chunk
    boundaries/text instead of re-chunking the merged translated file."""
    run_id = "storage_reuse"
    translation_dir = data_root / run_id / "translation"
    translation_dir.mkdir(parents=True)
    # Deliberately different from the chunk translations below, to prove the
    # merged file is NOT re-chunked when a valid chunks.json is present.
    (translation_dir / "A.txt").write_text("merged output text, ignored", encoding="utf-8")

    chunks_dir = data_root / run_id / "chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "A.chunks.json").write_text(
        json.dumps(
            [
                {
                    "chunk_index": 0,
                    "section_id": 0,
                    "type": "structural",
                    "context": None,
                    "text": "Pasal 1\nIsi pasal satu.",
                    "translated_text": "Article 1\nThe content of article one.",
                },
                {
                    "chunk_index": 0,
                    "section_id": 1,
                    "type": "structural",
                    "context": None,
                    "text": "Pasal 2\nIsi pasal dua.",
                    "translated_text": "Article 2\nThe content of article two.",
                },
            ]
        ),
        encoding="utf-8",
    )

    embedder = _fake_embedder()
    limiter = ConcurrencyLimiter(max_workers=1, enabled=False)

    result = run_storage_step(run_id, pipeline_config, embedder, limiter=limiter)
    assert result["counts"]["succeeded"] == 1

    store = json.loads((data_root / run_id / "rag_store" / "A.json").read_text(encoding="utf-8"))
    assert [c["text"] for c in store["chunks"]] == [
        "Article 1\nThe content of article one.",
        "Article 2\nThe content of article two.",
    ]
    assert [c["section_id"] for c in store["chunks"]] == [0, 1]


def test_run_storage_step_falls_back_when_chunks_json_invalid(pipeline_config, data_root):
    run_id = "storage_invalid_chunks"
    translation_dir = data_root / run_id / "translation"
    translation_dir.mkdir(parents=True)
    (translation_dir / "A.txt").write_text(
        "CHAPTER I\nGENERAL PROVISIONS\nArticle 1\nContent of article one.", encoding="utf-8"
    )

    chunks_dir = data_root / run_id / "chunks"
    chunks_dir.mkdir(parents=True)
    # Missing translated_text -> invalid, should trigger the fallback path.
    (chunks_dir / "A.chunks.json").write_text(
        json.dumps([{"chunk_index": 0, "section_id": 0, "type": "structural", "context": None, "text": "x"}]),
        encoding="utf-8",
    )

    embedder = _fake_embedder()
    limiter = ConcurrencyLimiter(max_workers=1, enabled=False)

    result = run_storage_step(run_id, pipeline_config, embedder, limiter=limiter)
    assert result["counts"]["succeeded"] == 1

    store = json.loads((data_root / run_id / "rag_store" / "A.json").read_text(encoding="utf-8"))
    assert store["chunks"][0]["text"].startswith("CHAPTER I\nGENERAL PROVISIONS")


def test_load_translation_chunks_rejects_empty_or_missing(tmp_path):
    assert _load_translation_chunks(tmp_path / "missing.json") is None

    empty_path = tmp_path / "empty.json"
    empty_path.write_text("[]", encoding="utf-8")
    assert _load_translation_chunks(empty_path) is None

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("not json", encoding="utf-8")
    assert _load_translation_chunks(malformed_path) is None


def test_run_storage_step_skips_existing_unless_forced(pipeline_config, data_root):
    run_id = "storage_skip"
    translation_dir = data_root / run_id / "translation"
    translation_dir.mkdir(parents=True)
    (translation_dir / "A.txt").write_text("Article 1\nSome text.", encoding="utf-8")

    embedder = _fake_embedder()
    limiter = ConcurrencyLimiter(max_workers=1, enabled=False)

    first = run_storage_step(run_id, pipeline_config, embedder, limiter=limiter)
    assert first["counts"]["succeeded"] == 1
    assert embedder.embed_texts.call_count == 1

    second = run_storage_step(run_id, pipeline_config, embedder, limiter=limiter)
    assert second["counts"]["skipped"] == 1
    assert embedder.embed_texts.call_count == 1  # not called again

    third = run_storage_step(run_id, pipeline_config, embedder, limiter=limiter, force=True)
    assert third["counts"]["succeeded"] == 1
    assert embedder.embed_texts.call_count == 2


def test_store_path_for_uses_stem():
    assert store_path_for(Path("/tmp/store"), "ACEH_BIREUEN.txt") == Path("/tmp/store/ACEH_BIREUEN.json")
