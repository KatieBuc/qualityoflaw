import json
import logging
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
from automation.src.translate import run_translation_step

RAW_POLICY_TEXT = (
    "BAB I\nKETENTUAN UMUM\nPasal 1\nIsi pasal satu selesai.\n"
    "BAB II\nMAKSUD DAN TUJUAN\n"
    "Kalimat pertama yang cukup panjang untuk diuji.\n"
    "Kalimat kedua yang cukup panjang untuk diuji juga.\n"
)


@pytest.fixture
def chunked_pipeline_config(tmp_path):
    translate_model = ModelProfile(name="test-translate", deployment="gpt-5.2", temperature=0.2)
    eval_model = ModelProfile(name="test-eval", deployment="gpt-4o", temperature=0.1)
    return ResolvedPipelineConfig(
        experiment_name="test",
        translation_model=translate_model,
        evaluation_model=eval_model,
        translation_prompt_path=AUTOMATION_ROOT / "prompts" / "translation" / "v1" / "prompt.txt",
        evaluation_criteria_dir=AUTOMATION_ROOT / "prompts" / "quality_eval" / "v1",
        evaluation_template_path=AUTOMATION_ROOT / "prompts" / "quality_eval" / "v1" / "prompt_template.txt",
        paths=PipelinePaths(
            input_dir=tmp_path / "input",
            golden_csv=PROJECT_ROOT / "data" / "processed" / "long_policy_encoding.csv",
            index_schema=PROJECT_ROOT / "data" / "mapping" / "index_schema.yaml",
        ),
        concurrency=ConcurrencyConfig(enabled=True, max_workers=2),
        chunking=ChunkingConfig(
            enabled=True, safe_limit=100, fallback_prompt_path=CHUNKING_FALLBACK_PROMPT
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


def test_run_translation_step_chunked_splits_translates_and_combines(
    chunked_pipeline_config, data_root
):
    run_id = "chunked_run"
    input_dir = chunked_pipeline_config.paths.input_dir
    input_dir.mkdir(parents=True)
    (input_dir / "POLICY.txt").write_text(RAW_POLICY_TEXT, encoding="utf-8")

    usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    translated_outputs = ["T0", "T1A", "T1B"]
    wrapper = MagicMock()
    wrapper.complete_text.side_effect = [(t, usage) for t in translated_outputs]

    limiter = ConcurrencyLimiter(max_workers=1, enabled=True)

    result = run_translation_step(
        run_id=run_id,
        config=chunked_pipeline_config,
        wrapper=wrapper,
        limiter=limiter,
        small_scale=False,
        force=False,
    )

    assert result["counts"] == {"total": 1, "succeeded": 1, "skipped": 0, "failed": 0}
    # BAB I/KETENTUAN UMUM/Pasal 1 merge into one structural chunk (section 0);
    # BAB II/MAKSUD DAN TUJUAN merge with the oversized body into section 1,
    # which fallback-splits into 2 sub-chunks. 3 calls total, not 6.
    assert wrapper.complete_text.call_count == 3
    assert result["token_usage"]["total_tokens"] == 6

    out_path = data_root / run_id / "translation" / "POLICY.txt"
    assert out_path.read_text(encoding="utf-8") == "T0\n\nT1AT1B"

    prompts = [call.args[0] for call in wrapper.complete_text.call_args_list]
    # First call is the structural chunk (section 0): no TARGET markers.
    assert "TARGET_START" not in prompts[0]
    # Remaining calls are the fallback sub-chunks for the oversized section.
    for prompt in prompts[1:]:
        assert "TARGET_START" in prompt
    # The second fallback call carries the first fallback sub-chunk's text as context.
    assert "Kalimat pertama" in prompts[2]
    assert "Kalimat kedua" in prompts[2]


def test_run_translation_step_chunked_logs_suspicious_fallback_output(
    chunked_pipeline_config, data_root, caplog
):
    run_id = "chunked_suspicious"
    input_dir = chunked_pipeline_config.paths.input_dir
    input_dir.mkdir(parents=True)
    (input_dir / "POLICY.txt").write_text(RAW_POLICY_TEXT, encoding="utf-8")

    usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    # Make the first fallback chunk's output implausibly long (as if context leaked in).
    translated_outputs = ["T0", "X" * 500, "T1B"]
    wrapper = MagicMock()
    wrapper.complete_text.side_effect = [(t, usage) for t in translated_outputs]

    limiter = ConcurrencyLimiter(max_workers=1, enabled=True)

    with caplog.at_level(logging.WARNING):
        run_translation_step(
            run_id=run_id,
            config=chunked_pipeline_config,
            wrapper=wrapper,
            limiter=limiter,
            small_scale=False,
            force=False,
        )

    assert any("suspiciously long" in record.message for record in caplog.records)


def test_run_translation_step_chunked_keep_chunk_result_writes_artifacts(
    chunked_pipeline_config, data_root
):
    run_id = "chunked_keep_result"
    input_dir = chunked_pipeline_config.paths.input_dir
    input_dir.mkdir(parents=True)
    (input_dir / "POLICY.txt").write_text(RAW_POLICY_TEXT, encoding="utf-8")

    usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    translated_outputs = ["T0", "T1A", "T1B"]
    wrapper = MagicMock()
    wrapper.complete_text.side_effect = [(t, usage) for t in translated_outputs]

    limiter = ConcurrencyLimiter(max_workers=1, enabled=True)

    run_translation_step(
        run_id=run_id,
        config=chunked_pipeline_config,
        wrapper=wrapper,
        limiter=limiter,
        small_scale=False,
        force=False,
        keep_chunk_result=True,
    )

    chunks_dir = data_root / run_id / "chunks"
    cleaned_path = chunks_dir / "POLICY.cleaned.txt"
    chunks_path = chunks_dir / "POLICY.chunks.json"
    assert cleaned_path.exists()
    assert chunks_path.exists()

    cleaned_text = cleaned_path.read_text(encoding="utf-8")
    assert "BAB I" in cleaned_text
    assert "Kalimat kedua yang cukup panjang untuk diuji juga." in cleaned_text

    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
    assert len(chunks_data) == 3
    assert [c["type"] for c in chunks_data] == ["structural", "fallback", "fallback"]
    assert chunks_data[1]["context"] is None
    assert chunks_data[2]["context"] is not None
    assert all(
        "text" in c and "translated_text" in c and "section_id" in c and "chunk_index" in c
        for c in chunks_data
    )
    assert [c["translated_text"] for c in chunks_data] == translated_outputs


def test_run_translation_step_chunked_without_keep_chunk_result_skips_cleaned_debug_only(
    chunked_pipeline_config, data_root
):
    """chunks.json (needed by the RAG storage step) is always written when
    chunking is enabled; only the .cleaned.txt debug artifact is gated behind
    --keep-chunk-result."""
    run_id = "chunked_no_keep_result"
    input_dir = chunked_pipeline_config.paths.input_dir
    input_dir.mkdir(parents=True)
    (input_dir / "POLICY.txt").write_text(RAW_POLICY_TEXT, encoding="utf-8")

    usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    translated_outputs = ["T0", "T1A", "T1B"]
    wrapper = MagicMock()
    wrapper.complete_text.side_effect = [(t, usage) for t in translated_outputs]

    limiter = ConcurrencyLimiter(max_workers=1, enabled=True)

    run_translation_step(
        run_id=run_id,
        config=chunked_pipeline_config,
        wrapper=wrapper,
        limiter=limiter,
        small_scale=False,
        force=False,
        keep_chunk_result=False,
    )

    chunks_dir = data_root / run_id / "chunks"
    assert (chunks_dir / "POLICY.chunks.json").exists()
    assert not (chunks_dir / "POLICY.cleaned.txt").exists()
