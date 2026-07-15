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
from automation.src.translate import run_translation_step


@pytest.fixture
def pipeline_config(tmp_path):
    translate_model = ModelProfile(name="test-translate", deployment="gpt-5.2", temperature=0.2)
    eval_model = ModelProfile(name="test-eval", deployment="gpt-4o", temperature=0.1)
    diagnosis_model = ModelProfile(name="test-diagnosis", deployment="gpt-5.2", temperature=0.2)
    prompt_path = AUTOMATION_ROOT / "prompts" / "translation" / "v1" / "prompt.txt"
    return ResolvedPipelineConfig(
        experiment_name="test",
        translation_model=translate_model,
        evaluation_model=eval_model,
        discrepancy_diagnosis_model=diagnosis_model,
        translation_prompt_path=prompt_path,
        evaluation_criteria_dir=AUTOMATION_ROOT / "prompts" / "quality_eval" / "v1",
        evaluation_template_path=AUTOMATION_ROOT / "prompts" / "quality_eval" / "v1" / "prompt_template.txt",
        discrepancy_diagnosis_template_path=(
            AUTOMATION_ROOT / "prompts" / "discrepancy_diagnosis" / "v1" / "prompt_template.txt"
        ),
        paths=PipelinePaths(
            input_dir=tmp_path / "input",
            golden_csv=PROJECT_ROOT / "data" / "processed" / "long_policy_encoding.csv",
            index_schema=PROJECT_ROOT / "data" / "mapping" / "index_schema.yaml",
        ),
        concurrency=ConcurrencyConfig(enabled=True, max_workers=3),
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


def test_run_translation_step_parallel(pipeline_config, data_root):
    run_id = "translate_parallel"
    input_dir = pipeline_config.paths.input_dir
    input_dir.mkdir(parents=True)
    for name in ["A.txt", "B.txt", "C.txt"]:
        (input_dir / name).write_text(f"source-{name}", encoding="utf-8")

    wrapper = MagicMock()
    wrapper.complete_text.side_effect = lambda prompt: (
        f"translated-{prompt.split('source-')[1].split()[0]}",
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    result = run_translation_step(
        run_id=run_id,
        config=pipeline_config,
        wrapper=wrapper,
        limiter=limiter,
        small_scale=False,
        force=False,
    )

    assert result["counts"] == {"total": 3, "succeeded": 3, "skipped": 0, "failed": 0}
    assert result["token_usage"]["total_tokens"] == 6
    out_dir = data_root / run_id / "translation"
    assert (out_dir / "A.txt").read_text(encoding="utf-8") == "translated-A.txt"
    assert (out_dir / "B.txt").read_text(encoding="utf-8") == "translated-B.txt"
    assert (out_dir / "C.txt").read_text(encoding="utf-8") == "translated-C.txt"
    assert wrapper.complete_text.call_count == 3


def test_run_translation_step_skips_existing(pipeline_config, data_root):
    run_id = "translate_skip"
    input_dir = pipeline_config.paths.input_dir
    input_dir.mkdir(parents=True)
    (input_dir / "A.txt").write_text("source", encoding="utf-8")

    out_dir = data_root / run_id / "translation"
    out_dir.mkdir(parents=True)
    (out_dir / "A.txt").write_text("existing", encoding="utf-8")

    wrapper = MagicMock()
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    result = run_translation_step(
        run_id=run_id,
        config=pipeline_config,
        wrapper=wrapper,
        limiter=limiter,
        small_scale=False,
        force=False,
    )

    assert result["counts"]["skipped"] == 1
    assert result["counts"]["succeeded"] == 0
    wrapper.complete_text.assert_not_called()


def test_run_translation_step_force_retranslates(pipeline_config, data_root):
    run_id = "translate_force"
    input_dir = pipeline_config.paths.input_dir
    input_dir.mkdir(parents=True)
    (input_dir / "A.txt").write_text("source", encoding="utf-8")

    out_dir = data_root / run_id / "translation"
    out_dir.mkdir(parents=True)
    (out_dir / "A.txt").write_text("existing", encoding="utf-8")

    wrapper = MagicMock()
    wrapper.complete_text.return_value = (
        "new-translation",
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    result = run_translation_step(
        run_id=run_id,
        config=pipeline_config,
        wrapper=wrapper,
        limiter=limiter,
        small_scale=False,
        force=True,
    )

    assert result["counts"]["succeeded"] == 1
    assert (out_dir / "A.txt").read_text(encoding="utf-8") == "new-translation"
