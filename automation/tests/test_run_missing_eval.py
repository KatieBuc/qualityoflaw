import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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
from automation.src.run_eval import _has_eval_report, run_evaluation_step


@pytest.fixture
def pipeline_config(tmp_path):
    translate_model = ModelProfile(name="test-translate", deployment="gpt-5.2", temperature=0.2)
    eval_model = ModelProfile(name="test-eval", deployment="gpt-4o", temperature=0.1)
    diagnosis_model = ModelProfile(name="test-diagnosis", deployment="gpt-5.2", temperature=0.2)
    translation_qa_model = ModelProfile(name="test-translation-qa", deployment="gpt-5.2", temperature=0.1)
    return ResolvedPipelineConfig(
        experiment_name="test",
        translation_model=translate_model,
        translation_qa_model=translation_qa_model,
        evaluation_model=eval_model,
        discrepancy_diagnosis_model=diagnosis_model,
        translation_prompt_path=AUTOMATION_ROOT / "prompts" / "translation" / "v1" / "prompt.txt",
        translation_qa_template_path=AUTOMATION_ROOT / "prompts" / "translation_qa" / "v1" / "prompt_template.txt",
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


def test_has_eval_report_by_policy_file(data_root):
    eval_dir = data_root / "eval"
    eval_dir.mkdir()
    report = {"policy_file": "A.txt", "evaluation_results": {}}
    (eval_dir / "01012025120000-gpt-4o-A.json").write_text(json.dumps(report), encoding="utf-8")

    assert _has_eval_report(eval_dir, "A.txt") is True
    assert _has_eval_report(eval_dir, "B.txt") is False


def test_has_eval_report_by_filename_suffix(data_root):
    eval_dir = data_root / "eval"
    eval_dir.mkdir()
    (eval_dir / "01012025120000-gpt-4o-ACEH_BIREUEN.json").write_text("{}", encoding="utf-8")

    assert _has_eval_report(eval_dir, "ACEH_BIREUEN.txt") is True


@patch("automation.src.run_eval._evaluate_policy_parallel")
def test_run_evaluation_step_skips_existing_by_default(mock_eval_parallel, pipeline_config, data_root):
    run_id = "eval_missing"
    run_dir = data_root / run_id
    translation_dir = run_dir / "results" / "translation"
    evaluation_dir = run_dir / "results" / "evaluation"
    translation_dir.mkdir(parents=True)
    evaluation_dir.mkdir(parents=True)

    (translation_dir / "A.txt").write_text("policy A", encoding="utf-8")
    (translation_dir / "B.txt").write_text("policy B", encoding="utf-8")
    (evaluation_dir / "01012025120000-gpt-4o-A.json").write_text(
        json.dumps({"policy_file": "A.txt", "evaluation_results": {}}),
        encoding="utf-8",
    )

    mock_eval_parallel.return_value = (
        {
            "policy_file": "B.txt",
            "model": "gpt-4o",
            "evaluated_at": "2026-01-01T00:00:00+00:00",
            "prompt_version": "v1",
            "completed_dimensions": ["01.txt"],
            "failed_dimensions": [],
            "errors": [],
            "evaluation_results": {"1.1": {"id": "1.1", "included": "Yes"}},
        },
        {},
    )

    wrapper = MagicMock()
    wrapper.profile.deployment = "gpt-4o"
    wrapper.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    embedder = MagicMock()
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    with patch("automation.src.run_eval.finalize_and_save_report", return_value=(True, "/tmp/B.json")):
        result = run_evaluation_step(
            run_id=run_id,
            config=pipeline_config,
            wrapper=wrapper,
            embedder=embedder,
            limiter=limiter,
            small_scale=False,
        )

    assert result["counts"]["skipped"] == 1
    assert result["counts"]["total"] == 2
    assert mock_eval_parallel.call_count == 1


@patch("automation.src.run_eval._evaluate_policy_parallel")
def test_run_evaluation_step_force_reruns_existing(mock_eval_parallel, pipeline_config, data_root):
    run_id = "eval_force"
    run_dir = data_root / run_id
    translation_dir = run_dir / "results" / "translation"
    evaluation_dir = run_dir / "results" / "evaluation"
    translation_dir.mkdir(parents=True)
    evaluation_dir.mkdir(parents=True)

    (translation_dir / "A.txt").write_text("policy A", encoding="utf-8")
    (evaluation_dir / "01012025120000-gpt-4o-A.json").write_text(
        json.dumps({"policy_file": "A.txt", "evaluation_results": {}}),
        encoding="utf-8",
    )

    mock_eval_parallel.return_value = (
        {
            "policy_file": "A.txt",
            "model": "gpt-4o",
            "evaluated_at": "2026-01-01T00:00:00+00:00",
            "prompt_version": "v1",
            "completed_dimensions": ["01.txt"],
            "failed_dimensions": [],
            "errors": [],
            "evaluation_results": {"1.1": {"id": "1.1", "included": "Yes"}},
        },
        {},
    )

    wrapper = MagicMock()
    wrapper.profile.deployment = "gpt-4o"
    wrapper.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    embedder = MagicMock()
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    with patch("automation.src.run_eval.finalize_and_save_report", return_value=(True, "/tmp/A.json")):
        result = run_evaluation_step(
            run_id=run_id,
            config=pipeline_config,
            wrapper=wrapper,
            embedder=embedder,
            limiter=limiter,
            small_scale=False,
            force=True,
        )

    assert result["counts"]["skipped"] == 0
    assert result["counts"]["total"] == 1
    assert mock_eval_parallel.call_count == 1
