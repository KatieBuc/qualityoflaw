from pathlib import Path

import pytest

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
from automation.src.translate import run_markdown_step

RAW_POLICY_TEXT = (
    "BAB I\nKETENTUAN UMUM\nPasal 1\nIsi pasal satu selesai.\n"
    "BAB II\nMAKSUD DAN TUJUAN\n"
    "Kalimat pertama yang cukup panjang untuk diuji.\n"
    "Kalimat kedua yang cukup panjang untuk diuji juga.\n"
)


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


def test_markdown_step_renders_cleaned_text_without_translation(pipeline_config, data_root):
    # cleaned_text/*.cleaned.md only depends on the raw input, not on a
    # translation having run yet.
    run_id = "markdown_cleaned_only"
    input_dir = pipeline_config.paths.input_dir
    input_dir.mkdir(parents=True)
    (input_dir / "POLICY.txt").write_text(RAW_POLICY_TEXT, encoding="utf-8")

    result = run_markdown_step(run_id=run_id, config=pipeline_config, small_scale=False, force=False)

    assert result["counts"] == {"total": 1, "succeeded": 1, "skipped": 0, "failed": 0}
    cleaned_path = data_root / run_id / "results" / "cleaned_text" / "POLICY.cleaned.md"
    assert cleaned_path.exists()
    cleaned_text = cleaned_path.read_text(encoding="utf-8")
    assert "# BAB I" in cleaned_text
    assert "#### Pasal 1" in cleaned_text
    assert "Kalimat kedua yang cukup panjang untuk diuji juga." in cleaned_text
    assert "Isi pasal satu selesai.\n\n# BAB II" in cleaned_text
    assert "\n\n\n" not in cleaned_text
    assert not (data_root / run_id / "results" / "translation_markdown").exists()


def test_markdown_step_renders_translation_markdown_when_translation_exists(pipeline_config, data_root):
    run_id = "markdown_with_translation"
    input_dir = pipeline_config.paths.input_dir
    input_dir.mkdir(parents=True)
    (input_dir / "POLICY.txt").write_text(RAW_POLICY_TEXT, encoding="utf-8")

    translation_dir = data_root / run_id / "results" / "translation"
    translation_dir.mkdir(parents=True)
    (translation_dir / "POLICY.txt").write_text(
        "CHAPTER I\nGENERAL PROVISIONS\nArticle 1\nThe content is complete.", encoding="utf-8"
    )

    result = run_markdown_step(run_id=run_id, config=pipeline_config, small_scale=False, force=False)

    assert result["counts"] == {"total": 2, "succeeded": 2, "skipped": 0, "failed": 0}
    md_text = (data_root / run_id / "results" / "translation_markdown" / "POLICY.md").read_text(
        encoding="utf-8"
    )
    assert "# CHAPTER I" in md_text
    assert "## GENERAL PROVISIONS" in md_text
    assert "#### Article 1" in md_text


def test_markdown_step_skips_existing_output_unless_forced(pipeline_config, data_root):
    run_id = "markdown_skip_existing"
    input_dir = pipeline_config.paths.input_dir
    input_dir.mkdir(parents=True)
    (input_dir / "POLICY.txt").write_text(RAW_POLICY_TEXT, encoding="utf-8")

    cleaned_dir = data_root / run_id / "results" / "cleaned_text"
    cleaned_dir.mkdir(parents=True)
    (cleaned_dir / "POLICY.cleaned.md").write_text("stale content", encoding="utf-8")

    result = run_markdown_step(run_id=run_id, config=pipeline_config, small_scale=False, force=False)
    assert result["counts"] == {"total": 1, "succeeded": 0, "skipped": 1, "failed": 0}
    assert (cleaned_dir / "POLICY.cleaned.md").read_text(encoding="utf-8") == "stale content"

    result = run_markdown_step(run_id=run_id, config=pipeline_config, small_scale=False, force=True)
    assert result["counts"] == {"total": 1, "succeeded": 1, "skipped": 0, "failed": 0}
    assert "# BAB I" in (cleaned_dir / "POLICY.cleaned.md").read_text(encoding="utf-8")


def test_markdown_step_backfills_only_missing_files(pipeline_config, data_root):
    # Simulates re-running the step on an existing run where one policy
    # already has its markdown and another doesn't yet.
    run_id = "markdown_backfill"
    input_dir = pipeline_config.paths.input_dir
    input_dir.mkdir(parents=True)
    (input_dir / "A.txt").write_text("source A", encoding="utf-8")
    (input_dir / "B.txt").write_text("source B", encoding="utf-8")

    cleaned_dir = data_root / run_id / "results" / "cleaned_text"
    cleaned_dir.mkdir(parents=True)
    (cleaned_dir / "A.cleaned.md").write_text("already there", encoding="utf-8")

    result = run_markdown_step(run_id=run_id, config=pipeline_config, small_scale=False, force=False)

    assert result["counts"] == {"total": 2, "succeeded": 1, "skipped": 1, "failed": 0}
    assert (cleaned_dir / "A.cleaned.md").read_text(encoding="utf-8") == "already there"
    assert (cleaned_dir / "B.cleaned.md").read_text(encoding="utf-8") == "source B\n"
