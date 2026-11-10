import json
from unittest.mock import MagicMock

import pytest

from automation.src.concurrency import ConcurrencyLimiter
from automation.src.config_loader import (
    ChunkingConfig,
    ConcurrencyConfig,
    HybridBM25Config,
    MarkdownTranslationConfig,
    PipelinePaths,
    RerankerConfig,
    ResolvedPipelineConfig,
    RetrievalConfig,
    StorageConfig,
)
from automation.src.constants import (
    AUTOMATION_ROOT,
    CHUNKING_FALLBACK_PROMPT,
    MARKDOWN_FALLBACK_PROMPT,
    PROJECT_ROOT,
)
from automation.src.llm.model_profile import ModelProfile
from automation.src.markdown.translate import build_prompt, run_md_translation_step
from automation.src.markdown.chunking import MarkdownChunk

POLICY_MD = """# BAB I

## KETENTUAN UMUM

#### Pasal 1

Isi pasal satu selesai.

#### Pasal 2

Isi pasal dua selesai.

# BAB II

#### Pasal 3

Isi pasal tiga selesai.
"""


def make_config(tmp_path, target_chars: int, safe_limit: int = 32000):
    model = ModelProfile(name="test", deployment="gpt-5.2", temperature=0.2)
    return ResolvedPipelineConfig(
        experiment_name="test",
        translation_model=model,
        translation_qa_model=model,
        evaluation_model=model,
        discrepancy_diagnosis_model=model,
        translation_prompt_path=AUTOMATION_ROOT / "prompts" / "translation" / "v1" / "prompt.txt",
        translation_qa_template_path=(
            AUTOMATION_ROOT / "prompts" / "translation_qa" / "v1" / "prompt_template.txt"
        ),
        evaluation_criteria_dir=AUTOMATION_ROOT / "prompts" / "quality_eval" / "v1",
        evaluation_template_path=(
            AUTOMATION_ROOT / "prompts" / "quality_eval" / "v1" / "prompt_template.txt"
        ),
        discrepancy_diagnosis_template_path=(
            AUTOMATION_ROOT / "prompts" / "discrepancy_diagnosis" / "v1" / "prompt_template.txt"
        ),
        paths=PipelinePaths(
            input_dir=tmp_path / "raw",
            golden_csv=PROJECT_ROOT / "data" / "processed" / "long_policy_encoding.csv",
            index_schema=PROJECT_ROOT / "data" / "mapping" / "index_schema.yaml",
            markdown_input_dir=tmp_path / "markdown_input",
            translation_markdown_dir=tmp_path / "translation_out",
            processed_cleaned_markdown_dir=tmp_path / "markdown_input",
            processed_translation_markdown_dir=tmp_path / "translation_out",
        ),
        concurrency=ConcurrencyConfig(enabled=True, max_workers=2),
        chunking=ChunkingConfig(
            enabled=True, safe_limit=32000, fallback_prompt_path=CHUNKING_FALLBACK_PROMPT
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
        markdown=MarkdownTranslationConfig(
            prompt_path=AUTOMATION_ROOT / "prompts" / "translation" / "v3" / "prompt.txt",
            fallback_prompt_path=MARKDOWN_FALLBACK_PROMPT,
            qa_template_path=(
                AUTOMATION_ROOT / "prompts" / "translation_qa" / "v2" / "prompt_template.txt"
            ),
            target_chars=target_chars,
            safe_limit=safe_limit,
        ),
    )


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.config_loader.OUTPUT_DIR_OVERRIDE", tmp_path)
    return tmp_path


@pytest.fixture
def corpus(tmp_path):
    input_dir = tmp_path / "markdown_input"
    input_dir.mkdir()
    (input_dir / "ACEH_BIREUEN.cleaned.md").write_text(POLICY_MD, encoding="utf-8")
    return input_dir


def make_wrapper():
    wrapper = MagicMock()
    wrapper.complete_text.side_effect = lambda prompt: (
        f"T{wrapper.complete_text.call_count - 1}",
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    )
    return wrapper


def run(config, data_root, force=False):
    wrapper = make_wrapper()
    result = run_md_translation_step(
        config=config,
        wrapper=wrapper,
        limiter=ConcurrencyLimiter(max_workers=2, enabled=False),
        force=force,
    )
    return result, wrapper, data_root


def test_packing_collapses_many_sections_into_few_calls(corpus, tmp_path, data_root):
    config = make_config(tmp_path, target_chars=8000)
    result, wrapper, run_dir = run(config, data_root)

    # Three heading sections, all well under the target: one call, not three.
    assert wrapper.complete_text.call_count == 1
    assert result["counts"] == {"total": 1, "succeeded": 1, "skipped": 0, "failed": 0}
    assert result["token_usage"]["total_tokens"] == 3


def test_a_small_target_splits_into_one_call_per_section(corpus, tmp_path, data_root):
    config = make_config(tmp_path, target_chars=60)
    result, wrapper, run_dir = run(config, data_root)

    assert wrapper.complete_text.call_count == 3
    combined = (tmp_path / "translation_out" / "ACEH_BIREUEN.md").read_text(
        encoding="utf-8"
    )
    assert combined == "T0\n\nT1\n\nT2"


def test_output_is_named_without_the_cleaned_suffix(corpus, tmp_path, data_root):
    config = make_config(tmp_path, target_chars=8000)
    _, _, run_dir = run(config, data_root)

    assert (tmp_path / "translation_out" / "ACEH_BIREUEN.md").exists()
    assert not list((tmp_path / "translation_out").glob("*.cleaned.*"))


def test_nothing_but_the_translation_is_saved(corpus, tmp_path, data_root):
    config = make_config(tmp_path, target_chars=60)
    _, _, run_dir = run(config, data_root)

    # No chunk artifact, no source snapshot, nothing under mid_product/.
    assert not (run_dir / "mid_product").exists()
    assert not (run_dir / "results" / "source_markdown").exists()
    assert [p.name for p in (tmp_path / "translation_out").iterdir()] == ["ACEH_BIREUEN.md"]


def test_existing_output_is_skipped_unless_forced(corpus, tmp_path, data_root):
    config = make_config(tmp_path, target_chars=8000)
    run(config, data_root)

    second, wrapper, _ = run(config, data_root)
    assert second["counts"]["skipped"] == 1
    assert wrapper.complete_text.call_count == 0

    third, wrapper, _ = run(config, data_root, force=True)
    assert third["counts"]["succeeded"] == 1
    assert wrapper.complete_text.call_count == 1


def test_missing_input_directory_raises(tmp_path, data_root):
    config = make_config(tmp_path, target_chars=8000)
    with pytest.raises(FileNotFoundError):
        run(config, data_root)


def test_structural_prompt_carries_the_breadcrumb_and_structure_hint():
    template = "Translate:\n\n{text}"
    chunk = MarkdownChunk(
        text="#### Pasal 9\n\nIsi.",
        type="structural",
        context=None,
        section_id=0,
        heading_path=["BAB IV", "Bagian Kesatu"],
    )
    prompt = build_prompt(template, chunk)

    assert "#### Pasal 9" in prompt
    assert "BAB IV > Bagian Kesatu" in prompt
    assert "same number of leading # characters" in prompt
    assert "TARGET_START" not in prompt


def test_fallback_prompt_fills_context_and_heading_path():
    template = MARKDOWN_FALLBACK_PROMPT.read_text(encoding="utf-8")
    chunk = MarkdownChunk(
        text="Isi lanjutan.",
        type="fallback",
        context="kalimat sebelumnya",
        section_id=0,
        chunk_index=1,
        heading_path=["BAB IV"],
    )
    prompt = build_prompt(template, chunk)

    assert "[TARGET_START]" in prompt
    assert "kalimat sebelumnya" in prompt
    assert "BAB IV" in prompt
    assert "{context}" not in prompt
    assert "{heading_path}" not in prompt
