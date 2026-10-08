import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from automation.src.config_loader import load_model_profiles, load_pipeline_config
from automation.src.constants import AUTOMATION_ROOT, SMALL_SCALE_FILES
from automation.src.llm.model_profile import ModelProfile
from automation.src.llm.wrapper import AzureLLMWrapper, LLMCallError, clean_translation_response, format_api_error
from automation.src.config_loader import clean_output_dir
from automation.src.metadata import (
    init_metadata,
    load_metadata,
    metadata_exists,
    update_metadata,
    validate_for_steps,
)
from automation.src.run_pipeline import parse_steps
from automation.src.legacy.translate import resolve_input_files


@pytest.fixture
def model_config_path(tmp_path):
    data = {
        "models": {
            "test-translate": {
                "deployment": "gpt-5.2",
                "temperature": 0.2,
                "max_tokens": None,
                "max_retries": 2,
            },
            "test-eval": {
                "deployment": "gpt-4o",
                "temperature": 0.1,
                "max_tokens": 128,
                "max_retries": 2,
            },
            "test-diagnosis": {
                "deployment": "gpt-5.2",
                "temperature": 0.2,
                "max_tokens": None,
                "max_retries": 2,
            },
            "test-translation-qa": {
                "deployment": "gpt-5.2",
                "temperature": 0.1,
                "max_tokens": None,
                "max_retries": 2,
            },
        }
    }
    path = tmp_path / "model_config.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


@pytest.fixture
def pipeline_config_path(tmp_path, model_config_path):
    data = {
        "experiment_name": "test_experiment",
        "translation": {"model": "test-translate", "prompt_version": "v1"},
        "translation_qa": {"model": "test-translation-qa", "prompt_version": "v1"},
        "evaluation": {"model": "test-eval", "prompt_version": "v1"},
        "discrepancy_diagnosis": {"model": "test-diagnosis", "prompt_version": "v1"},
        "paths": {
            "input_dir": "data/raw/localpolicies",
            "golden_csv": "data/processed/long_policy_encoding.csv",
            "index_schema": "data/mapping/index_schema.yaml",
        },
    }
    path = tmp_path / "pipeline_config.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


def test_load_model_profiles(model_config_path):
    profiles = load_model_profiles(model_config_path)
    assert "test-translate" in profiles
    assert profiles["test-translate"].deployment == "gpt-5.2"
    assert profiles["test-translate"].temperature == 0.2
    assert profiles["test-eval"].max_tokens == 128


def test_load_pipeline_config(pipeline_config_path, model_config_path):
    config = load_pipeline_config(pipeline_config_path, model_config_path)
    assert config.experiment_name == "test_experiment"
    assert config.translation_model.name == "test-translate"
    assert config.translation_qa_model.name == "test-translation-qa"
    assert config.evaluation_model.name == "test-eval"
    assert config.discrepancy_diagnosis_model.name == "test-diagnosis"
    assert config.concurrency.enabled is True
    assert config.concurrency.max_workers == 5
    assert config.translation_prompt_path.exists()
    assert config.translation_qa_template_path.exists()
    assert config.evaluation_template_path.exists()
    assert config.discrepancy_diagnosis_template_path.exists()
    assert config.chunking.enabled is False
    assert config.chunking.safe_limit == 32000


def test_load_pipeline_config_unknown_model(pipeline_config_path, model_config_path):
    data = yaml.safe_load(pipeline_config_path.read_text(encoding="utf-8"))
    data["translation"]["model"] = "missing-model"
    pipeline_config_path.write_text(yaml.dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown model key"):
        load_pipeline_config(pipeline_config_path, model_config_path)


def test_load_pipeline_config_chunking_enabled(pipeline_config_path, model_config_path):
    data = yaml.safe_load(pipeline_config_path.read_text(encoding="utf-8"))
    data["translation"]["chunking"] = {"enabled": True, "safe_limit": 1000}
    pipeline_config_path.write_text(yaml.dump(data), encoding="utf-8")

    config = load_pipeline_config(pipeline_config_path, model_config_path)

    assert config.chunking.enabled is True
    assert config.chunking.safe_limit == 1000
    assert config.chunking.fallback_prompt_path.exists()


def test_parse_chunking_config_invalid_safe_limit():
    from automation.src.config_loader import parse_chunking_config

    with pytest.raises(ValueError, match="safe_limit"):
        parse_chunking_config({"enabled": True, "safe_limit": 0})


def test_resolve_input_files_small_scale(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in SMALL_SCALE_FILES:
        (input_dir / name).write_text("x", encoding="utf-8")
    (input_dir / "OTHER.txt").write_text("x", encoding="utf-8")

    files = resolve_input_files(input_dir, small_scale=True)
    assert [f.name for f in files] == SMALL_SCALE_FILES


def test_resolve_input_files_full_corpus(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in SMALL_SCALE_FILES:
        (input_dir / name).write_text("x", encoding="utf-8")
    (input_dir / "OTHER.txt").write_text("x", encoding="utf-8")

    files = resolve_input_files(input_dir, small_scale=False)
    assert [f.name for f in files] == sorted([*SMALL_SCALE_FILES, "OTHER.txt"])


def test_resolve_input_files_full_corpus_skips_non_txt_files(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "POLICY.txt").write_text("x", encoding="utf-8")
    (input_dir / "SCAN.pdf").write_text("not a real pdf, just bytes", encoding="utf-8")
    (input_dir / "notes.docx").write_text("x", encoding="utf-8")
    (input_dir / "subdir").mkdir()

    files = resolve_input_files(input_dir, small_scale=False)

    assert [f.name for f in files] == ["POLICY.txt"]


def test_clean_translation_response():
    assert clean_translation_response("```\nHello\n```") == "Hello"
    assert clean_translation_response("Here's the translation:\nHi") == "Hi"


def test_azure_llm_wrapper_build_request_kwargs():
    profile = ModelProfile(name="t", deployment="m", temperature=0.3, max_tokens=100)
    wrapper = AzureLLMWrapper(profile=profile, client=MagicMock(), api_style="chat")
    kwargs = wrapper._build_request_kwargs()
    assert kwargs == {"temperature": 0.3, "max_tokens": 100}


def test_azure_llm_wrapper_complete_text_retries():
    profile = ModelProfile(name="t", deployment="m", temperature=0.2, max_retries=2)
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=" translated "))]
    response.usage = MagicMock(prompt_tokens=1, completion_tokens=2, total_tokens=3)

    client.chat.completions.create.side_effect = [RuntimeError("fail"), response]
    wrapper = AzureLLMWrapper(profile=profile, client=client, api_style="chat")

    with patch("automation.src.llm.wrapper.time.sleep"):
        text, usage = wrapper.complete_text("prompt")

    assert text == "translated"
    assert usage["total_tokens"] == 3
    assert client.chat.completions.create.call_count == 2


def test_format_api_error_generic():
    error_type, details = format_api_error(RuntimeError("connection reset"))
    assert error_type == "RuntimeError"
    assert details["message"] == "connection reset"


def test_format_api_error_api_status():
    exc = MagicMock()
    exc.status_code = 429
    exc.message = "Rate limit exceeded"
    exc.response = MagicMock(headers={"x-request-id": "req-123"})
    exc.body = {"error": {"code": "429"}}

    error_type, details = format_api_error(exc)
    assert error_type == "MagicMock"
    assert details["status_code"] == 429
    assert details["request_id"] == "req-123"


def test_azure_llm_wrapper_raises_llm_call_error_after_retries():
    profile = ModelProfile(name="t", deployment="m", temperature=0.2, max_retries=2)
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("fail")
    wrapper = AzureLLMWrapper(profile=profile, client=client, api_style="chat")

    with patch("automation.src.llm.wrapper.time.sleep"):
        with pytest.raises(LLMCallError) as exc_info:
            wrapper.complete_text("prompt")

    assert exc_info.value.error_type == "RuntimeError"
    assert exc_info.value.attempts == 2


def test_metadata_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.config_loader.OUTPUT_DIR_OVERRIDE", tmp_path)

    assert not metadata_exists()
    init_metadata(
        experiment_name="exp",
        small_scale=True,
        config_summary={"translation_model": "test-translate"},
        pipeline_config_path=AUTOMATION_ROOT / "config" / "pipeline_config.yaml",
        model_config_path=AUTOMATION_ROOT / "config" / "model_config.yaml",
    )

    assert metadata_exists()
    meta = load_metadata()
    assert "run_id" not in meta
    assert meta["execution_scope"]["small_scale"] is True
    assert (tmp_path / "config" / "pipeline_config.yaml").exists()

    update_metadata(
        execution_scope={"steps_executed": ["translation"]},
        timing_seconds={"translation": 1.5},
    )
    updated = load_metadata()
    assert updated["execution_scope"]["steps_executed"] == ["translation"]
    assert updated["timing_seconds"]["translation"] == 1.5


def test_clean_output_dir_empties_the_folder_but_keeps_it(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.config_loader.OUTPUT_DIR_OVERRIDE", tmp_path)
    (tmp_path / "results" / "evaluation").mkdir(parents=True)
    (tmp_path / "results" / "evaluation" / "A.json").write_text("{}", encoding="utf-8")
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")

    assert clean_output_dir() == 2
    assert tmp_path.is_dir() and list(tmp_path.iterdir()) == []
    assert clean_output_dir() == 0


def test_clean_output_dir_refuses_a_path_outside_the_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.config_loader.OUTPUT_DIR_OVERRIDE", None)
    monkeypatch.setattr("automation.src.config_loader._active_project", "../../..")
    with pytest.raises(ValueError, match="Refusing to clean"):
        clean_output_dir()


def test_validate_for_steps_missing_translation(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.config_loader.OUTPUT_DIR_OVERRIDE", tmp_path)

    with pytest.raises(FileNotFoundError, match="Translation output required"):
        validate_for_steps(["evaluation"])


def test_validate_for_steps_accepts_pre_refactor_flat_layout(tmp_path, monkeypatch):
    """Output created before the results/mid_product split (flat <name> layout)
    must still pass validation without being physically migrated."""
    monkeypatch.setattr("automation.src.config_loader.OUTPUT_DIR_OVERRIDE", tmp_path)

    translation_dir = tmp_path / "translation"  # flat, pre-refactor layout
    translation_dir.mkdir(parents=True)
    (translation_dir / "A.txt").write_text("text", encoding="utf-8")
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "report.json").write_text("{}", encoding="utf-8")

    # Should not raise, even though there's no results/ or mid_product/ subfolder.
    validate_for_steps(["comparison"], retrieval_enabled=False)


def test_validate_for_steps_requires_rag_store_when_retrieval_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.config_loader.OUTPUT_DIR_OVERRIDE", tmp_path)

    translation_dir = tmp_path / "results" / "translation"
    translation_dir.mkdir(parents=True)
    (translation_dir / "A.txt").write_text("text", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="RAG store required"):
        validate_for_steps(["evaluation"], retrieval_enabled=True)


def test_validate_for_steps_allows_comparison_when_evaluation_also_requested(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.config_loader.OUTPUT_DIR_OVERRIDE", tmp_path)

    translation_dir = tmp_path / "results" / "translation"
    translation_dir.mkdir(parents=True)
    (translation_dir / "A.txt").write_text("text", encoding="utf-8")
    rag_store_dir = tmp_path / "mid_product" / "rag_store"
    rag_store_dir.mkdir(parents=True)
    (rag_store_dir / "A.json").write_text("{}", encoding="utf-8")

    # evaluation/ has no reports yet, but "evaluation" is also requested in this
    # invocation and will produce them before "comparison" runs — must not raise.
    validate_for_steps(["evaluation", "comparison"], retrieval_enabled=True)


def test_validate_for_steps_rejects_comparison_alone_without_evaluation_output(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.config_loader.OUTPUT_DIR_OVERRIDE", tmp_path)

    with pytest.raises(FileNotFoundError, match="Evaluation output required"):
        validate_for_steps(["comparison"], retrieval_enabled=True)


def test_validate_for_steps_skips_rag_store_when_retrieval_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.config_loader.OUTPUT_DIR_OVERRIDE", tmp_path)

    translation_dir = tmp_path / "results" / "translation"
    translation_dir.mkdir(parents=True)
    (translation_dir / "A.txt").write_text("text", encoding="utf-8")

    # No rag_store/ directory exists — this must not raise when retrieval is disabled.
    validate_for_steps(["evaluation"], retrieval_enabled=False)


def test_assume_empty_ignores_existing_outputs(tmp_path, monkeypatch):
    """What --force checks before wiping: existing outputs do not count."""
    monkeypatch.setattr("automation.src.config_loader.OUTPUT_DIR_OVERRIDE", tmp_path)
    evaluation_dir = tmp_path / "results" / "evaluation"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "A.json").write_text("{}", encoding="utf-8")

    validate_for_steps(["comparison"])  # fine: the evaluation output exists
    with pytest.raises(FileNotFoundError, match="Evaluation output required"):
        validate_for_steps(["comparison"], assume_empty=True)  # but --force would delete it
