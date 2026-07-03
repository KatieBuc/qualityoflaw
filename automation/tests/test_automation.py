import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from automation.src.config_loader import load_model_profiles, load_pipeline_config
from automation.src.constants import AUTOMATION_ROOT, SMALL_SCALE_FILES
from automation.src.llm.model_profile import ModelProfile
from automation.src.llm.wrapper import AzureLLMWrapper, clean_translation_response
from automation.src.metadata import generate_run_id, init_run_metadata, load_metadata, update_metadata
from automation.src.run_pipeline import parse_steps, requires_run_id
from automation.src.translate import resolve_input_files


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
        "evaluation": {"model": "test-eval", "prompt_version": "v1"},
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
    assert config.evaluation_model.name == "test-eval"
    assert config.concurrency.enabled is True
    assert config.concurrency.max_workers == 5
    assert config.translation_prompt_path.exists()
    assert config.evaluation_template_path.exists()


def test_load_pipeline_config_unknown_model(pipeline_config_path, model_config_path):
    data = yaml.safe_load(pipeline_config_path.read_text(encoding="utf-8"))
    data["translation"]["model"] = "missing-model"
    pipeline_config_path.write_text(yaml.dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown model key"):
        load_pipeline_config(pipeline_config_path, model_config_path)


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


def test_metadata_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.metadata.DEFAULT_DATA_ROOT", tmp_path)
    monkeypatch.setattr("automation.src.config_loader.DEFAULT_DATA_ROOT", tmp_path)

    run_id = "20250101_120000"
    init_run_metadata(
        run_id=run_id,
        experiment_name="exp",
        small_scale=True,
        config_summary={"translation_model": "test-translate"},
        pipeline_config_path=AUTOMATION_ROOT / "config" / "pipeline_config.yaml",
        model_config_path=AUTOMATION_ROOT / "config" / "model_config.yaml",
    )

    meta = load_metadata(run_id)
    assert meta["run_id"] == run_id
    assert meta["small_scale"] is True
    assert (tmp_path / run_id / "config" / "pipeline_config.yaml").exists()

    update_metadata(run_id, steps_executed=["translation"], timing={"translation_s": 1.5})
    updated = load_metadata(run_id)
    assert updated["steps_executed"] == ["translation"]
    assert updated["timing"]["translation_s"] == 1.5


def test_generate_run_id_collision(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.metadata.DEFAULT_DATA_ROOT", tmp_path)
    monkeypatch.setattr("automation.src.config_loader.DEFAULT_DATA_ROOT", tmp_path)

    with patch("automation.src.metadata.datetime") as mock_dt:
        mock_dt.now.return_value.strftime.return_value = "20250101_120000"
        first = generate_run_id()
        (tmp_path / first).mkdir()
        second = generate_run_id()

    assert first == "20250101_120000"
    assert second == "20250101_120000_2"


def test_requires_run_id_helpers():
    assert requires_run_id(parse_steps(None), None) is False
    assert requires_run_id(parse_steps("evaluation,comparison"), None) is True


def test_validate_run_for_steps_missing_translation(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.metadata.DEFAULT_DATA_ROOT", tmp_path)
    monkeypatch.setattr("automation.src.config_loader.DEFAULT_DATA_ROOT", tmp_path)

    run_id = "test_run"
    (tmp_path / run_id).mkdir()

    from automation.src.metadata import validate_run_for_steps

    with pytest.raises(FileNotFoundError, match="Translation output required"):
        validate_run_for_steps(run_id, ["evaluation"])
