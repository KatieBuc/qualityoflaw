from pathlib import Path

import pytest
import yaml

from automation.src.config_loader import load_pipeline_config


@pytest.fixture
def model_config_path(tmp_path):
    data = {
        "models": {
            "test-translate": {"deployment": "gpt-5.2", "temperature": 0.2},
            "test-eval": {"deployment": "gpt-4o", "temperature": 0.1},
            "test-diagnosis": {"deployment": "gpt-5.2", "temperature": 0.2},
        }
    }
    path = tmp_path / "model_config.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


def _write_pipeline_config(tmp_path, evaluation_overrides: dict) -> Path:
    data = {
        "experiment_name": "test",
        "translation": {"model": "test-translate", "prompt_version": "v1"},
        "evaluation": {"model": "test-eval", "prompt_version": "v1", **evaluation_overrides},
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


def test_defaults_to_rag_method(tmp_path, model_config_path):
    pipeline_config_path = _write_pipeline_config(tmp_path, {})
    config = load_pipeline_config(pipeline_config_path, model_config_path)
    assert config.evaluation_method == "rag"
    assert config.storage.enabled is True


def test_sliding_window_method_disables_storage(tmp_path, model_config_path):
    pipeline_config_path = _write_pipeline_config(
        tmp_path, {"method": "sliding_window", "rag": {"enabled": True}}
    )
    config = load_pipeline_config(pipeline_config_path, model_config_path)
    assert config.evaluation_method == "sliding_window"
    assert config.storage.enabled is False


def test_sliding_window_template_path_resolved(tmp_path, model_config_path):
    pipeline_config_path = _write_pipeline_config(tmp_path, {"method": "sliding_window"})
    config = load_pipeline_config(pipeline_config_path, model_config_path)
    assert config.sliding_window_template_path.name == "prompt_template.txt"
    assert config.sliding_window_template_path.parent.name == "sliding_window_v1"
    assert config.sliding_window_template_path.exists()


def test_rejects_unknown_evaluation_method(tmp_path, model_config_path):
    pipeline_config_path = _write_pipeline_config(tmp_path, {"method": "bogus"})
    with pytest.raises(ValueError, match="evaluation.method"):
        load_pipeline_config(pipeline_config_path, model_config_path)
