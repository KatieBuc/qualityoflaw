import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

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
from automation.src.constants import AUTOMATION_ROOT, CHUNKING_FALLBACK_PROMPT, DEFAULT_STEPS, PROJECT_ROOT
from automation.src.llm.model_profile import ModelProfile
from automation.src.run_pipeline import main, parse_steps, requires_run_id

TRANSLATION_RESULT = {
    "counts": {"total": 5, "succeeded": 5, "skipped": 0, "failed": 0},
    "failed_files": [],
    "elapsed_s": 1.0,
    "token_usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    "output_dir": "",
}

MD_TRANSLATION_RESULT = {
    "counts": {"total": 5, "succeeded": 5, "skipped": 0, "failed": 0},
    "failed_files": [],
    "elapsed_s": 1.0,
    "token_usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    "output_dir": "",
}

MD_TRANSLATION_QA_RESULT = {
    "counts": {
        "total": 5,
        "succeeded": 5,
        "corrected": 0,
        "incomplete": 0,
        "not_converged": 0,
        "clauses_lost": 0,
        "qa_passes": 5,
        "skipped": 0,
        "failed": 0,
    },
    "failed_files": [],
    "elapsed_s": 0.5,
    "token_usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    "output_dir": "",
}

MD_TO_TEXT_RESULT = {
    "counts": {
        "total": 5,
        "succeeded": 5,
        "skipped": 0,
        "failed": 0,
        "clauses_lost": 0,
        "repaired": 0,
    },
    "failed_files": [],
    "elapsed_s": 0.1,
}

STORAGE_RESULT = {
    "counts": {"total": 5, "succeeded": 5, "skipped": 0, "failed": 0},
    "failed_files": [],
    "elapsed_s": 0.5,
    "output_dir": "",
}

EVALUATION_RESULT = {
    "counts": {"total": 5, "succeeded": 5, "failed": 0, "saved_reports": 5},
    "failed_policies": [],
    "elapsed_s": 2.0,
    "token_usage": {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
    "output_dir": "",
}

COMPARISON_RESULT = {
    "counts": {"matched_pairs": 280, "evaluated_policies": 5, "accuracy": 0.82},
    "evaluated_policy_files": ["ACEH_BIREUEN.txt"],
    "elapsed_s": 0.1,
    "metrics_path": "/tmp/metrics.csv",
    "errors_path": None,
    "output_dir": "",
}

DIAGNOSIS_RESULT = {
    "counts": {
        "total": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "saved_reports": 0,
        "discrepancies_total": 0,
    },
    "failed_policies": [],
    "elapsed_s": 0.05,
    "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    "output_dir": "",
}


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
                "max_tokens": None,
                "max_retries": 2,
            },
            "test-diagnosis": {
                "deployment": "gpt-5.2",
                "temperature": 0.2,
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


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.metadata.DEFAULT_DATA_ROOT", tmp_path)
    monkeypatch.setattr("automation.src.config_loader.DEFAULT_DATA_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def mock_config():
    translate_model = ModelProfile(name="test-translate", deployment="gpt-5.2", temperature=0.2)
    eval_model = ModelProfile(name="test-eval", deployment="gpt-4o", temperature=0.1)
    diagnosis_model = ModelProfile(name="test-diagnosis", deployment="gpt-5.2", temperature=0.2)
    translation_qa_model = ModelProfile(name="test-translation-qa", deployment="gpt-5.2", temperature=0.1)
    return ResolvedPipelineConfig(
        experiment_name="test_experiment",
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
            input_dir=PROJECT_ROOT / "data" / "raw" / "localpolicies",
            golden_csv=PROJECT_ROOT / "data" / "processed" / "long_policy_encoding.csv",
            index_schema=PROJECT_ROOT / "data" / "mapping" / "index_schema.yaml",
        ),
        concurrency=ConcurrencyConfig(enabled=True, max_workers=5),
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


def test_parse_steps_defaults_to_all():
    assert parse_steps(None) == list(DEFAULT_STEPS)


def test_parse_steps_default_includes_discrepancy_diagnosis():
    assert "discrepancy_diagnosis" in parse_steps(None)


def test_parse_steps_accepts_discrepancy_diagnosis_explicitly():
    assert parse_steps("discrepancy_diagnosis") == ["discrepancy_diagnosis"]


def test_parse_steps_accepts_single_step():
    assert parse_steps("translation") == ["translation"]


def test_parse_steps_trims_whitespace():
    assert parse_steps(" translation , evaluation ") == ["translation", "evaluation"]


def test_parse_steps_rejects_invalid_step():
    with pytest.raises(ValueError, match="Invalid steps: bogus"):
        parse_steps("translation,bogus")


def test_requires_run_id_for_eval_only():
    assert requires_run_id(["evaluation"], None) is True
    assert requires_run_id(["comparison"], None) is True
    assert requires_run_id(["evaluation", "comparison"], None) is True
    assert requires_run_id(["discrepancy_diagnosis"], None) is True


def test_requires_run_id_false_for_full_pipeline():
    assert requires_run_id(list(DEFAULT_STEPS), None) is False
    assert requires_run_id(["translation", "evaluation"], None) is False


def test_requires_run_id_false_when_run_id_provided():
    assert requires_run_id(["evaluation"], "existing_run") is False


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.init_run_metadata")
@patch("automation.src.run_pipeline.generate_run_id", return_value="20250101_120000")
@patch("automation.src.run_pipeline.run_diagnosis_step", return_value=DIAGNOSIS_RESULT)
@patch("automation.src.run_pipeline.run_comparison_step", return_value=COMPARISON_RESULT)
@patch("automation.src.run_pipeline.run_evaluation_step", return_value=EVALUATION_RESULT)
@patch("automation.src.run_pipeline.run_storage_step", return_value=STORAGE_RESULT)
@patch("automation.src.run_pipeline.run_md_to_text_step", return_value=MD_TO_TEXT_RESULT)
@patch(
    "automation.src.run_pipeline.run_md_translation_qa_step",
    return_value=MD_TRANSLATION_QA_RESULT,
)
@patch(
    "automation.src.run_pipeline.run_md_translation_step",
    return_value=MD_TRANSLATION_RESULT,
)
@patch("automation.src.run_pipeline.AzureEmbedder")
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_full_pipeline_small_scale_without_run_id(
    mock_load_config,
    mock_wrapper,
    mock_embedder,
    mock_translate,
    mock_translation_qa,
    mock_to_text,
    mock_storage,
    mock_eval,
    mock_compare,
    mock_diagnose,
    mock_generate_run_id,
    mock_init_metadata,
    mock_update_metadata,
    mock_config,
    data_root,
):
    mock_load_config.return_value = mock_config

    with patch.object(sys, "argv", ["run_pipeline", "--small-scale"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    mock_translate.assert_called_once()
    mock_translation_qa.assert_called_once()
    mock_to_text.assert_called_once()
    mock_storage.assert_called_once()
    mock_eval.assert_called_once()
    mock_compare.assert_called_once()
    mock_diagnose.assert_called_once()
    assert mock_translate.call_args.kwargs["small_scale"] is True
    assert mock_to_text.call_args.kwargs["small_scale"] is True
    assert mock_translate.call_args.kwargs["force"] is False
    assert mock_eval.call_args.kwargs["small_scale"] is True
    assert mock_eval.call_args.kwargs["allow_partial"] is False
    assert mock_diagnose.call_args.kwargs["small_scale"] is True
    assert mock_diagnose.call_args.kwargs["allow_partial"] is False


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.init_run_metadata")
@patch("automation.src.run_pipeline.generate_run_id", return_value="20250101_120000")
@patch("automation.src.run_pipeline.run_diagnosis_step", return_value=DIAGNOSIS_RESULT)
@patch("automation.src.run_pipeline.run_comparison_step", return_value=COMPARISON_RESULT)
@patch("automation.src.run_pipeline.run_evaluation_step", return_value=EVALUATION_RESULT)
@patch("automation.src.run_pipeline.run_storage_step", return_value=STORAGE_RESULT)
@patch("automation.src.run_pipeline.run_md_to_text_step", return_value=MD_TO_TEXT_RESULT)
@patch(
    "automation.src.run_pipeline.run_md_translation_qa_step",
    return_value=MD_TRANSLATION_QA_RESULT,
)
@patch(
    "automation.src.run_pipeline.run_md_translation_step",
    return_value=MD_TRANSLATION_RESULT,
)
@patch("automation.src.run_pipeline.AzureEmbedder")
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_full_pipeline_without_small_scale_passes_false(
    mock_load_config,
    mock_wrapper,
    mock_embedder,
    mock_translate,
    mock_translation_qa,
    mock_to_text,
    mock_storage,
    mock_eval,
    mock_compare,
    mock_diagnose,
    mock_generate_run_id,
    mock_init_metadata,
    mock_update_metadata,
    mock_config,
    data_root,
):
    mock_load_config.return_value = mock_config

    with patch.object(sys, "argv", ["run_pipeline"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    mock_translate.assert_called_once()
    mock_translation_qa.assert_called_once()
    mock_to_text.assert_called_once()
    mock_storage.assert_called_once()
    mock_eval.assert_called_once()
    mock_compare.assert_called_once()
    mock_diagnose.assert_called_once()
    assert mock_translate.call_args.kwargs["small_scale"] is False
    assert mock_to_text.call_args.kwargs["small_scale"] is False
    assert mock_eval.call_args.kwargs["small_scale"] is False
    assert mock_diagnose.call_args.kwargs["small_scale"] is False
    mock_init_metadata.assert_called_once()
    assert mock_init_metadata.call_args.kwargs["small_scale"] is False


@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_eval_only_without_run_id_exits(mock_load_config, mock_config):
    mock_load_config.return_value = mock_config

    with patch.object(sys, "argv", ["run_pipeline", "--steps", "evaluation"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1
    mock_load_config.assert_not_called()


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.validate_run_for_steps")
@patch("automation.src.run_pipeline.run_comparison_step", return_value=COMPARISON_RESULT)
@patch("automation.src.run_pipeline.run_evaluation_step", return_value=EVALUATION_RESULT)
@patch("automation.src.run_pipeline.run_translation_step")
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_comparison_only_with_existing_run_id(
    mock_load_config,
    mock_wrapper,
    mock_translate,
    mock_eval,
    mock_compare,
    mock_validate,
    mock_update_metadata,
    mock_config,
    data_root,
):
    mock_load_config.return_value = mock_config
    run_id = "existing_run"
    (data_root / run_id).mkdir()
    (data_root / run_id / "metadata.json").write_text('{"run_id": "existing_run"}', encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        ["run_pipeline", "--run-id", run_id, "--steps", "comparison"],
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    mock_translate.assert_not_called()
    mock_eval.assert_not_called()
    mock_compare.assert_called_once_with(run_id=run_id, config=mock_config)
    mock_validate.assert_called_once_with(
        run_id,
        ["comparison"],
        retrieval_enabled=mock_config.retrieval.enabled,
        evaluation_method=mock_config.evaluation_method,
    )


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.init_run_metadata")
@patch("automation.src.run_pipeline.run_translation_step", return_value=TRANSLATION_RESULT)
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_translation_only_passes_force_flag(
    mock_load_config,
    mock_wrapper,
    mock_translate,
    mock_init_metadata,
    mock_update_metadata,
    mock_config,
    data_root,
):
    mock_load_config.return_value = mock_config

    with patch.object(
        sys,
        "argv",
        ["run_pipeline", "--steps", "translation", "--run-id", "forced_run", "--force"],
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    mock_translate.assert_called_once()
    assert mock_translate.call_args.kwargs["force"] is True


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.init_run_metadata")
@patch("automation.src.run_pipeline.generate_run_id", return_value="20250101_120000")
@patch("automation.src.run_pipeline.run_evaluation_step")
@patch("automation.src.run_pipeline.run_translation_step", return_value=TRANSLATION_RESULT)
@patch("automation.src.run_pipeline.AzureEmbedder")
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_eval_failure_without_allow_partial_exits_nonzero(
    mock_load_config,
    mock_wrapper,
    mock_embedder,
    mock_translate,
    mock_eval,
    mock_generate_run_id,
    mock_init_metadata,
    mock_update_metadata,
    mock_config,
    data_root,
):
    mock_load_config.return_value = mock_config
    mock_eval.return_value = {
        **EVALUATION_RESULT,
        "counts": {"total": 5, "succeeded": 4, "failed": 1, "saved_reports": 4},
        "failed_policies": ["ACEH_BIREUEN.txt"],
    }

    with patch.object(
        sys,
        "argv",
        ["run_pipeline", "--steps", "translation,evaluation", "--small-scale"],
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1
    mock_eval.assert_called_once()
    assert mock_eval.call_args.kwargs["allow_partial"] is False


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.init_run_metadata")
@patch("automation.src.run_pipeline.generate_run_id", return_value="20250101_120000")
@patch("automation.src.run_pipeline.run_evaluation_step")
@patch("automation.src.run_pipeline.run_translation_step", return_value=TRANSLATION_RESULT)
@patch("automation.src.run_pipeline.AzureEmbedder")
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_allow_partial_keeps_zero_exit_on_eval_failure(
    mock_load_config,
    mock_wrapper,
    mock_embedder,
    mock_translate,
    mock_eval,
    mock_generate_run_id,
    mock_init_metadata,
    mock_update_metadata,
    mock_config,
    data_root,
):
    mock_load_config.return_value = mock_config
    mock_eval.return_value = {
        **EVALUATION_RESULT,
        "counts": {"total": 5, "succeeded": 4, "failed": 1, "saved_reports": 4},
        "failed_policies": ["ACEH_BIREUEN.txt"],
    }

    with patch.object(
        sys,
        "argv",
        [
            "run_pipeline",
            "--steps",
            "translation,evaluation",
            "--allow-partial",
        ],
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    assert mock_eval.call_args.kwargs["allow_partial"] is True


@patch("automation.src.run_pipeline.load_pipeline_config", side_effect=FileNotFoundError("missing config"))
def test_main_invalid_config_path_exits(mock_load_config):
    with patch.object(
        sys,
        "argv",
        ["run_pipeline", "--pipeline-config", "missing.yaml"],
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.validate_run_for_steps")
@patch("automation.src.run_pipeline.run_comparison_step", return_value=COMPARISON_RESULT)
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_custom_config_paths(
    mock_load_config,
    mock_compare,
    mock_validate,
    mock_update_metadata,
    mock_config,
    pipeline_config_path,
    model_config_path,
    data_root,
):
    mock_load_config.return_value = mock_config
    run_id = "cfg_run"
    (data_root / run_id).mkdir()
    (data_root / run_id / "metadata.json").write_text("{}", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "run_pipeline",
            "--run-id",
            run_id,
            "--steps",
            "comparison",
            "--pipeline-config",
            str(pipeline_config_path),
            "--model-config",
            str(model_config_path),
        ],
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    mock_load_config.assert_called_once_with(
        pipeline_config_path=Path(pipeline_config_path),
        model_config_path=Path(model_config_path),
    )


def test_main_rejects_unknown_run_missing_flag():
    with patch.object(sys, "argv", ["run_pipeline", "--run-missing", "--steps", "translation"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 2
