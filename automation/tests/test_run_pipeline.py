import dataclasses
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
from automation.src.constants import (
    AUTOMATION_ROOT,
    CHUNKING_FALLBACK_PROMPT,
    DEFAULT_STEPS,
    OPTIONAL_STEPS,
    PROJECT_ROOT,
)
from automation.src.llm.model_profile import ModelProfile
import json

from automation.src.run_pipeline import (
    _print_overall_alignment_rate,
    build_config_summary,
    main,
    parse_steps,
    resolve_concurrency_config,
)

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

SLIDING_WINDOW_RESULT = {
    "counts": {"total": 5, "succeeded": 5, "failed": 0, "skipped": 0, "saved_reports": 5},
    "failed_policies": [],
    "elapsed_s": 3.0,
    "token_usage": {"prompt_tokens": 80, "completion_tokens": 120, "total_tokens": 200},
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


@pytest.fixture(autouse=True)
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.config_loader.OUTPUT_DIR_OVERRIDE", tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _no_preflight_validation(monkeypatch):
    """Tests that care about validation patch it themselves (see the --force tests)."""
    monkeypatch.setattr("automation.src.run_pipeline.validate_for_steps", MagicMock())


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


def _write_eval_report(eval_dir: Path, name: str, stat: dict | None) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    report = {"policy_file": name}
    if stat is not None:
        report["evidence_original_aligned_rate"] = stat
    (eval_dir / name).write_text(json.dumps(report), encoding="utf-8")


def test_print_overall_alignment_rate_aggregates_across_reports(tmp_path, capsys):
    eval_dir = tmp_path / "results" / "evaluation"
    _write_eval_report(eval_dir, "A.json", {"sentence_aligned": 40, "resolved": 50, "rate": 0.8})
    _write_eval_report(eval_dir, "B.json", {"sentence_aligned": 5, "resolved": 10, "rate": 0.5})
    _write_eval_report(eval_dir, "C.json", None)  # report without the stat is ignored

    with patch("automation.src.run_pipeline.get_output_dir", return_value=tmp_path):
        _print_overall_alignment_rate()

    out = capsys.readouterr().out
    assert "Evidence original-language alignment: 45/60 locally aligned (75.0%)" in out


def test_print_overall_alignment_rate_silent_when_nothing_resolved(tmp_path, capsys):
    eval_dir = tmp_path / "results" / "evaluation"
    _write_eval_report(eval_dir, "A.json", {"sentence_aligned": 0, "resolved": 0, "rate": None})

    with patch("automation.src.run_pipeline.get_output_dir", return_value=tmp_path):
        _print_overall_alignment_rate()

    assert capsys.readouterr().out == ""


def test_print_overall_alignment_rate_silent_when_no_eval_dir(tmp_path, capsys):
    with patch("automation.src.run_pipeline.get_output_dir", return_value=tmp_path):
        _print_overall_alignment_rate()

    assert capsys.readouterr().out == ""


def test_parse_steps_defaults_to_all():
    assert parse_steps(None) == list(DEFAULT_STEPS)


def test_parse_steps_default_excludes_optional_steps():
    # translation_qa_md and discrepancy_diagnosis are valid --steps values but
    # deliberately left out of the default chain (extra QA pass / diagnosis
    # cost isn't wanted on every run).
    steps = parse_steps(None)
    assert "translation_qa_md" not in steps
    assert "discrepancy_diagnosis" not in steps


def test_parse_steps_accepts_discrepancy_diagnosis_explicitly():
    assert parse_steps("discrepancy_diagnosis") == ["discrepancy_diagnosis"]


def test_parse_steps_accepts_translation_qa_md_explicitly():
    assert parse_steps("translation_qa_md") == ["translation_qa_md"]


def test_parse_steps_accepts_single_step():
    assert parse_steps("translation") == ["translation"]


def test_parse_steps_trims_whitespace():
    assert parse_steps(" translation , evaluation ") == ["translation", "evaluation"]


def test_parse_steps_rejects_invalid_step():
    with pytest.raises(ValueError, match="Invalid steps: bogus"):
        parse_steps("translation,bogus")


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.init_metadata")
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
def test_main_full_pipeline_small_scale_on_fresh_output(
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
    mock_to_text.assert_called_once()
    mock_storage.assert_called_once()
    mock_eval.assert_called_once()
    mock_compare.assert_not_called()  # comparison is optional now
    # translation_qa_md and discrepancy_diagnosis are valid --steps but not
    # part of the default chain, so the default run must not invoke them.
    mock_translation_qa.assert_not_called()
    mock_diagnose.assert_not_called()
    assert mock_translate.call_args.kwargs["small_scale"] is True
    assert mock_to_text.call_args.kwargs["small_scale"] is True
    assert mock_eval.call_args.kwargs["small_scale"] is True
    assert mock_eval.call_args.kwargs["allow_partial"] is False


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.init_metadata")
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
    mock_to_text.assert_called_once()
    mock_storage.assert_called_once()
    mock_eval.assert_called_once()
    mock_compare.assert_not_called()  # comparison is optional now
    mock_translation_qa.assert_not_called()
    mock_diagnose.assert_not_called()
    assert mock_translate.call_args.kwargs["small_scale"] is False
    assert mock_to_text.call_args.kwargs["small_scale"] is False
    assert mock_eval.call_args.kwargs["small_scale"] is False
    mock_init_metadata.assert_called_once()
    assert mock_init_metadata.call_args.kwargs["small_scale"] is False


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.validate_for_steps")
@patch("automation.src.run_pipeline.run_comparison_step", return_value=COMPARISON_RESULT)
@patch("automation.src.run_pipeline.run_evaluation_step", return_value=EVALUATION_RESULT)
@patch("automation.src.run_pipeline.run_translation_step")
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_comparison_only_on_existing_output(
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
    (data_root / "metadata.json").write_text("{}", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        ["run_pipeline", "--steps", "comparison"],
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    mock_translate.assert_not_called()
    mock_eval.assert_not_called()
    mock_compare.assert_called_once_with(config=mock_config)
    mock_validate.assert_called_once_with(
        ["comparison"],
        retrieval_enabled=mock_config.retrieval.enabled,
        evaluation_method=mock_config.evaluation_method,
        translation_markdown_dir=mock_config.paths.translation_markdown_dir,
        processed_translation_markdown_dir=mock_config.paths.processed_translation_markdown_dir,
    )


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.init_metadata")
@patch("automation.src.run_pipeline.clean_output_dir", return_value=3)
@patch("automation.src.run_pipeline.validate_for_steps")
@patch("automation.src.run_pipeline.run_translation_step", return_value=TRANSLATION_RESULT)
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_force_validates_then_cleans_then_runs(
    mock_load_config,
    mock_wrapper,
    mock_translate,
    mock_validate,
    mock_clean,
    mock_init_metadata,
    mock_update_metadata,
    mock_config,
    data_root,
):
    mock_load_config.return_value = mock_config
    order = MagicMock()
    order.attach_mock(mock_validate, "validate")
    order.attach_mock(mock_clean, "clean")
    order.attach_mock(mock_translate, "translate")

    with patch.object(sys, "argv", ["run_pipeline", "--steps", "translation", "--force"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    assert mock_validate.call_args.kwargs["assume_empty"] is True
    assert [c[0] for c in order.mock_calls] == ["validate", "clean", "translate"]
    # --force wipes the folder; steps themselves are no longer forced.
    assert "force" not in mock_translate.call_args.kwargs
    mock_init_metadata.assert_called_once()


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.clean_output_dir")
@patch(
    "automation.src.run_pipeline.validate_for_steps",
    side_effect=FileNotFoundError("Evaluation output required for comparison"),
)
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_force_does_not_clean_when_the_steps_cannot_rebuild(
    mock_load_config,
    mock_validate,
    mock_clean,
    mock_update_metadata,
    mock_config,
    data_root,
):
    mock_load_config.return_value = mock_config

    with patch.object(sys, "argv", ["run_pipeline", "--steps", "comparison", "--force"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1
    mock_clean.assert_not_called()


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.init_metadata")
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
@patch("automation.src.run_pipeline.init_metadata")
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
@patch("automation.src.run_pipeline.validate_for_steps")
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
    (data_root / "metadata.json").write_text("{}", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "run_pipeline",
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


# ---------------------------------------------------------------------------
# parse_steps: optional steps restored to VALID_STEPS but kept out of default
# ---------------------------------------------------------------------------


def test_parse_steps_accepts_full_markdown_chain_including_optional_steps():
    steps = parse_steps(
        "translation_md,translation_qa_md,md_to_text,storage,evaluation,comparison,discrepancy_diagnosis"
    )
    assert steps == [
        "translation_md",
        "translation_qa_md",
        "md_to_text",
        "storage",
        "evaluation",
        "comparison",
        "discrepancy_diagnosis",
    ]


def test_optional_steps_are_not_part_of_default_steps():
    assert set(OPTIONAL_STEPS).isdisjoint(DEFAULT_STEPS)


# ---------------------------------------------------------------------------
# build_config_summary: metadata snapshot reflects the resolved config
# ---------------------------------------------------------------------------


def test_build_config_summary_reports_key_settings(mock_config):
    summary = build_config_summary(mock_config)

    assert summary["translation_model"] == "test-translate"
    assert summary["evaluation_model"] == "test-eval"
    assert summary["discrepancy_diagnosis_model"] == "test-diagnosis"
    assert summary["evaluation_method"] == "rag"
    assert summary["concurrency"] == {"enabled": True, "max_workers": 5}
    assert summary["chunking"] == {"enabled": False, "safe_limit": 32000}
    assert summary["storage"] == {"enabled": True, "batch_size": 16}
    assert summary["retrieval"]["top_k"] == 10
    assert summary["retrieval"]["reranker_enabled"] is False
    assert summary["retrieval"]["evidence_verification_enabled"] is True
    assert summary["markdown_translation"]["target_chars"] == mock_config.markdown.target_chars
    assert summary["markdown_translation"]["qa_max_passes"] == mock_config.markdown.qa_max_passes
    assert summary["sliding_window"]["window_sentences"] == mock_config.sliding_window.window_sentences


def test_build_config_summary_reflects_sliding_window_method(mock_config):
    config = dataclasses.replace(mock_config, evaluation_method="sliding_window")
    summary = build_config_summary(config)
    assert summary["evaluation_method"] == "sliding_window"


# ---------------------------------------------------------------------------
# Step failures halt the pipeline: downstream steps must not run
# ---------------------------------------------------------------------------


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.init_metadata")
@patch("automation.src.run_pipeline.run_comparison_step")
@patch("automation.src.run_pipeline.run_evaluation_step")
@patch("automation.src.run_pipeline.run_storage_step")
@patch("automation.src.run_pipeline.run_md_to_text_step")
@patch(
    "automation.src.run_pipeline.run_md_translation_step",
    side_effect=RuntimeError("translation API unreachable"),
)
@patch("automation.src.run_pipeline.AzureEmbedder")
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_stops_pipeline_when_translation_md_fails(
    mock_load_config,
    mock_wrapper,
    mock_embedder,
    mock_translate,
    mock_to_text,
    mock_storage,
    mock_eval,
    mock_compare,
    mock_init_metadata,
    mock_update_metadata,
    mock_config,
    data_root,
    capsys,
):
    mock_load_config.return_value = mock_config

    with patch.object(sys, "argv", ["run_pipeline"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1
    mock_to_text.assert_not_called()
    mock_storage.assert_not_called()
    mock_eval.assert_not_called()
    mock_compare.assert_not_called()
    err = capsys.readouterr().err
    assert "Step 'translation_md' failed: translation API unreachable" in err


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.validate_for_steps")
@patch("automation.src.run_pipeline.run_comparison_step")
@patch("automation.src.run_pipeline.run_evaluation_step", side_effect=RuntimeError("quota exceeded"))
@patch("automation.src.run_pipeline.AzureEmbedder")
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_stops_pipeline_when_evaluation_fails_before_comparison(
    mock_load_config,
    mock_wrapper,
    mock_embedder,
    mock_eval,
    mock_compare,
    mock_validate,
    mock_update_metadata,
    mock_config,
    data_root,
    capsys,
):
    mock_load_config.return_value = mock_config
    (data_root / "metadata.json").write_text("{}", encoding="utf-8")

    with patch.object(
        sys, "argv", ["run_pipeline", "--steps", "evaluation,comparison"]
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1
    mock_compare.assert_not_called()
    err = capsys.readouterr().err
    assert "Step 'evaluation' failed: quota exceeded" in err


# ---------------------------------------------------------------------------
# evaluation.method switches which evaluation step runs, and storage follows
# ---------------------------------------------------------------------------


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.init_metadata")
@patch("automation.src.run_pipeline.run_comparison_step", return_value=COMPARISON_RESULT)
@patch("automation.src.run_pipeline.run_evaluation_step")
@patch(
    "automation.src.run_pipeline.run_sliding_window_evaluation_step",
    return_value=SLIDING_WINDOW_RESULT,
)
@patch("automation.src.run_pipeline.run_storage_step")
@patch("automation.src.run_pipeline.run_md_to_text_step", return_value=MD_TO_TEXT_RESULT)
@patch(
    "automation.src.run_pipeline.run_md_translation_step",
    return_value=MD_TRANSLATION_RESULT,
)
@patch("automation.src.run_pipeline.AzureEmbedder")
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_sliding_window_method_skips_rag_evaluation_and_storage(
    mock_load_config,
    mock_wrapper,
    mock_embedder,
    mock_translate,
    mock_to_text,
    mock_storage,
    mock_sliding_eval,
    mock_eval,
    mock_compare,
    mock_init_metadata,
    mock_update_metadata,
    mock_config,
    data_root,
    capsys,
):
    config = dataclasses.replace(
        mock_config,
        evaluation_method="sliding_window",
        storage=StorageConfig(enabled=False, batch_size=16),
    )
    mock_load_config.return_value = config

    with patch.object(sys, "argv", ["run_pipeline", "--small-scale"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    mock_sliding_eval.assert_called_once()
    mock_eval.assert_not_called()
    mock_storage.assert_not_called()
    out = capsys.readouterr().out
    assert "Step: storage (skipped" in out
    assert "Step: evaluation (method=sliding_window)" in out


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.validate_for_steps")
@patch("automation.src.run_pipeline.run_evaluation_step", return_value=EVALUATION_RESULT)
@patch("automation.src.run_pipeline.AzureEmbedder")
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_rag_evaluation_gets_no_embedder_when_retrieval_disabled(
    mock_load_config,
    mock_wrapper,
    mock_embedder,
    mock_eval,
    mock_validate,
    mock_update_metadata,
    mock_config,
    data_root,
):
    retrieval = dataclasses.replace(mock_config.retrieval, enabled=False)
    config = dataclasses.replace(mock_config, retrieval=retrieval)
    mock_load_config.return_value = config
    (data_root / "metadata.json").write_text("{}", encoding="utf-8")

    with patch.object(sys, "argv", ["run_pipeline", "--steps", "evaluation"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    mock_embedder.from_env.assert_not_called()
    assert mock_eval.call_args.kwargs["embedder"] is None


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.validate_for_steps")
@patch("automation.src.run_pipeline.run_evaluation_step", return_value=EVALUATION_RESULT)
@patch("automation.src.run_pipeline.AzureEmbedder")
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_rag_evaluation_gets_embedder_when_retrieval_enabled(
    mock_load_config,
    mock_wrapper,
    mock_embedder,
    mock_eval,
    mock_validate,
    mock_update_metadata,
    mock_config,
    data_root,
):
    mock_load_config.return_value = mock_config  # retrieval.enabled is True by default
    (data_root / "metadata.json").write_text("{}", encoding="utf-8")

    with patch.object(sys, "argv", ["run_pipeline", "--steps", "evaluation"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    mock_embedder.from_env.assert_called_once_with(batch_size=mock_config.storage.batch_size)
    assert mock_eval.call_args.kwargs["embedder"] is mock_embedder.from_env.return_value


# ---------------------------------------------------------------------------
# reranker.enabled fails the evaluation step fast on missing credentials,
# before any worker thread would otherwise hit the same error independently
# ---------------------------------------------------------------------------


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.validate_for_steps")
@patch("automation.src.run_pipeline.run_evaluation_step")
@patch("automation.src.run_pipeline.get_cohere_rerank_api_key")
@patch(
    "automation.src.run_pipeline.get_cohere_rerank_endpoint",
    side_effect=RuntimeError("AZURE_COHERE_RERANK_ENDPOINT must be set in .env for the reranker step."),
)
@patch("automation.src.run_pipeline.AzureEmbedder")
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_reranker_enabled_fails_fast_on_missing_credentials(
    mock_load_config,
    mock_wrapper,
    mock_embedder,
    mock_get_endpoint,
    mock_get_key,
    mock_eval,
    mock_validate,
    mock_update_metadata,
    mock_config,
    data_root,
    capsys,
):
    reranker = dataclasses.replace(mock_config.retrieval.reranker, enabled=True, model="rerank-v3.5")
    retrieval = dataclasses.replace(mock_config.retrieval, enabled=True, reranker=reranker)
    config = dataclasses.replace(mock_config, retrieval=retrieval)
    mock_load_config.return_value = config
    (data_root / "metadata.json").write_text("{}", encoding="utf-8")

    with patch.object(sys, "argv", ["run_pipeline", "--steps", "evaluation"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1
    mock_get_endpoint.assert_called_once()
    mock_get_key.assert_not_called()
    mock_eval.assert_not_called()
    err = capsys.readouterr().err
    assert "Step 'evaluation' failed" in err


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.validate_for_steps")
@patch("automation.src.run_pipeline.run_evaluation_step", return_value=EVALUATION_RESULT)
@patch("automation.src.run_pipeline.get_cohere_rerank_api_key")
@patch("automation.src.run_pipeline.get_cohere_rerank_endpoint")
@patch("automation.src.run_pipeline.AzureEmbedder")
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_reranker_disabled_skips_credential_check(
    mock_load_config,
    mock_wrapper,
    mock_embedder,
    mock_get_endpoint,
    mock_get_key,
    mock_eval,
    mock_validate,
    mock_update_metadata,
    mock_config,
    data_root,
):
    mock_load_config.return_value = mock_config  # reranker.enabled is False by default
    (data_root / "metadata.json").write_text("{}", encoding="utf-8")

    with patch.object(sys, "argv", ["run_pipeline", "--steps", "evaluation"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    mock_get_endpoint.assert_not_called()
    mock_get_key.assert_not_called()


# ---------------------------------------------------------------------------
# discrepancy_diagnosis: skip messaging and allow_partial interaction
# ---------------------------------------------------------------------------


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.validate_for_steps")
@patch("automation.src.run_pipeline.run_diagnosis_step", return_value=DIAGNOSIS_RESULT)
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_discrepancy_diagnosis_prints_nothing_to_diagnose(
    mock_load_config,
    mock_wrapper,
    mock_diagnose,
    mock_validate,
    mock_update_metadata,
    mock_config,
    data_root,
    capsys,
):
    mock_load_config.return_value = mock_config
    (data_root / "metadata.json").write_text("{}", encoding="utf-8")

    with patch.object(
        sys, "argv", ["run_pipeline", "--steps", "discrepancy_diagnosis"]
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Discrepancy diagnosis: nothing to diagnose (no mismatches found by comparison)" in out


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.validate_for_steps")
@patch("automation.src.run_pipeline.run_diagnosis_step")
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_discrepancy_diagnosis_failure_without_allow_partial_exits_nonzero(
    mock_load_config,
    mock_wrapper,
    mock_diagnose,
    mock_validate,
    mock_update_metadata,
    mock_config,
    data_root,
):
    mock_load_config.return_value = mock_config
    mock_diagnose.return_value = {
        **DIAGNOSIS_RESULT,
        "counts": {**DIAGNOSIS_RESULT["counts"], "discrepancies_total": 3, "succeeded": 2, "failed": 1},
        "failed_policies": ["ACEH_BIREUEN.txt"],
    }
    (data_root / "metadata.json").write_text("{}", encoding="utf-8")

    with patch.object(
        sys, "argv", ["run_pipeline", "--steps", "discrepancy_diagnosis"]
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.validate_for_steps")
@patch("automation.src.run_pipeline.run_diagnosis_step")
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_discrepancy_diagnosis_failure_with_allow_partial_exits_zero(
    mock_load_config,
    mock_wrapper,
    mock_diagnose,
    mock_validate,
    mock_update_metadata,
    mock_config,
    data_root,
):
    mock_load_config.return_value = mock_config
    mock_diagnose.return_value = {
        **DIAGNOSIS_RESULT,
        "counts": {**DIAGNOSIS_RESULT["counts"], "discrepancies_total": 3, "succeeded": 2, "failed": 1},
        "failed_policies": ["ACEH_BIREUEN.txt"],
    }
    (data_root / "metadata.json").write_text("{}", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        ["run_pipeline", "--steps", "discrepancy_diagnosis", "--allow-partial"],
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# Output folder without metadata.json yet
# ---------------------------------------------------------------------------


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.init_metadata")
@patch("automation.src.run_pipeline.run_translation_step", return_value=TRANSLATION_RESULT)
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_creates_metadata_when_the_output_folder_is_new(
    mock_load_config,
    mock_wrapper,
    mock_translate,
    mock_init_metadata,
    mock_update_metadata,
    mock_config,
    data_root,
    capsys,
):
    mock_load_config.return_value = mock_config

    with patch.object(sys, "argv", ["run_pipeline", "--steps", "translation"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    mock_init_metadata.assert_called_once()
    assert "run_id" not in mock_init_metadata.call_args.kwargs
    assert "Created output folder" in capsys.readouterr().out


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.init_metadata")
@patch("automation.src.run_pipeline.run_translation_step", return_value=TRANSLATION_RESULT)
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_reuses_existing_metadata(
    mock_load_config,
    mock_wrapper,
    mock_translate,
    mock_init_metadata,
    mock_update_metadata,
    mock_config,
    data_root,
    capsys,
):
    mock_load_config.return_value = mock_config
    (data_root / "metadata.json").write_text("{}", encoding="utf-8")

    with patch.object(sys, "argv", ["run_pipeline", "--steps", "translation"]):
        with pytest.raises(SystemExit):
            main()

    mock_init_metadata.assert_not_called()
    assert "Using existing output folder" in capsys.readouterr().out


@patch(
    "automation.src.run_pipeline.validate_for_steps",
    side_effect=FileNotFoundError("Translation output required for storage"),
)
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_missing_prerequisite_exits_nonzero(
    mock_load_config,
    mock_validate,
    mock_config,
    data_root,
):
    mock_load_config.return_value = mock_config

    with patch.object(sys, "argv", ["run_pipeline", "--steps", "storage"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# The final failure summary line is printed once any step recorded a failure
# ---------------------------------------------------------------------------


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.validate_for_steps")
@patch("automation.src.run_pipeline.run_evaluation_step")
@patch("automation.src.run_pipeline.AzureEmbedder")
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_prints_failure_summary_line_when_allow_partial_swallows_exit_code(
    mock_load_config,
    mock_wrapper,
    mock_embedder,
    mock_eval,
    mock_validate,
    mock_update_metadata,
    mock_config,
    data_root,
    capsys,
):
    mock_load_config.return_value = mock_config
    mock_eval.return_value = {
        **EVALUATION_RESULT,
        "counts": {"total": 5, "succeeded": 4, "failed": 1, "saved_reports": 4},
        "failed_policies": ["ACEH_BIREUEN.txt"],
    }
    (data_root / "metadata.json").write_text("{}", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        ["run_pipeline", "--steps", "evaluation", "--allow-partial"],
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    err = capsys.readouterr().err
    # allow_partial suppresses the nonzero exit code, but the failed policy
    # must still surface to the operator.
    assert "Failed policies: ACEH_BIREUEN.txt" in err


# ---------------------------------------------------------------------------
# Explicit full Markdown chain, including the two opt-in steps, end to end
# ---------------------------------------------------------------------------


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.init_metadata")
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
def test_main_explicit_full_markdown_chain_runs_optional_steps_too(
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
    mock_init_metadata,
    mock_update_metadata,
    mock_config,
    data_root,
):
    mock_load_config.return_value = mock_config
    full_chain = (
        "translation_md,translation_qa_md,md_to_text,storage,evaluation,comparison,"
        "discrepancy_diagnosis"
    )

    with patch.object(sys, "argv", ["run_pipeline", "--steps", full_chain, "--small-scale"]):
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
    for call in (mock_translate, mock_translation_qa, mock_to_text, mock_eval, mock_diagnose):
        assert call.call_args.kwargs["small_scale"] is True


# ---------------------------------------------------------------------------
# Legacy raw-OCR-text chain: translation, translation_qa, markdown connect
# without ever touching the Markdown-path functions
# ---------------------------------------------------------------------------


@patch("automation.src.run_pipeline.update_metadata")
@patch("automation.src.run_pipeline.init_metadata")
@patch("automation.src.run_pipeline.run_md_translation_step")
@patch("automation.src.run_pipeline.run_markdown_step")
@patch("automation.src.run_pipeline.run_translation_qa_step")
@patch("automation.src.run_pipeline.run_translation_step", return_value=TRANSLATION_RESULT)
@patch("automation.src.run_pipeline.AzureLLMWrapper")
@patch("automation.src.run_pipeline.load_pipeline_config")
def test_main_legacy_text_chain_runs_translation_qa_and_markdown_only(
    mock_load_config,
    mock_wrapper,
    mock_translate,
    mock_translation_qa,
    mock_markdown,
    mock_md_translate,
    mock_init_metadata,
    mock_update_metadata,
    mock_config,
    data_root,
):
    mock_translation_qa.return_value = {
        "counts": {"total": 5, "succeeded": 5, "corrected": 0, "incomplete": 0, "skipped": 0, "failed": 0},
        "failed_files": [],
        "elapsed_s": 0.5,
        "token_usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        "output_dir": "",
    }
    mock_markdown.return_value = {
        "counts": {"total": 5, "succeeded": 5, "skipped": 0, "failed": 0},
        "failed_files": [],
        "elapsed_s": 0.2,
    }
    mock_load_config.return_value = mock_config

    with patch.object(
        sys, "argv", ["run_pipeline", "--steps", "translation,translation_qa,markdown"]
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    mock_translate.assert_called_once()
    mock_translation_qa.assert_called_once()
    mock_markdown.assert_called_once()
    mock_md_translate.assert_not_called()
