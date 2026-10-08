import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from automation.src.concurrency import ConcurrencyLimiter
from automation.src.diagnose import run_diagnosis_step

TEMPLATE_TEXT = "ORIGINAL:\n{{ORIGINAL_TEXT}}\n\nDISCREPANCIES:\n{{DISCREPANCIES}}\n"


class _Paths:
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir


class _Config:
    """Minimal stand-in for ResolvedPipelineConfig — run_diagnosis_step only
    reads config.paths.input_dir and config.discrepancy_diagnosis_template_path.
    """

    def __init__(self, input_dir: Path, template_path: Path):
        self.paths = _Paths(input_dir)
        self.discrepancy_diagnosis_template_path = template_path


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.config_loader.OUTPUT_DIR_OVERRIDE", tmp_path)
    return tmp_path


@pytest.fixture
def input_dir(tmp_path):
    d = tmp_path / "raw_input"
    d.mkdir()
    return d


@pytest.fixture
def template_path(tmp_path):
    versioned_dir = tmp_path / "prompts" / "discrepancy_diagnosis" / "v1"
    versioned_dir.mkdir(parents=True)
    path = versioned_dir / "prompt_template.txt"
    path.write_text(TEMPLATE_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def config(input_dir, template_path):
    return _Config(input_dir, template_path)


def _write_error_analysis(comparison_dir: Path, rows: list[dict]) -> None:
    comparison_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "fullname", "filename", "indicator_id", "indicator_value", "dimension",
        "golden_label", "pred_label", "value", "pred_value", "error_type",
    ]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row[c]) for c in columns))
    (comparison_dir / "error_analysis.csv").write_text("\n".join(lines), encoding="utf-8")


def _write_evaluation_report(evaluation_dir: Path, policy_file: str, results: dict) -> None:
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "policy_file": policy_file,
        "evaluated_at": "2026-01-01T00:00:00+00:00",
        "evaluation_results": results,
    }
    stem = Path(policy_file).stem
    (evaluation_dir / f"01012026000000-gpt-4o-{stem}.json").write_text(
        json.dumps(report), encoding="utf-8"
    )


def _write_candidates(rag_candidates_dir: Path, policy_file: str, candidates: dict) -> None:
    rag_candidates_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(policy_file).stem
    (rag_candidates_dir / f"{stem}.json").write_text(
        json.dumps({"policy_file": policy_file, "candidates": candidates}), encoding="utf-8"
    )


def _make_wrapper(complete_side_effect):
    wrapper = MagicMock()
    wrapper.profile.deployment = "gpt-5.2-diagnosis"
    wrapper.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    wrapper.complete_structured.side_effect = complete_side_effect
    return wrapper


DISCREPANCY_ROW = {
    "fullname": "Policy A",
    "filename": "A.txt",
    "indicator_id": "1.1",
    "indicator_value": "Domestic violence",
    "dimension": "scope_of_violence",
    "golden_label": "Yes",
    "pred_label": "No",
    "value": 1.0,
    "pred_value": 0.0,
    "error_type": "false_negative",
}


def test_missing_error_analysis_is_noop(config, data_root):
    wrapper = _make_wrapper(lambda *a, **k: {"diagnosis_results": []})
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    result = run_diagnosis_step(config=config, wrapper=wrapper, limiter=limiter)

    assert result["counts"]["discrepancies_total"] == 0
    assert result["counts"]["saved_reports"] == 0
    assert "skipped_reason" in result
    assert not (data_root / "results" / "diagnosis").exists()
    wrapper.complete_structured.assert_not_called()


def test_missing_rag_candidates_fails_only_that_policy(config, data_root):
    run_dir = data_root
    (config.paths.input_dir / "A.txt").write_text("Teks asli A.", encoding="utf-8")
    (config.paths.input_dir / "B.txt").write_text("Teks asli B.", encoding="utf-8")

    row_a = {**DISCREPANCY_ROW, "filename": "A.txt"}
    row_b = {**DISCREPANCY_ROW, "fullname": "Policy B", "filename": "B.txt"}
    _write_error_analysis(run_dir / "results" / "comparison", [row_a, row_b])
    _write_evaluation_report(run_dir / "results" / "evaluation", "A.txt", {"1.1": {"id": "1.1", "included": "No"}})
    _write_evaluation_report(run_dir / "results" / "evaluation", "B.txt", {"1.1": {"id": "1.1", "included": "No"}})
    # Only B has a rag_candidates file — A's is entirely missing.
    _write_candidates(
        run_dir / "mid_product" / "rag_candidates", "B.txt", {"1.1": [{"chunk_id": 0, "text": "evidence", "score": 0.5}]}
    )

    wrapper = _make_wrapper(
        lambda prompt, *_: {
            "diagnosis_results": [{"id": "1.1", "root_causes": ["rag_candidate_issue"], "rationale": "r"}]
        }
    )
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    result = run_diagnosis_step(config=config, wrapper=wrapper, limiter=limiter)

    assert result["failed_policies"] == ["A.txt"]
    assert result["counts"]["succeeded"] == 1
    assert result["counts"]["failed"] == 1
    assert not (run_dir / "results" / "diagnosis" / "A.json").exists()
    assert (run_dir / "results" / "diagnosis" / "B.json").exists()

    failures = json.loads((run_dir / "failures.json").read_text(encoding="utf-8"))
    assert failures["discrepancy_diagnosis"][0]["policy_file"] == "A.txt"


def test_unresolved_indicator_excluded_and_reported(config, data_root):
    run_dir = data_root
    (config.paths.input_dir / "C.txt").write_text("Teks asli C.", encoding="utf-8")

    row_resolvable = {**DISCREPANCY_ROW, "filename": "C.txt", "indicator_id": "1.1"}
    row_unresolved = {**DISCREPANCY_ROW, "filename": "C.txt", "indicator_id": "9.9", "indicator_value": "Unknown"}
    _write_error_analysis(run_dir / "results" / "comparison", [row_resolvable, row_unresolved])
    _write_evaluation_report(
        run_dir / "results" / "evaluation",
        "C.txt",
        {"1.1": {"id": "1.1", "included": "No"}, "9.9": {"id": "9.9", "included": "No"}},
    )
    # rag_candidates exists but only covers "1.1" — "9.9" has no entry at all.
    _write_candidates(
        run_dir / "mid_product" / "rag_candidates", "C.txt", {"1.1": [{"chunk_id": 0, "text": "evidence", "score": 0.5}]}
    )

    seen_prompts = []

    def complete_fn(prompt, *_):
        seen_prompts.append(prompt)
        return {"diagnosis_results": [{"id": "1.1", "root_causes": ["evaluation_failure"], "rationale": "r"}]}

    wrapper = _make_wrapper(complete_fn)
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    # Without --allow-partial: unresolved_indicators makes the whole policy incomplete.
    result = run_diagnosis_step(config=config, wrapper=wrapper, limiter=limiter, allow_partial=False)
    assert result["failed_policies"] == ["C.txt"]
    assert not (run_dir / "results" / "diagnosis" / "C.json").exists()
    # "9.9" must never reach the LLM prompt since it has no candidates.
    assert "9.9" not in seen_prompts[0]
    assert "1.1" in seen_prompts[0]


def test_partial_report_written_with_allow_partial(config, data_root):
    run_dir = data_root
    (config.paths.input_dir / "D.txt").write_text("Teks asli D.", encoding="utf-8")

    row_1 = {**DISCREPANCY_ROW, "filename": "D.txt", "indicator_id": "1.1"}
    row_2 = {**DISCREPANCY_ROW, "filename": "D.txt", "indicator_id": "1.2", "indicator_value": "Sexual violence"}
    _write_error_analysis(run_dir / "results" / "comparison", [row_1, row_2])
    _write_evaluation_report(
        run_dir / "results" / "evaluation",
        "D.txt",
        {"1.1": {"id": "1.1", "included": "No"}, "1.2": {"id": "1.2", "included": "No"}},
    )
    _write_candidates(
        run_dir / "mid_product" / "rag_candidates",
        "D.txt",
        {
            "1.1": [{"chunk_id": 0, "text": "evidence 1", "score": 0.5}],
            "1.2": [{"chunk_id": 1, "text": "evidence 2", "score": 0.4}],
        },
    )

    # LLM only returns a diagnosis for "1.1", omitting "1.2".
    wrapper = _make_wrapper(
        lambda prompt, *_: {
            "diagnosis_results": [{"id": "1.1", "root_causes": ["evaluation_failure"], "rationale": "r"}]
        }
    )
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    result = run_diagnosis_step(config=config, wrapper=wrapper, limiter=limiter, allow_partial=True)

    assert result["failed_policies"] == []
    assert result["counts"]["succeeded"] == 1
    report_path = run_dir / "results" / "diagnosis" / "D.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["missing_diagnoses"] == ["1.2"]
    assert report["diagnoses"]["1.2"]["diagnosis_status"] == "missing_from_llm_response"
    assert report["diagnoses"]["1.1"]["diagnosis_status"] == "ok"
    assert report["diagnoses"]["1.1"]["root_causes"] == ["evaluation_failure"]


def test_run_diagnosis_step_skips_existing_report_by_default(config, data_root):
    run_dir = data_root
    (config.paths.input_dir / "A.txt").write_text("Teks asli A.", encoding="utf-8")

    row = {**DISCREPANCY_ROW, "filename": "A.txt"}
    _write_error_analysis(run_dir / "results" / "comparison", [row])
    _write_evaluation_report(run_dir / "results" / "evaluation", "A.txt", {"1.1": {"id": "1.1", "included": "No"}})
    _write_candidates(
        run_dir / "mid_product" / "rag_candidates", "A.txt", {"1.1": [{"chunk_id": 0, "text": "evidence", "score": 0.5}]}
    )
    diagnosis_dir = run_dir / "results" / "diagnosis"
    diagnosis_dir.mkdir(parents=True)
    (diagnosis_dir / "A.json").write_text(json.dumps({"policy_file": "A.txt", "diagnoses": {}}), encoding="utf-8")

    wrapper = _make_wrapper(lambda *a, **k: {"diagnosis_results": []})
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    result = run_diagnosis_step(config=config, wrapper=wrapper, limiter=limiter)

    assert result["counts"]["skipped"] == 1
    assert result["counts"]["succeeded"] == 0
    wrapper.complete_structured.assert_not_called()


def test_run_diagnosis_step_force_reruns_existing(config, data_root):
    run_dir = data_root
    (config.paths.input_dir / "A.txt").write_text("Teks asli A.", encoding="utf-8")

    row = {**DISCREPANCY_ROW, "filename": "A.txt"}
    _write_error_analysis(run_dir / "results" / "comparison", [row])
    _write_evaluation_report(run_dir / "results" / "evaluation", "A.txt", {"1.1": {"id": "1.1", "included": "No"}})
    _write_candidates(
        run_dir / "mid_product" / "rag_candidates", "A.txt", {"1.1": [{"chunk_id": 0, "text": "evidence", "score": 0.5}]}
    )
    diagnosis_dir = run_dir / "results" / "diagnosis"
    diagnosis_dir.mkdir(parents=True)
    (diagnosis_dir / "A.json").write_text(json.dumps({"policy_file": "A.txt", "diagnoses": {}}), encoding="utf-8")

    wrapper = _make_wrapper(
        lambda prompt, *_: {
            "diagnosis_results": [{"id": "1.1", "root_causes": ["evaluation_failure"], "rationale": "r"}]
        }
    )
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    result = run_diagnosis_step(config=config, wrapper=wrapper, limiter=limiter, force=True)

    assert result["counts"]["skipped"] == 0
    assert result["counts"]["succeeded"] == 1
    wrapper.complete_structured.assert_called_once()
