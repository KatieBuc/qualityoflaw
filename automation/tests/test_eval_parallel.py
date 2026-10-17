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
from automation.src.rag.retriever import RetrievedChunk
from automation.src.criteria import EXPECTED_INDICATOR_IDS
from automation.src.run_eval import (
    DimensionResult,
    _evaluate_policies_flat,
    _merge_dimension_results,
    run_evaluation_step,
    validate_completeness,
)


def fake_flat(outcome_for, seen=None):
    """Stand-in for `_evaluate_policies_flat`: settle each policy with `outcome_for(policy_path)`
    (a `(report, candidates)` tuple or an exception), the way the real scheduler does."""

    def run(policy_files, *args):
        settle = args[-1]
        for path in policy_files:
            if seen is not None:
                seen.append(path.name)
            settle(path, outcome_for(path))

    return run


@pytest.fixture
def pipeline_config():
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
        evaluation_criteria_dir=AUTOMATION_ROOT / "prompts" / "quality_eval" / "v2",
        evaluation_template_path=AUTOMATION_ROOT / "prompts" / "quality_eval" / "v2" / "prompt_template.txt",
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


def _write_store(store_path: Path, chunks: list[str]) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(
        json.dumps(
            {
                "policy_file": store_path.stem + ".txt",
                "chunks": [
                    {
                        "chunk_id": i,
                        "section_id": 0,
                        "chunk_index": 0,
                        "type": "structural",
                        "text": text,
                        "embedding": [1.0, 0.0],
                    }
                    for i, text in enumerate(chunks)
                ],
            }
        ),
        encoding="utf-8",
    )


def test_merge_dimension_results_handles_failures():
    policy_path = Path("ACEH_BIREUEN.txt")
    results = [
        DimensionResult("01_scope_of_violence.txt", {"1.1": {"id": "1.1", "indicator": "x", "included": "Yes"}}),
        DimensionResult("02_institutional_mechanism.txt", {}, error="02_institutional_mechanism.txt: boom"),
    ]
    report, candidates_by_id = _merge_dimension_results(policy_path, "gpt-4o", "v3", results)
    assert report["completed_dimensions"] == ["01_scope_of_violence.txt"]
    assert report["failed_dimensions"] == ["02_institutional_mechanism.txt"]
    assert report["prompt_version"] == "v3"
    assert "1.1" in report["evaluation_results"]
    assert candidates_by_id == {}


def test_merge_dimension_results_reports_evidence_original_aligned_rate():
    policy_path = Path("ACEH_BIREUEN.txt")
    results = [
        DimensionResult(
            "01_scope_of_violence.txt",
            {
                "1.1": {"id": "1.1", "indicator": "x", "included": "Yes", "evidence_original_aligned": True},
                "1.2": {"id": "1.2", "indicator": "x", "included": "Yes", "evidence_original_aligned": True},
                "1.3": {"id": "1.3", "indicator": "x", "included": "Yes", "evidence_original_aligned": False},
                # Not counted: no original-language pairing resolved.
                "1.4": {"id": "1.4", "indicator": "x", "included": "No", "evidence_original_aligned": None},
                "1.5": {"id": "1.5", "indicator": "x", "included": "Yes"},
            },
        ),
    ]
    report, _ = _merge_dimension_results(policy_path, "gpt-4o", "v3", results)
    assert report["evidence_original_aligned_rate"] == {
        "sentence_aligned": 2,
        "resolved": 3,
        "rate": round(2 / 3, 4),
    }


def test_merge_dimension_results_evidence_original_aligned_rate_none_when_no_pairing():
    policy_path = Path("ACEH_BIREUEN.txt")
    results = [
        DimensionResult(
            "01_scope_of_violence.txt",
            {"1.1": {"id": "1.1", "indicator": "x", "included": "No", "evidence_original_aligned": None}},
        ),
    ]
    report, _ = _merge_dimension_results(policy_path, "gpt-4o", "v3", results)
    assert report["evidence_original_aligned_rate"] == {
        "sentence_aligned": 0,
        "resolved": 0,
        "rate": None,
    }


def test_validate_completeness_detects_missing():
    partial = {"1.1": {"id": "1.1", "included": "Yes"}}
    is_complete, issues = validate_completeness(partial)
    assert not is_complete
    assert any("Missing indicators" in issue for issue in issues)


def test_validate_completeness_passes_for_full_set():
    full = {indicator_id: {"id": indicator_id} for indicator_id in EXPECTED_INDICATOR_IDS}
    is_complete, issues = validate_completeness(full)
    assert is_complete
    assert issues == []


def test_flat_scheduler_merges_dimensions(tmp_path):
    import re

    from automation.src.criteria import CRITERIA_FILES, EXPECTED_INDICATOR_COUNT

    policy_path = Path("policy.txt")
    rag_store_dir = tmp_path / "rag_store"
    _write_store(rag_store_dir / "policy.json", ["Some evidence sentence about the policy."])

    embedder = MagicMock()
    embedder.embed_texts.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]

    def complete_fn(prompt: str) -> dict:
        # Extract the real indicator ids rendered into this dimension's prompt
        # (avoids relying on call order, which is not guaranteed across threads),
        # and cite the sole candidate's sole sentence tag for each ("{cid}-0.0",
        # since the store has exactly one chunk with chunk_id 0, one sentence).
        ids = re.findall(r"^- (\S+) \(", prompt, re.MULTILINE)
        return {
            "evaluation_results": {
                cid: {
                    "id": cid,
                    "indicator": f"ind-{cid}",
                    "included": "Yes",
                    "evidence": [f"{cid}-0.0"],
                    "rationale": "because",
                }
                for cid in ids
            }
        }

    limiter = ConcurrencyLimiter(max_workers=3, enabled=True)
    settled = {}
    _evaluate_policies_flat(
        [policy_path],
        AUTOMATION_ROOT / "prompts" / "quality_eval" / "v2",
        "{{CRITERIA_WITH_CANDIDATES}}",
        "gpt-4o",
        rag_store_dir,
        embedder,
        RetrievalConfig(
            top_k=10,
            hybrid_bm25=HybridBM25Config(enabled=False, rrf_k=60),
            reranker=RerankerConfig(enabled=False),
            evidence_verification_enabled=True,
        ),
        complete_fn,
        limiter,
        lambda path, outcome: settled.update({path: outcome}),
    )
    report, candidates_by_id = settled[policy_path]

    assert len(report["completed_dimensions"]) == len(CRITERIA_FILES)
    assert report["indicator_count"] == EXPECTED_INDICATOR_COUNT
    assert report["evidence_verification_failures"] == []
    assert all(item["evidence_verified"] is True for item in report["evaluation_results"].values())
    assert set(candidates_by_id.keys()) == set(report["evaluation_results"].keys())


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.config_loader.OUTPUT_DIR_OVERRIDE", tmp_path)
    return tmp_path


@patch("automation.src.run_eval.finalize_and_save_report")
def test_run_evaluation_step_parallel(mock_finalize, pipeline_config, data_root):
    policy_dir = data_root / "results" / "translation"
    policy_dir.mkdir(parents=True)
    (policy_dir / "A.txt").write_text("policy A", encoding="utf-8")
    (policy_dir / "B.txt").write_text("policy B", encoding="utf-8")

    outcomes = {
        "A.txt": (
            {"policy_file": "A.txt", "evaluation_results": {}, "failed_dimensions": [], "errors": []},
            {"1.1": [RetrievedChunk(chunk_id=0, text="Evidence for A.", score=0.9)]},
        ),
        "B.txt": ({"policy_file": "B.txt", "evaluation_results": {}, "failed_dimensions": [], "errors": []}, {}),
    }
    seen = []
    mock_finalize.return_value = (True, str(data_root / "results" / "evaluation" / "report.json"))

    wrapper = MagicMock()
    wrapper.profile.deployment = "gpt-4o"
    wrapper.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    embedder = MagicMock()
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    with patch("automation.src.run_eval._evaluate_policies_flat", fake_flat(lambda p: outcomes[p.name], seen)):
        result = run_evaluation_step(config=pipeline_config,
            wrapper=wrapper,
            embedder=embedder,
            limiter=limiter,
            small_scale=False,
            allow_partial=False,
        )

    assert result["counts"]["total"] == 2
    assert result["counts"]["succeeded"] == 2
    assert sorted(seen) == ["A.txt", "B.txt"]

    rag_candidates_dir = data_root / "mid_product" / "rag_candidates"
    a_candidates = json.loads((rag_candidates_dir / "A.json").read_text(encoding="utf-8"))
    assert a_candidates["policy_file"] == "A.txt"
    assert a_candidates["candidates"] == {
        "1.1": [{"chunk_id": 0, "text": "Evidence for A.", "score": 0.9}]
    }
    b_candidates = json.loads((rag_candidates_dir / "B.json").read_text(encoding="utf-8"))
    assert b_candidates["candidates"] == {}


def _wrapper():
    wrapper = MagicMock()
    wrapper.profile.deployment = "gpt-4o"
    wrapper.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return wrapper


def _dimension(criteria_file):
    from automation.src.run_eval import DimensionResult

    return DimensionResult(criteria_file=criteria_file, batch_evals={})


def _run(pipeline_config, **kwargs):
    return run_evaluation_step(
        config=pipeline_config,
        wrapper=_wrapper(),
        embedder=MagicMock(),
        limiter=ConcurrencyLimiter(max_workers=kwargs.pop("max_workers", 2), enabled=True),
        small_scale=False,
        allow_partial=False,
    )


def _policies(data_root, *names):
    policy_dir = data_root / "results" / "translation"
    policy_dir.mkdir(parents=True)
    for name in names:
        (policy_dir / f"{name}.txt").write_text(name, encoding="utf-8")


def _finalize_saving(data_root):
    def finalize(final_report, policy_path, output_folder, **kwargs):
        out = Path(output_folder) / f"{Path(policy_path).stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}", encoding="utf-8")
        return not final_report.get("failed_dimensions"), (str(out) if not final_report.get("failed_dimensions") else None)

    return finalize


def test_a_finished_policy_is_saved_while_the_others_are_still_running(pipeline_config, data_root):
    import threading

    _policies(data_root, "FAST", "SLOW")
    fast_saved = threading.Event()
    saved_while_slow_runs = []

    def evaluate_dimension(policy_path, criteria_file, *args, **kwargs):
        if policy_path.stem == "SLOW":
            saved_while_slow_runs.append(fast_saved.wait(timeout=10))     # SLOW goes on only after FAST is on disk
        return _dimension(criteria_file)

    save = _finalize_saving(data_root)

    def finalize(final_report, policy_path, output_folder, **kwargs):
        outcome = save(final_report, policy_path, output_folder, **kwargs)
        if Path(policy_path).stem == "FAST":
            fast_saved.set()
        return outcome

    with patch("automation.src.run_eval._evaluate_dimension", side_effect=evaluate_dimension), patch(
        "automation.src.run_eval.load_policy_store", return_value=[]
    ), patch("automation.src.run_eval.finalize_and_save_report", side_effect=finalize):
        result = _run(pipeline_config)

    assert saved_while_slow_runs and all(saved_while_slow_runs)
    assert result["counts"]["saved_reports"] == 2 and result["counts"]["failed"] == 0


def test_a_failed_policy_is_recorded_without_stopping_the_others(pipeline_config, data_root):
    _policies(data_root, "A", "B")

    def evaluate_dimension(policy_path, criteria_file, *args, **kwargs):
        if policy_path.stem == "A":
            raise RuntimeError("model unreachable")
        return _dimension(criteria_file)

    with patch("automation.src.run_eval._evaluate_dimension", side_effect=evaluate_dimension), patch(
        "automation.src.run_eval.load_policy_store", return_value=[]
    ), patch("automation.src.run_eval.finalize_and_save_report", side_effect=_finalize_saving(data_root)):
        result = _run(pipeline_config)

    assert result["failed_policies"] == ["A.txt"] and result["counts"]["saved_reports"] == 1


def test_a_policy_whose_store_cannot_be_loaded_fails_alone(pipeline_config, data_root):
    _policies(data_root, "A", "B")

    def load_store(path):
        if "A" in Path(path).name:
            raise FileNotFoundError("no store for A")
        return []

    with patch("automation.src.run_eval._evaluate_dimension", side_effect=lambda p, cf, *a, **k: _dimension(cf)), patch(
        "automation.src.run_eval.load_policy_store", side_effect=load_store
    ), patch("automation.src.run_eval.finalize_and_save_report", side_effect=_finalize_saving(data_root)):
        result = _run(pipeline_config)

    assert result["failed_policies"] == ["A.txt"] and result["counts"]["saved_reports"] == 1


def test_total_threads_never_exceed_max_workers_across_policies_and_dimensions(pipeline_config, data_root):
    """The old nested pools allowed max_workers ** 2 threads (the cause of the memory crash)."""
    import threading
    import time

    _policies(data_root, *[f"P{i}" for i in range(6)])
    running = peak = 0
    lock = threading.Lock()

    def evaluate_dimension(policy_path, criteria_file, *args, **kwargs):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.02)
        with lock:
            running -= 1
        return _dimension(criteria_file)

    with patch("automation.src.run_eval._evaluate_dimension", side_effect=evaluate_dimension), patch(
        "automation.src.run_eval.load_policy_store", return_value=[]
    ), patch("automation.src.run_eval.finalize_and_save_report", side_effect=_finalize_saving(data_root)):
        result = _run(pipeline_config, max_workers=3)

    assert result["counts"]["saved_reports"] == 6
    assert 1 < peak <= 3


def test_policies_finish_in_queue_order_and_each_store_is_loaded_once(pipeline_config, data_root):
    from automation.src.criteria import CRITERIA_FILES

    paths = [Path(f"{name}.txt") for name in ("A", "B", "C")]
    loads = []
    settled = []
    store_ids = {}

    def load_store(path):
        loads.append(Path(path).name)
        return [{"chunk": Path(path).name}]

    def evaluate_dimension(policy_path, criteria_file, criteria_folder, template_text, store_chunks, *args, **kwargs):
        store_ids.setdefault(policy_path.stem, set()).add(id(store_chunks))
        return _dimension(criteria_file)

    with patch("automation.src.run_eval._evaluate_dimension", side_effect=evaluate_dimension), patch(
        "automation.src.run_eval.load_policy_store", side_effect=load_store
    ):
        _evaluate_policies_flat(
            paths,
            AUTOMATION_ROOT / "prompts" / "quality_eval" / "v2",
            "",
            "gpt-4o",
            data_root / "rag_store",
            MagicMock(),
            pipeline_config.retrieval,
            lambda prompt: {},
            ConcurrencyLimiter(max_workers=1, enabled=True),      # one thread: the queue order is the finishing order
            lambda path, outcome: settled.append(path.stem),
        )

    assert settled == ["A", "B", "C"]
    assert len(loads) == 3                                        # one load per policy, not one per dimension
    assert all(len(ids) == 1 for ids in store_ids.values()) and len(CRITERIA_FILES) > 1
