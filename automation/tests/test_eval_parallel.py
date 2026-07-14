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
from automation.src.run_eval import DimensionResult, _evaluate_policy_parallel, _merge_dimension_results, run_evaluation_step


@pytest.fixture
def pipeline_config():
    translate_model = ModelProfile(name="test-translate", deployment="gpt-5.2", temperature=0.2)
    eval_model = ModelProfile(name="test-eval", deployment="gpt-4o", temperature=0.1)
    diagnosis_model = ModelProfile(name="test-diagnosis", deployment="gpt-5.2", temperature=0.2)
    return ResolvedPipelineConfig(
        experiment_name="test",
        translation_model=translate_model,
        evaluation_model=eval_model,
        discrepancy_diagnosis_model=diagnosis_model,
        translation_prompt_path=AUTOMATION_ROOT / "prompts" / "translation" / "v1" / "prompt.txt",
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
    report, candidates_by_id = _merge_dimension_results(policy_path, "gpt-4o", results)
    assert report["completed_dimensions"] == ["01_scope_of_violence.txt"]
    assert report["failed_dimensions"] == ["02_institutional_mechanism.txt"]
    assert "1.1" in report["evaluation_results"]
    assert candidates_by_id == {}


def test_evaluate_policy_parallel_merges_dimensions(tmp_path):
    import re

    from quality_eval.v1.criteria import CRITERIA_FILES, EXPECTED_INDICATOR_COUNT

    policy_path = Path("policy.txt")
    rag_store_dir = tmp_path / "rag_store"
    _write_store(rag_store_dir / "policy.json", ["Some evidence sentence about the policy."])

    embedder = MagicMock()
    embedder.embed_texts.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]

    def complete_fn(prompt: str) -> dict:
        # Extract the real indicator ids rendered into this dimension's prompt
        # (avoids relying on call order, which is not guaranteed across threads),
        # and cite the sole candidate's tag for each ("{cid}-0", since the
        # store has exactly one chunk with chunk_id 0).
        ids = re.findall(r"^- (\S+) \(", prompt, re.MULTILINE)
        return {
            "evaluation_results": {
                cid: {
                    "id": cid,
                    "indicator": f"ind-{cid}",
                    "included": "Yes",
                    "evidence": f"{cid}-0",
                    "rationale": "because",
                }
                for cid in ids
            }
        }

    limiter = ConcurrencyLimiter(max_workers=3, enabled=True)
    report, candidates_by_id = _evaluate_policy_parallel(
        policy_path=policy_path,
        criteria_folder=AUTOMATION_ROOT / "prompts" / "quality_eval" / "v2",
        template_text="{{CRITERIA_WITH_CANDIDATES}}",
        deployment_name="gpt-4o",
        rag_store_dir=rag_store_dir,
        embedder=embedder,
        retrieval_config=RetrievalConfig(
            top_k=10,
            hybrid_bm25=HybridBM25Config(enabled=False, rrf_k=60),
            reranker=RerankerConfig(enabled=False),
            evidence_verification_enabled=True,
        ),
        complete_fn=complete_fn,
        limiter=limiter,
    )

    assert len(report["completed_dimensions"]) == len(CRITERIA_FILES)
    assert report["indicator_count"] == EXPECTED_INDICATOR_COUNT
    assert report["evidence_verification_failures"] == []
    assert all(item["evidence_verified"] is True for item in report["evaluation_results"].values())
    assert set(candidates_by_id.keys()) == set(report["evaluation_results"].keys())


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.metadata.DEFAULT_DATA_ROOT", tmp_path)
    monkeypatch.setattr("automation.src.config_loader.DEFAULT_DATA_ROOT", tmp_path)
    return tmp_path


@patch("automation.src.run_eval.finalize_and_save_report")
@patch("automation.src.run_eval._evaluate_policy_parallel")
def test_run_evaluation_step_parallel(mock_eval_policy, mock_finalize, pipeline_config, data_root):
    run_id = "eval_parallel"
    policy_dir = data_root / run_id / "translation"
    policy_dir.mkdir(parents=True)
    (policy_dir / "A.txt").write_text("policy A", encoding="utf-8")
    (policy_dir / "B.txt").write_text("policy B", encoding="utf-8")

    mock_eval_policy.side_effect = [
        (
            {"policy_file": "A.txt", "evaluation_results": {}, "failed_dimensions": [], "errors": []},
            {"1.1": [RetrievedChunk(chunk_id=0, text="Evidence for A.", score=0.9)]},
        ),
        ({"policy_file": "B.txt", "evaluation_results": {}, "failed_dimensions": [], "errors": []}, {}),
    ]
    mock_finalize.return_value = (True, str(data_root / run_id / "evaluation" / "report.json"))

    wrapper = MagicMock()
    wrapper.profile.deployment = "gpt-4o"
    wrapper.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    embedder = MagicMock()
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    result = run_evaluation_step(
        run_id=run_id,
        config=pipeline_config,
        wrapper=wrapper,
        embedder=embedder,
        limiter=limiter,
        small_scale=False,
        allow_partial=False,
    )

    assert result["counts"]["total"] == 2
    assert result["counts"]["succeeded"] == 2
    assert mock_eval_policy.call_count == 2

    rag_candidates_dir = data_root / run_id / "rag_candidates"
    a_candidates = json.loads((rag_candidates_dir / "A.json").read_text(encoding="utf-8"))
    assert a_candidates["policy_file"] == "A.txt"
    assert a_candidates["candidates"] == {
        "1.1": [{"chunk_id": 0, "text": "Evidence for A.", "score": 0.9}]
    }
    b_candidates = json.loads((rag_candidates_dir / "B.json").read_text(encoding="utf-8"))
    assert b_candidates["candidates"] == {}
