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
from automation.src.criteria import list_criteria_files
from automation.src.run_eval import (
    DimensionResult,
    indicator_ids_in,
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


V2 = AUTOMATION_ROOT / "prompts" / "quality_eval" / "v2"
V2_IDS = indicator_ids_in(V2)


def test_the_expected_indicators_are_the_ones_in_the_prompt_used():
    assert len(V2_IDS) == 56
    assert len(indicator_ids_in(AUTOMATION_ROOT / "prompts" / "quality_eval" / "global_v1")) == 47


def test_completeness_follows_the_prompt_not_the_active_project():
    global_ids = indicator_ids_in(AUTOMATION_ROOT / "prompts" / "quality_eval" / "global_v1")
    answers = {i: {"id": i} for i in global_ids}
    assert validate_completeness(answers, global_ids) == (True, [])
    ok, issues = validate_completeness(answers, V2_IDS)                 # the same answers against the wrong prompt
    assert not ok and any("Missing" in i for i in issues) and any("Unexpected" in i for i in issues)


def test_validate_completeness_detects_missing():
    partial = {"1.1": {"id": "1.1", "included": "Yes"}}
    is_complete, issues = validate_completeness(partial, V2_IDS)
    assert not is_complete
    assert any("Missing indicators" in issue for issue in issues)


def test_validate_completeness_passes_for_full_set():
    full = {indicator_id: {"id": indicator_id} for indicator_id in V2_IDS}
    is_complete, issues = validate_completeness(full, V2_IDS)
    assert is_complete
    assert issues == []


def test_flat_scheduler_merges_dimensions(tmp_path):
    import re

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

    assert len(report["completed_dimensions"]) == len(list_criteria_files(V2))
    assert report["indicator_count"] == len(V2_IDS)
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
    assert all(len(ids) == 1 for ids in store_ids.values()) and len(list_criteria_files(V2)) > 1


# --------------------------------------------------------------------------- only the cited chunks are aligned
def _chunks(*ids):
    return [RetrievedChunk(chunk_id=i, text=f"Sentence of chunk {i}.", score=1.0 - i / 10, source_text=f"Kalimat {i}.") for i in ids]


def test_cited_candidates_keeps_only_what_a_yes_answer_cites():
    from automation.src.rag.prompt_builder import cited_candidates

    candidates = {"1.1": _chunks(1, 2, 3), "1.2": _chunks(1, 2), "1.3": _chunks(4)}
    batch = {
        "1.1": {"included": "Yes", "evidence": ["1.1-2.0", "[1.1-2.3]", " 1.1-3.1 "]},
        "1.2": {"included": "No", "evidence": ["1.2-1.0"]},                      # a No cites nothing that is shown
        "1.3": {"included": "Yes", "evidence": ["some transcribed text"]},       # not a tag: nothing to align
    }

    kept = cited_candidates(batch, candidates)

    assert {cid: [c.chunk_id for c in cs] for cid, cs in kept.items()} == {"1.1": [2, 3]}


def test_a_tag_only_cites_a_chunk_of_its_own_criterion():
    from automation.src.rag.prompt_builder import cited_candidates

    kept = cited_candidates(
        {"1.1": {"included": "Yes", "evidence": ["1.2-1.0"]}},                  # cites criterion 1.2's chunk 1
        {"1.1": _chunks(1), "1.2": _chunks(1)},
    )

    assert {cid: [c.chunk_id for c in cs] for cid, cs in kept.items()} == {"1.2": [1]}


def test_nothing_cited_means_nothing_to_align():
    from automation.src.rag.prompt_builder import cited_candidates

    assert cited_candidates({"1.1": {"included": "Yes", "evidence": None}, "1.2": {"included": "No"}}, {"1.1": _chunks(1)}) == {}


def _dimension_run(tmp_path, *, verification, cited_tags):
    """Run one dimension with 4 retrieved chunks per criterion; return the chunk ids that were aligned."""
    from automation.src.criteria import list_criteria_files
    from automation.src.run_eval import _evaluate_dimension

    folder = AUTOMATION_ROOT / "prompts" / "quality_eval" / "v2"
    criteria_file = list_criteria_files(folder)[0]
    store = [
        {"chunk_id": i, "section_id": 0, "chunk_index": i, "type": "pasal", "text": f"Evidence sentence {i}.",
         "source_text": f"Kalimat bukti {i}.", "embedding": [1.0, 0.0]}
        for i in range(4)
    ]
    embedder = MagicMock()
    embedder.embed_texts.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    aligned: list[int] = []

    def fake_align(candidate, method="vecalign"):
        aligned.append(candidate.chunk_id)
        return {0: (candidate.source_text, "sentence")}

    from automation.src.rag.prompt_builder import parse_criteria_lines

    first_id = parse_criteria_lines(str(folder / criteria_file))[0].id        # an indicator of THIS dimension

    def complete_fn(prompt):
        return {"evaluation_results": {first_id: {"id": first_id, "indicator": "x", "included": "Yes",
                                                  "evidence": [t.replace("ID", first_id) for t in cited_tags], "rationale": "r"}}}

    retrieval = RetrievalConfig(
        top_k=4, hybrid_bm25=HybridBM25Config(enabled=False, rrf_k=60), reranker=RerankerConfig(enabled=False),
        evidence_verification_enabled=verification,
    )
    with patch("automation.src.rag.prompt_builder.align_chunk_evidence", side_effect=fake_align):
        result = _evaluate_dimension(Path("p.txt"), criteria_file, folder, "{{CRITERIA_WITH_CANDIDATES}}", store,
                                     embedder, retrieval, complete_fn)
    return result, sorted(set(aligned))


def test_only_the_chunks_the_judge_cited_are_aligned(tmp_path):
    result, aligned = _dimension_run(tmp_path, verification=True, cited_tags=["ID-1.0"])

    assert aligned == [1]                                   # 4 chunks were retrieved for every indicator; 1 was cited
    item = next(iter(result.batch_evals.values()))
    assert item["evidence"] == "Evidence sentence 1." and item["evidence_original"] == "Kalimat bukti 1."
    assert item["evidence_verified"] is True and item["evidence_original_aligned"] is True


def test_no_alignment_at_all_when_evidence_verification_is_off(tmp_path):
    result, aligned = _dimension_run(tmp_path, verification=False, cited_tags=["ID-1.0"])

    assert aligned == []
    assert next(iter(result.batch_evals.values()))["evidence_original"] is None


def test_a_citation_that_does_not_resolve_costs_no_alignment(tmp_path):
    result, aligned = _dimension_run(tmp_path, verification=True, cited_tags=["ID-9.0"])

    assert aligned == []
    assert next(iter(result.batch_evals.values()))["evidence_verified"] is False


# --------------------------------------------------------------------------- a prompt too long for one call is split
def _rows_and_candidates(n_chunks=3, chunk_chars=4000):
    from automation.src.criteria import list_criteria_files
    from automation.src.rag.prompt_builder import parse_criteria_lines

    folder = AUTOMATION_ROOT / "prompts" / "quality_eval" / "v2"
    rows = parse_criteria_lines(str(folder / list_criteria_files(folder)[0]))
    candidates = {
        row.id: [
            RetrievedChunk(
                chunk_id=i,
                text=" ".join(f"Sentence {k} of chunk {i} for {row.id}." for k in range(chunk_chars // 40)),
                score=1.0 - i / 10,
            )
            for i in range(n_chunks)
        ]
        for row in rows
    }
    return rows, candidates


class _Recorder:
    """A fake judge: answers every indicator whose heading is in the prompt, and remembers the prompts."""

    def __init__(self, rows, refuse_if=None):
        self.rows, self.prompts, self.refuse_if = rows, [], refuse_if

    def __call__(self, prompt):
        from automation.src.llm.wrapper import ContextLengthExceeded

        self.prompts.append(prompt)
        if self.refuse_if and self.refuse_if(prompt):
            raise ContextLengthExceeded("too long", limit=272000, actual=413247, details={}, attempts=1)
        return {"evaluation_results": {r.id: {"id": r.id, "included": "No"} for r in self.rows if f"- {r.id} (" in prompt}}


def test_a_prompt_that_fits_is_sent_whole_and_unchanged():
    from automation.src.rag.prompt_builder import format_criteria_with_candidates
    from automation.src.run_eval import _judge_in_parts

    rows, candidates = _rows_and_candidates(n_chunks=1, chunk_chars=400)
    judge = _Recorder(rows)

    judged = _judge_in_parts(rows, candidates, "HEAD {{CRITERIA_WITH_CANDIDATES}} TAIL", judge, 200_000)

    assert judged.calls == 1 and judged.trim.trimmed is False
    assert judge.prompts == ["HEAD " + format_criteria_with_candidates(rows, candidates) + " TAIL"]   # byte-identical to before
    assert set(judged.batch_evals) == {r.id for r in rows}


def test_a_prompt_over_the_budget_is_split_and_nothing_is_dropped():
    from automation.src.rag.prompt_builder import CHARS_PER_TOKEN
    from automation.src.run_eval import _judge_in_parts

    rows, candidates = _rows_and_candidates()
    judge = _Recorder(rows)
    budget_tokens = 12_000                                              # one whole dimension is several times that

    judged = _judge_in_parts(rows, candidates, "{{CRITERIA_WITH_CANDIDATES}}", judge, budget_tokens)

    assert judged.calls == len(judge.prompts) > 1
    assert all(len(p) <= budget_tokens * CHARS_PER_TOKEN for p in judge.prompts)       # every part fits
    assert set(judged.batch_evals) == {r.id for r in rows}                             # every indicator answered once
    assert judged.trim.trimmed is False                                                # and none of its candidates cut
    for row in rows:                                                                   # all candidates of an indicator are in its part
        part = next(p for p in judge.prompts if f"- {row.id} (" in p)
        assert all(f"[{row.id}-{c.chunk_id}.0]" in part for c in candidates[row.id])
    sent = [r.id for p in judge.prompts for r in rows if f"- {r.id} (" in p]
    assert sorted(sent) == sorted(r.id for r in rows)                                  # an indicator is in exactly one part


def test_the_split_keeps_the_order_and_balances_the_parts():
    from automation.src.run_eval import _split_in_two

    class Row:
        def __init__(self, id):
            self.id = id

    rows = [Row(str(i)) for i in range(5)]
    left, right = _split_in_two(rows, {"0": 10, "1": 10, "2": 10, "3": 10, "4": 60})
    assert [r.id for r in left] == ["0", "1", "2", "3"] and [r.id for r in right] == ["4"]       # 40 vs 60
    left, right = _split_in_two(rows[:2], {"0": 1, "1": 1000})
    assert [r.id for r in left] == ["0"] and [r.id for r in right] == ["1"]                        # never an empty side


def test_a_prompt_the_model_refuses_is_split_even_when_the_estimate_said_it_fits():
    from automation.src.run_eval import _judge_in_parts

    rows, candidates = _rows_and_candidates(n_chunks=1, chunk_chars=400)
    judge = _Recorder(rows, refuse_if=lambda prompt: sum(f"- {r.id} (" in prompt for r in rows) > 3)

    judged = _judge_in_parts(rows, candidates, "{{CRITERIA_WITH_CANDIDATES}}", judge, 200_000)

    assert judged.calls >= 3 and set(judged.batch_evals) == {r.id for r in rows}
    accepted = [p for p in judge.prompts if sum(f"- {r.id} (" in p for r in rows) <= 3]
    assert len(accepted) == judged.calls                                                           # only the small parts were answered


def test_one_indicator_too_long_even_alone_has_its_candidates_trimmed_and_reported():
    from automation.src.rag.prompt_builder import CHARS_PER_TOKEN
    from automation.src.run_eval import _judge_in_parts

    rows, candidates = _rows_and_candidates(n_chunks=6, chunk_chars=6000)
    one = rows[:1]
    judge = _Recorder(one)

    judged = _judge_in_parts(one, {one[0].id: candidates[one[0].id]}, "{{CRITERIA_WITH_CANDIDATES}}", judge, 3_000)

    assert judged.calls == 1 and judged.trim.trimmed
    assert judged.trim.indicators == [one[0].id] and judged.trim.dropped_candidates + judged.trim.truncated_candidates > 0
    assert len(judge.prompts[0]) <= 3_000 * CHARS_PER_TOKEN


def test_a_single_indicator_the_model_keeps_refusing_gives_up_after_a_few_smaller_tries():
    from automation.src.llm.wrapper import ContextLengthExceeded
    from automation.src.run_eval import _MAX_SHRINKS, _judge_in_parts

    rows, candidates = _rows_and_candidates(n_chunks=2, chunk_chars=2000)
    one = rows[:1]
    judge = _Recorder(one, refuse_if=lambda prompt: True)

    with pytest.raises(ContextLengthExceeded):
        _judge_in_parts(one, {one[0].id: candidates[one[0].id]}, "{{CRITERIA_WITH_CANDIDATES}}", judge, 200_000)

    sizes = [len(p) for p in judge.prompts]
    assert len(sizes) == _MAX_SHRINKS + 1 and sizes == sorted(sizes, reverse=True) and sizes[-1] < sizes[0]


def test_the_report_says_which_dimensions_were_split_or_trimmed_and_only_then():
    from automation.src.run_eval import DimensionResult, _merge_dimension_results

    plain = DimensionResult(criteria_file="01_a.txt", batch_evals={"1.1": {"id": "1.1", "included": "No"}})
    split = DimensionResult(criteria_file="02_b.txt", batch_evals={"2.1": {"id": "2.1", "included": "No"}}, prompt_parts=2)
    trimmed = DimensionResult(criteria_file="03_c.txt", batch_evals={}, prompt_parts=1, prompt_trim={"dropped_candidates": 3})

    report, _ = _merge_dimension_results(Path("p.txt"), "gpt", "v", [plain, split, trimmed])
    assert report["split_dimensions"] == {"02_b.txt": 2} and report["prompt_trimmed"] == {"03_c.txt": {"dropped_candidates": 3}}

    report, _ = _merge_dimension_results(Path("p.txt"), "gpt", "v", [plain])
    assert "split_dimensions" not in report and "prompt_trimmed" not in report            # unchanged for ordinary runs


def test_evaluate_dimension_splits_when_the_budget_is_small(tmp_path):
    from automation.src.criteria import list_criteria_files
    from automation.src.run_eval import _evaluate_dimension

    folder = AUTOMATION_ROOT / "prompts" / "quality_eval" / "v2"
    criteria_file = list_criteria_files(folder)[0]
    store = [
        {"chunk_id": i, "section_id": 0, "chunk_index": i, "type": "pasal",
         "text": " ".join(f"Evidence sentence {k} of chunk {i}." for k in range(150)), "source_text": None, "embedding": [1.0, 0.0]}
        for i in range(4)
    ]
    embedder = MagicMock()
    embedder.embed_texts.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    rows, _ = _rows_and_candidates(n_chunks=1, chunk_chars=40)
    judge = _Recorder(rows)
    retrieval = RetrievalConfig(
        top_k=4, hybrid_bm25=HybridBM25Config(enabled=False, rrf_k=60), reranker=RerankerConfig(enabled=False),
        evidence_verification_enabled=False, max_prompt_tokens=12_000,
    )

    result = _evaluate_dimension(Path("p.txt"), criteria_file, folder, "{{CRITERIA_WITH_CANDIDATES}}", store, embedder, retrieval, judge)

    assert result.error is None and result.prompt_parts == len(judge.prompts) > 1
    assert set(result.batch_evals) == {r.id for r in rows}


def test_evaluation_stops_before_any_model_call_when_the_criteria_files_are_malformed(pipeline_config, data_root, tmp_path):
    import dataclasses

    bad = tmp_path / "bad_prompt"
    bad.mkdir()
    (bad / "01_dim.txt").write_text("1.1 | A | Q?\n5. Not a criterion | but it looks like one\n", encoding="utf-8")
    (bad / "prompt_template.txt").write_text("{{CRITERIA_WITH_CANDIDATES}}", encoding="utf-8")
    config = dataclasses.replace(pipeline_config, evaluation_criteria_dir=bad, evaluation_template_path=bad / "prompt_template.txt")
    policy_dir = data_root / "results" / "translation"
    policy_dir.mkdir(parents=True)
    (policy_dir / "A.txt").write_text("policy A", encoding="utf-8")
    wrapper = MagicMock()
    wrapper.profile.deployment = "gpt-4o"

    with pytest.raises(ValueError, match="malformed"):
        run_evaluation_step(config=config, wrapper=wrapper, embedder=MagicMock(),
                            limiter=ConcurrencyLimiter(max_workers=2, enabled=True), small_scale=False, allow_partial=False)

    wrapper.complete_structured.assert_not_called()
