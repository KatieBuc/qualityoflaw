from pathlib import Path

from automation.src.rag.retriever import RetrievedChunk
from automation.src.run_eval_sliding_window import (
    _evaluate_dimension_sliding_window,
    _merge_window_results,
    _resolve_window_evidence,
)


def test_resolve_window_evidence_marks_no_as_unverified_none():
    batch_evals = {"1.1": {"included": "No", "evidence": None, "rationale": "not present"}}
    _resolve_window_evidence(batch_evals, candidate_lookup={})
    assert batch_evals["1.1"]["evidence_verified"] is None


def test_resolve_window_evidence_resolves_known_citation():
    batch_evals = {
        "1.1": {"included": "Yes", "evidence": ["0.0"], "rationale": "stated"}
    }
    _resolve_window_evidence(batch_evals, candidate_lookup={"0.0": "The policy covers DV."})
    assert batch_evals["1.1"]["evidence"] == ["The policy covers DV."]
    assert batch_evals["1.1"]["evidence_verified"] is True


def test_resolve_window_evidence_nulls_unresolvable_citation():
    batch_evals = {
        "1.1": {"included": "Yes", "evidence": ["bogus-tag"], "rationale": "stated"}
    }
    _resolve_window_evidence(batch_evals, candidate_lookup={"0.0": "text"})
    assert batch_evals["1.1"]["evidence"] is None
    assert batch_evals["1.1"]["evidence_verified"] is False
    assert "unresolved evidence citation removed" in batch_evals["1.1"]["rationale"]


def test_merge_yes_if_any_window_says_yes():
    window_results = [
        {"1.1": {"id": "1.1", "indicator": "DV", "included": "No", "evidence": None, "rationale": "absent here"}},
        {"1.1": {"id": "1.1", "indicator": "DV", "included": "Yes", "evidence": ["The policy covers DV."], "rationale": "found here"}},
    ]
    merged = _merge_window_results(["1.1"], window_results)
    assert merged["1.1"]["included"] == "Yes"
    assert merged["1.1"]["evidence"] == "The policy covers DV."
    assert merged["1.1"]["evidence_verified"] is True


def test_merge_evidence_is_deduplicated_union():
    window_results = [
        {"1.1": {"id": "1.1", "indicator": "DV", "included": "Yes", "evidence": ["Sentence A."], "rationale": "r1"}},
        {"1.1": {"id": "1.1", "indicator": "DV", "included": "Yes", "evidence": ["Sentence A.", "Sentence B."], "rationale": "r2"}},
    ]
    merged = _merge_window_results(["1.1"], window_results)
    assert merged["1.1"]["evidence"] == "Sentence A.\nSentence B."
    assert merged["1.1"]["rationale"] == "r1 | r2"


def test_merge_all_no_produces_no_with_evidence_verified_none():
    window_results = [
        {"1.1": {"id": "1.1", "indicator": "DV", "included": "No", "evidence": None, "rationale": "absent"}},
        {"1.1": {"id": "1.1", "indicator": "DV", "included": "No", "evidence": None, "rationale": "still absent"}},
    ]
    merged = _merge_window_results(["1.1"], window_results)
    assert merged["1.1"]["included"] == "No"
    assert merged["1.1"]["evidence"] is None
    assert merged["1.1"]["evidence_verified"] is None
    assert merged["1.1"]["rationale"] == "absent"


def test_merge_yes_but_unresolved_evidence_marks_verified_false():
    window_results = [
        {"1.1": {"id": "1.1", "indicator": "DV", "included": "Yes", "evidence": None, "rationale": "citation failed"}},
    ]
    merged = _merge_window_results(["1.1"], window_results)
    assert merged["1.1"]["included"] == "Yes"
    assert merged["1.1"]["evidence"] is None
    assert merged["1.1"]["evidence_verified"] is False


def test_merge_missing_indicator_in_all_windows_defaults_to_no():
    merged = _merge_window_results(["9.9"], [{}, {}])
    assert merged["9.9"]["included"] == "No"
    assert merged["9.9"]["evidence_verified"] is None


def test_evaluate_dimension_sliding_window_merges_across_windows(tmp_path):
    criteria_dir = tmp_path / "criteria"
    criteria_dir.mkdir()
    (criteria_dir / "01_scope.txt").write_text(
        "1.1 | Domestic violence | does it cover DV?\n", encoding="utf-8"
    )

    windows = [
        RetrievedChunk(chunk_id=0, text="Budget allocation for roads.", score=0.0),
        RetrievedChunk(chunk_id=1, text="The policy explicitly covers domestic violence.", score=0.0),
    ]

    calls = []

    def complete_fn(prompt: str) -> dict:
        calls.append(prompt)
        # Both windows' prompts mention "domestic violence" in the criterion
        # header, so key off the window's own sentence text instead.
        if "explicitly covers domestic violence" in prompt.lower():
            return {
                "evaluation_results": {
                    "1.1": {
                        "id": "1.1",
                        "indicator": "Domestic violence",
                        "included": "Yes",
                        "evidence": ["1.0"],
                        "rationale": "stated directly",
                    }
                }
            }
        return {
            "evaluation_results": {
                "1.1": {
                    "id": "1.1",
                    "indicator": "Domestic violence",
                    "included": "No",
                    "evidence": None,
                    "rationale": "not addressed",
                }
            }
        }

    result = _evaluate_dimension_sliding_window(
        policy_path=Path("policy.txt"),
        criteria_file="01_scope.txt",
        criteria_folder=criteria_dir,
        template_text="{{WINDOW_SENTENCES}}\n---\n{{CRITERIA_LIST}}",
        windows=windows,
        complete_fn=complete_fn,
    )

    assert result.error is None
    assert len(calls) == 2  # one call per window
    item = result.batch_evals["1.1"]
    assert item["included"] == "Yes"
    assert item["evidence"] == "The policy explicitly covers domestic violence."
    assert item["evidence_verified"] is True
    assert result.candidates_by_id["1.1"] == windows


def test_window_sentences_appear_exactly_once_regardless_of_indicator_count(tmp_path):
    """Regression test for the duplication bug: a window's sentences must be
    shown once per prompt, not once per indicator in the dimension."""
    criteria_dir = tmp_path / "criteria"
    criteria_dir.mkdir()
    (criteria_dir / "01_scope.txt").write_text(
        "1.1 | Domestic violence | does it cover DV?\n"
        "1.2 | Sexual violence | does it cover SV?\n"
        "1.3 | Physical violence | does it cover PV?\n",
        encoding="utf-8",
    )

    windows = [
        RetrievedChunk(
            chunk_id=0, text="This unique sentence marker appears in the window.", score=0.0
        )
    ]

    captured_prompts = []

    def complete_fn(prompt: str) -> dict:
        captured_prompts.append(prompt)
        return {
            "evaluation_results": {
                cid: {"id": cid, "indicator": "x", "included": "No", "evidence": None, "rationale": "r"}
                for cid in ("1.1", "1.2", "1.3")
            }
        }

    _evaluate_dimension_sliding_window(
        policy_path=Path("policy.txt"),
        criteria_file="01_scope.txt",
        criteria_folder=criteria_dir,
        template_text="{{WINDOW_SENTENCES}}\n---\n{{CRITERIA_LIST}}",
        windows=windows,
        complete_fn=complete_fn,
    )

    assert len(captured_prompts) == 1  # one call per window, regardless of indicator count
    occurrences = captured_prompts[0].count("This unique sentence marker appears in the window.")
    assert occurrences == 1
    # But all three criteria are still listed for the LLM to judge against it.
    assert "1.1" in captured_prompts[0]
    assert "1.2" in captured_prompts[0]
    assert "1.3" in captured_prompts[0]
