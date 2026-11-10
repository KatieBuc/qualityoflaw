import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from automation.src.config_loader import DEFAULT_COVERAGE_LABELS, parse_confidence_config, parse_coverage_labels
from automation.src.confidence_report import build_confidence_report, build_level_report
from automation.src.evaluate_accuracy import coverage_level, reports_to_dataframe
from automation.src.llm.confidence import (
    ConfidenceSpec,
    attach_confidence,
    attach_coverage_confidence,
    normalize_coverage,
)
from automation.src.llm.logprobs import (
    canonical_coverage,
    classify_coverage_label,
    extract_answer_confidences,
    extract_coverage_confidences,
)
from automation.src.run_eval import (
    CoveragePolicyEvaluationResponse,
    PolicyEvaluationResponse,
    VerbalizedCoveragePolicyEvaluationResponse,
    VerbalizedPolicyEvaluationResponse,
    response_model_for,
)
from automation.tests.test_logprobs import FakeToken, FakeTopLogprob, choice_for

LABELS = ("Substantive", "Referenced only")


# --------------------------------------------------------------------------- config
def test_coverage_is_off_unless_enabled():
    assert parse_coverage_labels(None) == ()
    assert parse_coverage_labels({"enabled": False, "labels": ["A", "B"]}) == ()
    assert parse_confidence_config({}, None).has_coverage is False


def test_enabled_coverage_has_default_labels_and_the_first_is_the_highest_level():
    assert parse_coverage_labels({"enabled": True}) == DEFAULT_COVERAGE_LABELS == LABELS
    assert parse_confidence_config({}, {"enabled": True}).coverage_labels == LABELS
    assert parse_coverage_labels({"enabled": True, "labels": ["Full", "Partial", "Mention"]}) == ("Full", "Partial", "Mention")


@pytest.mark.parametrize("labels", [["only one"], ["Same", "same"], ["A", ""], "Substantive"])
def test_bad_coverage_labels_are_rejected(labels):
    with pytest.raises(ValueError):
        parse_coverage_labels({"enabled": True, "labels": labels})


# --------------------------------------------------------------------------- schema
def test_without_coverage_the_schema_is_exactly_what_it_was():
    off = ConfidenceSpec(methods=("logprobs",))
    assert response_model_for(off) is PolicyEvaluationResponse
    assert response_model_for(ConfidenceSpec(methods=("verbalized",), primary="verbalized")) is VerbalizedPolicyEvaluationResponse
    assert "coverage" not in PolicyEvaluationResponse.model_json_schema()["$defs"]["CriterionResult"]["properties"]


def test_with_coverage_the_schema_gains_the_coverage_fields_only_when_asked():
    plain = ConfidenceSpec(methods=("logprobs",), coverage_labels=LABELS)
    verbal = ConfidenceSpec(methods=("verbalized",), primary="verbalized", coverage_labels=LABELS)
    assert response_model_for(plain) is CoveragePolicyEvaluationResponse
    assert response_model_for(verbal) is VerbalizedCoveragePolicyEvaluationResponse
    props = VerbalizedCoveragePolicyEvaluationResponse.model_json_schema()["$defs"]["VerbalizedCoverageCriterionResult"]["properties"]
    assert {"included", "coverage", "self_reported_confidence", "coverage_self_reported_confidence"} <= set(props)
    # a No answer leaves them out
    parsed = VerbalizedCoveragePolicyEvaluationResponse(
        evaluation_results=[{"id": "1.1", "indicator": "x", "included": "No", "rationale": "r", "self_reported_confidence": 0.9}]
    )
    assert parsed.evaluation_results[0].coverage is None


# --------------------------------------------------------------------------- normalising the answer
def test_normalize_coverage_canonicalises_and_nulls_what_cannot_be_used():
    items = {
        "1.1": {"included": "Yes", "coverage": " substantive "},
        "1.2": {"included": "Yes", "coverage": "referenced ONLY"},
        "1.3": {"included": "Yes", "coverage": "somewhat"},
        "1.4": {"included": "Yes"},
        "1.5": {"included": "No", "coverage": "Substantive"},
    }

    unusable = normalize_coverage(items, LABELS)

    assert [items[k]["coverage"] for k in ("1.1", "1.2", "1.3", "1.4", "1.5")] == ["Substantive", "Referenced only", None, None, None]
    assert unusable == ["1.3", "1.4"]


def test_label_helpers():
    assert canonical_coverage("REFERENCED  only", LABELS) == "Referenced only"
    assert canonical_coverage("x", LABELS) is None and canonical_coverage(None, LABELS) is None
    assert classify_coverage_label('"Sub', LABELS) == "Substantive"
    assert classify_coverage_label("Ref", LABELS) == "Referenced only"
    assert classify_coverage_label("Su", ("Substantive", "Summary")) is None          # ambiguous prefix
    assert classify_coverage_label("", LABELS) is None and classify_coverage_label("zzz", LABELS) is None


# --------------------------------------------------------------------------- token confidence of the coverage value
def _tokens():
    return [
        FakeToken('{"evaluation_results":[{"id":"1.1","included":"', 0.0),
        FakeToken("Yes", -0.01, [FakeTopLogprob("Yes", -0.01), FakeTopLogprob("No", -4.6)]),
        FakeToken('","coverage":"', 0.0),
        FakeToken("Sub", -0.69, [FakeTopLogprob("Sub", -0.69), FakeTopLogprob("Ref", -0.69)]),   # a coin flip
        FakeToken("stantive", -0.01),
        FakeToken('"},{"id":"1.2","included":"', 0.0),
        FakeToken("No", -0.02, [FakeTopLogprob("No", -0.02), FakeTopLogprob("Yes", -4.0)]),
        FakeToken('","coverage":null},{"id":"1.3","included":"', 0.0),
        FakeToken("Yes", -0.05, [FakeTopLogprob("Yes", -0.05), FakeTopLogprob("No", -3.0)]),
        FakeToken('","coverage":"', 0.0),
        FakeToken("Ref", -0.01, [FakeTopLogprob("Ref", -0.01), FakeTopLogprob("Sub", -6.0)]),
        FakeToken("erenced only", -0.0),
        FakeToken('"}]}', 0.0),
    ]


def test_the_coverage_value_has_its_own_probability_and_margin_apart_from_yes_no():
    choice = choice_for(_tokens())

    yes_no = {c.indicator_id: c for c in extract_answer_confidences(choice)}
    coverage = {c.indicator_id: c for c in extract_coverage_confidences(choice, LABELS)}

    assert set(coverage) == {"1.1", "1.3"}                          # the "No" (coverage null) has none
    # 1.1: a sure Yes, a coin-flip coverage -- two different answers about the same indicator
    assert yes_no["1.1"].probability > 0.98
    assert math.isclose(coverage["1.1"].probability, math.exp(-0.69 - 0.01))
    assert coverage["1.1"].margin == pytest.approx(0.0, abs=1e-6)
    assert coverage["1.1"].p_by_label["Substantive"] == pytest.approx(0.5, abs=1e-6)
    # 1.3: a clear coverage
    assert coverage["1.3"].p_by_label["Referenced only"] > 0.99 and coverage["1.3"].margin > 0.98
    assert coverage["1.3"].counterpart_observed is True


def test_no_logprobs_gives_no_coverage_confidence():
    assert extract_coverage_confidences(choice_for([]), LABELS) == []
    assert extract_coverage_confidences(choice_for(_tokens()), ()) == []


# --------------------------------------------------------------------------- attaching the two confidences
def test_yes_no_and_coverage_confidence_are_attached_independently():
    spec = ConfidenceSpec(methods=("logprobs", "margin", "verbalized"), primary="logprobs", coverage_labels=LABELS)
    choice = choice_for(_tokens())
    items = {
        "1.1": {"included": "Yes", "coverage": "Substantive", "self_reported_confidence": 0.9, "coverage_self_reported_confidence": 0.4},
        "1.2": {"included": "No", "coverage": None, "self_reported_confidence": 0.8, "coverage_self_reported_confidence": None},
        "1.3": {"included": "Yes", "coverage": "Referenced only", "self_reported_confidence": 0.7, "coverage_self_reported_confidence": 0.95},
    }

    attach_confidence(items, extract_answer_confidences(choice), spec, logprobs_available=True)
    yes_no_before = {k: (v["confidence"], dict(v["confidence_scores"])) for k, v in items.items()}
    attach_coverage_confidence(items, extract_coverage_confidences(choice, LABELS), spec, logprobs_available=True)

    # attaching the coverage changed nothing of the Yes / No scores
    assert {k: (v["confidence"], dict(v["confidence_scores"])) for k, v in items.items()} == yes_no_before
    # 1.1: confident Yes (0.9 verbalised, ~0.99 logprob) but unsure of the coverage (0.4 verbalised, a 0 margin)
    assert items["1.1"]["confidence_scores"]["verbalized"] == 0.9
    assert items["1.1"]["coverage_confidence_scores"]["verbalized"] == 0.4
    assert items["1.1"]["coverage_confidence_scores"]["margin"] == pytest.approx(0.0, abs=1e-6)
    assert items["1.1"]["coverage_confidence"] == pytest.approx(math.exp(-0.70))          # logprobs is the primary
    assert items["1.3"]["coverage_confidence_scores"]["verbalized"] == 0.95
    # an indicator answered No has no coverage, hence nothing to be confident about
    assert items["1.2"]["coverage_confidence"] is None
    assert items["1.2"]["coverage_confidence_source"] == "not_applicable" and items["1.2"]["coverage_confidence_scores"] == {}
    assert "coverage_self_reported_confidence" not in items["1.1"]                          # consumed, like the Yes / No one


def test_a_verbalized_only_run_has_no_token_fields_for_the_coverage_either():
    spec = ConfidenceSpec(methods=("verbalized",), primary="verbalized", coverage_labels=LABELS)
    items = {"1.1": {"included": "Yes", "coverage": "Substantive", "coverage_self_reported_confidence": 80}}

    attach_coverage_confidence(items, [], spec, logprobs_available=False)

    assert items["1.1"]["coverage_confidence"] == pytest.approx(0.8)                       # 0-100 rescaled, as for Yes / No
    assert items["1.1"]["coverage_confidence_source"] == "verbalized"
    assert "coverage_margin" not in items["1.1"] and "coverage_answer_logprob" not in items["1.1"]


def test_missing_coverage_confidence_is_marked_not_guessed():
    spec = ConfidenceSpec(methods=("verbalized",), primary="verbalized", coverage_labels=LABELS)
    items = {"1.1": {"included": "Yes", "coverage": "Substantive"}}                       # the model gave no number

    attach_coverage_confidence(items, [], spec, logprobs_available=False)

    assert items["1.1"]["coverage_confidence"] is None and items["1.1"]["coverage_confidence_source"] == "unavailable"


# --------------------------------------------------------------------------- reports -> dataframe
def test_levels_are_ranked_from_the_first_label():
    assert coverage_level("No", None, LABELS) == 0.0
    assert coverage_level("Yes", "Referenced only", LABELS) == 1.0
    assert coverage_level("Yes", "Substantive", LABELS) == 2.0
    assert coverage_level("Yes", "substantive", LABELS) == 2.0
    assert math.isnan(coverage_level("Yes", None, LABELS)) and math.isnan(coverage_level("Yes", "other", LABELS))
    assert coverage_level("Yes", "Partial", ("Full", "Partial", "Mention")) == 2.0


def _report(policy, items):
    return (datetime.now(timezone.utc), f"{policy}.json", {"policy_file": policy, "evaluation_results": items})


def test_reports_without_coverage_produce_exactly_the_columns_they_always_did():
    df = reports_to_dataframe([_report("A.txt", {"1.1": {"id": "1.1", "included": "Yes", "evidence": [], "rationale": "r", "confidence": 0.9}})])
    assert not any(c.startswith(("pred_coverage", "pred_level", "coverage_", "covconf_")) for c in df.columns)
    assert df.loc[0, "pred_value"] == 1.0


def test_reports_with_coverage_add_level_and_coverage_confidence_columns():
    items = {
        "1.1": {"id": "1.1", "included": "Yes", "coverage": "Substantive", "confidence": 0.9, "coverage_confidence": 0.4,
                "coverage_confidence_scores": {"verbalized": 0.4}, "confidence_scores": {"verbalized": 0.9}},
        "1.2": {"id": "1.2", "included": "No", "coverage": None, "confidence": 0.8, "coverage_confidence": None},
    }

    df = reports_to_dataframe([_report("A.txt", items)]).set_index("indicator_id")

    assert df.loc["1.1", "pred_value"] == 1.0 and df.loc["1.1", "pred_level"] == 2.0
    assert df.loc["1.2", "pred_value"] == 0.0 and df.loc["1.2", "pred_level"] == 0.0
    assert df.loc["1.1", "coverage_confidence"] == 0.4 and df.loc["1.1", "confidence"] == 0.9    # independent columns
    assert df.loc["1.1", "covconf_verbalized"] == 0.4 and df.loc["1.1", "conf_verbalized"] == 0.9


# --------------------------------------------------------------------------- the report: Yes / No and level, apart
def _merged(rows):
    """rows: (golden_level, pred_level, confidence, coverage_confidence)"""
    records = []
    for i, (golden, pred, conf, cov_conf) in enumerate(rows):
        records.append({
            "filename": f"P{i}.txt", "indicator_id": "1.1", "value": float(golden > 0), "golden_level": float(golden),
            "pred_value": float(pred > 0), "pred_level": float(pred),
            "confidence": conf, "confidence_source": "verbalized", "conf_verbalized": conf,
            "coverage_confidence": cov_conf, "coverage_confidence_source": "verbalized", "covconf_verbalized": cov_conf,
        })
    return pd.DataFrame(records)


def test_the_level_section_is_scored_on_its_own_confidence_and_its_own_correctness():
    # The Yes / No answers are all right and sure of themselves; the levels are not.
    rows = [(2, 2, 0.95, 0.9), (2, 1, 0.95, 0.9), (1, 1, 0.95, 0.9), (1, 2, 0.95, 0.5), (2, 1, 0.95, 0.4), (0, 0, 0.95, None)]
    merged = _merged(rows)

    report = build_confidence_report(merged, primary_method="verbalized")

    assert report["baseline"]["accuracy"] == 1.0                              # Yes / No: every answer is right
    assert report["calibration"] is not None
    level = report["level"]
    assert level["status"] == "ok" and level["pairs"] == 5                      # the No / No pair cannot be judged
    assert level["baseline"]["accuracy"] == pytest.approx(2 / 5)
    assert level["baseline"]["by_error_type"] == {"correct": 2, "level_under": 2, "level_over": 1}
    # coverage confidence is what is calibrated, not the 0.95 of the Yes / No answers
    assert level["distribution"]["mean"] == pytest.approx(np.mean([0.9, 0.9, 0.9, 0.5, 0.4]))
    assert level["methods"]["verbalized"]["pairs_scored"] == 5
    # low coverage confidence goes with the wrong levels: it discriminates
    assert level["discrimination"] is not None


def test_a_run_without_coverage_gets_the_report_it_always_got():
    merged = _merged([(1, 1, 0.9, 0.9)]).drop(columns=["pred_level", "golden_level", "coverage_confidence", "covconf_verbalized"])
    assert "level" not in build_confidence_report(merged, primary_method="verbalized")
    assert build_level_report(merged) is None


def test_a_golden_without_levels_cannot_score_the_level_and_says_so():
    merged = _merged([(1, 2, 0.9, 0.8), (1, 1, 0.9, 0.7)]).drop(columns=["golden_level"])

    level = build_confidence_report(merged, primary_method="verbalized")["level"]

    assert level["status"] == "no_level_labels"


def test_when_nothing_is_above_no_there_is_no_level_to_score():
    level = build_level_report(_merged([(0, 0, 0.9, None), (0, 1, 0.9, 0.5)]), primary_method="verbalized")
    assert level["status"] == "no_pairs_above_no"
