import pytest

from automation.src.config_loader import parse_confidence_config
from automation.src.llm.confidence import (
    METHOD_LOGPROBS,
    METHOD_MARGIN,
    METHOD_VERBALIZED,
    SOURCE_UNALIGNED,
    SOURCE_UNAVAILABLE,
    ConfidenceSpec,
    attach_confidence,
    merge_scores,
)
from automation.src.llm.logprobs import AnswerConfidence


def answer(indicator_id="1.1", probability=0.9, margin=0.8) -> AnswerConfidence:
    return AnswerConfidence(
        indicator_id=indicator_id,
        answer="Yes",
        logprob=-0.1,
        probability=probability,
        p_yes=0.9,
        p_no=0.1,
        margin=margin,
        counterpart_observed=True,
        token_count=1,
    )


def items(*ids) -> dict[str, dict]:
    return {i: {"id": i, "included": "Yes"} for i in ids}


def test_every_enabled_method_is_recorded():
    spec = ConfidenceSpec(methods=(METHOD_LOGPROBS, METHOD_MARGIN), primary=METHOD_LOGPROBS)
    by_id = items("1.1")

    attach_confidence(by_id, [answer(probability=0.95, margin=0.6)], spec, logprobs_available=True)

    scores = by_id["1.1"]["confidence_scores"]
    assert scores == {METHOD_LOGPROBS: 0.95, METHOD_MARGIN: 0.6}
    assert by_id["1.1"]["confidence"] == 0.95
    assert by_id["1.1"]["confidence_source"] == METHOD_LOGPROBS


def test_primary_selects_which_score_leads():
    spec = ConfidenceSpec(methods=(METHOD_LOGPROBS, METHOD_MARGIN), primary=METHOD_MARGIN)
    by_id = items("1.1")

    attach_confidence(by_id, [answer(probability=0.95, margin=0.6)], spec, logprobs_available=True)

    assert by_id["1.1"]["confidence"] == 0.6
    assert by_id["1.1"]["confidence_source"] == METHOD_MARGIN
    # Both are still recorded, so the report can score them against each other.
    assert by_id["1.1"]["confidence_scores"][METHOD_LOGPROBS] == 0.95


def test_disabled_method_is_not_recorded():
    spec = ConfidenceSpec(methods=(METHOD_LOGPROBS,), primary=METHOD_LOGPROBS)
    by_id = items("1.1")

    attach_confidence(by_id, [answer()], spec, logprobs_available=True)

    assert METHOD_MARGIN not in by_id["1.1"]["confidence_scores"]


def test_verbalized_is_read_from_the_response_field():
    spec = ConfidenceSpec(
        methods=(METHOD_LOGPROBS, METHOD_VERBALIZED), primary=METHOD_VERBALIZED
    )
    by_id = {"1.1": {"id": "1.1", "included": "Yes", "self_reported_confidence": 0.7}}

    attach_confidence(by_id, [answer()], spec, logprobs_available=True)

    assert by_id["1.1"]["confidence"] == 0.7
    assert by_id["1.1"]["confidence_source"] == METHOD_VERBALIZED
    # The raw schema field is consumed, not left to duplicate the score.
    assert "self_reported_confidence" not in by_id["1.1"]


@pytest.mark.parametrize(
    "raw,expected",
    [(0.85, 0.85), (85, 0.85), (1, 1.0), (0, 0.0), (-1, None), ("x", None), (None, None)],
)
def test_verbalized_values_are_sanity_checked(raw, expected):
    spec = ConfidenceSpec(methods=(METHOD_VERBALIZED,), primary=METHOD_VERBALIZED)
    by_id = {"1.1": {"id": "1.1", "included": "Yes", "self_reported_confidence": raw}}

    attach_confidence(by_id, [], spec, logprobs_available=False)

    assert by_id["1.1"]["confidence"] == expected


def test_verbalized_works_without_logprobs_at_all():
    # A deployment that rejects logprobs can still report verbalized confidence.
    spec = ConfidenceSpec(methods=(METHOD_VERBALIZED,), primary=METHOD_VERBALIZED)
    by_id = {"1.1": {"id": "1.1", "included": "Yes", "self_reported_confidence": 0.6}}

    attach_confidence(by_id, [], spec, logprobs_available=False)

    assert by_id["1.1"]["confidence"] == 0.6
    assert by_id["1.1"]["answer_logprob"] is None


def test_unaligned_and_unavailable_are_distinguished():
    spec = ConfidenceSpec(methods=(METHOD_LOGPROBS,), primary=METHOD_LOGPROBS)

    partial = items("1.1", "1.2")
    attach_confidence(partial, [answer("1.1")], spec, logprobs_available=True)
    assert partial["1.2"]["confidence_source"] == SOURCE_UNALIGNED

    nothing = items("1.1")
    attach_confidence(nothing, [], spec, logprobs_available=False)
    assert nothing["1.1"]["confidence_source"] == SOURCE_UNAVAILABLE
    assert nothing["1.1"]["confidence_scores"] == {}


def test_merge_scores_labels_the_source_from_the_chosen_window():
    chosen = {
        "confidence": 0.8,
        "confidence_source": METHOD_MARGIN,
        "confidence_scores": {METHOD_MARGIN: 0.8},
        "answer_logprob": -0.2,
    }

    merged = merge_scores([chosen], chosen)

    assert merged["confidence"] == 0.8
    assert merged["confidence_source"] == "margin_merged"
    assert merged["confidence_scores"] == {METHOD_MARGIN: 0.8}


def test_merge_scores_without_any_scored_window():
    merged = merge_scores([{"confidence_source": SOURCE_UNAVAILABLE}], None)

    assert merged["confidence"] is None
    assert merged["confidence_source"] == SOURCE_UNAVAILABLE


def test_spec_reports_what_the_request_needs():
    logprob_only = ConfidenceSpec(methods=(METHOD_LOGPROBS,), primary=METHOD_LOGPROBS)
    assert logprob_only.needs_logprobs
    assert not logprob_only.needs_verbalized

    verbal_only = ConfidenceSpec(methods=(METHOD_VERBALIZED,), primary=METHOD_VERBALIZED)
    assert not verbal_only.needs_logprobs
    assert verbal_only.needs_verbalized

    margin_only = ConfidenceSpec(methods=(METHOD_MARGIN,), primary=METHOD_MARGIN)
    assert margin_only.needs_logprobs


def test_config_parses_methods_and_primary():
    spec = parse_confidence_config(
        {"enabled": True, "methods": ["margin", "verbalized"], "primary": "verbalized"}
    )

    assert spec.methods == (METHOD_MARGIN, METHOD_VERBALIZED)
    assert spec.primary == METHOD_VERBALIZED


def test_config_defaults_primary_to_the_first_method():
    assert parse_confidence_config({"methods": ["margin"]}).primary == METHOD_MARGIN


def test_config_rejects_unknown_and_inconsistent_settings():
    with pytest.raises(ValueError, match="Unknown"):
        parse_confidence_config({"methods": ["telepathy"]})

    with pytest.raises(ValueError, match="must be one of the enabled"):
        parse_confidence_config({"methods": ["margin"], "primary": "logprobs"})

    with pytest.raises(ValueError, match="top_logprobs"):
        parse_confidence_config({"top_logprobs": 50})


def test_config_with_no_methods_is_disabled():
    assert parse_confidence_config({"methods": []}).enabled is False
