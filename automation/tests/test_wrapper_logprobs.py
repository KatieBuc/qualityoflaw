import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List

import pytest
from pydantic import BaseModel

from automation.src.llm.confidence import (
    METHOD_LOGPROBS,
    METHOD_MARGIN,
    METHOD_VERBALIZED,
    ConfidenceSpec,
)
from automation.src.llm.model_profile import ModelProfile
from automation.src.llm.wrapper import (
    AzureLLMWrapper,
    is_logprobs_unsupported,
    top_logprobs_limit,
)


class CriterionResult(BaseModel):
    id: str
    included: str


class PolicyEvaluationResponse(BaseModel):
    evaluation_results: List[CriterionResult]


@dataclass
class FakeToken:
    token: str
    logprob: float = 0.0
    top_logprobs: list = field(default_factory=list)


class UnsupportedParameterError(Exception):
    """Mirrors the shape of an openai 400 for an unsupported parameter."""

    def __init__(self):
        super().__init__("Unsupported parameter: 'logprobs' is not supported with this model.")
        self.status_code = 400
        self.message = "Unsupported parameter: 'logprobs' is not supported with this model."


def build_response(*, with_logprobs: bool):
    parsed = PolicyEvaluationResponse(
        evaluation_results=[
            CriterionResult(id="1.1", included="Yes"),
            CriterionResult(id="1.2", included="No"),
        ]
    )
    tokens = [
        FakeToken(token='{"id":"1.1","included":"'),
        FakeToken(token="Yes", logprob=-0.01),
        FakeToken(token='"},{"id":"1.2","included":"'),
        FakeToken(token="No", logprob=-0.5),
        FakeToken(token='"}'),
    ]
    logprobs = SimpleNamespace(content=tokens) if with_logprobs else None
    choice = SimpleNamespace(message=SimpleNamespace(parsed=parsed), logprobs=logprobs)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return SimpleNamespace(choices=[choice], usage=usage)


class RecordingClient:
    """Records every parse() call, optionally rejecting the logprobs param."""

    def __init__(self, *, reject_logprobs: bool):
        self.reject_logprobs = reject_logprobs
        self.calls: list[dict] = []
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=self))

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("logprobs") and self.reject_logprobs:
            raise UnsupportedParameterError()
        return build_response(with_logprobs=bool(kwargs.get("logprobs")))


LOGPROB_SPEC = ConfidenceSpec(
    methods=(METHOD_LOGPROBS, METHOD_MARGIN), primary=METHOD_LOGPROBS, top_logprobs=5
)


def make_wrapper(client, **kwargs) -> AzureLLMWrapper:
    profile = ModelProfile(name="test", deployment="gpt-5.2", temperature=0.2, max_retries=3)
    return AzureLLMWrapper(profile=profile, client=client, api_style="chat", **kwargs)


def test_is_logprobs_unsupported_only_matches_relevant_400s():
    assert is_logprobs_unsupported(UnsupportedParameterError())

    other_400 = UnsupportedParameterError()
    other_400.message = "Invalid content filter result"
    assert not is_logprobs_unsupported(other_400)

    server_error = UnsupportedParameterError()
    server_error.status_code = 500
    assert not is_logprobs_unsupported(server_error)


def test_confidence_attached_when_logprobs_supported():
    client = RecordingClient(reject_logprobs=False)
    wrapper = make_wrapper(client, confidence=LOGPROB_SPEC)

    result = wrapper.complete_structured("prompt", PolicyEvaluationResponse)

    assert client.calls[0]["logprobs"] is True
    assert client.calls[0]["top_logprobs"] == 5
    results = result["evaluation_results"]
    assert math.isclose(results["1.1"]["confidence"], math.exp(-0.01))
    assert results["1.2"]["confidence_source"] == "logprobs"


def test_unsupported_logprobs_retries_once_without_them_and_stays_disabled():
    client = RecordingClient(reject_logprobs=True)
    wrapper = make_wrapper(client, confidence=LOGPROB_SPEC)

    first = wrapper.complete_structured("prompt", PolicyEvaluationResponse)

    # One rejected call, then an immediate retry without the parameter — the
    # request never reaches _call_with_retry, so no attempts are burned.
    assert len(client.calls) == 2
    assert client.calls[0].get("logprobs") is True
    assert "logprobs" not in client.calls[1]
    assert first["evaluation_results"]["1.1"]["confidence"] is None
    assert first["evaluation_results"]["1.1"]["confidence_source"] == "unavailable"

    wrapper.complete_structured("prompt", PolicyEvaluationResponse)

    assert not wrapper.logprobs_active()
    assert "logprobs" not in client.calls[2]
    assert len(client.calls) == 3


def test_capture_disabled_leaves_items_untouched():
    client = RecordingClient(reject_logprobs=False)
    wrapper = make_wrapper(client)

    result = wrapper.complete_structured("prompt", PolicyEvaluationResponse)

    assert "logprobs" not in client.calls[0]
    assert "confidence" not in result["evaluation_results"]["1.1"]


def test_profile_opt_out_skips_logprobs_entirely():
    client = RecordingClient(reject_logprobs=True)
    profile = ModelProfile(
        name="test", deployment="gpt-5.2", temperature=0.2, supports_logprobs=False
    )
    wrapper = AzureLLMWrapper(
        profile=profile, client=client, api_style="chat", confidence=LOGPROB_SPEC
    )

    wrapper.complete_structured("prompt", PolicyEvaluationResponse)

    assert "logprobs" not in client.calls[0]
    assert len(client.calls) == 1


class TopLogprobsRangeError(Exception):
    """Azure's 400 when more alternatives are requested than it allows."""

    def __init__(self, limit: int = 5):
        message = f"Invalid value for 'top_logprobs': must be less than or equal to {limit}."
        super().__init__(message)
        self.status_code = 400
        self.message = message


def test_top_logprobs_limit_is_parsed_from_the_error():
    assert top_logprobs_limit(TopLogprobsRangeError(5)) == 5
    assert top_logprobs_limit(TopLogprobsRangeError(3)) == 3
    # An unsupported-parameter 400 is a different problem with a different fix.
    assert top_logprobs_limit(UnsupportedParameterError()) is None


def test_too_many_alternatives_retries_with_the_cap_not_without_logprobs():
    class CappingClient(RecordingClient):
        def parse(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("top_logprobs", 0) > 5:
                raise TopLogprobsRangeError(5)
            return build_response(with_logprobs=bool(kwargs.get("logprobs")))

    client = CappingClient(reject_logprobs=False)
    spec = ConfidenceSpec(
        methods=(METHOD_LOGPROBS, METHOD_MARGIN), primary=METHOD_LOGPROBS, top_logprobs=20
    )
    wrapper = make_wrapper(client, confidence=spec)

    result = wrapper.complete_structured("prompt", PolicyEvaluationResponse)

    assert client.calls[0]["top_logprobs"] == 20
    assert client.calls[1]["top_logprobs"] == 5
    # Crucially, logprobs stayed on — confidence survives a merely-too-high value.
    assert client.calls[1]["logprobs"] is True
    assert result["evaluation_results"]["1.1"]["confidence"] is not None
    assert wrapper.logprobs_active()

    wrapper.complete_structured("prompt", PolicyEvaluationResponse)

    # The cap sticks, so later calls don't repeat the rejected request.
    assert client.calls[2]["top_logprobs"] == 5
    assert len(client.calls) == 3


def test_range_error_that_cannot_be_fixed_by_capping_propagates():
    class AlwaysRejectingClient(RecordingClient):
        def parse(self, **kwargs):
            self.calls.append(kwargs)
            raise TopLogprobsRangeError(5)

    client = AlwaysRejectingClient(reject_logprobs=False)
    spec = ConfidenceSpec(
        methods=(METHOD_LOGPROBS,), primary=METHOD_LOGPROBS, top_logprobs=5
    )
    wrapper = make_wrapper(client, confidence=spec)

    with pytest.raises(Exception):
        wrapper.complete_structured("prompt", PolicyEvaluationResponse)


def test_verbalized_score_flows_from_the_response_schema():
    class VerbalizedCriterionResult(CriterionResult):
        self_reported_confidence: float

    class VerbalizedResponse(BaseModel):
        evaluation_results: List[VerbalizedCriterionResult]

    class VerbalizedClient(RecordingClient):
        def parse(self, **kwargs):
            self.calls.append(kwargs)
            parsed = VerbalizedResponse(
                evaluation_results=[
                    VerbalizedCriterionResult(
                        id="1.1", included="Yes", self_reported_confidence=0.65
                    )
                ]
            )
            choice = SimpleNamespace(
                message=SimpleNamespace(parsed=parsed),
                logprobs=SimpleNamespace(
                    content=[
                        FakeToken(token='{"id":"1.1","included":"'),
                        FakeToken(token="Yes", logprob=-0.01),
                        FakeToken(token='"}'),
                    ]
                ),
            )
            usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            return SimpleNamespace(choices=[choice], usage=usage)

    client = VerbalizedClient(reject_logprobs=False)
    spec = ConfidenceSpec(
        methods=(METHOD_LOGPROBS, METHOD_VERBALIZED), primary=METHOD_VERBALIZED
    )
    wrapper = make_wrapper(client, confidence=spec)

    item = wrapper.complete_structured("prompt", VerbalizedResponse)["evaluation_results"]["1.1"]

    assert item["confidence"] == 0.65
    assert item["confidence_source"] == METHOD_VERBALIZED
    # Both signals recorded from a single call, so they can be compared later.
    assert math.isclose(item["confidence_scores"][METHOD_LOGPROBS], math.exp(-0.01))


def test_non_logprob_errors_still_propagate():
    class FailingClient(RecordingClient):
        def parse(self, **kwargs):
            self.calls.append(kwargs)
            raise ValueError("something else went wrong")

    client = FailingClient(reject_logprobs=False)
    wrapper = make_wrapper(client, confidence=LOGPROB_SPEC)

    with pytest.raises(Exception):
        wrapper.complete_structured("prompt", PolicyEvaluationResponse)
