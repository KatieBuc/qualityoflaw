from types import SimpleNamespace

import pytest

from automation.src.config_loader import parse_model_profile
from automation.src.llm.model_profile import ModelProfile
from automation.src.llm.wrapper import (
    AzureLLMWrapper,
    LLMCallError,
    is_temperature_unsupported,
)

MESSAGE = "Unsupported parameter: 'temperature' is not supported with this model."


class BadRequest(Exception):
    """Mirrors the shape of an openai 400."""

    def __init__(self, message=MESSAGE):
        super().__init__(message)
        self.status_code = 400
        self.message = message


class RecordingClient:
    """chat.completions.create() that rejects `temperature` like a reasoning model."""

    def __init__(self, *, accepts_temperature: bool, always_fail: bool = False):
        self.calls: list[dict] = []
        self.accepts_temperature = accepts_temperature
        self.always_fail = always_fail
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.always_fail:
            raise BadRequest("Unsupported parameter: 'temperature' is not supported.")
        if "temperature" in kwargs and not self.accepts_temperature:
            raise BadRequest()
        message = SimpleNamespace(content="Hello.")
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def make_wrapper(client, **profile_kwargs):
    profile = ModelProfile(name="m", deployment="d", temperature=0.2, max_retries=3, **profile_kwargs)
    return AzureLLMWrapper(profile, client, "chat")


def test_detects_the_unsupported_temperature_error():
    assert is_temperature_unsupported(BadRequest())
    assert not is_temperature_unsupported(BadRequest("Unsupported parameter: 'logprobs'"))
    assert not is_temperature_unsupported(Exception("temperature is unsupported"))  # no 400


def test_a_rejected_temperature_is_dropped_and_resent_without_waiting(monkeypatch):
    sleeps = []
    monkeypatch.setattr("automation.src.llm.wrapper.time.sleep", sleeps.append)
    client = RecordingClient(accepts_temperature=False)
    wrapper = make_wrapper(client)

    text, _usage = wrapper.complete_text("hi")

    assert text == "Hello."
    assert "temperature" in client.calls[0] and "temperature" not in client.calls[1]
    assert len(client.calls) == 2
    assert sleeps == []  # not treated as a transient failure


def test_the_run_keeps_going_without_temperature_after_the_first_rejection():
    client = RecordingClient(accepts_temperature=False)
    wrapper = make_wrapper(client)

    wrapper.complete_text("one")
    wrapper.complete_text("two")
    wrapper.complete_text("three")

    assert len(client.calls) == 4  # 1 rejected + 3 successful
    assert all("temperature" not in call for call in client.calls[1:])


def test_a_deployment_that_accepts_temperature_still_gets_it():
    client = RecordingClient(accepts_temperature=True)
    make_wrapper(client).complete_text("hi")
    assert client.calls[0]["temperature"] == 0.2


def test_supports_temperature_false_never_sends_it():
    client = RecordingClient(accepts_temperature=False)
    make_wrapper(client, supports_temperature=False).complete_text("hi")
    assert len(client.calls) == 1 and "temperature" not in client.calls[0]


def test_a_persistent_error_still_fails_after_max_retries(monkeypatch):
    monkeypatch.setattr("automation.src.llm.wrapper.time.sleep", lambda _s: None)
    client = RecordingClient(accepts_temperature=False, always_fail=True)
    wrapper = make_wrapper(client)

    with pytest.raises(LLMCallError):
        wrapper.complete_text("hi")

    # One free resend without temperature, then the normal retries: no endless loop.
    assert len(client.calls) <= 1 + wrapper.profile.max_retries


def test_model_config_supports_temperature_is_parsed():
    assert parse_model_profile("m", {"deployment": "d"}).supports_temperature is True
    off = parse_model_profile("m", {"deployment": "d", "supports_temperature": False})
    assert off.supports_temperature is False
