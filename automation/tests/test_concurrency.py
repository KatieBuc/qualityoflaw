import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from automation.src.concurrency import ConcurrencyLimiter
from automation.src.config_loader import parse_concurrency_config
from automation.src.run_pipeline import build_limiter, resolve_concurrency_config
from automation.src.config_loader import ConcurrencyConfig


def test_concurrency_limiter_serial_when_disabled():
    limiter = ConcurrencyLimiter(max_workers=5, enabled=False)
    calls: list[int] = []

    def task(value: int) -> int:
        calls.append(value)
        return value

    results = limiter.run_parallel([lambda v=i: task(v) for i in range(3)])
    assert results == [0, 1, 2]
    assert calls == [0, 1, 2]


def test_concurrency_limiter_serial_when_max_workers_one():
    limiter = ConcurrencyLimiter(max_workers=1, enabled=True)
    results = limiter.run_parallel([lambda i=i: i * 2 for i in range(4)])
    assert results == [0, 2, 4, 6]


def test_concurrency_limiter_respects_max_workers():
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)
    active = 0
    peak = 0
    lock = threading.Lock()

    def slow_task() -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return 1

    results = limiter.run_parallel([slow_task for _ in range(6)])
    assert results == [1, 1, 1, 1, 1, 1]
    assert peak <= 2


def test_concurrency_limiter_isolates_task_failures():
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    def failing() -> int:
        raise ValueError("boom")

    results = limiter.run_parallel([failing, lambda: 42])
    assert isinstance(results[0], ValueError)
    assert results[1] == 42


def test_concurrency_limiter_rejects_invalid_max_workers():
    with pytest.raises(ValueError, match="max_workers must be >= 1"):
        ConcurrencyLimiter(max_workers=0)


def test_parse_concurrency_config_defaults():
    config = parse_concurrency_config(None)
    assert config.enabled is True
    assert config.max_workers == 5


def test_parse_concurrency_config_custom():
    config = parse_concurrency_config({"enabled": False, "max_workers": 3})
    assert config.enabled is False
    assert config.max_workers == 3


def test_parse_concurrency_config_invalid_max_workers():
    with pytest.raises(ValueError, match="concurrency.max_workers must be >= 1"):
        parse_concurrency_config({"max_workers": 0})


def test_resolve_concurrency_config_no_concurrency_flag():
    base = ConcurrencyConfig(enabled=True, max_workers=5)
    resolved = resolve_concurrency_config(base, max_workers_override=None, no_concurrency=True)
    assert resolved.enabled is False
    assert resolved.max_workers == 1


def test_resolve_concurrency_config_max_workers_override():
    base = ConcurrencyConfig(enabled=True, max_workers=5)
    resolved = resolve_concurrency_config(base, max_workers_override=8, no_concurrency=False)
    assert resolved.max_workers == 8
    assert resolved.enabled is True


def test_resolve_concurrency_config_no_concurrency_overrides_max_workers_flag():
    # --no-concurrency must win even when --max-workers is also passed.
    base = ConcurrencyConfig(enabled=True, max_workers=5)
    resolved = resolve_concurrency_config(base, max_workers_override=8, no_concurrency=True)
    assert resolved.enabled is False
    assert resolved.max_workers == 1


def test_resolve_concurrency_config_rejects_zero_override():
    base = ConcurrencyConfig(enabled=True, max_workers=5)
    with pytest.raises(ValueError, match="--max-workers must be >= 1"):
        resolve_concurrency_config(base, max_workers_override=0, no_concurrency=False)


def test_resolve_concurrency_config_rejects_negative_override():
    base = ConcurrencyConfig(enabled=True, max_workers=5)
    with pytest.raises(ValueError, match="--max-workers must be >= 1"):
        resolve_concurrency_config(base, max_workers_override=-1, no_concurrency=False)


def test_resolve_concurrency_config_preserves_disabled_base_with_override():
    # An override changes worker count but must not silently re-enable
    # concurrency the config had turned off.
    base = ConcurrencyConfig(enabled=False, max_workers=5)
    resolved = resolve_concurrency_config(base, max_workers_override=3, no_concurrency=False)
    assert resolved.enabled is False
    assert resolved.max_workers == 3


def test_build_limiter():
    limiter = build_limiter(ConcurrencyConfig(enabled=True, max_workers=4))
    assert limiter.max_workers == 4
    assert limiter.enabled is True


def test_azure_llm_wrapper_uses_limiter_semaphore():
    from automation.src.llm.model_profile import ModelProfile
    from automation.src.llm.wrapper import AzureLLMWrapper

    mock_semaphore = MagicMock()
    limiter = MagicMock()
    limiter.semaphore = mock_semaphore
    profile = ModelProfile(name="t", deployment="m", temperature=0.2, max_retries=1)
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="ok"))]
    response.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    client.chat.completions.create.return_value = response

    wrapper = AzureLLMWrapper(profile=profile, client=client, api_style="chat", limiter=limiter)
    text, _ = wrapper.complete_text("prompt")
    assert text == "ok"
    mock_semaphore.__enter__.assert_called_once()
    mock_semaphore.__exit__.assert_called_once()
