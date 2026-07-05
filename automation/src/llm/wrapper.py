import logging
import re
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from openai import APIConnectionError, APIStatusError, AzureOpenAI, OpenAI, RateLimitError
from pydantic import BaseModel

from automation.src.llm.client import ApiStyle
from automation.src.llm.model_profile import ModelProfile

if TYPE_CHECKING:
    from automation.src.concurrency import ConcurrencyLimiter

logger = logging.getLogger(__name__)

RETRY_BASE_DELAY = 2.0
T = TypeVar("T")


class LLMCallError(Exception):
    """Raised when all API retry attempts are exhausted."""

    def __init__(self, message: str, *, error_type: str, details: dict[str, Any], attempts: int):
        super().__init__(message)
        self.error_type = error_type
        self.details = details
        self.attempts = attempts


def format_api_error(exc: Exception) -> tuple[str, dict[str, Any]]:
    error_type = type(exc).__name__
    details: dict[str, Any] = {}

    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        details["status_code"] = status_code

    message = getattr(exc, "message", None) or str(exc)
    details["message"] = message

    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers is not None:
            request_id = headers.get("x-request-id") or headers.get("x-ms-request-id")
            if request_id:
                details["request_id"] = request_id

    body = getattr(exc, "body", None)
    if body is not None:
        details["body"] = body

    return error_type, details

PREAMBLE_PATTERNS = [
    re.compile(r"^here(?:'s| is) the translation[:\s]*", re.IGNORECASE),
    re.compile(r"^translation[:\s]*", re.IGNORECASE),
]


def _extract_response_text(response: Any, api_style: ApiStyle) -> str:
    if api_style == "responses":
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
        output = response.output[0]
        content = output.content[0]
        return content.text

    return response.choices[0].message.content or ""


def _extract_usage(response: Any, api_style: ApiStyle) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if not usage:
        return {}

    if api_style == "responses":
        return {
            "prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "output_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

    return {
        "prompt_tokens": usage.prompt_tokens or 0,
        "completion_tokens": usage.completion_tokens or 0,
        "total_tokens": usage.total_tokens or 0,
    }


def clean_translation_response(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:\w+)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    for pattern in PREAMBLE_PATTERNS:
        cleaned = pattern.sub("", cleaned).lstrip()
    return cleaned


class AzureLLMWrapper:
    def __init__(
        self,
        profile: ModelProfile,
        client: OpenAI | AzureOpenAI,
        api_style: ApiStyle,
        limiter: "ConcurrencyLimiter | None" = None,
    ):
        self.profile = profile
        self.client = client
        self.api_style = api_style
        self.limiter = limiter
        self._usage_lock = threading.Lock()
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    @classmethod
    def from_profile(
        cls,
        profile: ModelProfile,
        *,
        limiter: "ConcurrencyLimiter | None" = None,
    ) -> "AzureLLMWrapper":
        from automation.src.llm.client import get_api_style, get_azure_client

        return cls(
            profile=profile,
            client=get_azure_client(),
            api_style=get_api_style(),
            limiter=limiter,
        )

    def _build_request_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"temperature": self.profile.temperature}
        if self.profile.max_tokens is not None:
            kwargs["max_tokens"] = self.profile.max_tokens
        return kwargs

    def _record_usage(self, usage: dict[str, int]) -> None:
        if not usage:
            return
        with self._usage_lock:
            for key in self.token_usage:
                self.token_usage[key] += usage.get(key, 0)

    def _call_with_retry(self, fn: Callable[[], T]) -> T:
        last_error: Exception | None = None
        last_error_type = "Exception"
        last_details: dict[str, Any] = {}
        for attempt in range(self.profile.max_retries):
            try:
                if self.limiter is not None:
                    with self.limiter.semaphore:
                        return fn()
                return fn()
            except (RateLimitError, APIConnectionError, APIStatusError, Exception) as exc:
                last_error = exc
                last_error_type, last_details = format_api_error(exc)
                if attempt == self.profile.max_retries - 1:
                    message = last_details.get("message", str(exc))
                    status_code = last_details.get("status_code", "n/a")
                    request_id = last_details.get("request_id", "n/a")
                    logger.error(
                        "API error (attempt %d/%d) [%s]: %s | status=%s request_id=%s",
                        attempt + 1,
                        self.profile.max_retries,
                        last_error_type,
                        message,
                        status_code,
                        request_id,
                    )
                    raise LLMCallError(
                        message,
                        error_type=last_error_type,
                        details=last_details,
                        attempts=attempt + 1,
                    ) from exc
                delay = RETRY_BASE_DELAY ** (attempt + 1)
                status_code = last_details.get("status_code", "n/a")
                request_id = last_details.get("request_id", "n/a")
                logger.warning(
                    "API error (attempt %d/%d) [%s]: %s | status=%s request_id=%s — retrying in %.1fs",
                    attempt + 1,
                    self.profile.max_retries,
                    last_error_type,
                    last_details.get("message", str(exc)),
                    status_code,
                    request_id,
                    delay,
                )
                time.sleep(delay)
        raise LLMCallError(
            str(last_error),
            error_type=last_error_type,
            details=last_details,
            attempts=self.profile.max_retries,
        ) from last_error  # pragma: no cover

    def complete_text(self, prompt: str) -> tuple[str, dict[str, int]]:
        def _call() -> tuple[str, dict[str, int]]:
            messages = [{"role": "user", "content": prompt}]
            kwargs = self._build_request_kwargs()

            if self.api_style == "responses":
                response = self.client.responses.create(
                    model=self.profile.deployment,
                    input=messages,
                    **kwargs,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.profile.deployment,
                    messages=messages,
                    **kwargs,
                )

            text = _extract_response_text(response, self.api_style)
            usage = _extract_usage(response, self.api_style)
            self._record_usage(usage)
            return text, usage

        text, usage = self._call_with_retry(_call)
        return clean_translation_response(text), usage

    def complete_structured(
        self,
        prompt: str,
        response_format: type[BaseModel],
    ) -> dict:
        def _call() -> dict:
            kwargs = self._build_request_kwargs()
            response = self.client.beta.chat.completions.parse(
                model=self.profile.deployment,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format,
                **kwargs,
            )

            parsed_output = response.choices[0].message.parsed
            if not parsed_output:
                raise ValueError("LLM return structure error.")

            output_dict = parsed_output.model_dump()
            if "evaluation_results" in output_dict:
                output_dict["evaluation_results"] = {
                    item["id"]: item for item in output_dict["evaluation_results"]
                }
            self._record_usage(_extract_usage(response, "chat"))
            return output_dict

        return self._call_with_retry(_call)
