import logging
import re
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from openai import APIConnectionError, APIStatusError, AzureOpenAI, OpenAI, RateLimitError
from pydantic import BaseModel

from automation.src.llm.client import ApiStyle
from automation.src.llm.confidence import DISABLED, ConfidenceSpec, attach_confidence
from automation.src.llm.logprobs import extract_answer_confidences
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

LOGPROB_UNSUPPORTED_MARKERS = ("logprob", "top_logprobs")
TOP_LOGPROBS_LIMIT_PATTERN = re.compile(
    r"top_logprobs.*?(?:less than or equal to|at most|maximum of|<=)\s*(\d+)", re.IGNORECASE
)
#: Fallback when a deployment rejects the requested value without naming its
#: own ceiling. Azure's chat completions cap top_logprobs at 5.
TOP_LOGPROBS_FALLBACK = 5


def top_logprobs_limit(exc: Exception) -> int | None:
    """The deployment's own `top_logprobs` ceiling, if a 400 just complained
    about the requested value being too high.

    This is a fixable request rather than an unsupported one, so it must be
    checked before `is_logprobs_unsupported` — otherwise asking for one
    alternative too many silently costs the whole run its confidence scores.
    """
    if getattr(exc, "status_code", None) != 400:
        return None
    message = (getattr(exc, "message", None) or str(exc)).lower()
    if "top_logprobs" not in message:
        return None
    match = TOP_LOGPROBS_LIMIT_PATTERN.search(message)
    if match:
        return max(1, int(match.group(1)))
    if "must be" in message or "invalid value" in message:
        return TOP_LOGPROBS_FALLBACK
    return None


def is_logprobs_unsupported(exc: Exception) -> bool:
    """True when a 400 says this deployment will not accept `logprobs`.

    Reasoning-family deployments reject the parameter outright. Detecting that
    here lets `complete_structured` retry once without it, instead of letting
    `_call_with_retry` burn every attempt on a request that can never succeed.
    """
    if getattr(exc, "status_code", None) != 400:
        return False
    message = (getattr(exc, "message", None) or str(exc)).lower()
    return any(marker in message for marker in LOGPROB_UNSUPPORTED_MARKERS)


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
        confidence: ConfidenceSpec = DISABLED,
    ):
        self.profile = profile
        self.client = client
        self.api_style = api_style
        self.limiter = limiter
        self.confidence = confidence
        self._logprobs_lock = threading.Lock()
        self._logprobs_disabled = False
        self._top_logprobs_cap: int | None = None
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
        confidence: ConfidenceSpec = DISABLED,
    ) -> "AzureLLMWrapper":
        from automation.src.llm.client import get_api_style, get_azure_client

        return cls(
            profile=profile,
            client=get_azure_client(),
            api_style=get_api_style(),
            limiter=limiter,
            confidence=confidence,
        )

    def _build_request_kwargs(self, *, with_logprobs: bool = False) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"temperature": self.profile.temperature}
        if self.profile.max_tokens is not None:
            kwargs["max_tokens"] = self.profile.max_tokens
        if with_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = self.effective_top_logprobs()
        return kwargs

    def effective_top_logprobs(self) -> int:
        with self._logprobs_lock:
            if self._top_logprobs_cap is None:
                return self.confidence.top_logprobs
            return min(self.confidence.top_logprobs, self._top_logprobs_cap)

    def _cap_top_logprobs(self, limit: int) -> bool:
        """Lower the requested alternatives to what the deployment allows.

        Returns False when the cap wouldn't change anything, so the caller
        knows retrying is pointless and the error is real.
        """
        with self._logprobs_lock:
            current = self.confidence.top_logprobs if self._top_logprobs_cap is None else self._top_logprobs_cap
            if limit >= current:
                return False
            already_capped = self._top_logprobs_cap is not None
            self._top_logprobs_cap = limit
        if not already_capped:
            logger.warning(
                "Deployment %s caps top_logprobs at %d; requesting %d instead of %d "
                "for the rest of this run.",
                self.profile.deployment,
                limit,
                limit,
                self.confidence.top_logprobs,
            )
        return True

    def logprobs_active(self) -> bool:
        if not (self.confidence.needs_logprobs and self.profile.supports_logprobs):
            return False
        with self._logprobs_lock:
            return not self._logprobs_disabled

    def _disable_logprobs(self, reason: str) -> None:
        """Stop asking for logprobs for the rest of the run, warning once.

        Evaluation runs 20-wide, so several threads can hit the same 400 before
        the first one lands here; the lock keeps that to a single warning.
        """
        with self._logprobs_lock:
            already_disabled = self._logprobs_disabled
            self._logprobs_disabled = True
        if not already_disabled:
            logger.warning(
                "Deployment %s rejected logprobs; continuing without confidence "
                "scores for the rest of this run: %s",
                self.profile.deployment,
                reason,
            )

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

    def _parse_request(
        self,
        prompt: str,
        response_format: type[BaseModel],
        *,
        with_logprobs: bool,
    ) -> Any:
        return self.client.beta.chat.completions.parse(
            model=self.profile.deployment,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_format,
            **self._build_request_kwargs(with_logprobs=with_logprobs),
        )

    def complete_structured(
        self,
        prompt: str,
        response_format: type[BaseModel],
    ) -> dict:
        def _call() -> dict:
            with_logprobs = self.logprobs_active()
            try:
                response = self._parse_request(
                    prompt, response_format, with_logprobs=with_logprobs
                )
            except Exception as exc:
                if not with_logprobs:
                    raise
                limit = top_logprobs_limit(exc)
                if limit is not None and self._cap_top_logprobs(limit):
                    # Too many alternatives requested, not an unsupported
                    # parameter — retry with logprobs still on.
                    response = self._parse_request(
                        prompt, response_format, with_logprobs=True
                    )
                elif is_logprobs_unsupported(exc):
                    self._disable_logprobs(str(exc))
                    with_logprobs = False
                    response = self._parse_request(
                        prompt, response_format, with_logprobs=False
                    )
                else:
                    raise

            choice = response.choices[0]
            parsed_output = choice.message.parsed
            if not parsed_output:
                raise ValueError("LLM return structure error.")

            output_dict = parsed_output.model_dump()
            if "evaluation_results" in output_dict:
                items_by_id = {
                    item["id"]: item for item in output_dict["evaluation_results"]
                }
                if self.confidence.enabled:
                    confidences = (
                        extract_answer_confidences(choice) if with_logprobs else []
                    )
                    attach_confidence(
                        items_by_id,
                        confidences,
                        self.confidence,
                        logprobs_available=with_logprobs,
                    )
                output_dict["evaluation_results"] = items_by_id
            self._record_usage(_extract_usage(response, "chat"))
            return output_dict

        return self._call_with_retry(_call)
