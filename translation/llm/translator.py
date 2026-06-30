import logging
import re
import time
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, AzureOpenAI, OpenAI, RateLimitError

from translation.llm.azure_client import ApiStyle, get_api_style

logger = logging.getLogger(__name__)

STRUCTURE_HINT = (
    "Preserve the original paragraph and line structure where possible."
)

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


class PolicyLLMTranslator:
    def __init__(
        self,
        client: OpenAI | AzureOpenAI,
        model: str,
        prompt_path: Path,
        temperature: float = 0.2,
        max_retries: int = 3,
        api_style: ApiStyle | None = None,
    ):
        self.client = client
        self.model = model
        self.prompt_template = prompt_path.read_text(encoding="utf-8")
        self.temperature = temperature
        self.max_retries = max_retries
        self.api_style = api_style or get_api_style()

    def _build_prompt(self, source_text: str) -> str:
        prompt = self.prompt_template.replace("{text}", source_text)
        return f"{prompt}\n\n{STRUCTURE_HINT}"

    @staticmethod
    def _clean_response(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:\w+)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        for pattern in PREAMBLE_PATTERNS:
            cleaned = pattern.sub("", cleaned).lstrip()
        return cleaned

    def _call_api(self, prompt: str) -> tuple[str, dict[str, int]]:
        messages = [{"role": "user", "content": prompt}]

        if self.api_style == "responses":
            response = self.client.responses.create(
                model=self.model,
                input=messages,
                temperature=self.temperature,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )

        text = _extract_response_text(response, self.api_style)
        return text, _extract_usage(response, self.api_style)

    def translate_text(self, source_text: str) -> tuple[str, dict[str, int]]:
        prompt = self._build_prompt(source_text)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                text, usage = self._call_api(prompt)
                return self._clean_response(text), usage
            except (RateLimitError, APIConnectionError, APIStatusError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                delay = 2 ** attempt
                logger.warning(
                    "API error on attempt %d/%d: %s. Retrying in %ds.",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError(f"Translation failed after {self.max_retries} attempts.") from last_error

    def translate_file(self, input_path: Path, output_path: Path, force: bool = False) -> dict:
        if output_path.exists() and not force:
            return {"status": "skipped", "input": str(input_path), "output": str(output_path)}

        source_text = input_path.read_text(encoding="utf-8")
        start = time.time()
        translated, usage = self.translate_text(source_text)
        elapsed = time.time() - start

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(translated, encoding="utf-8")

        return {
            "status": "succeeded",
            "input": str(input_path),
            "output": str(output_path),
            "chars": len(source_text),
            "elapsed_s": round(elapsed, 2),
            "usage": usage,
        }
