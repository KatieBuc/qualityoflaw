"""Helpers shared by the Markdown pipeline (`automation.src.markdown`), the RAG
store, and the legacy raw-text pipeline (`automation.src.legacy`).

Moved here unchanged from the legacy `translate.py`, `translation_qa.py` and
`translation_qa_prompt_builder.py` so that current code no longer imports from
the legacy stack.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from automation.src.llm.wrapper import LLMCallError, format_api_error

logger = logging.getLogger(__name__)


@dataclass
class TranslateResult:
    filename: str
    status: str
    token_usage: dict[str, int]
    source_chars: int = 0
    error: str | None = None
    error_type: str | None = None
    error_details: dict | None = None
    attempts: int | None = None


def failure_entry_from_exc(filename: str, exc: Exception) -> dict:
    if isinstance(exc, LLMCallError):
        return {
            "filename": filename,
            "error_type": exc.error_type,
            "message": str(exc),
            "details": exc.details,
            "attempts": exc.attempts,
        }
    error_type, details = format_api_error(exc)
    return {
        "filename": filename,
        "error_type": error_type,
        "message": str(exc),
        "details": details,
        "attempts": 1,
    }


def chunks_artifact_path(chunks_dir: Path, filename: str) -> Path:
    return chunks_dir / f"{Path(filename).stem}.chunks.json"


class TranslationQaResult(BaseModel):
    action_required: bool
    issues: list[str]
    corrected_text: str | None


@dataclass
class QaFileResult:
    filename: str
    status: str  # "succeeded" | "skipped" | "failed"
    corrected: bool = False
    response_incomplete: bool = False
    report: dict | None = None
    error: str | None = None
    error_type: str | None = None
    error_details: dict | None = None


def load_chunk_records(chunks_path: Path) -> list[dict] | None:
    """Load the translation step's chunks.json, if present and usable.

    Same validity checks as `rag/store.py:_load_translation_chunks` (exists,
    valid JSON, non-empty, every record has a translated_text) but returns
    the raw stored records unchanged -- chunk_index/section_id/type/context/
    text/translated_text -- instead of store.py's renumbered/flattened shape,
    since this step needs to write corrections back into the same file.
    """
    if not chunks_path.exists():
        return None
    try:
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, list) or not data:
        return None
    if not all(isinstance(c, dict) and c.get("translated_text") for c in data):
        return None
    return data


def qa_unit(
    original_text: str,
    translated_text: str,
    template_text: str,
    complete_fn: Callable[[str], dict],
    context: str | None = None,
) -> tuple[bool, list[str], str | None, bool]:
    """Returns (action_required, issues, corrected_text, response_incomplete).

    Occasionally the LLM sets action_required=true but doesn't actually
    supply corrected_text -- a content-quality slip, not an API failure. A
    single policy file can be split into dozens or hundreds of chunks, each
    QA'd with its own LLM call (see _qa_policy_file's atomic per-file
    behavior below); treating this as a hard failure would mean even a small
    per-chunk chance of it happening compounds into most files failing
    outright and discarding every other chunk's already-good result. So:
    retry once, and if it's still missing, degrade this one unit to "no
    action" (keep its original text) rather than raising -- response_incomplete
    is set so the caller can flag it in the report for a human to review.
    """
    prompt = build_translation_qa_prompt(original_text, translated_text, template_text, context=context)

    response: dict = {}
    for attempt in range(2):
        response = complete_fn(prompt)
        action_required = bool(response.get("action_required"))
        corrected_text = response.get("corrected_text")
        if not action_required or corrected_text:
            issues = list(response.get("issues") or [])
            return action_required, issues, (corrected_text if action_required else None), False
        logger.warning(
            "LLM reported action_required=true but returned no corrected_text (attempt %d/2)%s",
            attempt + 1,
            "; retrying" if attempt == 0 else "; keeping original text for this unit",
        )

    issues = list(response.get("issues") or []) + [
        "LLM flagged this text for correction but did not return corrected text after retry; "
        "original text kept unchanged."
    ]
    return False, issues, None, True


def build_translation_qa_prompt(
    original_text: str,
    translated_text: str,
    template_text: str,
    context: str | None = None,
) -> str:
    context_block = (
        f"### Preceding Context (already-translated tail of the previous chunk, for reference only — do not re-fix or duplicate it)\n{context}\n"
        if context
        else ""
    )
    return (
        template_text.replace("{{ORIGINAL_TEXT}}", original_text)
        .replace("{{TRANSLATED_TEXT}}", translated_text)
        .replace("{{CONTEXT}}", context_block)
    )
