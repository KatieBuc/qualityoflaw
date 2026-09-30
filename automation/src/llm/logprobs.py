"""Token-level confidence for the Yes/No answers inside a structured judge response.

One evaluation call covers a whole dimension (~8-10 indicators), so the model's
answer for each indicator is a field inside one large JSON object rather than a
standalone token. Recovering a per-indicator probability therefore means
aligning the returned token stream back onto the JSON text: rebuild the text
from the tokens, locate each `"included": "<answer>"` value, and sum the
logprobs of the tokens that spell that value.

    probability = exp(logprob)

The same alignment yields the margin between the winning and losing label, from
the answer token's `top_logprobs`. See `llm/confidence.py` for how these become
the configured confidence methods.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

ID_PATTERN = re.compile(r'"id"\s*:\s*"([^"]*)"')
INCLUDED_PATTERN = re.compile(r'"included"\s*:\s*"([^"]*)"')

LABEL_YES = "yes"
LABEL_NO = "no"


@dataclass
class AnswerConfidence:
    """Confidence for one indicator's Yes/No answer.

    `logprob` is summed over every token that spells the answer, so
    `probability` is the joint probability of the whole answer string even when
    the tokenizer splits it ("Y" + "es") or merges the opening quote ('"Yes').

    `p_yes` / `p_no` are normalised over the two labels and answer a different
    question than `probability`, so they can differ from it slightly.
    `margin` is their absolute difference: how decisively the winning label beat
    the losing one.
    """

    indicator_id: str
    answer: str
    logprob: float
    probability: float
    p_yes: float | None = None
    p_no: float | None = None
    margin: float | None = None
    #: False when the losing label never appeared in `top_logprobs` and its
    #: probability had to be bounded rather than read. Margin is then a
    #: conservative estimate, not a measurement.
    counterpart_observed: bool = False
    token_count: int = 0


def _to_probability(logprob: float) -> float:
    """exp(logprob), clamped to 1.0.

    For an essentially-certain token the API returns a logprob of +4e-05 or so
    rather than exactly 0, and exp() of that exceeds 1. Left unclamped, such a
    value falls outside the last calibration bin and drops out of the report
    entirely. The raw logprob is still reported unmodified.
    """
    return min(math.exp(logprob), 1.0)


def _token_entries(choice: Any) -> list[Any]:
    logprobs = getattr(choice, "logprobs", None)
    if logprobs is None:
        return []
    content = getattr(logprobs, "content", None)
    if not content:
        return []
    return list(content)


def _token_text(entry: Any) -> str:
    return getattr(entry, "token", "") or ""


def _build_text(entries: Sequence[Any]) -> tuple[str, list[tuple[int, int]]]:
    """Rebuild the response text and record each token's [start, end) char span."""
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    position = 0
    for entry in entries:
        text = _token_text(entry)
        parts.append(text)
        spans.append((position, position + len(text)))
        position += len(text)
    return "".join(parts), spans


def _tokens_covering(spans: Sequence[tuple[int, int]], start: int, end: int) -> list[int]:
    return [index for index, (s, e) in enumerate(spans) if s < end and e > start]


def _classify_label(text: str) -> str | None:
    """Map a candidate token to "yes"/"no", tolerating partial and quoted forms."""
    normalized = text.strip().strip('"').lower()
    if not normalized:
        return None
    if normalized.startswith(LABEL_YES) or LABEL_YES.startswith(normalized):
        return LABEL_YES
    if normalized.startswith(LABEL_NO) or LABEL_NO.startswith(normalized):
        return LABEL_NO
    return None


def _label_margin(
    entry: Any, answer_label: str | None, answer_probability: float
) -> tuple[float | None, float | None, float | None, bool]:
    """Normalised P(Yes) / P(No) / margin from one token's top_logprobs.

    The losing label is often missing from the top-k: when the judge is certain,
    the runners-up are whitespace and casing variants rather than the opposite
    answer. Rather than giving up on the margin, its probability is bounded by
    the smallest alternative that *was* returned -- anything outside the top-k
    can be no larger than that -- which makes the margin a conservative
    estimate instead of an unknown.
    """
    if answer_label is None:
        return None, None, None, False

    best: dict[str, float] = {}
    observed: list[float] = []
    for alternative in getattr(entry, "top_logprobs", None) or []:
        logprob = getattr(alternative, "logprob", None)
        if logprob is None:
            continue
        probability = _to_probability(logprob)
        observed.append(probability)
        label = _classify_label(_token_text(alternative))
        if label is not None and probability > best.get(label, 0.0):
            best[label] = probability

    other_label = LABEL_NO if answer_label == LABEL_YES else LABEL_YES
    p_answer = best.get(answer_label, answer_probability)
    p_other = best.get(other_label)
    counterpart_observed = p_other is not None
    if p_other is None:
        bound = min(observed) if observed else 0.0
        p_other = min(bound, max(0.0, 1.0 - p_answer))

    total = p_answer + p_other
    if total <= 0:
        return None, None, None, counterpart_observed

    p_answer /= total
    p_other /= total
    margin = abs(p_answer - p_other)

    if answer_label == LABEL_YES:
        return p_answer, p_other, margin, counterpart_observed
    return p_other, p_answer, margin, counterpart_observed


def _pair_with_id(id_matches: list[re.Match], included_start: int) -> str | None:
    """The indicator id belonging to an `included` value is the last id emitted
    before it -- more robust than zipping by position, and it survives the model
    reordering fields within an object."""
    candidate = None
    for match in id_matches:
        if match.start() < included_start:
            candidate = match.group(1)
        else:
            break
    return candidate


def extract_answer_confidences(choice: Any) -> list[AnswerConfidence]:
    """Per-indicator confidence from one chat completion choice.

    Returns an empty list when the response carries no logprobs (the model or
    deployment did not return them), which callers treat as "unavailable"
    rather than as an error.
    """
    entries = _token_entries(choice)
    if not entries:
        return []

    text, spans = _build_text(entries)
    id_matches = list(ID_PATTERN.finditer(text))

    results: list[AnswerConfidence] = []
    for match in INCLUDED_PATTERN.finditer(text):
        indicator_id = _pair_with_id(id_matches, match.start())
        if indicator_id is None:
            continue

        value_start, value_end = match.span(1)
        token_indices = _tokens_covering(spans, value_start, value_end)
        if not token_indices:
            continue

        logprob = 0.0
        for index in token_indices:
            token_logprob = getattr(entries[index], "logprob", None)
            if token_logprob is None:
                continue
            logprob += token_logprob

        probability = _to_probability(logprob)
        p_yes, p_no, margin, counterpart_observed = _label_margin(
            entries[token_indices[0]], _classify_label(match.group(1)), probability
        )
        results.append(
            AnswerConfidence(
                indicator_id=indicator_id,
                answer=match.group(1),
                logprob=logprob,
                probability=probability,
                p_yes=p_yes,
                p_no=p_no,
                margin=margin,
                counterpart_observed=counterpart_observed,
                token_count=len(token_indices),
            )
        )
    return results
