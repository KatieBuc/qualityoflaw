"""Confidence methods and how they are attached to evaluation results.

Three signals are available today, and they cost different things:

- `logprobs`   -- P(the answer the judge gave) = exp(summed answer logprob).
                  Free: it rides along on the evaluation call.
- `margin`     -- how far the winning label beat the losing one, from the same
                  call's `top_logprobs`. Also free. Separates "Yes at 0.99 with
                  No at 0.01" from "Yes at 0.99 with No at 0.98 before
                  normalising", which raw probability cannot.
- `verbalized` -- the judge's own stated certainty, requested as a field in the
                  response. Needs the v4 prompt and the extended schema.

Every enabled method is recorded per indicator under `confidence_scores`, and
one configured `primary` method is copied to the `confidence` field that the
comparison step treats as the headline signal. Adding a method later (e.g.
self-consistency, which needs k samples per indicator) means adding a name here
and a writer for its score -- the report scores whatever it finds.
"""

from dataclasses import dataclass, field

from automation.src.llm.logprobs import AnswerConfidence

METHOD_LOGPROBS = "logprobs"
METHOD_MARGIN = "margin"
METHOD_VERBALIZED = "verbalized"

#: Methods this module can produce. Self-consistency is deliberately absent --
#: it needs repeated sampling rather than a single call, so it belongs to a
#: separate step whenever it gets built.
AVAILABLE_METHODS = (METHOD_LOGPROBS, METHOD_MARGIN, METHOD_VERBALIZED)

#: Methods derived from the answer tokens, so the call must request logprobs.
LOGPROB_METHODS = frozenset({METHOD_LOGPROBS, METHOD_MARGIN})

SOURCE_UNAVAILABLE = "unavailable"
SOURCE_UNALIGNED = "unaligned"

#: Field the verbalized prompt asks the model to fill. Deliberately not
#: `confidence` -- that name holds the primary score on the same item.
VERBALIZED_FIELD = "self_reported_confidence"

CONFIDENCE_FIELDS = (
    "confidence",
    "confidence_source",
    "confidence_scores",
    "answer_logprob",
    "p_yes",
    "p_no",
    "margin",
    "margin_counterpart_observed",
)


@dataclass(frozen=True)
class ConfidenceSpec:
    """Which confidence signals to capture, and which one leads."""

    enabled: bool = True
    methods: tuple[str, ...] = (METHOD_LOGPROBS, METHOD_MARGIN)
    primary: str = METHOD_LOGPROBS
    top_logprobs: int = 5

    def wants(self, method: str) -> bool:
        return self.enabled and method in self.methods

    @property
    def needs_logprobs(self) -> bool:
        return self.enabled and bool(LOGPROB_METHODS.intersection(self.methods))

    @property
    def needs_verbalized(self) -> bool:
        return self.wants(METHOD_VERBALIZED)


#: Capture nothing -- the default for wrappers that aren't the evaluation judge.
DISABLED = ConfidenceSpec(enabled=False, methods=())


@dataclass
class _Scores:
    values: dict[str, float] = field(default_factory=dict)
    logprob: float | None = None
    p_yes: float | None = None
    p_no: float | None = None
    margin: float | None = None
    #: None when there is no margin at all; False when the losing label was
    #: absent from top_logprobs and its probability had to be bounded. Recorded
    #: so the report can say how much of the margin signal was measured rather
    #: than inferred.
    margin_counterpart_observed: bool | None = None


def _scores_from_answer(confidence: AnswerConfidence, spec: ConfidenceSpec) -> _Scores:
    scores = _Scores(
        logprob=confidence.logprob,
        p_yes=confidence.p_yes,
        p_no=confidence.p_no,
        margin=confidence.margin,
        margin_counterpart_observed=(
            confidence.counterpart_observed if confidence.margin is not None else None
        ),
    )
    if spec.wants(METHOD_LOGPROBS):
        scores.values[METHOD_LOGPROBS] = confidence.probability
    if spec.wants(METHOD_MARGIN) and confidence.margin is not None:
        scores.values[METHOD_MARGIN] = confidence.margin
    return scores


def _verbalized_score(item: dict) -> float | None:
    """Read and sanity-check the judge's self-reported certainty.

    Models occasionally answer on a 0-100 scale despite being asked for 0-1, so
    a value above 1 is rescaled rather than discarded; anything else unusable
    becomes None.
    """
    raw = item.pop(VERBALIZED_FIELD, None)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 0.0:
        return None
    if value > 1.0:
        value = value / 100.0
    return min(value, 1.0)


def attach_confidence(
    items_by_id: dict[str, dict],
    confidences: list[AnswerConfidence],
    spec: ConfidenceSpec,
    *,
    logprobs_available: bool,
) -> None:
    """Record every enabled method's score on each item, in place.

    Items the extractor could not align still get the full field set, with
    nulls and a `confidence_source` saying why, so the comparison step never
    has to guess whether a missing key means "no signal" or "old report".
    """
    by_id = {confidence.indicator_id: confidence for confidence in confidences}

    for indicator_id, item in items_by_id.items():
        answer = by_id.get(indicator_id)
        scores = _scores_from_answer(answer, spec) if answer else _Scores()

        if spec.needs_verbalized:
            verbalized = _verbalized_score(item)
            if verbalized is not None:
                scores.values[METHOD_VERBALIZED] = verbalized

        primary = scores.values.get(spec.primary)
        if primary is not None:
            source = spec.primary
        elif not logprobs_available and not scores.values:
            source = SOURCE_UNAVAILABLE
        elif answer is None:
            source = SOURCE_UNALIGNED
        else:
            # The answer aligned but the primary method produced nothing --
            # e.g. margin is primary and the opposing label never appeared.
            source = SOURCE_UNALIGNED

        item["confidence"] = primary
        item["confidence_source"] = source
        item["confidence_scores"] = scores.values
        item["answer_logprob"] = scores.logprob
        item["p_yes"] = scores.p_yes
        item["p_no"] = scores.p_no
        item["margin"] = scores.margin
        item["margin_counterpart_observed"] = scores.margin_counterpart_observed


def merge_scores(items: list[dict], chosen: dict | None) -> dict:
    """Confidence fields for one result merged from several sub-results.

    Used by the sliding-window path, where one indicator's answer is folded
    together from several windows: `chosen` is the window whose confidence
    represents the merged answer. The method name is taken from that window
    rather than passed in, so this needs no knowledge of the active config.
    """
    if chosen is None:
        sources = {item.get("confidence_source") for item in items if item.get("confidence_source")}
        return {
            "confidence": None,
            "confidence_source": sources.pop() if len(sources) == 1 else SOURCE_UNAVAILABLE,
            "confidence_scores": {},
            "answer_logprob": None,
            "p_yes": None,
            "p_no": None,
            "margin": None,
            "margin_counterpart_observed": None,
        }

    source = chosen.get("confidence_source") or SOURCE_UNAVAILABLE
    return {
        "confidence": chosen.get("confidence"),
        "confidence_source": f"{source}_merged",
        "confidence_scores": dict(chosen.get("confidence_scores") or {}),
        "answer_logprob": chosen.get("answer_logprob"),
        "p_yes": chosen.get("p_yes"),
        "p_no": chosen.get("p_no"),
        "margin": chosen.get("margin"),
        "margin_counterpart_observed": chosen.get("margin_counterpart_observed"),
    }
