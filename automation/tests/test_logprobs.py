import math
from dataclasses import dataclass, field

from automation.src.llm.logprobs import extract_answer_confidences


@dataclass
class FakeTopLogprob:
    token: str
    logprob: float


@dataclass
class FakeToken:
    token: str
    logprob: float = 0.0
    top_logprobs: list = field(default_factory=list)


@dataclass
class FakeLogprobs:
    content: list


@dataclass
class FakeChoice:
    logprobs: object


def tokens_from(pieces: list[tuple[str, float]]) -> list[FakeToken]:
    return [FakeToken(token=text, logprob=logprob) for text, logprob in pieces]


def choice_for(tokens: list[FakeToken]) -> FakeChoice:
    return FakeChoice(logprobs=FakeLogprobs(content=tokens))


def test_single_token_answer():
    tokens = tokens_from(
        [
            ('{"evaluation_results":[{"id":', 0.0),
            ('"1.1"', 0.0),
            (',"included":"', 0.0),
            ("Yes", -0.01),
            ('","rationale":"x"}]}', 0.0),
        ]
    )

    (result,) = extract_answer_confidences(choice_for(tokens))

    assert result.indicator_id == "1.1"
    assert result.answer == "Yes"
    assert result.token_count == 1
    assert math.isclose(result.probability, math.exp(-0.01))


def test_answer_split_across_tokens_sums_logprobs():
    tokens = tokens_from(
        [
            ('{"id":"2.3","included":"', 0.0),
            ("Y", -0.2),
            ("es", -0.3),
            ('","rationale":"x"}', 0.0),
        ]
    )

    (result,) = extract_answer_confidences(choice_for(tokens))

    assert result.token_count == 2
    assert math.isclose(result.logprob, -0.5)
    assert math.isclose(result.probability, math.exp(-0.5))


def test_quote_merged_into_answer_token():
    tokens = tokens_from(
        [
            ('{"id":"3.1","included":', 0.0),
            ('"No', -0.05),
            ('","rationale":"x"}', 0.0),
        ]
    )

    (result,) = extract_answer_confidences(choice_for(tokens))

    assert result.answer == "No"
    # The token carrying the answer also carries the opening quote, so its
    # logprob is the only one available for that answer.
    assert math.isclose(result.logprob, -0.05)


def test_multiple_indicators_pair_with_their_own_ids():
    tokens = tokens_from(
        [
            ('{"id":"1.1","included":"', 0.0),
            ("Yes", -0.01),
            ('","rationale":"a"},{"id":"1.2","included":"', 0.0),
            ("No", -0.7),
            ('","rationale":"b"}', 0.0),
        ]
    )

    first, second = extract_answer_confidences(choice_for(tokens))

    assert (first.indicator_id, first.answer) == ("1.1", "Yes")
    assert (second.indicator_id, second.answer) == ("1.2", "No")
    assert second.probability < first.probability


def test_top_logprobs_give_normalised_p_yes_and_p_no():
    answer_token = FakeToken(
        token="Yes",
        logprob=math.log(0.8),
        top_logprobs=[
            FakeTopLogprob(token="Yes", logprob=math.log(0.8)),
            FakeTopLogprob(token="No", logprob=math.log(0.1)),
        ],
    )
    tokens = [
        FakeToken(token='{"id":"4.1","included":"'),
        answer_token,
        FakeToken(token='","rationale":"x"}'),
    ]

    (result,) = extract_answer_confidences(choice_for(tokens))

    assert math.isclose(result.p_yes, 0.8 / 0.9)
    assert math.isclose(result.p_no, 0.1 / 0.9)
    assert math.isclose(result.margin, abs(result.p_yes - result.p_no))


def test_missing_counterpart_bounds_the_margin():
    # When the judge is certain, the runners-up are casing and whitespace
    # variants rather than the opposite answer, so "No" never appears. Its
    # probability is bounded by the smallest alternative that did appear.
    answer_token = FakeToken(
        token="Yes",
        logprob=math.log(0.99),
        top_logprobs=[
            FakeTopLogprob(token="Yes", logprob=math.log(0.99)),
            FakeTopLogprob(token=" Yes", logprob=math.log(0.005)),
        ],
    )
    tokens = [
        FakeToken(token='{"id":"4.2","included":"'),
        answer_token,
        FakeToken(token='"}'),
    ]

    (result,) = extract_answer_confidences(choice_for(tokens))

    assert result.counterpart_observed is False
    assert math.isclose(result.p_no, 0.005 / 0.995)
    assert math.isclose(result.margin, abs(result.p_yes - result.p_no))
    assert 0.0 <= result.margin <= 1.0


def test_margin_separates_a_close_call_from_a_certain_one():
    def margin_for(p_yes: float, p_no: float) -> float:
        answer = FakeToken(
            token="Yes",
            logprob=math.log(p_yes),
            top_logprobs=[
                FakeTopLogprob(token="Yes", logprob=math.log(p_yes)),
                FakeTopLogprob(token="No", logprob=math.log(p_no)),
            ],
        )
        tokens = [FakeToken(token='{"id":"1.1","included":"'), answer, FakeToken(token='"}')]
        (result,) = extract_answer_confidences(choice_for(tokens))
        return result.margin

    # Both answers carry nearly the same raw probability, but one was a
    # coin flip and the other wasn't -- which is the point of margin.
    assert margin_for(0.51, 0.49) < 0.05
    assert margin_for(0.99, 0.001) > 0.99


def test_counterpart_observed_flag_marks_measured_margins():
    answer = FakeToken(
        token="No",
        logprob=math.log(0.7),
        top_logprobs=[
            FakeTopLogprob(token="No", logprob=math.log(0.7)),
            FakeTopLogprob(token="Yes", logprob=math.log(0.3)),
        ],
    )
    tokens = [FakeToken(token='{"id":"2.1","included":"'), answer, FakeToken(token='"}')]

    (result,) = extract_answer_confidences(choice_for(tokens))

    assert result.counterpart_observed is True
    assert math.isclose(result.p_no, 0.7)
    assert math.isclose(result.p_yes, 0.3)
    assert math.isclose(result.margin, 0.4)


def test_positive_logprob_is_clamped_to_probability_one():
    # Observed against a live deployment: an essentially-certain token comes
    # back with a logprob of +4e-05, whose exp() exceeds 1.
    tokens = tokens_from(
        [
            ('{"id":"5.1","included":"', 0.0),
            ("No", 4.1961669921875e-05),
            ('"}', 0.0),
        ]
    )

    (result,) = extract_answer_confidences(choice_for(tokens))

    assert result.probability == 1.0
    assert result.logprob > 0  # the raw value is reported unmodified


def test_no_logprobs_returns_empty():
    assert extract_answer_confidences(FakeChoice(logprobs=None)) == []
    assert extract_answer_confidences(FakeChoice(logprobs=FakeLogprobs(content=[]))) == []
