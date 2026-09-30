from unittest.mock import MagicMock

import pytest
import requests

from automation.src.rag import reranker


@pytest.fixture(autouse=True)
def fake_credentials(monkeypatch):
    monkeypatch.setattr(
        "automation.src.llm.client.get_cohere_rerank_endpoint",
        lambda: "https://fake.models.ai.azure.com",
    )
    monkeypatch.setattr(
        "automation.src.llm.client.get_cohere_rerank_api_key", lambda: "fake-key"
    )


def _mock_response(results: list[dict]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"results": results}
    return response


def test_score_candidates_returns_scores_in_original_order(monkeypatch):
    # Cohere returns results sorted by relevance, each tagged with its
    # original input index — score_candidates must remap back to input order.
    response = _mock_response(
        [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
        ]
    )
    mock_post = MagicMock(return_value=response)
    monkeypatch.setattr(requests, "post", mock_post)

    scores = reranker.score_candidates("rerank-v3.5", "query", ["doc0", "doc1"])

    assert scores == [0.1, 0.9]


def test_score_candidates_empty_texts_skips_request(monkeypatch):
    mock_post = MagicMock()
    monkeypatch.setattr(requests, "post", mock_post)

    assert reranker.score_candidates("rerank-v3.5", "query", []) == []
    mock_post.assert_not_called()


def test_score_candidates_sends_expected_request(monkeypatch):
    response = _mock_response([{"index": 0, "relevance_score": 0.5}])
    mock_post = MagicMock(return_value=response)
    monkeypatch.setattr(requests, "post", mock_post)

    reranker.score_candidates("rerank-v3.5", "my query", ["doc0"])

    args, kwargs = mock_post.call_args
    assert args[0] == "https://fake.models.ai.azure.com/v2/rerank"
    assert kwargs["headers"]["Authorization"] == "Bearer fake-key"
    assert kwargs["json"] == {"model": "rerank-v3.5", "query": "my query", "documents": ["doc0"]}


def test_score_candidates_retries_on_request_exception(monkeypatch):
    response = _mock_response([{"index": 0, "relevance_score": 0.5}])
    mock_post = MagicMock(side_effect=[requests.ConnectionError("boom"), response])
    monkeypatch.setattr(requests, "post", mock_post)
    monkeypatch.setattr(reranker.time, "sleep", lambda _: None)

    scores = reranker.score_candidates("rerank-v3.5", "q", ["doc0"])

    assert scores == [0.5]
    assert mock_post.call_count == 2


def test_score_candidates_raises_after_exhausting_retries(monkeypatch):
    mock_post = MagicMock(side_effect=requests.ConnectionError("boom"))
    monkeypatch.setattr(requests, "post", mock_post)
    monkeypatch.setattr(reranker.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="Cohere rerank call failed"):
        reranker.score_candidates("rerank-v3.5", "q", ["doc0"])

    assert mock_post.call_count == reranker._MAX_RETRIES


def test_score_candidates_retries_on_malformed_response(monkeypatch):
    bad_response = MagicMock()
    bad_response.raise_for_status.return_value = None
    bad_response.json.return_value = {"unexpected": "shape"}

    good_response = _mock_response([{"index": 0, "relevance_score": 0.7}])
    mock_post = MagicMock(side_effect=[bad_response, good_response])
    monkeypatch.setattr(requests, "post", mock_post)
    monkeypatch.setattr(reranker.time, "sleep", lambda _: None)

    scores = reranker.score_candidates("rerank-v3.5", "q", ["doc0"])

    assert scores == [0.7]
    assert mock_post.call_count == 2
