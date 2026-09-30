"""Cohere rerank API client for the retrieval reranker.

Calls a Cohere rerank model hosted as an Azure AI Foundry serverless
endpoint — same request/response wire format as Cohere's native `/v1/rerank`
API, just pointed at your Azure deployment's endpoint + key (see
AZURE_COHERE_RERANK_ENDPOINT / AZURE_COHERE_RERANK_API_KEY in .env).

This runs as an ordinary API call on whichever worker thread is already
evaluating a dimension/window — it does not need its own concurrency
guard or local caching; the pipeline's existing `concurrency.max_workers`
bounds it exactly like any other Azure call.
"""

from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0
_REQUEST_TIMEOUT_S = 30.0


def score_candidates(model_name: str, query: str, texts: list[str]) -> list[float]:
    if not texts:
        return []

    from automation.src.llm.client import get_cohere_rerank_api_key, get_cohere_rerank_endpoint

    endpoint = get_cohere_rerank_endpoint().rstrip("/")
    headers = {
        "Authorization": f"Bearer {get_cohere_rerank_api_key()}",
        "Content-Type": "application/json",
    }
    payload = {"model": model_name, "query": query, "documents": texts}

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = requests.post(
                f"{endpoint}/v2/rerank", headers=headers, json=payload, timeout=_REQUEST_TIMEOUT_S
            )
            response.raise_for_status()
            results = response.json()["results"]
            scores = [0.0] * len(texts)
            for item in results:
                scores[item["index"]] = float(item["relevance_score"])
            return scores
        except (requests.RequestException, KeyError, ValueError) as exc:
            last_exc = exc
            if attempt == _MAX_RETRIES - 1:
                break
            delay = _RETRY_BASE_DELAY ** (attempt + 1)
            logger.warning(
                "Cohere rerank call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                _MAX_RETRIES,
                exc,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(f"Cohere rerank call failed after {_MAX_RETRIES} attempts: {last_exc}") from last_exc
