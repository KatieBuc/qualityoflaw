"""Cohere rerank API client for the retrieval reranker.

Calls a Cohere rerank model hosted as an Azure AI Foundry serverless
endpoint — same request/response wire format as Cohere's native `/v1/rerank`
API, just pointed at your Azure deployment's endpoint + key (see
AZURE_COHERE_RERANK_ENDPOINT / AZURE_COHERE_RERANK_API_KEY in .env).

Calls go through one shared, pooled `requests.Session` (`_get_session`). This runs as an ordinary API call on whichever worker thread is already
evaluating a dimension/window — it does not need its own concurrency
guard or local caching; the pipeline's existing `concurrency.max_workers`
bounds it exactly like any other Azure call.
"""

from __future__ import annotations

import logging
import threading
import time

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0
_REQUEST_TIMEOUT_S = 30.0
_POOL_SIZE = 32          # >= the number of rerank calls that can be in flight at once

_session: requests.Session | None = None
_session_lock = threading.Lock()


def _get_session() -> requests.Session:
    """One `requests.Session` for the whole process, shared by every worker thread.

    A bare `requests.post` opens a new TCP connection and does a new TLS handshake for every call. With many
    threads that is a burst of handshakes, and some of them get cut (SSLEOFError, WinError 10053). A session keeps
    the connections alive in a pool and reuses them, so far fewer handshakes happen.
    """
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                session = requests.Session()
                adapter = HTTPAdapter(pool_connections=4, pool_maxsize=_POOL_SIZE)
                session.mount("https://", adapter)
                session.mount("http://", adapter)
                _session = session
    return _session


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
            response = _get_session().post(
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
