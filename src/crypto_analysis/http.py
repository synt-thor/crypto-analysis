"""Shared HTTP client with retry/backoff for public endpoints."""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import HTTP_MAX_RETRIES, HTTP_TIMEOUT_S

_client: httpx.Client | None = None


def client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            timeout=HTTP_TIMEOUT_S,
            headers={"User-Agent": "crypto-analysis/0.1"},
        )
    return _client


@retry(
    reraise=True,
    stop=stop_after_attempt(HTTP_MAX_RETRIES),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
)
def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    resp = client().get(url, params=params)
    if resp.status_code == 429:
        resp.raise_for_status()
    resp.raise_for_status()
    return resp.json()
