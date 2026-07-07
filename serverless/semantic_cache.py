"""GPTCache-style semantic cache in front of the OpenAI proxy (U2.2 threshold 0.92)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from difflib import SequenceMatcher
from typing import Any

log = logging.getLogger("xcelsior.serverless.semantic_cache")

SIMILARITY_THRESHOLD = float(os.environ.get("XCELSIOR_SEMANTIC_CACHE_THRESHOLD", "0.92"))
MAX_ENTRIES_PER_ENDPOINT = int(os.environ.get("XCELSIOR_SEMANTIC_CACHE_MAX", "500"))


def semantic_cache_enabled() -> bool:
    return os.environ.get("XCELSIOR_SEMANTIC_CACHE_ENABLED", "1").lower() in (
        "1",
        "true",
        "yes",
    )


def _normalize_prompt(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    return t[:8000]


def extract_prompt_text(body: dict[str, Any]) -> str:
    messages = body.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for m in messages:
            if isinstance(m, dict):
                parts.append(str(m.get("content") or ""))
        return _normalize_prompt("\n".join(parts))
    if "input" in body:
        inp = body["input"]
        if isinstance(inp, list):
            return _normalize_prompt("\n".join(str(x) for x in inp))
        return _normalize_prompt(str(inp))
    return _normalize_prompt(str(body.get("prompt") or ""))


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def prompt_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def try_cache_hit(repo: Any, endpoint_id: str, body: dict[str, Any]) -> dict[str, Any] | None:
    if not semantic_cache_enabled() or not endpoint_id:
        return None
    prompt = extract_prompt_text(body)
    if not prompt:
        return None
    if hasattr(repo, "semantic_cache_lookup"):
        row = repo.semantic_cache_lookup(endpoint_id, prompt, threshold=SIMILARITY_THRESHOLD)
        if row:
            return {
                "response": row.get("response_json") or row.get("response"),
                "usage": row.get("usage_json") or row.get("usage") or {},
                "cache_hit": True,
                "similarity": float(row.get("similarity") or 1.0),
                "cache_id": row.get("cache_id"),
            }
    return None


def store_cache_entry(
    repo: Any,
    endpoint_id: str,
    body: dict[str, Any],
    response: dict[str, Any],
    usage: dict[str, int],
) -> None:
    if not semantic_cache_enabled() or not hasattr(repo, "semantic_cache_store"):
        return
    prompt = extract_prompt_text(body)
    if not prompt:
        return
    repo.semantic_cache_store(
        endpoint_id,
        prompt,
        response=response,
        usage=usage,
        max_entries=MAX_ENTRIES_PER_ENDPOINT,
    )


def meter_cache_savings(
    repo: Any,
    endpoint_id: str,
    *,
    saved_cost_cad: float,
    similarity: float,
) -> None:
    if saved_cost_cad <= 0 or not hasattr(repo, "record_semantic_cache_savings"):
        return
    repo.record_semantic_cache_savings(endpoint_id, saved_cost_cad, similarity=similarity)


def accrue_cache_hit_usage(
    repo: Any,
    endpoint: dict,
    usage: dict[str, int],
    *,
    idempotency_key: str | None,
    saved_cost_cad: float = 0.0,
    similarity: float = 1.0,
) -> dict[str, Any]:
    """Record near-zero token accrual for semantic cache hits (meter savings)."""
    from serverless.openai_proxy import accrue_proxy_token_usage

    cached_usage = dict(usage)
    cached_usage["cached_tokens"] = int(cached_usage.get("input_tokens") or 0)
    result = accrue_proxy_token_usage(
        repo,
        endpoint,
        cached_usage,
        idempotency_key=idempotency_key,
    )
    if saved_cost_cad > 0:
        meter_cache_savings(
            repo,
            str(endpoint.get("endpoint_id") or ""),
            saved_cost_cad=saved_cost_cad,
            similarity=similarity,
        )
    result["semantic_cache_hit"] = True
    return result