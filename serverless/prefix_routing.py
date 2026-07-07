"""KV-aware routing: session affinity by prompt-prefix hash (LMCache small-scale)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

log = logging.getLogger("xcelsior.serverless.prefix_routing")

DYNAMO_HOST_THRESHOLD = int(os.environ.get("XCELSIOR_DYNAMO_HOST_THRESHOLD", "4"))


def prefix_routing_enabled() -> bool:
    return os.environ.get("XCELSIOR_PREFIX_ROUTING_ENABLED", "1").lower() in (
        "1",
        "true",
        "yes",
    )


def prefix_hash_from_payload(payload: dict[str, Any]) -> str:
    """Stable hash of prompt prefix for KV affinity."""
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        first = messages[0] if isinstance(messages[0], dict) else {}
        text = str(first.get("content") or "")[:2048]
    else:
        text = str(payload.get("input") or payload.get("prompt") or "")[:2048]
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def lmcache_remote_env() -> dict[str, str]:
    """Optional LMCache remote tier (Mooncake-compatible URI when configured)."""
    remote = os.environ.get("XCELSIOR_LMCACHE_REMOTE_URL", "").strip()
    if not remote:
        return {}
    return {
        "LMCACHE_REMOTE_URL": remote,
        "LMCACHE_REMOTE_SERDE": os.environ.get("XCELSIOR_LMCACHE_REMOTE_SERDE", "naive"),
    }


def use_dynamo_router(llm_host_count: int) -> bool:
    return llm_host_count > DYNAMO_HOST_THRESHOLD


def rank_workers_for_prefix(
    workers: list[dict],
    prefix_hash: str,
    *,
    affinity: dict[str, str] | None = None,
) -> list[dict]:
    """Prefer worker that last served this prefix hash."""
    if not prefix_routing_enabled() or not workers:
        return workers
    aff = affinity or {}
    preferred_id = aff.get(prefix_hash)
    if not preferred_id:
        return workers
    preferred = [w for w in workers if str(w.get("worker_id")) == preferred_id]
    others = [w for w in workers if str(w.get("worker_id")) != preferred_id]
    return preferred + others


def select_worker_with_prefix(
    repo: Any,
    endpoint_id: str,
    workers: list[dict],
    payload: dict[str, Any],
) -> dict | None:
    if not workers:
        return None
    prefix_hash = prefix_hash_from_payload(payload)
    affinity: dict[str, str] = {}
    if hasattr(repo, "get_prefix_affinities"):
        affinity = repo.get_prefix_affinities(endpoint_id)
    ranked = rank_workers_for_prefix(workers, prefix_hash, affinity=affinity)
    for w in ranked:
        if str(w.get("state") or "") in ("ready", "idle"):
            if hasattr(repo, "record_prefix_affinity"):
                repo.record_prefix_affinity(endpoint_id, prefix_hash, str(w["worker_id"]))
            return w
    return ranked[0] if ranked else None