"""OpenAI-style async Batch API for non-urgent inference at a discount."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

log = logging.getLogger("xcelsior.serverless.batch_api")

BATCH_DISCOUNT_RATE = float(__import__("os").environ.get("XCELSIOR_BATCH_DISCOUNT_RATE", "0.5"))
BATCH_STATUSES = frozenset({"validating", "in_progress", "completed", "failed", "cancelled"})


def batch_discount_multiplier() -> float:
    return max(0.1, min(1.0, 1.0 - BATCH_DISCOUNT_RATE))


def create_batch(
    repo: Any,
    *,
    endpoint_id: str,
    owner_id: str,
    requests: list[dict[str, Any]],
    completion_window: str = "24h",
) -> dict[str, Any]:
    batch_id = f"batch_{uuid.uuid4().hex[:24]}"
    now = time.time()
    if hasattr(repo, "create_batch_job"):
        return repo.create_batch_job(
            batch_id=batch_id,
            endpoint_id=endpoint_id,
            owner_id=owner_id,
            requests=requests,
            discount_rate=BATCH_DISCOUNT_RATE,
            completion_window=completion_window,
            created_at=now,
        )
    return {
        "batch_id": batch_id,
        "endpoint_id": endpoint_id,
        "status": "validating",
        "request_counts": {"total": len(requests), "completed": 0, "failed": 0},
        "discount_rate": BATCH_DISCOUNT_RATE,
        "completion_window": completion_window,
        "created_at": now,
    }


def enqueue_batch_requests(repo: Any, batch: dict, endpoint: dict) -> int:
    """Queue batch line items as serverless jobs with batch metadata."""
    if not hasattr(repo, "enqueue_batch_line_items"):
        return 0
    return repo.enqueue_batch_line_items(batch, endpoint)


def get_batch(repo: Any, batch_id: str, *, owner_id: str | None = None) -> dict | None:
    if hasattr(repo, "get_batch_job"):
        row = repo.get_batch_job(batch_id, owner_id=owner_id)
        return row
    return None


def apply_batch_token_discount(cost_cad: float, *, discount_rate: float = BATCH_DISCOUNT_RATE) -> float:
    return round(cost_cad * (1.0 - discount_rate), 6)