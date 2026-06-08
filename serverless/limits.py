# Xcelsior — Serverless per-key RPM limits and queue backpressure

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless.repo import ServerlessRepo

DEFAULT_MAX_QUEUE_SIZE = 100
_RATE_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
_RATE_WINDOW_SEC = 60


@dataclass(frozen=True)
class RateLimitInfo:
    limit: int
    remaining: int
    reset_at: float


class RateLimitExceeded(Exception):
    def __init__(self, info: RateLimitInfo):
        self.info = info
        super().__init__("Rate limit exceeded")


class QueueFullError(Exception):
    def __init__(self, *, limit: int, depth: int):
        self.limit = limit
        self.depth = depth
        super().__init__(f"Queue full ({depth}/{limit})")


def _bucket_key(key_id: str) -> str:
    return f"serverless:key:{key_id}"


def check_key_rate_limit(key_id: str, rpm: int) -> RateLimitInfo:
    """Sliding-window per-key RPM check. Raises RateLimitExceeded when over limit."""
    limit = max(1, int(rpm))
    now = time.time()
    bucket = _RATE_BUCKETS[_bucket_key(key_id)]
    while bucket and bucket[0] <= now - _RATE_WINDOW_SEC:
        bucket.popleft()
    reset_at = (bucket[0] + _RATE_WINDOW_SEC) if bucket else (now + _RATE_WINDOW_SEC)
    remaining = max(0, limit - len(bucket))
    if len(bucket) >= limit:
        raise RateLimitExceeded(
            RateLimitInfo(limit=limit, remaining=0, reset_at=reset_at)
        )
    bucket.append(now)
    return RateLimitInfo(
        limit=limit,
        remaining=max(0, limit - len(bucket)),
        reset_at=reset_at,
    )


def rate_limit_headers(info: RateLimitInfo | None) -> dict[str, str]:
    if info is None:
        return {}
    return {
        "X-RateLimit-Limit": str(info.limit),
        "X-RateLimit-Remaining": str(info.remaining),
        "X-RateLimit-Reset": str(int(info.reset_at)),
    }


def endpoint_max_queue_size(ep: dict) -> int:
    raw = ep.get("max_queue_size")
    if raw is None:
        return DEFAULT_MAX_QUEUE_SIZE
    return max(1, int(raw))


def check_queue_capacity(repo: ServerlessRepo, endpoint_id: str, ep: dict) -> None:
    limit = endpoint_max_queue_size(ep)
    depth = repo.queue_depth(endpoint_id)
    if depth >= limit:
        raise QueueFullError(limit=limit, depth=depth)