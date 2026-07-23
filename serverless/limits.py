# Xcelsior — Serverless per-key RPM limits and queue backpressure

from __future__ import annotations

import enum
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless.repo import ServerlessRepo

log = logging.getLogger("xcelsior.serverless.limits")

DEFAULT_MAX_QUEUE_SIZE = 100
_RATE_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
_RATE_WINDOW_SEC = 60


# ── Rate-limit degradation policy (data-architecture companion §2.3/§5.5) ──
# The Redis limiter is the only *global* rate limiter — process-local deque
# buckets grant one quota per replica (a nominal 60 RPM becomes ~60 × the
# replica count). Production must therefore declare what happens when the
# global limiter is unavailable instead of silently degrading to
# per-process memory.
class RateLimitPolicy(str, enum.Enum):
    # Fail closed: without the global limiter, deny the protected
    # operation rather than admit unbounded per-process traffic.
    STRICT_DENY = "strict-deny"
    # A separately-available gateway already owns the global limit; the
    # app-level check passes through (valid ONLY when such a gateway
    # actually enforces it — an explicit operator assertion).
    UPSTREAM_ENFORCED = "upstream-enforced"
    # Local per-process deque is acceptable — development/test only.
    DISABLED_FOR_DEVELOPMENT = "disabled-for-development"


_VALID_POLICIES = {p.value for p in RateLimitPolicy}


class RateLimitPolicyError(RuntimeError):
    """Production serverless rate-limit policy is undefined or invalid."""


@dataclass(frozen=True)
class RateLimitInfo:
    limit: int
    remaining: int
    reset_at: float


class RateLimitExceeded(Exception):
    def __init__(self, info: RateLimitInfo):
        self.info = info
        super().__init__("Rate limit exceeded")


class RateLimiterUnavailable(RateLimitExceeded):
    """Global limiter is unavailable under a strict-deny policy — the
    request is failed closed. Subclasses RateLimitExceeded so existing
    handlers deny it (429) with no route change, while remaining
    distinguishable in logs/metrics from a genuine over-limit."""


class QueueFullError(Exception):
    def __init__(self, *, limit: int, depth: int):
        self.limit = limit
        self.depth = depth
        super().__init__(f"Queue full ({depth}/{limit})")


def _is_production() -> bool:
    return os.environ.get("XCELSIOR_ENV", "").strip().lower() == "production"


def rate_limit_policy() -> RateLimitPolicy:
    """Resolve the configured degradation policy.

    In production an undefined/invalid policy raises (fail closed — there
    is no safe silent default). Outside production the local memory
    limiter is acceptable, so it defaults to ``disabled-for-development``.
    """
    raw = os.environ.get("XCELSIOR_SERVERLESS_RATE_LIMIT_POLICY", "").strip().lower()
    if raw in _VALID_POLICIES:
        return RateLimitPolicy(raw)
    if _is_production():
        raise RateLimitPolicyError(
            "XCELSIOR_SERVERLESS_RATE_LIMIT_POLICY must be one of "
            f"{sorted(_VALID_POLICIES)} in production; got {raw!r}. "
            "Refusing to silently fall back to per-process rate limiting."
        )
    return RateLimitPolicy.DISABLED_FOR_DEVELOPMENT


def validate_rate_limit_policy() -> None:
    """Startup check: reject an undefined production policy early.

    Call from service startup / readiness so a misconfigured production
    deployment fails fast instead of at the first rate-limited request.
    """
    rate_limit_policy()


def _bucket_key(key_id: str) -> str:
    """Key for the process-local dev/test bucket.

    This one never reaches Redis, but it goes through the same contract
    helper anyway: `key_id` can be a `dashboard-test:{owner_id}` composite
    and owner ids are frequently email addresses, and having exactly one
    way to build a cache key means there is no exception to remember if
    this store is ever swapped for a shared one.
    """
    from cache_keys import cache_key

    return cache_key("ratelimit", "serverless_local", secret=str(key_id))


def _local_deque_check(key_id: str, rpm: int) -> RateLimitInfo:
    """Process-local sliding window. Per-replica, non-global (dev/test)."""
    limit = max(1, int(rpm))
    now = time.time()
    bucket = _RATE_BUCKETS[_bucket_key(key_id)]
    while bucket and bucket[0] <= now - _RATE_WINDOW_SEC:
        bucket.popleft()
    reset_at = (bucket[0] + _RATE_WINDOW_SEC) if bucket else (now + _RATE_WINDOW_SEC)
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


def check_key_rate_limit(key_id: str, rpm: int) -> RateLimitInfo:
    """Sliding-window per-key RPM check. Raises RateLimitExceeded when over
    limit (or RateLimiterUnavailable when the global limiter is down under
    a strict-deny policy).

    When the Redis (global) limiter is in effect its verdict is
    authoritative for every policy. Only when it is NOT in effect does the
    degradation policy decide the behavior — never a silent per-process
    fallback in production (companion §2.3/§5.5).
    """
    from serverless.rate_limit_store import check_key_rate_limit_redis

    redis_info = check_key_rate_limit_redis(key_id, rpm)
    if redis_info is not None:
        return redis_info

    policy = rate_limit_policy()
    if policy is RateLimitPolicy.DISABLED_FOR_DEVELOPMENT:
        return _local_deque_check(key_id, rpm)

    limit = max(1, int(rpm))
    if policy is RateLimitPolicy.UPSTREAM_ENFORCED:
        # A tested gateway owns the global limit; do not enforce (and do
        # not fabricate per-process quotas). Report permissive info.
        return RateLimitInfo(
            limit=limit, remaining=limit, reset_at=time.time() + _RATE_WINDOW_SEC
        )

    # strict-deny: fail closed rather than admit unbounded per-process load.
    log.error(
        "serverless rate limiter unavailable (policy=strict-deny) — denying "
        "key=%s; the global Redis limiter is not in effect", key_id,
    )
    raise RateLimiterUnavailable(
        RateLimitInfo(limit=limit, remaining=0, reset_at=time.time() + _RATE_WINDOW_SEC)
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