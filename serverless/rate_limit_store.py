# Xcelsior — Redis-backed serverless rate limits (Phase 15 hot path)

from __future__ import annotations

import logging
import os
import time

from cache_keys import cache_key

from serverless.limits import RateLimitExceeded, RateLimitInfo

log = logging.getLogger("xcelsior.serverless.rate_limit")

_REDIS_CLIENT = None
_REDIS_TRIED = False


def _redis_url() -> str:
    return (
        os.environ.get("XCELSIOR_SERVERLESS_REDIS_URL", "").strip()
        or os.environ.get("XCELSIOR_AUTH_REDIS_URL", "").strip()
    )


def redis_rate_limits_enabled() -> bool:
    return os.environ.get("XCELSIOR_SERVERLESS_REDIS_RATE_LIMITS", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _redis_timeout_sec() -> float:
    """Socket timeout bound for the rate-limit Redis path.

    A Redis outage must fail fast — well under the API request deadline —
    so a stalled limiter can never hold the request open or trigger retry
    storms that exhaust the connection pool (companion §5.5).
    """
    try:
        return max(0.1, float(os.environ.get("XCELSIOR_SERVERLESS_REDIS_TIMEOUT_SEC", "2")))
    except (TypeError, ValueError):
        return 2.0


def _get_redis():
    global _REDIS_CLIENT, _REDIS_TRIED
    if _REDIS_TRIED:
        return _REDIS_CLIENT
    _REDIS_TRIED = True
    if not redis_rate_limits_enabled():
        return None
    url = _redis_url()
    if not url:
        return None
    try:
        import redis

        timeout = _redis_timeout_sec()
        # Bounded socket timeouts + a single (non-retrying) attempt so a
        # Redis hang surfaces quickly instead of blocking the request.
        _REDIS_CLIENT = redis.from_url(
            url,
            decode_responses=True,
            socket_timeout=timeout,
            socket_connect_timeout=timeout,
            retry_on_timeout=False,
        )
        _REDIS_CLIENT.ping()
        return _REDIS_CLIENT
    except Exception as exc:
        log.warning("Serverless Redis rate limits unavailable: %s", exc)
        _REDIS_CLIENT = None
        return None


def check_key_rate_limit_redis(key_id: str, rpm: int) -> RateLimitInfo | None:
    """Sliding-window RPM in Redis. Returns None when Redis is not in use."""
    client = _get_redis()
    if client is None:
        return None
    limit = max(1, int(rpm))
    now = time.time()
    # `key_id` is an API key id or a `dashboard-test:{owner_id}` composite,
    # and owner ids are frequently email addresses — companion §5.4 forbids
    # those in key names, so it is hashed. The zadd/expire pair below runs
    # in one MULTI, so the TTL is set with the key (§5.4 again).
    bucket_key = cache_key("ratelimit", "serverless", secret=str(key_id))
    window_start = now - 60.0
    pipe = client.pipeline()
    pipe.zremrangebyscore(bucket_key, 0, window_start)
    pipe.zcard(bucket_key)
    pipe.zadd(bucket_key, {str(now): now})
    pipe.expire(bucket_key, 120)
    _, count, _, _ = pipe.execute()
    reset_at = now + 60.0
    if int(count) >= limit:
        raise RateLimitExceeded(RateLimitInfo(limit=limit, remaining=0, reset_at=reset_at))
    remaining = max(0, limit - int(count) - 1)
    return RateLimitInfo(limit=limit, remaining=remaining, reset_at=reset_at)