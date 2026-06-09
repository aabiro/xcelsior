# Xcelsior — Redis-backed serverless rate limits (Phase 15 hot path)

from __future__ import annotations

import logging
import os
import time

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

        _REDIS_CLIENT = redis.from_url(url, decode_responses=True)
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
    bucket_key = f"serverless:rpm:{key_id}"
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