# Xcelsior — Redis-backed serverless rate limits (Phase 15 hot path)

from __future__ import annotations

import logging
import os
import threading
import time
import uuid

from cache_keys import cache_key

from serverless.limits import RateLimitExceeded, RateLimitInfo

log = logging.getLogger("xcelsior.serverless.rate_limit")

_REDIS_CLIENT = None
_REDIS_TRIED = False
_REDIS_RETRY_AT = 0.0
_REDIS_LOCK = threading.Lock()

_TAKE_SLOT_LUA = """
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window)
local count = redis.call('ZCARD', KEYS[1])
local allowed = 0
if count < limit then
    redis.call('ZADD', KEYS[1], now, ARGV[4])
    redis.call('EXPIRE', KEYS[1], 120)
    count = count + 1
    allowed = 1
end
local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
return {allowed, count, oldest[2]}
"""


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
    global _REDIS_CLIENT, _REDIS_TRIED, _REDIS_RETRY_AT
    if not redis_rate_limits_enabled():
        return None
    url = _redis_url()
    if not url:
        return None
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT
    if _REDIS_TRIED and time.monotonic() < _REDIS_RETRY_AT:
        return None
    with _REDIS_LOCK:
        if _REDIS_CLIENT is not None:
            return _REDIS_CLIENT
        if _REDIS_TRIED and time.monotonic() < _REDIS_RETRY_AT:
            return None
        _REDIS_TRIED = True
        client = None
        try:
            import redis
            from redis.backoff import NoBackoff
            from redis.retry import Retry

            timeout = _redis_timeout_sec()
            client = redis.from_url(
                url,
                decode_responses=True,
                socket_timeout=timeout,
                socket_connect_timeout=timeout,
                retry_on_timeout=False,
                retry=Retry(NoBackoff(), 0),
            )
            client.ping()
            _REDIS_CLIENT = client
            _REDIS_RETRY_AT = 0.0
            return client
        except Exception as exc:
            log.warning("Serverless Redis rate limits unavailable: %s", type(exc).__name__)
            _REDIS_RETRY_AT = time.monotonic() + 5.0
            if client is not None:
                try:
                    client.close()
                except Exception as close_error:
                    log.debug("Redis client cleanup failed: %s", type(close_error).__name__)
            return None


def check_key_rate_limit_redis(key_id: str, rpm: int) -> RateLimitInfo | None:
    """Atomic sliding-window admission; None delegates an outage to policy."""
    client = _get_redis()
    if client is None:
        return None
    limit = max(1, int(rpm))
    now = time.time()
    # `key_id` is an API key id or a `dashboard-test:{owner_id}` composite,
    # and owner ids are frequently email addresses — companion §5.4 forbids
    # those in key names, so it is hashed. Lua checks the quota before adding
    # a unique request member, and sets the TTL in the same atomic operation.
    bucket_key = cache_key("ratelimit", "serverless", secret=str(key_id))
    from redis.exceptions import RedisError

    try:
        allowed, count, oldest = client.eval(
            _TAKE_SLOT_LUA, 1, bucket_key, now, 60.0, limit, uuid.uuid4().hex
        )
    except RedisError as exc:
        log.warning("Serverless Redis rate limit check failed: %s", type(exc).__name__)
        return None
    reset_at = float(oldest) + 60.0
    if not int(allowed):
        raise RateLimitExceeded(RateLimitInfo(limit=limit, remaining=0, reset_at=reset_at))
    remaining = max(0, limit - int(count))
    return RateLimitInfo(limit=limit, remaining=remaining, reset_at=reset_at)
