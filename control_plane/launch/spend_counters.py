"""Atomic rolling spend reservations for MCP client policies.

PostgreSQL remains authoritative for plans and ledger entries. Redis is the
coordination layer for fast, cross-replica hourly/daily policy ceilings. A
reservation is idempotent per action plan and deliberately conservative on a
process crash: it expires with the budget window instead of risking overspend.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import redis


class SpendCounterUnavailable(RuntimeError):
    pass


class SpendLimitExceeded(RuntimeError):
    def __init__(self, window: str, limit_micros: int, projected_micros: int):
        self.window = window
        self.limit_micros = limit_micros
        self.projected_micros = projected_micros
        super().__init__(
            f"{window} spend ceiling {limit_micros} micro-CAD would be exceeded "
            f"by projected spend {projected_micros}"
        )


@dataclass(frozen=True)
class SpendReservation:
    plan_id: str
    hourly_key: str
    daily_key: str
    marker_key: str
    amount_micros: int
    backend: str
    replay: bool = False


_LUA_RESERVE = """
local marker = redis.call('GET', KEYS[3])
if marker then return {2, tonumber(redis.call('GET', KEYS[1]) or '0'),
                          tonumber(redis.call('GET', KEYS[2]) or '0')} end
local hourly = tonumber(redis.call('GET', KEYS[1]) or '0')
local daily = tonumber(redis.call('GET', KEYS[2]) or '0')
local amount = tonumber(ARGV[1])
local hourly_limit = tonumber(ARGV[2])
local daily_limit = tonumber(ARGV[3])
if hourly_limit >= 0 and hourly + amount > hourly_limit then
  return {-1, hourly + amount, daily + amount}
end
if daily_limit >= 0 and daily + amount > daily_limit then
  return {-2, hourly + amount, daily + amount}
end
redis.call('INCRBY', KEYS[1], amount)
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[4]))
redis.call('INCRBY', KEYS[2], amount)
redis.call('EXPIRE', KEYS[2], tonumber(ARGV[5]))
redis.call('SET', KEYS[3], amount, 'EX', tonumber(ARGV[5]), 'NX')
return {1, hourly + amount, daily + amount}
"""

_LUA_RELEASE = """
local amount = redis.call('GET', KEYS[3])
if not amount then return 0 end
amount = tonumber(amount)
local hourly = tonumber(redis.call('GET', KEYS[1]) or '0')
local daily = tonumber(redis.call('GET', KEYS[2]) or '0')
redis.call('SET', KEYS[1], math.max(0, hourly - amount), 'KEEPTTL')
redis.call('SET', KEYS[2], math.max(0, daily - amount), 'KEEPTTL')
redis.call('DEL', KEYS[3])
return 1
"""

_memory_lock = threading.Lock()
_memory_values: dict[str, tuple[int, float]] = {}
_redis_client: Any | None = None


def reset_for_tests() -> None:
    global _redis_client
    with _memory_lock:
        _memory_values.clear()
    _redis_client = None


def _backend() -> str:
    explicit = os.environ.get("MCP_RATE_LIMIT_BACKEND", "").strip().lower()
    if explicit in {"redis", "memory"}:
        return explicit
    return (
        "redis"
        if (
            os.environ.get("XCELSIOR_MCP_REDIS_URL")
            or os.environ.get("MCP_REDIS_URL")
            or os.environ.get("REDIS_URL")
        )
        else "memory"
    )


def _redis() -> Any:
    global _redis_client
    if _redis_client is None:
        url = (
            os.environ.get("XCELSIOR_MCP_REDIS_URL")
            or os.environ.get("MCP_REDIS_URL")
            or os.environ.get("REDIS_URL")
            or "redis://127.0.0.1:6379/0"
        )
        _redis_client = redis.Redis.from_url(
            url, socket_connect_timeout=1.0, socket_timeout=1.0, decode_responses=True
        )
    return _redis_client


def _keys(client_id: str, tenant_id: str, plan_id: str, now: float) -> tuple[str, str, str]:
    hour = time.strftime("%Y%m%d%H", time.gmtime(now))
    day = time.strftime("%Y%m%d", time.gmtime(now))
    prefix = f"mcp:spend:{tenant_id}:{client_id}"
    return f"{prefix}:hour:{hour}", f"{prefix}:day:{day}", f"{prefix}:plan:{plan_id}"


def _window_ttls(now: float) -> tuple[int, int]:
    next_hour = (int(now) // 3600 + 1) * 3600
    next_day = (int(now) // 86400 + 1) * 86400
    # Keep a small grace period for retries straddling the boundary.
    return max(60, next_hour - int(now) + 60), max(60, next_day - int(now) + 60)


def reserve(
    *,
    plan_id: str,
    client_id: str,
    tenant_id: str,
    amount_micros: int,
    hourly_limit_micros: int | None,
    daily_limit_micros: int | None,
    now: float | None = None,
) -> SpendReservation | None:
    """Atomically reserve spend or raise before any wallet/job side effect."""
    if hourly_limit_micros is None and daily_limit_micros is None:
        return None
    amount = max(0, int(amount_micros))
    now = time.time() if now is None else now
    hourly_key, daily_key, marker_key = _keys(client_id, tenant_id, plan_id, now)
    hour_ttl, day_ttl = _window_ttls(now)
    hourly_limit = -1 if hourly_limit_micros is None else int(hourly_limit_micros)
    daily_limit = -1 if daily_limit_micros is None else int(daily_limit_micros)
    backend = _backend()

    if backend == "memory":
        with _memory_lock:
            for key in (hourly_key, daily_key, marker_key):
                value = _memory_values.get(key)
                if value and value[1] <= now:
                    _memory_values.pop(key, None)
            if marker_key in _memory_values:
                return SpendReservation(
                    plan_id, hourly_key, daily_key, marker_key, amount, backend, replay=True
                )
            hourly = _memory_values.get(hourly_key, (0, now + hour_ttl))[0]
            daily = _memory_values.get(daily_key, (0, now + day_ttl))[0]
            if hourly_limit >= 0 and hourly + amount > hourly_limit:
                raise SpendLimitExceeded("hourly", hourly_limit, hourly + amount)
            if daily_limit >= 0 and daily + amount > daily_limit:
                raise SpendLimitExceeded("daily", daily_limit, daily + amount)
            _memory_values[hourly_key] = (hourly + amount, now + hour_ttl)
            _memory_values[daily_key] = (daily + amount, now + day_ttl)
            _memory_values[marker_key] = (amount, now + day_ttl)
        return SpendReservation(plan_id, hourly_key, daily_key, marker_key, amount, backend)

    try:
        result = _redis().eval(
            _LUA_RESERVE,
            3,
            hourly_key,
            daily_key,
            marker_key,
            amount,
            hourly_limit,
            daily_limit,
            hour_ttl,
            day_ttl,
        )
    except Exception as exc:
        raise SpendCounterUnavailable("distributed spend counters unavailable") from exc
    code, projected_hourly, projected_daily = map(int, result)
    if code == -1:
        raise SpendLimitExceeded("hourly", hourly_limit, projected_hourly)
    if code == -2:
        raise SpendLimitExceeded("daily", daily_limit, projected_daily)
    return SpendReservation(
        plan_id, hourly_key, daily_key, marker_key, amount, backend, replay=code == 2
    )


def release(reservation: SpendReservation | None) -> None:
    """Compensate a reservation when execution fails before durable success."""
    if reservation is None or reservation.replay:
        return
    if reservation.backend == "memory":
        with _memory_lock:
            marker = _memory_values.pop(reservation.marker_key, None)
            if marker is None:
                return
            for key in (reservation.hourly_key, reservation.daily_key):
                current = _memory_values.get(key)
                if current:
                    _memory_values[key] = (
                        max(0, current[0] - reservation.amount_micros),
                        current[1],
                    )
        return
    try:
        _redis().eval(
            _LUA_RELEASE,
            3,
            reservation.hourly_key,
            reservation.daily_key,
            reservation.marker_key,
        )
    except Exception:
        # Conservative crash behavior: the reservation expires at the window.
        return
