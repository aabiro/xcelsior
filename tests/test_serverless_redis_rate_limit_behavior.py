"""Exercise Redis admission and outage recovery, including concurrent callers.

Set XCELSIOR_TEST_REDIS_URL to an isolated Redis instance for the store tests.
Each test deletes only its own randomly named key; it never flushes a database.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
import redis

from cache_keys import cache_key
from serverless import limits
from serverless import rate_limit_store as store


@pytest.fixture
def redis_bucket(monkeypatch):
    url = os.environ.get("XCELSIOR_TEST_REDIS_URL")
    if not url:
        pytest.skip("XCELSIOR_TEST_REDIS_URL is needed for real Redis tests")
    client = redis.from_url(url, decode_responses=True, socket_timeout=2, socket_connect_timeout=2)
    client.ping()
    key_id = f"audit-{uuid4().hex}"
    bucket_key = cache_key("ratelimit", "serverless", secret=key_id)
    monkeypatch.setattr(store, "_get_redis", lambda: client)
    try:
        yield key_id, bucket_key, client
    finally:
        client.delete(bucket_key)
        client.close()


def test_attempts_with_the_same_timestamp_each_consume_a_slot(redis_bucket, monkeypatch):
    key_id, _, _ = redis_bucket
    monkeypatch.setattr(store, "time", SimpleNamespace(time=lambda: 1_000_000.0))
    assert store.check_key_rate_limit_redis(key_id, 2).remaining == 1
    assert store.check_key_rate_limit_redis(key_id, 2).remaining == 0
    with pytest.raises(limits.RateLimitExceeded):
        store.check_key_rate_limit_redis(key_id, 2)


def test_rejected_attempts_do_not_extend_the_rolling_window(redis_bucket, monkeypatch):
    key_id, bucket_key, client = redis_bucket
    now = 1_000_000.0
    monkeypatch.setattr(store, "time", SimpleNamespace(time=lambda: now))
    store.check_key_rate_limit_redis(key_id, 1)
    now += 1
    for _ in range(5):
        with pytest.raises(limits.RateLimitExceeded):
            store.check_key_rate_limit_redis(key_id, 1)
    assert client.zcard(bucket_key) == 1
    assert 0 < client.ttl(bucket_key) <= 120
    now += 59
    assert store.check_key_rate_limit_redis(key_id, 1).remaining == 0


def test_reset_time_is_when_the_oldest_slot_expires(redis_bucket, monkeypatch):
    key_id, _, _ = redis_bucket
    now = 1_000_000.25
    monkeypatch.setattr(store, "time", SimpleNamespace(time=lambda: now))
    store.check_key_rate_limit_redis(key_id, 3)
    now += 1
    assert store.check_key_rate_limit_redis(key_id, 3).reset_at == 1_000_060.25


def test_concurrent_workers_admit_only_the_configured_quota(redis_bucket, monkeypatch):
    key_id, bucket_key, client = redis_bucket
    monkeypatch.setattr(store, "time", SimpleNamespace(time=lambda: 1_000_000.0))

    def attempt(_):
        try:
            store.check_key_rate_limit_redis(key_id, 7)
            return True
        except limits.RateLimitExceeded:
            return False

    with ThreadPoolExecutor(max_workers=4) as workers:
        assert sum(workers.map(attempt, range(40))) == 7
    assert client.zcard(bucket_key) == 7


def test_command_errors_reach_the_configured_outage_policy(monkeypatch):
    client = Mock()
    client.pipeline.return_value.execute.side_effect = redis.ConnectionError("test outage")
    client.eval.side_effect = redis.ConnectionError("test outage")
    monkeypatch.setattr(store, "_get_redis", lambda: client)
    monkeypatch.setenv("XCELSIOR_SERVERLESS_RATE_LIMIT_POLICY", "strict-deny")
    with pytest.raises(limits.RateLimiterUnavailable):
        limits.check_key_rate_limit("outage-key", 10)


def test_an_initial_connection_failure_can_recover(monkeypatch):
    client = Mock()
    client.ping.side_effect = [redis.ConnectionError("initial outage"), True]
    monkeypatch.setattr(redis, "from_url", lambda *a, **k: client)
    monkeypatch.setattr(store, "_REDIS_CLIENT", None)
    monkeypatch.setattr(store, "_REDIS_TRIED", False)
    monkeypatch.setattr(store, "_REDIS_RETRY_AT", 0.0, raising=False)
    monkeypatch.setenv("XCELSIOR_SERVERLESS_REDIS_RATE_LIMITS", "true")
    monkeypatch.setenv("XCELSIOR_SERVERLESS_REDIS_URL", "redis://localhost:6379/0")
    now = 100.0
    monkeypatch.setattr(store, "time", SimpleNamespace(time=time.time, monotonic=lambda: now))
    assert store._get_redis() is None
    now += 10
    assert store._get_redis() is client
