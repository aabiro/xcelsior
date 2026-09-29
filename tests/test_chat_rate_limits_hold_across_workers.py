"""Paid LLM calls must consume a shared quota, not a quota per API worker."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

import ai_assistant
import chat
import routes._deps as deps


@pytest.fixture(params=["chat", "assistant"])
def limiter(request, monkeypatch):
    if request.param == "chat":
        module, check, buckets, limit_name, window = (
            chat,
            chat.check_chat_rate_limit,
            chat._chat_rate_buckets,
            "CHAT_RATE_LIMIT",
            60,
        )
    else:
        module, check, buckets, limit_name, window = (
            ai_assistant,
            ai_assistant.check_ai_rate_limit,
            ai_assistant._ai_rate_buckets,
            "AI_RATE_LIMIT",
            ai_assistant.RATE_LIMIT_WINDOW_SEC,
        )
    monkeypatch.setattr(module, limit_name, 3)
    monkeypatch.setattr(deps, "_USE_SHARED_RUNTIME_LIMITS", True)
    buckets.clear()
    yield SimpleNamespace(module=module, check=check, buckets=buckets, window=window)
    buckets.clear()


def test_clearing_one_workers_memory_does_not_grant_a_new_quota(limiter):
    for _ in range(3):
        assert limiter.check("same-client") is True
    limiter.buckets.clear()
    assert limiter.check("same-client") is False
    assert limiter.check("different-client") is True


def test_concurrent_requests_share_one_budget(limiter):
    barrier = Barrier(2)

    def requests():
        barrier.wait(timeout=5)
        return sum(limiter.check("concurrent-client") for _ in range(3))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(requests) for _ in range(2)]
        assert sum(future.result(timeout=10) for future in futures) == 3


@pytest.mark.parametrize("shared", [True, False], ids=["shared", "fallback"])
def test_only_the_expired_slots_become_available(limiter, monkeypatch, shared):
    if not shared:
        monkeypatch.setattr(deps, "_shared_state_update", lambda *a, **k: (False, None))
    now = base = 1_000_000.0
    monkeypatch.setattr(
        limiter.module, "time", SimpleNamespace(time=lambda: now, monotonic=lambda: now)
    )
    assert limiter.check("expiry-client")
    assert limiter.check("expiry-client")
    now += 1
    assert limiter.check("expiry-client")
    assert not limiter.check("expiry-client")
    now = base + limiter.window
    assert limiter.check("expiry-client")
    assert limiter.check("expiry-client")
    assert not limiter.check("expiry-client")


def test_shared_store_failure_still_enforces_a_local_limit(limiter, monkeypatch):
    monkeypatch.setattr(deps, "_shared_state_update", lambda *a, **k: (False, None))
    for _ in range(3):
        assert limiter.check("fallback-client") is True
    assert limiter.check("fallback-client") is False
