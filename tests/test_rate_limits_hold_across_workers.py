"""A limit counted in one worker's memory is not the limit the code states.

The default deployment runs `GUNICORN_WORKERS=2`. A client's requests
round-robin, so a limiter backed by a module-level dict counted only the
requests its own worker happened to serve, and the real budget was the
configured one times the worker count.

Measured against the running two-worker stack with
`XCELSIOR_AUTH_RATE_LIMIT_REQUESTS=10`, hitting `/api/auth/login` with a wrong
password:

    401 401 401 401 401 401 401 401 401 401 401 401 401 401 401 429
    -> first 429 at attempt 16

Fifteen attempts against a stated ten. That matters more than the ratio
suggests, because `_check_auth_rate_limit` is the *only* brute-force control
on login — there is no account lockout anywhere in this codebase.

Clearing the per-process dict after exhausting the quota is exactly what the
second worker sees: same client, same shared state, an empty local bucket. If
the limit is counted where it should be, the refusal still comes.
"""

from __future__ import annotations

from collections import defaultdict, deque
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import routes._deps as deps


@pytest.fixture(autouse=True)
def _rate_limit_config(monkeypatch, _pin_test_auth_env):
    """Exercise normal quotas after conftest raises them for endpoint tests.

    Without the explicit fixture dependency, exhausting the auth quota can
    mean 100,000 database transactions instead of ten. Shared state is reset
    by conftest; replace the fallback buckets so those are isolated too.
    """
    monkeypatch.setattr(deps, "_AUTH_RATE_LIMIT_REQUESTS", 10)
    monkeypatch.setattr(deps, "_AUTH_RATE_LIMIT_WINDOW_SEC", 300)
    monkeypatch.setattr(deps, "_BILLING_PAYMENT_RATE_LIMIT_REQUESTS", 20)
    monkeypatch.setattr(deps, "_BILLING_PAYMENT_RATE_LIMIT_WINDOW_SEC", 300)
    monkeypatch.setattr(deps, "_USE_SHARED_RUNTIME_LIMITS", True)
    monkeypatch.setattr(deps, "_AUTH_RATE_BUCKETS", defaultdict(deque))
    monkeypatch.setattr(deps, "_BILLING_PAYMENT_RATE_BUCKETS", defaultdict(deque))


class _FakeRequest:
    """Enough of a Request for `_get_real_client_ip`."""

    def __init__(self, ip: str) -> None:
        self.headers = {"x-real-ip": ip}
        self.client = None


def _exhaust_auth(ip: str) -> None:
    request = _FakeRequest(ip)
    for _ in range(deps._AUTH_RATE_LIMIT_REQUESTS):
        deps._check_auth_rate_limit(request)


@pytest.mark.parametrize("limiter", ["auth", "billing"])
def test_the_shared_path_is_the_one_being_used(limiter) -> None:
    """The precondition every test below rests on.

    With shared state unavailable the limiters fall back to the per-process
    dict, clearing it really would reset the quota, and the cross-worker tests
    would fail for a reason that has nothing to do with the defect.
    """
    if limiter == "auth":
        key = "203.0.113.1"
        namespace = deps._AUTH_RATE_STATE_NAMESPACE
        deps._check_auth_rate_limit(_FakeRequest(key))
    else:
        key = "cus_shared_probe:payment-intent"
        namespace = deps._BILLING_PAYMENT_RATE_STATE_NAMESPACE
        deps._check_billing_payment_rate_limit("cus_shared_probe", "payment-intent")

    ok, buckets = deps._shared_state_update(
        namespace, lambda: {"buckets": {}}, lambda s: (s, s.get("buckets", {}))
    )
    assert ok, (
        "shared runtime state is unavailable here, so the limiters are on their "
        "per-process fallback and the tests below cannot be trusted"
    )
    assert len(buckets.get(key, [])) == 1, "the attempt was not recorded in shared state"


def test_the_auth_limit_is_not_multiplied_by_the_worker_count() -> None:
    _exhaust_auth("203.0.113.10")

    deps._AUTH_RATE_BUCKETS.clear()  # the next attempt lands on worker 2

    with pytest.raises(HTTPException) as exc:
        deps._check_auth_rate_limit(_FakeRequest("203.0.113.10"))
    assert exc.value.status_code == 429, (
        "a second worker granted a fresh brute-force budget to a client that "
        "had already spent theirs, and login has no other rate control"
    )


def test_the_auth_limit_still_refuses_on_the_worker_that_counted() -> None:
    """The ordinary case has to keep working, not just the cross-worker one."""
    _exhaust_auth("203.0.113.11")
    with pytest.raises(HTTPException) as exc:
        deps._check_auth_rate_limit(_FakeRequest("203.0.113.11"))
    assert exc.value.status_code == 429


def test_the_auth_limit_is_per_client() -> None:
    """Over-blocking would turn one abusive IP into an outage for everyone."""
    _exhaust_auth("203.0.113.12")
    deps._check_auth_rate_limit(_FakeRequest("203.0.113.13"))


def test_the_billing_payment_limit_is_not_multiplied_by_the_worker_count() -> None:
    """Each call that gets through creates a real object at a payment provider."""
    for _ in range(deps._BILLING_PAYMENT_RATE_LIMIT_REQUESTS):
        deps._check_billing_payment_rate_limit("cus_probe", "payment-intent")

    deps._BILLING_PAYMENT_RATE_BUCKETS.clear()

    with pytest.raises(HTTPException) as exc:
        deps._check_billing_payment_rate_limit("cus_probe", "payment-intent")
    assert exc.value.status_code == 429


def test_the_billing_limit_is_per_customer_and_action() -> None:
    for _ in range(deps._BILLING_PAYMENT_RATE_LIMIT_REQUESTS):
        deps._check_billing_payment_rate_limit("cus_a", "payment-intent")
    # A different action for the same customer, and a different customer, are
    # both separate budgets.
    deps._check_billing_payment_rate_limit("cus_a", "wallet-deposit")
    deps._check_billing_payment_rate_limit("cus_b", "payment-intent")


def test_the_billing_limit_still_refuses_on_the_worker_that_counted() -> None:
    for _ in range(deps._BILLING_PAYMENT_RATE_LIMIT_REQUESTS):
        deps._check_billing_payment_rate_limit("cus_same_worker", "payment-intent")
    with pytest.raises(HTTPException) as exc:
        deps._check_billing_payment_rate_limit("cus_same_worker", "payment-intent")
    assert exc.value.status_code == 429


@pytest.mark.parametrize("limit", [0, -1])
def test_a_disabled_billing_limit_still_lets_everything_through(monkeypatch, limit) -> None:
    monkeypatch.setattr(deps, "_BILLING_PAYMENT_RATE_LIMIT_REQUESTS", limit)

    def unexpected_shared_update(*args, **kwargs):
        pytest.fail("a disabled limiter should not access shared state")

    monkeypatch.setattr(deps, "_shared_state_update", unexpected_shared_update)
    for _ in range(50):
        deps._check_billing_payment_rate_limit("cus_unlimited", "payment-intent")


@pytest.mark.parametrize("limiter", ["auth", "billing"])
def test_the_limiters_survive_shared_state_being_down(monkeypatch, limiter) -> None:
    """Degrade to per-process, never to a 500 or to no limit at all.

    `_shared_state_update` already swallows its own exceptions and reports
    unavailability, so this pins the caller's half of that contract: the
    fallback must still refuse past the quota rather than letting everything
    through while the database is unhappy.
    """
    monkeypatch.setattr(deps, "_shared_state_update", lambda *a, **k: (False, None))
    if limiter == "auth":
        check = lambda: deps._check_auth_rate_limit(_FakeRequest("203.0.113.14"))
        limit = deps._AUTH_RATE_LIMIT_REQUESTS
    else:
        check = lambda: deps._check_billing_payment_rate_limit("cus_fallback", "payment-intent")
        limit = deps._BILLING_PAYMENT_RATE_LIMIT_REQUESTS
    for _ in range(limit):
        check()
    with pytest.raises(HTTPException) as exc:
        check()
    assert exc.value.status_code == 429


@pytest.mark.parametrize("shared", [True, False], ids=["shared", "fallback"])
@pytest.mark.parametrize("limiter", ["auth", "billing"])
def test_only_expired_attempts_release_slots(monkeypatch, limiter, shared) -> None:
    """Both stores enforce the same rolling window, including its boundary."""
    if not shared:
        monkeypatch.setattr(deps, "_shared_state_update", lambda *a, **k: (False, None))
    if limiter == "auth":
        check = lambda: deps._check_auth_rate_limit(_FakeRequest("203.0.113.15"))
        limit = deps._AUTH_RATE_LIMIT_REQUESTS
        window = deps._AUTH_RATE_LIMIT_WINDOW_SEC
    else:
        check = lambda: deps._check_billing_payment_rate_limit("cus_expiry", "payment-intent")
        limit = deps._BILLING_PAYMENT_RATE_LIMIT_REQUESTS
        window = deps._BILLING_PAYMENT_RATE_LIMIT_WINDOW_SEC

    base = now = 1_000_000.0
    monkeypatch.setattr(deps, "time", SimpleNamespace(time=lambda: now))
    for _ in range(limit - 1):
        check()
    now = base + 1
    check()

    now = base + window - 0.01
    with pytest.raises(HTTPException) as exc:
        check()
    assert exc.value.status_code == 429

    now = base + window
    for _ in range(limit - 1):
        check()
    with pytest.raises(HTTPException) as exc:
        check()
    assert exc.value.status_code == 429

    now += 1
    check()
    with pytest.raises(HTTPException) as exc:
        check()
    assert exc.value.status_code == 429
