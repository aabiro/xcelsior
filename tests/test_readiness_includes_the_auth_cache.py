"""The readiness gates must see the dependency that gates every login.

`/readyz` checked the database schema, the serverless rate-limit policy,
worker identity, storage, and NFS. It did not check the auth cache — the one
dependency whose absence makes *every* authenticated request answer
`auth_cache_unavailable` (503). A replica in that state passed readiness, the
orchestrator routed traffic to it, and the only symptom was that nobody could
log in.

That is the same defect as a `/healthz` returning `{"ok": true}` next to a
completely absent cache, which
`tests/test_compose_provides_the_auth_cache.py` exists to prevent at the
compose layer. This file prevents it at the application layer.
"""

from __future__ import annotations

import sys
import types

import pytest

import oauth_service
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


class _BrokenCache:
    """A cache that connects but cannot serve — a password mismatch, in effect."""

    def get(self, key: str):
        raise oauth_service.AuthCacheUnavailableError("redis unavailable: connection refused")


# ── /readyz ───────────────────────────────────────────────────────────────


def test_readyz_reports_the_auth_cache_it_verified() -> None:
    r = client.get("/readyz")
    assert r.status_code == 200
    cache = r.json().get("auth_cache")
    assert cache is not None, (
        "/readyz does not report the auth cache; a gate whose result is invisible "
        "is one nobody can confirm ran"
    )
    assert cache["ok"] is True
    assert cache["backend"] == "memory"


def test_readyz_refuses_traffic_when_the_auth_cache_is_down(monkeypatch) -> None:
    """The failure this whole file exists for: ready, but nobody can log in."""
    monkeypatch.setattr(oauth_service, "get_auth_cache", lambda: _BrokenCache())
    r = client.get("/readyz")
    assert r.status_code == 503, (
        "readiness passed while the auth cache was unreachable — the orchestrator "
        "will send this replica traffic it can only answer with 503"
    )
    assert "auth cache" in r.json()["error"]["message"].lower()


def test_the_auth_cache_gate_can_be_disabled(monkeypatch) -> None:
    """Mirrors XCELSIOR_READYZ_SCHEMA_CHECK: serve the unauthenticated surface
    while the cache is being repaired."""
    monkeypatch.setattr(oauth_service, "get_auth_cache", lambda: _BrokenCache())
    monkeypatch.setenv("XCELSIOR_READYZ_AUTH_CACHE_CHECK", "false")
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json().get("auth_cache") is None


# ── /api/status (the wizard's preflight gate) ─────────────────────────────


def test_api_status_blocks_the_wizard_when_the_auth_cache_is_down(monkeypatch) -> None:
    """The preflight gate told users to proceed into a sign-in that could not work."""
    monkeypatch.setattr(oauth_service, "get_auth_cache", lambda: _BrokenCache())
    r = client.get("/api/status")
    assert r.status_code == 200  # this endpoint never raises, by design
    body = r.json()
    entry = next((s for s in body["services"] if s["name"] == "Auth cache"), None)
    assert entry is not None, "/api/status does not probe the auth cache at all"
    assert entry["state"] == "down"
    assert entry["required"] is True
    assert body["verdict"] == "blocked", (
        "a dead auth cache left the preflight verdict at "
        f"{body['verdict']!r}; the wizard would invite the user to sign in"
    )


def test_api_status_reports_a_healthy_auth_cache() -> None:
    r = client.get("/api/status")
    entry = next((s for s in r.json()["services"] if s["name"] == "Auth cache"), None)
    assert entry is not None
    assert entry["state"] == "operational"


# ── the probe itself ──────────────────────────────────────────────────────


def test_the_probe_never_raises(monkeypatch) -> None:
    """A readiness probe that can raise is a readiness probe that can 500.

    `get` is the only call it makes, but the cache is constructed lazily, so a
    surprise from *construction* reaches the probe too.
    """

    def _explode():
        raise ValueError("something entirely unexpected")

    monkeypatch.setattr(oauth_service, "get_auth_cache", _explode)
    result = oauth_service.auth_cache_healthcheck()
    assert result["ok"] is False
    assert "ValueError" in result["error"]


# ── the client the probe depends on ───────────────────────────────────────


def test_the_auth_cache_client_bounds_its_sockets(monkeypatch) -> None:
    """Unbounded sockets turn a Redis outage into a hang, not a 503.

    Every other Redis client in this codebase passes socket timeouts —
    `serverless/rate_limit_store.py`, `privacy_sinks.py`, and
    `control_plane/launch/spend_counters.py`. The auth cache, on the hot path
    of every authenticated request, was the one that did not. Without a bound,
    requests block in the threadpool holding `_cache_lock`, `/readyz` queues
    behind them and never answers, and the orchestrator sees a hang rather
    than an unready replica — so nothing restarts.
    """
    captured: dict[str, object] = {}

    class _FakeClient:
        def ping(self):
            return True

    fake = types.ModuleType("redis")

    def _from_url(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _FakeClient()

    fake.from_url = _from_url  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "redis", fake)

    oauth_service.RedisAuthCache("redis://localhost:6379/0")

    for field in ("socket_connect_timeout", "socket_timeout"):
        value = captured.get(field)
        assert isinstance(value, (int, float)) and value > 0, (
            f"RedisAuthCache passes {field}={value!r}; an unbounded auth-cache "
            "socket blocks the request thread until the OS gives up"
        )


def test_the_socket_timeout_is_configurable_and_survives_junk(monkeypatch) -> None:
    monkeypatch.setenv("XCELSIOR_AUTH_REDIS_TIMEOUT_SEC", "5")
    assert oauth_service.auth_cache_timeout_sec() == 5.0
    # A typo in the environment must not remove the bound it was setting.
    monkeypatch.setenv("XCELSIOR_AUTH_REDIS_TIMEOUT_SEC", "not-a-number")
    assert oauth_service.auth_cache_timeout_sec() == 2.0
    monkeypatch.setenv("XCELSIOR_AUTH_REDIS_TIMEOUT_SEC", "0")
    assert oauth_service.auth_cache_timeout_sec() > 0


# ── diagnosing the misconfiguration operators actually hit ────────────────


@pytest.mark.parametrize(
    "message, expected",
    [
        # Password in the URL, none on the server.
        ("Client sent AUTH, but no password is set", "XCELSIOR_REDIS_REQUIREPASS"),
        # Password on the server, none in the URL.
        ("NOAUTH Authentication required.", "XCELSIOR_AUTH_REDIS_URL"),
        # Both set, and different.
        ("WRONGPASS invalid username-password pair", "XCELSIOR_REDIS_REQUIREPASS"),
    ],
)
def test_a_password_mismatch_names_the_variable_to_change(message, expected) -> None:
    """The secret is written in three places and nothing checks that they agree.

    Redis reports the protocol exchange, not the fix. An operator otherwise
    sees `auth_cache_unavailable` in the response and a sentence about AUTH in
    the log, with no hint that a second and third variable even exist.
    """
    explained = oauth_service._explain_redis_error(RuntimeError(message))
    assert message in explained, "the original error must survive"
    assert expected in explained, (
        f"{message!r} was not translated into something actionable: {explained!r}"
    )


def test_an_unrelated_error_is_passed_through_unchanged() -> None:
    explained = oauth_service._explain_redis_error(RuntimeError("Connection refused"))
    assert explained == "Connection refused", (
        "guessing at a remediation for an error we do not recognise sends the "
        "operator to the wrong variable"
    )


# ── one connection attempt per outage, not one per request ────────────────


def test_a_failed_connection_is_remembered_briefly(monkeypatch) -> None:
    """Otherwise every waiting request pays a full connect timeout in turn.

    `get_auth_cache` builds the client under `_cache_lock`. With Redis down,
    ten concurrent requests each take the lock and each wait out the connect
    timeout — two seconds apiece, serially, so the tenth waits twenty. `/readyz`
    is one of those requests, which means the gate that exists to report this
    outage is queued behind it and times out instead of answering 503: the
    orchestrator sees a hang rather than an unready replica, and nothing
    restarts.
    """
    attempts = {"n": 0}

    class _DeadCache:
        def __init__(self, url):
            attempts["n"] += 1
            raise oauth_service.AuthCacheUnavailableError("redis unavailable: connection refused")

    oauth_service.reset_auth_cache()
    monkeypatch.setattr(oauth_service, "AUTH_CACHE_BACKEND", "redis")
    monkeypatch.setattr(oauth_service, "RedisAuthCache", _DeadCache)
    monkeypatch.setenv("XCELSIOR_AUTH_REDIS_RETRY_BACKOFF_SEC", "30")

    try:
        for _ in range(10):
            with pytest.raises(oauth_service.AuthCacheUnavailableError):
                oauth_service.get_auth_cache()
        assert attempts["n"] == 1, (
            f"{attempts['n']} connection attempts for 10 callers; each one blocks "
            "a request thread for a full connect timeout while holding the lock"
        )
    finally:
        oauth_service.reset_auth_cache()


def test_the_remembered_failure_still_says_what_went_wrong(monkeypatch) -> None:
    """A refused caller must get the real reason, not a generic placeholder.

    The whole point of `_explain_redis_error` is that the operator sees which
    variable to change; suppressing the retry must not also suppress that.
    """

    class _DeadCache:
        def __init__(self, url):
            raise oauth_service.AuthCacheUnavailableError(
                "redis unavailable: Client sent AUTH, but no password is set"
            )

    oauth_service.reset_auth_cache()
    monkeypatch.setattr(oauth_service, "AUTH_CACHE_BACKEND", "redis")
    monkeypatch.setattr(oauth_service, "RedisAuthCache", _DeadCache)
    monkeypatch.setenv("XCELSIOR_AUTH_REDIS_RETRY_BACKOFF_SEC", "30")

    try:
        with pytest.raises(oauth_service.AuthCacheUnavailableError) as first:
            oauth_service.get_auth_cache()
        with pytest.raises(oauth_service.AuthCacheUnavailableError) as second:
            oauth_service.get_auth_cache()
        assert str(second.value) == str(first.value), (
            "the suppressed retry reported something different from the attempt "
            f"it stood in for: {second.value!r} vs {first.value!r}"
        )
    finally:
        oauth_service.reset_auth_cache()


def test_the_cache_recovers_once_the_backoff_lapses(monkeypatch) -> None:
    """A backoff that never lapses is an outage that never ends."""
    state = {"fail": True, "attempts": 0}

    class _Flaky:
        def __init__(self, url):
            state["attempts"] += 1
            if state["fail"]:
                raise oauth_service.AuthCacheUnavailableError("redis unavailable")

        def get(self, key):
            return None

    oauth_service.reset_auth_cache()
    monkeypatch.setattr(oauth_service, "AUTH_CACHE_BACKEND", "redis")
    monkeypatch.setattr(oauth_service, "RedisAuthCache", _Flaky)
    monkeypatch.setenv("XCELSIOR_AUTH_REDIS_RETRY_BACKOFF_SEC", "0")

    try:
        with pytest.raises(oauth_service.AuthCacheUnavailableError):
            oauth_service.get_auth_cache()
        state["fail"] = False
        assert oauth_service.get_auth_cache() is not None, (
            "the cache stayed refused after Redis came back"
        )
        assert state["attempts"] == 2
    finally:
        oauth_service.reset_auth_cache()


def test_a_healthy_cache_is_built_once(monkeypatch) -> None:
    """The success path must not have acquired a per-request cost."""
    built = {"n": 0}

    class _Counting(oauth_service.MemoryAuthCache):
        def __init__(self):
            super().__init__()
            built["n"] += 1

    oauth_service.reset_auth_cache()
    monkeypatch.setattr(oauth_service, "AUTH_CACHE_BACKEND", "memory")
    monkeypatch.setattr(oauth_service, "MemoryAuthCache", _Counting)
    try:
        for _ in range(5):
            oauth_service.get_auth_cache()
        assert built["n"] == 1
    finally:
        oauth_service.reset_auth_cache()
