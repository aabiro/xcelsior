"""The provider network test downloads a fixed payload from the scheduler.

The wizard measures throughput against this endpoint, so it must return the
whole payload, refuse callers who cannot register hosts, and stay bounded.
"""

import os

import pytest
from fastapi.testclient import TestClient

import scheduler

os.environ.setdefault("XCELSIOR_API_TOKEN", "testtoken")
os.environ.setdefault("XCELSIOR_ENV", "test")

from api import app
from tests.test_mcp_quick_connect import _auth, _register_and_get_token

client = TestClient(app)
PATH = "/api/diagnostics/network-download"


@pytest.fixture(autouse=True)
def clean_state():
    import routes._deps as _deps_mod
    import routes.health as health
    from db import auth_connection
    from oauth_service import reset_auth_cache_for_tests

    with scheduler._atomic_mutation() as conn:
        conn.execute("DELETE FROM state")
    with auth_connection() as conn:
        conn.execute("DELETE FROM oauth_refresh_tokens")
        conn.execute("DELETE FROM oauth_clients")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM users")
    reset_auth_cache_for_tests()
    client.cookies.clear()
    _deps_mod._RATE_BUCKETS.clear()
    _deps_mod._AUTH_RATE_BUCKETS.clear()
    health._NETWORK_DIAGNOSTIC_BUCKETS.clear()
    yield


def test_returns_the_full_uncompressed_payload():
    token = _register_and_get_token("net-diag@xcelsior.ca")
    r = client.get(PATH, headers=_auth(token))
    assert r.status_code == 200, r.text
    assert len(r.content) == 8 * 1024 * 1024
    assert r.headers["content-type"] == "application/octet-stream"
    assert r.headers["cache-control"] == "no-store, no-transform"


def test_requires_a_signed_in_caller(monkeypatch):
    import routes._deps as _deps_mod

    monkeypatch.setattr(_deps_mod, "AUTH_REQUIRED", True)
    client.cookies.clear()
    assert client.get(PATH).status_code == 401


def test_a_key_without_host_scope_is_refused():
    token = _register_and_get_token("net-diag-scope@xcelsior.ca")
    key = client.get("/api/mcp/quick-connect", headers=_auth(token)).json()["access_token"]
    client.cookies.clear()
    r = client.get(PATH, headers=_auth(key))
    assert r.status_code == 403, r.text


def test_is_rate_limited_per_owner():
    token = _register_and_get_token("net-diag-rate@xcelsior.ca")
    for _ in range(4):
        assert client.get(PATH, headers=_auth(token)).status_code == 200
    limited = client.get(PATH, headers=_auth(token))
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "60"


def test_falls_back_to_a_local_window_when_shared_state_is_unavailable(monkeypatch):
    import routes._deps as _deps_mod

    token = _register_and_get_token("net-diag-local@xcelsior.ca")
    monkeypatch.setattr(_deps_mod, "_USE_SHARED_RUNTIME_LIMITS", False)
    for _ in range(4):
        assert client.get(PATH, headers=_auth(token)).status_code == 200
    assert client.get(PATH, headers=_auth(token)).status_code == 429
