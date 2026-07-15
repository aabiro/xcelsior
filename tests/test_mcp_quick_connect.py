"""Functional coverage for the /dashboard/mcp quick-connect endpoint.

Verifies the always-there copy-paste token flow: find-or-create a system-managed
MCP client, mint a live token every call, rotate on regenerate, and never surface
the client in the user-facing OAuth client list.
"""

import os

import pytest
from fastapi.testclient import TestClient

import scheduler

os.environ.setdefault("XCELSIOR_API_TOKEN", "testtoken")
os.environ.setdefault("XCELSIOR_ENV", "test")

from api import app

client = TestClient(app)


def _register_and_get_token(email: str, password: str = "testpass123") -> str:
    reg = client.post("/api/auth/register", json={"email": email, "password": password})
    assert reg.status_code == 200, reg.text
    body = reg.json()
    if body.get("access_token"):
        return body["access_token"]
    if body.get("email_verification_required"):
        import routes._deps as _deps_mod
        from db import auth_connection

        token = None
        if _deps_mod._USE_PERSISTENT_AUTH:
            with auth_connection() as conn:
                row = conn.execute(
                    "SELECT email_verification_token FROM users WHERE email = %s",
                    (email,),
                ).fetchone()
            token = row["email_verification_token"] if row else None
        else:
            token = _deps_mod._users_db.get(email, {}).get("email_verification_token")
        assert token, f"missing verification token for {email}"
        verified = client.post("/api/auth/verify-email", json={"token": token})
        assert verified.status_code == 200, verified.text
        if verified.json().get("access_token"):
            return verified.json()["access_token"]
    login = client.post("/api/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200, login.text
    return login.json()["access_token"]


@pytest.fixture(autouse=True)
def clean_state():
    import routes._deps as _deps_mod
    from oauth_service import reset_auth_cache_for_tests
    from db import auth_connection

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
    _deps_mod._users_db.clear()
    _deps_mod._sessions.clear()
    yield


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_quick_connect_returns_live_token():
    token = _register_and_get_token("qc-basic@xcelsior.ca")
    r = client.get("/api/mcp/quick-connect", headers=_auth(token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["access_token"]
    assert body["expires_in"] > 0
    assert body["mcp_url"].endswith("/mcp")
    assert "gpu:read" in body["scopes"]


def test_quick_connect_is_idempotent():
    token = _register_and_get_token("qc-idem@xcelsior.ca")
    first = client.get("/api/mcp/quick-connect", headers=_auth(token)).json()
    second = client.get("/api/mcp/quick-connect", headers=_auth(token)).json()
    # Same underlying client (find-or-create), fresh token each time.
    assert first["client_id"] == second["client_id"]
    assert first["access_token"] and second["access_token"]


def test_regenerate_rotates_the_client():
    token = _register_and_get_token("qc-regen@xcelsior.ca")
    first = client.get("/api/mcp/quick-connect", headers=_auth(token)).json()
    rotated = client.get(
        "/api/mcp/quick-connect?regenerate=true", headers=_auth(token)
    ).json()
    assert rotated["client_id"] != first["client_id"]


def test_quick_connect_client_excluded_from_client_list():
    token = _register_and_get_token("qc-hidden@xcelsior.ca")
    # Provision the quick-connect client.
    client.get("/api/mcp/quick-connect", headers=_auth(token))
    # It must not appear in the user's manual OAuth client list.
    listing = client.get("/api/oauth/clients", headers=_auth(token)).json()
    names = [c.get("client_name") for c in listing.get("clients", [])]
    assert "mcp-quick-connect" not in names, names


def test_quick_connect_requires_auth():
    r = client.get("/api/mcp/quick-connect")
    assert r.status_code in (401, 403)
