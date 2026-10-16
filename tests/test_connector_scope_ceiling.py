"""A connector can sign in when it asks for what discovery advertised.

VS Code's MCP client registers through `POST /oauth/register` without a
`scope`, so the registration gets the read-biased default, and then sends the
authorization server's whole `scopes_supported` to `/oauth/authorize`. Every
scope outside that default was refused:

    Error 400 — Unsupported permissions
    This application asked for permissions it is not registered for:
    billing:write, hosts:read, hosts:write, inference:write, instances:operate,
    instances:write, mcp_actions:approve

so the connector could not connect at all. A connector's registered scopes are
now its default, its ceiling is `CONNECTOR_ALLOWED_SCOPES`, and anything it asks
for beyond the ceiling is left out of the grant (RFC 6749 §3.3) instead of
failing the sign-in.

The ceiling had also drifted: SSH, volume, artifact, terminal and provider
scopes were added to the quick-connect MCP token in P2–P6 but never to it, so a
self-registered connector could not hold what its own tools require.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import secrets

import pytest
from fastapi.testclient import TestClient

import scheduler

os.environ.setdefault("XCELSIOR_API_TOKEN", "testtoken")
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

from api import app  # noqa: E402
from oauth_registration import CONNECTOR_ALLOWED_SCOPES  # noqa: E402
from oauth_service import MCP_QUICK_CONNECT_SCOPES, MCP_RESOURCE_AUDIENCE  # noqa: E402

client = TestClient(app)

#: Never grantable to a connector, whatever it asks for.
WITHHELD = {"billing:write", "hosts:read", "hosts:write"}


@pytest.fixture(autouse=True)
def _clean_state():
    from db import auth_connection
    from oauth_service import reset_auth_cache_for_tests
    import oauth_service as _oauth
    import routes._deps as _deps_mod

    with scheduler._atomic_mutation() as conn:
        conn.execute("DELETE FROM state")
    with auth_connection() as conn:
        conn.execute("DELETE FROM oauth_consent_grants")
        conn.execute("DELETE FROM oauth_refresh_tokens")
        conn.execute("DELETE FROM oauth_clients")
        conn.execute("DELETE FROM api_keys")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM users")
    _oauth._defaults_ready = False
    reset_auth_cache_for_tests()
    client.cookies.clear()
    _deps_mod._RATE_BUCKETS.clear()
    _deps_mod._AUTH_RATE_BUCKETS.clear()
    _deps_mod._users_db.clear()
    _deps_mod._sessions.clear()
    _deps_mod._api_keys.clear()
    yield


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )
    return verifier, challenge


def _sign_in(email: str, password: str = "ConnectorTest123!") -> None:
    """Register and leave the session cookie on this module's client."""
    reg = client.post("/api/auth/register", json={"email": email, "password": password})
    assert reg.status_code == 200, reg.text
    if reg.json().get("email_verification_required"):
        import routes._deps as _deps_mod
        from db import auth_connection

        if _deps_mod._USE_PERSISTENT_AUTH:
            with auth_connection() as conn:
                row = conn.execute(
                    "SELECT email_verification_token FROM users WHERE email = %s", (email,)
                ).fetchone()
            token = row["email_verification_token"] if row else None
        else:
            token = _deps_mod._users_db.get(email, {}).get("email_verification_token")
        assert token, f"missing verification token for {email}"
        verified = client.post("/api/auth/verify-email", json={"token": token})
        assert verified.status_code == 200, verified.text
    login = client.post("/api/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200, login.text


def _authorize(client_id: str, redirect_uri: str, scope: str, challenge: str):
    return client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": "s",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "scope": scope,
            "resource": MCP_RESOURCE_AUDIENCE,
        },
        headers={"Accept": "text/html"},
        follow_redirects=False,
    )


def _approve(page) -> str:
    if page.status_code == 302:
        approved = page
    else:
        assert page.status_code == 200, page.text[:600]
        key = re.search(r'name="consent_key"\s+value="([^"]+)"', page.text)
        assert key, page.text[:600]
        approved = client.post(
            "/oauth/authorize",
            data={"consent_key": key.group(1), "decision": "approve"},
            follow_redirects=False,
        )
    assert approved.status_code == 302, approved.text
    return re.search(r"[?&]code=([^&]+)", approved.headers["location"]).group(1)


def _exchange(client_id: str, code: str, redirect_uri: str, verifier: str) -> dict:
    token = client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
            "resource": MCP_RESOURCE_AUDIENCE,
        },
    )
    assert token.status_code == 200, token.text
    return token.json()


def test_connector_ceiling_covers_every_scope_an_mcp_token_carries():
    missing = sorted(set(MCP_QUICK_CONNECT_SCOPES) - set(CONNECTOR_ALLOWED_SCOPES))
    assert not missing, (
        f"{missing} are on the quick-connect MCP token but a connector can never "
        "be granted them, so its tools that require them always refuse."
    )
    assert not (WITHHELD & set(CONNECTOR_ALLOWED_SCOPES))


def test_vscode_shaped_registration_signs_in_with_the_advertised_scopes():
    _sign_in("vscode-connector@xcelsior.ca")
    redirect_uri = "http://127.0.0.1:33418/"
    registered = client.post(
        "/oauth/register",
        json={
            "client_name": "Visual Studio Code",
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        },
    )
    assert registered.status_code == 201, registered.text
    client_id = registered.json()["client_id"]

    advertised = client.get("/.well-known/oauth-authorization-server").json()["scopes_supported"]
    verifier, challenge = _pkce()
    page = _authorize(client_id, redirect_uri, " ".join(advertised), challenge)
    assert page.status_code != 400, "a connector asking for the advertised scopes was refused"
    bundle = _exchange(client_id, _approve(page), redirect_uri, verifier)

    granted = set(bundle["scope"].split())
    assert granted == set(advertised) - WITHHELD
    # The MCP tools behind these were unreachable for any self-registered client.
    assert {"instances:write", "instances:connect", "ssh:write", "volumes:read"} <= granted
    assert not granted & WITHHELD


def test_shared_connector_client_drops_what_it_cannot_hold():
    _sign_in("shared-connector@xcelsior.ca")
    redirect_uri = "https://claude.ai/api/mcp/auth_callback"
    verifier, challenge = _pkce()
    page = _authorize(
        "xcelsior-connector", redirect_uri, "instances:read volumes:read billing:write", challenge
    )
    bundle = _exchange("xcelsior-connector", _approve(page), redirect_uri, verifier)
    assert set(bundle["scope"].split()) == {"instances:read", "volumes:read"}


def test_a_connector_asking_only_for_withheld_scopes_gets_its_default():
    _sign_in("withheld-only@xcelsior.ca")
    redirect_uri = "https://claude.ai/api/mcp/auth_callback"
    verifier, challenge = _pkce()
    page = _authorize("xcelsior-connector", redirect_uri, "billing:write hosts:write", challenge)
    bundle = _exchange("xcelsior-connector", _approve(page), redirect_uri, verifier)
    granted = set(bundle["scope"].split())
    assert granted and not granted & WITHHELD
    assert "instances:read" in granted
