"""End-to-end connector OAuth over HTTP: discovery → consent → token → refresh.

Gate GX0's scripted-client assertion, run in-process against the real ASGI app
so it exercises the same routing, body parsing, and cookie handling a provider
does. `scripts/gx0_conformance.py` runs the same chain against a public
endpoint from an external vantage point; this file is what keeps it from
regressing between those runs.
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

from api import app  # noqa: E402
from oauth_service import MCP_RESOURCE_AUDIENCE  # noqa: E402

client = TestClient(app)

CLAUDE_CALLBACK = "https://claude.ai/api/mcp/auth_callback"


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    return verifier, challenge


def _register_user(email: str, password: str = "ConnectorTest123!") -> str:
    reg = client.post("/api/auth/register", json={"email": email, "password": password})
    assert reg.status_code == 200, reg.text
    body = reg.json()
    if body.get("access_token"):
        return body["access_token"]
    if body.get("email_verification_required"):
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
        if verified.json().get("access_token"):
            return verified.json()["access_token"]
    login = client.post("/api/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200, login.text
    return login.json()["access_token"]


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
    # The seeded connector client is created lazily; force a re-seed so each
    # test starts from the shipped default set rather than a prior test's rows.
    _oauth._defaults_ready = False
    reset_auth_cache_for_tests()
    client.cookies.clear()
    _deps_mod._RATE_BUCKETS.clear()
    _deps_mod._AUTH_RATE_BUCKETS.clear()
    _deps_mod._users_db.clear()
    _deps_mod._sessions.clear()
    _deps_mod._api_keys.clear()
    yield


def _consent_key(html: str) -> str:
    match = re.search(r'name="consent_key"\s+value="([^"]+)"', html)
    assert match, f"no consent form in response: {html[:400]}"
    return match.group(1)


# ── Discovery ─────────────────────────────────────────────────────────────


def test_authorization_server_metadata_advertises_both_registration_paths():
    metadata = client.get("/.well-known/oauth-authorization-server").json()
    # CIMD first — Anthropic and OpenAI prefer it, and it keeps a directory
    # listing from producing an unbounded client-registration table.
    assert metadata["client_id_metadata_document_supported"] is True
    assert metadata["registration_endpoint"].endswith("/oauth/register")
    assert metadata["resource_indicators_supported"] is True
    assert metadata["code_challenge_methods_supported"] == ["S256"]
    assert "authorization_code" in metadata["grant_types_supported"]
    assert "refresh_token" in metadata["grant_types_supported"]


# ── The full connector round trip ─────────────────────────────────────────


def _authorize_through_consent(*, challenge: str, redirect_uri: str, resource: str | None):
    params = {
        "response_type": "code",
        "client_id": "xcelsior-connector",
        "redirect_uri": redirect_uri,
        "state": "opaque-state-value",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "scope": "instances:read billing:read offline_access",
    }
    if resource:
        params["resource"] = resource
    # Never follow: an already-consented authorize redirects straight to the
    # client's callback, and following that into the test app would 404.
    page = client.get(
        "/oauth/authorize", params=params, headers={"Accept": "text/html"}, follow_redirects=False
    )
    if page.status_code == 302:
        approved = page
    else:
        assert page.status_code == 200, page.text
        approved = client.post(
            "/oauth/authorize",
            data={"consent_key": _consent_key(page.text), "decision": "approve"},
            follow_redirects=False,
        )
    assert approved.status_code == 302, approved.text
    location = approved.headers["location"]
    assert location.startswith(redirect_uri)
    assert "state=opaque-state-value" in location
    return re.search(r"[?&]code=([^&]+)", location).group(1)


def test_connector_completes_authorize_consent_token_and_refresh():
    _register_user("connector-flow@xcelsior.ca")
    verifier, challenge = _pkce()
    code = _authorize_through_consent(
        challenge=challenge, redirect_uri=CLAUDE_CALLBACK, resource=MCP_RESOURCE_AUDIENCE
    )

    # Form-encoded, exactly as RFC 6749 §4.1.3 requires and every connector
    # sends. A JSON-only token endpoint is a documented silent-failure cause.
    token = client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "client_id": "xcelsior-connector",
            "code": code,
            "redirect_uri": CLAUDE_CALLBACK,
            "code_verifier": verifier,
            "resource": MCP_RESOURCE_AUDIENCE,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert token.status_code == 200, token.text
    bundle = token.json()
    assert bundle["token_type"] == "Bearer"
    assert bundle["resource"] == MCP_RESOURCE_AUDIENCE
    assert bundle["expires_in"] >= 1800, "connector access tokens are ~1h, not 15m"
    assert bundle["refresh_expires_in"] == 86400 * 30
    assert bundle["refresh_token"]

    introspect = client.get(
        "/api/auth/introspect", headers={"Authorization": f"Bearer {bundle['access_token']}"}
    )
    assert introspect.status_code == 200, introspect.text
    # The MCP edge rejects any token whose audience is not the resource.
    assert introspect.json()["audience"] == MCP_RESOURCE_AUDIENCE

    refreshed = client.post(
        "/oauth/token",
        data={
            "grant_type": "refresh_token",
            "client_id": "xcelsior-connector",
            "refresh_token": bundle["refresh_token"],
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert refreshed.status_code == 200, refreshed.text
    assert refreshed.json()["access_token"] != bundle["access_token"]
    assert refreshed.json()["resource"] == MCP_RESOURCE_AUDIENCE


def test_token_endpoint_accepts_json_as_well_as_form_encoding():
    """Both, so neither a strict RFC client nor a JSON-only client is stranded."""
    _register_user("connector-json@xcelsior.ca")
    verifier, challenge = _pkce()
    code = _authorize_through_consent(
        challenge=challenge, redirect_uri=CLAUDE_CALLBACK, resource=MCP_RESOURCE_AUDIENCE
    )
    token = client.post(
        "/oauth/token",
        json={
            "grant_type": "authorization_code",
            "client_id": "xcelsior-connector",
            "code": code,
            "redirect_uri": CLAUDE_CALLBACK,
            "code_verifier": verifier,
        },
    )
    assert token.status_code == 200, token.text


def test_loopback_callback_succeeds_on_two_different_random_ports():
    """The GX0 assertion: a native client binds a fresh port every attempt."""
    _register_user("connector-loopback@xcelsior.ca")
    for port in (51423, 62197):
        verifier, challenge = _pkce()
        redirect_uri = f"http://127.0.0.1:{port}/callback"
        code = _authorize_through_consent(
            challenge=challenge, redirect_uri=redirect_uri, resource=MCP_RESOURCE_AUDIENCE
        )
        token = client.post(
            "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": "xcelsior-connector",
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
            },
        )
        assert token.status_code == 200, f"port {port}: {token.text}"


def test_legacy_origin_resource_still_yields_a_canonical_token():
    """Old-origin clients follow the documented compatibility path, not a 400."""
    from oauth_service import MCP_LEGACY_RESOURCE_AUDIENCE, legacy_mcp_audience_accepted

    if not legacy_mcp_audience_accepted():
        pytest.skip("legacy audience window has closed")
    _register_user("connector-legacy@xcelsior.ca")
    verifier, challenge = _pkce()
    code = _authorize_through_consent(
        challenge=challenge,
        redirect_uri=CLAUDE_CALLBACK,
        resource=MCP_LEGACY_RESOURCE_AUDIENCE,
    )
    token = client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "client_id": "xcelsior-connector",
            "code": code,
            "redirect_uri": CLAUDE_CALLBACK,
            "code_verifier": verifier,
            "resource": MCP_LEGACY_RESOURCE_AUDIENCE,
        },
    )
    assert token.status_code == 200, token.text
    assert token.json()["resource"] == MCP_RESOURCE_AUDIENCE


def test_resource_substitution_is_rejected_at_authorize():
    _register_user("connector-substitute@xcelsior.ca")
    _, challenge = _pkce()
    response = client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": "xcelsior-connector",
            "redirect_uri": CLAUDE_CALLBACK,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "resource": "https://attacker.example.test/mcp",
        },
        headers={"Accept": "text/html"},
    )
    assert response.status_code == 400


# ── Consent ───────────────────────────────────────────────────────────────


def test_declining_consent_returns_access_denied_and_issues_no_code():
    _register_user("connector-deny@xcelsior.ca")
    _, challenge = _pkce()
    page = client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": "xcelsior-connector",
            "redirect_uri": CLAUDE_CALLBACK,
            "state": "s",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        headers={"Accept": "text/html"},
    )
    denied = client.post(
        "/oauth/authorize",
        data={"consent_key": _consent_key(page.text), "decision": "deny"},
        follow_redirects=False,
    )
    assert denied.status_code == 302
    assert "error=access_denied" in denied.headers["location"]
    assert "code=" not in denied.headers["location"]


def test_consent_key_cannot_be_replayed():
    _register_user("connector-replay@xcelsior.ca")
    _, challenge = _pkce()
    page = client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": "xcelsior-connector",
            "redirect_uri": CLAUDE_CALLBACK,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        headers={"Accept": "text/html"},
    )
    key = _consent_key(page.text)
    first = client.post(
        "/oauth/authorize",
        data={"consent_key": key, "decision": "approve"},
        follow_redirects=False,
    )
    assert first.status_code == 302
    second = client.post(
        "/oauth/authorize",
        data={"consent_key": key, "decision": "approve"},
        follow_redirects=False,
        headers={"Accept": "text/html"},
    )
    assert second.status_code == 400


def test_second_authorization_reuses_the_recorded_grant():
    """A returning user reconnects without seeing the screen again."""
    _register_user("connector-remember@xcelsior.ca")
    _, challenge = _pkce()
    params = {
        "response_type": "code",
        "client_id": "xcelsior-connector",
        "redirect_uri": CLAUDE_CALLBACK,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "scope": "instances:read offline_access",
        "resource": MCP_RESOURCE_AUDIENCE,
    }
    page = client.get("/oauth/authorize", params=params, headers={"Accept": "text/html"})
    client.post(
        "/oauth/authorize",
        data={"consent_key": _consent_key(page.text), "decision": "approve"},
        follow_redirects=False,
    )
    again = client.get(
        "/oauth/authorize", params=params, headers={"Accept": "text/html"}, follow_redirects=False
    )
    assert again.status_code == 302, "an existing grant should skip the prompt"
    assert "code=" in again.headers["location"]


def test_widening_scopes_asks_again():
    _register_user("connector-widen@xcelsior.ca")
    _, challenge = _pkce()
    base = {
        "response_type": "code",
        "client_id": "xcelsior-connector",
        "redirect_uri": CLAUDE_CALLBACK,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "resource": MCP_RESOURCE_AUDIENCE,
    }
    page = client.get(
        "/oauth/authorize",
        params={**base, "scope": "instances:read"},
        headers={"Accept": "text/html"},
    )
    client.post(
        "/oauth/authorize",
        data={"consent_key": _consent_key(page.text), "decision": "approve"},
        follow_redirects=False,
    )
    widened = client.get(
        "/oauth/authorize",
        params={**base, "scope": "instances:read instances:write"},
        headers={"Accept": "text/html"},
        follow_redirects=False,
    )
    assert widened.status_code == 200, "a wider request must show the screen again"
    assert "consent_key" in widened.text


def test_first_party_login_flow_is_not_interrupted_by_consent():
    """`xcelsior-web` is us; a consent screen there would be theatre."""
    _register_user("connector-firstparty@xcelsior.ca")
    _, challenge = _pkce()
    response = client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": "xcelsior-web",
            "redirect_uri": os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca")
            + "/oauth/callback",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        headers={"Accept": "text/html"},
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert "code=" in response.headers["location"]


# ── Dynamic client registration ───────────────────────────────────────────


def test_dcr_registers_authorizes_and_calls():
    registration = client.post(
        "/oauth/register",
        json={
            "client_name": "Copilot Studio Test Connector",
            "redirect_uris": ["https://copilotstudio.microsoft.com/oauth/callback"],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        },
    )
    assert registration.status_code == 201, registration.text
    registered = registration.json()
    assert registered["token_endpoint_auth_method"] == "none"
    # MCP-audience-only: a self-registered client cannot become an API client.
    assert registered["resource"] == MCP_RESOURCE_AUDIENCE

    _register_user("connector-dcr@xcelsior.ca")
    verifier, challenge = _pkce()
    redirect_uri = "https://copilotstudio.microsoft.com/oauth/callback"
    page = client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": registered["client_id"],
            "redirect_uri": redirect_uri,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        headers={"Accept": "text/html"},
    )
    assert page.status_code == 200, page.text
    approved = client.post(
        "/oauth/authorize",
        data={"consent_key": _consent_key(page.text), "decision": "approve"},
        follow_redirects=False,
    )
    assert approved.status_code == 302, approved.text
    code = re.search(r"[?&]code=([^&]+)", approved.headers["location"]).group(1)
    token = client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "client_id": registered["client_id"],
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        },
    )
    assert token.status_code == 200, token.text
    # Pinned even though the client never sent a resource indicator.
    assert token.json()["resource"] == MCP_RESOURCE_AUDIENCE


def test_dcr_rejects_an_off_allowlist_redirect():
    response = client.post(
        "/oauth/register",
        json={
            "client_name": "Hostile",
            "redirect_uris": ["https://evil.example.test/steal"],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_redirect_uri"


def test_dcr_client_cannot_obtain_operator_scopes():
    response = client.post(
        "/oauth/register",
        json={
            "client_name": "Would-be operator",
            "redirect_uris": [CLAUDE_CALLBACK],
            "scope": "instances:read hosts:evict control_plane:operate",
        },
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_scope"


def test_dcr_client_cannot_request_client_credentials():
    """A self-registered client must never act without a user behind it."""
    response = client.post(
        "/oauth/register",
        json={
            "client_name": "Headless",
            "redirect_uris": [CLAUDE_CALLBACK],
            "grant_types": ["client_credentials"],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_client_metadata"


def test_dcr_defaults_to_read_biased_scopes():
    registered = client.post(
        "/oauth/register",
        json={"client_name": "Quiet", "redirect_uris": [CLAUDE_CALLBACK]},
    ).json()
    granted = set(registered["scope"].split())
    assert "instances:read" in granted
    assert "instances:write" not in granted
    assert "offline_access" in granted


def test_dcr_is_rate_limited_per_source():
    from oauth_registration import DCR_MAX_PER_IP_PER_HOUR

    last = None
    for index in range(DCR_MAX_PER_IP_PER_HOUR + 2):
        last = client.post(
            "/oauth/register",
            json={"client_name": f"Flood {index}", "redirect_uris": [CLAUDE_CALLBACK]},
        )
    assert last is not None and last.status_code == 429


def test_authorize_rejects_an_unresolvable_metadata_document_client():
    """GX0: an invalid/untrusted CIMD client id is refused, with a reason."""
    _register_user("connector-badcimd@xcelsior.ca")
    _, challenge = _pkce()
    response = client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": "https://cimd.invalid.test/does-not-exist.json",
            "redirect_uri": CLAUDE_CALLBACK,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        headers={"Accept": "text/html"},
    )
    assert response.status_code == 400
    assert "Untrusted application" in response.text
