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
from urllib.parse import parse_qs, urlsplit

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
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
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
    # Never follow the client's external callback into the test app.
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


@pytest.mark.parametrize("registration_source", ["connector", "dcr", "cimd", "manual"])
def test_returning_connector_reviews_permissions_before_a_new_authorization(registration_source):
    """An existing account/grant must not hide a fresh connector permission decision."""
    from oauth_delegation import SYSTEM_PRINCIPAL
    from oauth_service import create_oauth_client

    _register_user("connector-remember@xcelsior.ca")
    registered = create_oauth_client(
        actor=SYSTEM_PRINCIPAL,
        client_name="Returning connector",
        redirect_uris=[CLAUDE_CALLBACK],
        grant_types=["authorization_code", "refresh_token"],
        scopes=["instances:read", "offline_access"],
        created_by_email=None,
        client_type="public",
        registration_source=registration_source,
        resource_audience=MCP_RESOURCE_AUDIENCE,
    )
    _, challenge = _pkce()
    params = {
        "response_type": "code",
        "client_id": registered["client_id"],
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
    if registration_source == "manual":
        assert again.status_code == 302, "manually provisioned clients retain grant reuse"
        assert "code=" in again.headers["location"]
        return
    assert again.status_code == 200, "a new connector authorization must show permissions"
    assert "location" not in again.headers
    assert "connector-remember@xcelsior.ca" in again.text
    assert 'name="scope:offline_access"' in again.text
    assert _consent_key(again.text) != _consent_key(page.text)
    denied = client.post(
        "/oauth/authorize",
        data={"consent_key": _consent_key(again.text), "decision": "deny"},
        follow_redirects=False,
    )
    callback = parse_qs(urlsplit(denied.headers["location"]).query)
    assert denied.status_code == 302
    assert callback["error"] == ["access_denied"]
    assert "code" not in callback


def test_signed_out_connector_returns_from_login_to_its_permission_screen():
    _, challenge = _pkce()
    params = {
        "response_type": "code",
        "client_id": "xcelsior-connector",
        "redirect_uri": "http://127.0.0.1:33418/",
        "state": "retained-browser-state",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "scope": "instances:read offline_access",
        "resource": MCP_RESOURCE_AUDIENCE,
    }
    signed_out = client.get(
        "/oauth/authorize", params=params, headers={"Accept": "text/html"}, follow_redirects=False
    )
    assert signed_out.status_code == 302
    login_url = urlsplit(signed_out.headers["location"])
    assert login_url.path == "/login"
    return_to = parse_qs(login_url.query)["redirect"][0]
    assert urlsplit(return_to).path == "/oauth/authorize"
    assert parse_qs(urlsplit(return_to).query) == {key: [value] for key, value in params.items()}

    _register_user("connector-login-return@xcelsior.ca")
    permission_page = client.get(return_to, headers={"Accept": "text/html"}, follow_redirects=False)
    assert permission_page.status_code == 200
    assert "connector-login-return@xcelsior.ca" in permission_page.text
    approved = client.post(
        "/oauth/authorize",
        data={"consent_key": _consent_key(permission_page.text), "decision": "approve"},
        follow_redirects=False,
    )
    callback = urlsplit(approved.headers["location"])
    assert callback.netloc == "127.0.0.1:33418"
    assert parse_qs(callback.query)["state"] == [params["state"]]
    assert "code" in parse_qs(callback.query)


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
    """Operator scopes are dropped from the grant, and the response says so."""
    response = client.post(
        "/oauth/register",
        json={
            "client_name": "Would-be operator",
            "redirect_uris": [CLAUDE_CALLBACK],
            "scope": "instances:read hosts:evict control_plane:operate",
        },
    )
    assert response.status_code == 201
    granted = set(response.json()["scope"].split())
    assert "instances:read" in granted
    assert not granted & {"hosts:evict", "control_plane:operate"}


def test_dcr_refuses_a_request_with_nothing_grantable():
    response = client.post(
        "/oauth/register",
        json={
            "client_name": "Only an operator",
            "redirect_uris": [CLAUDE_CALLBACK],
            "scope": "hosts:evict control_plane:operate offline_access",
        },
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_scope"


def test_dcr_accepts_a_client_asking_for_withheld_scopes():
    """The exact request mcp-remote sent prod's stale resource metadata.

    Prod advertised `billing:write`, `providers:write` and `reputation:write`,
    which connectors may not hold, and refused the registration outright, so
    Claude Desktop could never connect (2026-11-01).
    """
    requested = (
        "artifacts:read billing:read billing:write events:read gpu:read inference:read "
        "inference:write instances:connect instances:operate instances:read instances:write "
        "marketplace:read providers:read providers:write reputation:read reputation:write "
        "ssh:read ssh:write volumes:read volumes:write"
    )
    response = client.post(
        "/oauth/register",
        json={
            "client_name": "MCP CLI Proxy",
            "redirect_uris": ["http://localhost:42418/oauth/callback"],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "scope": requested,
        },
    )
    assert response.status_code == 201, response.text
    granted = set(response.json()["scope"].split())
    assert not granted & {"billing:write", "providers:write", "reputation:write"}
    assert {"instances:read", "ssh:read", "ssh:write", "offline_access"} <= granted


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


def test_refused_registrations_do_not_spend_the_quota():
    """A client stuck retrying an error must not lock out its whole address."""
    from oauth_registration import DCR_MAX_PER_IP_PER_HOUR

    for index in range(DCR_MAX_PER_IP_PER_HOUR + 2):
        refused = client.post(
            "/oauth/register",
            json={
                "client_name": f"Retry {index}",
                "redirect_uris": [CLAUDE_CALLBACK],
                "scope": "hosts:evict",
            },
        )
        assert refused.status_code == 400, refused.text
    accepted = client.post(
        "/oauth/register",
        json={"client_name": "Finally valid", "redirect_uris": [CLAUDE_CALLBACK]},
    )
    assert accepted.status_code == 201, accepted.text


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


def test_consent_page_lists_each_requested_permission_as_a_choice():
    """The MCP baseline stays selected; additional requested scopes remain optional."""
    from routes.auth import _required_consent_scopes

    _register_user("connector-granular@xcelsior.ca")
    _, challenge = _pkce()
    scopes = {"profile", "instances:read", "instances:write", "ssh:read", "offline_access"}
    page = client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": "xcelsior-connector",
            "redirect_uri": "http://127.0.0.1:54321/callback",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "scope": " ".join(sorted(scopes)),
            "resource": MCP_RESOURCE_AUDIENCE,
        },
        headers={"Accept": "text/html"},
    )
    assert page.status_code == 200
    assert 'name="scope_selection" value="1"' in page.text
    choices = re.findall(r'<input\b[^>]*type="checkbox"[^>]*>', page.text)
    names = [re.search(r'name="scope:([^"]+)"', choice).group(1) for choice in choices]
    assert len(names) == len(scopes)
    assert set(names) == scopes
    assert all("checked" in choice for choice in choices)
    required = _required_consent_scopes(sorted(scopes), MCP_RESOURCE_AUDIENCE)
    assert required == {"instances:read", "instances:write"}
    for choice, scope in zip(choices, names):
        assert ("disabled" in choice) == (scope in required)
        if scope in required:
            assert f'type="hidden" name="scope:{scope}" value="1"' in page.text
    assert "See your GPU instances, their status, and their logs" in page.text
    assert "Launch, cancel, and terminate GPU instances" in page.text
    assert "public keys" in page.text


@pytest.mark.parametrize("decision", ["approve", "deny"])
@pytest.mark.parametrize(
    "loopback_uri",
    ["http://127.0.0.1:33418/", "http://127.0.0.1:54322/callback?client_nonce=retained"],
)
def test_browser_loopback_decision_delivers_actual_callback_directly(decision, loopback_uri):
    """The browser reaches the IDE listener without iframe or app-launch permission."""
    _register_user("connector-ui-decision@xcelsior.ca")
    verifier, challenge = _pkce()
    params = {
        "response_type": "code",
        "client_id": "xcelsior-connector",
        "redirect_uri": loopback_uri,
        "state": "ui state + & = opaque",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "scope": "instances:read offline_access",
        "resource": MCP_RESOURCE_AUDIENCE,
    }

    browser_headers = {"Accept": "text/html", "Sec-Fetch-Mode": "navigate"}
    page = client.get("/oauth/authorize", params=params, headers=browser_headers)
    response = client.post(
        "/oauth/authorize",
        data={"consent_key": _consent_key(page.text), "decision": decision, "ui": "1"},
        headers=browser_headers,
        follow_redirects=False,
    )
    assert response.status_code == 302, response.text
    assert "<iframe" not in response.text
    assert "vscode://" not in response.text
    location = urlsplit(response.headers["location"])
    registered = urlsplit(loopback_uri)
    assert (location.scheme, location.netloc, location.path) == (
        registered.scheme,
        registered.netloc,
        registered.path,
    )
    callback = parse_qs(location.query)
    assert callback["state"] == [params["state"]]
    if registered.query:
        assert callback["client_nonce"] == ["retained"]
    if decision == "deny":
        assert callback["error"] == ["access_denied"]
        assert "error_description" in callback
        assert "code" not in callback
        return

    assert "error" not in callback
    token = client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "client_id": "xcelsior-connector",
            "code": callback["code"][0],
            "redirect_uri": loopback_uri,
            "code_verifier": verifier,
            "resource": MCP_RESOURCE_AUDIENCE,
        },
    )
    assert token.status_code == 200, token.text
    assert set(token.json()["scope"].split()) == {"instances:read", "offline_access"}

    again = client.get(
        "/oauth/authorize",
        params=params,
        headers=browser_headers,
        follow_redirects=False,
    )
    assert again.status_code == 200, "a new browser authorization must show permissions again"
    repeated = client.post(
        "/oauth/authorize",
        data={"consent_key": _consent_key(again.text), "decision": "approve", "ui": "1"},
        headers=browser_headers,
        follow_redirects=False,
    )
    assert repeated.status_code == 302
    assert parse_qs(urlsplit(repeated.headers["location"]).query)["state"] == [params["state"]]


@pytest.mark.parametrize("approved", [True, False])
def test_decision_page_returns_to_registered_application_without_guessing_ide(approved):
    from routes.auth import _oauth_decision_page

    page = _oauth_decision_page(
        approved=approved,
        client={"client_name": "Codex", "client_uri": "https://chatgpt.com/codex"},
        callback_url=f"{CLAUDE_CALLBACK}?state=opaque&code=abc",
        is_loopback=False,
    )
    assert "Authentication complete" in page if approved else "cancelled" in page.lower()
    assert CLAUDE_CALLBACK in page
    assert "location.replace" in page
    assert "<iframe" not in page
    for invented_scheme in ("vscode://", "cursor://", "codex://"):
        assert invented_scheme not in page
    assert "Go to dashboard" not in page
    assert "Close window" not in page


def test_selected_permissions_are_the_only_grant_and_survive_token_refresh():
    from db import ConsentStore

    email = "connector-selected@xcelsior.ca"
    login_token = _register_user(email)
    login_principal = client.get(
        "/api/auth/introspect", headers={"Authorization": f"Bearer {login_token}"}
    )
    assert login_principal.status_code == 200, login_principal.text
    user_id = login_principal.json()["user_id"]
    verifier, challenge = _pkce()
    requested = {"instances:read", "instances:write", "billing:read", "email", "offline_access"}
    selected = {"instances:read", "instances:write", "billing:read", "offline_access"}
    params = {
        "response_type": "code",
        "client_id": "xcelsior-connector",
        "redirect_uri": "http://127.0.0.1:54323/callback",
        "state": "selected-state",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "scope": " ".join(sorted(requested)),
        "resource": MCP_RESOURCE_AUDIENCE,
    }
    page = client.get("/oauth/authorize", params=params, headers={"Accept": "text/html"})
    approved = client.post(
        "/oauth/authorize",
        data={
            "consent_key": _consent_key(page.text),
            "decision": "approve",
            "ui": "1",
            "scope_selection": "1",
            **{f"scope:{scope}": "1" for scope in selected},
        },
        follow_redirects=False,
    )
    assert approved.status_code == 302, approved.text
    callback = parse_qs(urlsplit(approved.headers["location"]).query)
    token = client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "client_id": "xcelsior-connector",
            "code": callback["code"][0],
            "redirect_uri": params["redirect_uri"],
            "code_verifier": verifier,
            "resource": MCP_RESOURCE_AUDIENCE,
        },
    )
    assert token.status_code == 200, token.text
    bundle = token.json()
    assert set(bundle["scope"].split()) == selected
    grant = ConsentStore.get(user_id, "xcelsior-connector", MCP_RESOURCE_AUDIENCE)
    assert set(grant["scopes"]) == selected
    principal = client.get(
        "/api/auth/introspect", headers={"Authorization": f"Bearer {bundle['access_token']}"}
    )
    assert principal.status_code == 200, principal.text
    assert set(principal.json()["scopes"]) == selected
    assert "email" not in principal.json()
    assert principal.json()["subject"] == user_id
    assert principal.json()["user_id"] == user_id
    profile = client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {bundle['access_token']}"}
    )
    assert profile.status_code == 403, profile.text
    assert "profile" in profile.text.lower()

    refreshed = client.post(
        "/oauth/token",
        data={
            "grant_type": "refresh_token",
            "client_id": "xcelsior-connector",
            "refresh_token": bundle["refresh_token"],
        },
    )
    assert refreshed.status_code == 200, refreshed.text
    assert set(refreshed.json()["scope"].split()) == selected

    again = client.get(
        "/oauth/authorize", params=params, headers={"Accept": "text/html"}, follow_redirects=False
    )
    assert again.status_code == 200, "unchecked permissions must require consent if requested again"
    assert 'name="consent_key"' in again.text


@pytest.mark.parametrize("selected", [set(), {"instances:write"}, {"offline_access"}])
def test_scope_selection_rejects_empty_or_unrequested_permissions_without_a_grant(selected):
    from db import ConsentStore

    email = "connector-invalid-selection@xcelsior.ca"
    login_token = _register_user(email)
    login_principal = client.get(
        "/api/auth/introspect", headers={"Authorization": f"Bearer {login_token}"}
    )
    assert login_principal.status_code == 200, login_principal.text
    user_id = login_principal.json()["user_id"]
    _, challenge = _pkce()
    page = client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": "xcelsior-connector",
            "redirect_uri": "http://127.0.0.1:54323/callback",
            "code_challenge": challenge,
            "scope": "instances:read offline_access",
            "resource": MCP_RESOURCE_AUDIENCE,
        },
        headers={"Accept": "text/html"},
    )
    response = client.post(
        "/oauth/authorize",
        data={
            "consent_key": _consent_key(page.text),
            "decision": "approve",
            "scope_selection": "1",
            **{f"scope:{scope}": "1" for scope in selected},
        },
        headers={"Accept": "text/html"},
        follow_redirects=False,
    )
    assert response.status_code == 400, response.text
    assert "location" not in response.headers
    assert ConsentStore.get(user_id, "xcelsior-connector", MCP_RESOURCE_AUDIENCE) is None


def test_required_mcp_permissions_never_add_to_the_clients_request():
    from routes.auth import _required_consent_scopes

    assert _required_consent_scopes(["instances:read"], MCP_RESOURCE_AUDIENCE) == {"instances:read"}
    assert _required_consent_scopes(["profile", "email"], MCP_RESOURCE_AUDIENCE) == set()
    assert _required_consent_scopes(["instances:read", "instances:write"], "xcelsior-api") == set()


def test_deselecting_offline_access_keeps_mcp_access_without_refresh_credentials():
    from oauth_service import resolve_opaque_access_token

    _register_user("connector-no-offline@xcelsior.ca")
    verifier, challenge = _pkce()
    redirect_uri = "http://127.0.0.1:33418/"
    selected = {"instances:read", "instances:write"}
    page = client.get(
        "/oauth/authorize",
        params={
            "response_type": "code",
            "client_id": "xcelsior-connector",
            "redirect_uri": redirect_uri,
            "code_challenge": challenge,
            "scope": "instances:read instances:write offline_access",
            "resource": MCP_RESOURCE_AUDIENCE,
        },
        headers={"Accept": "text/html"},
    )
    approved = client.post(
        "/oauth/authorize",
        data={
            "consent_key": _consent_key(page.text),
            "decision": "approve",
            "scope_selection": "1",
            **{f"scope:{scope}": "1" for scope in selected},
        },
        follow_redirects=False,
    )
    assert approved.status_code == 302, approved.text
    callback = parse_qs(urlsplit(approved.headers["location"]).query)
    token = client.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "client_id": "xcelsior-connector",
            "code": callback["code"][0],
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
            "resource": MCP_RESOURCE_AUDIENCE,
        },
    )
    assert token.status_code == 200, token.text
    bundle = token.json()
    assert set(bundle["scope"].split()) == selected
    assert not {"refresh_token", "session_token", "refresh_expires_in"} & bundle.keys()
    assert "set-cookie" not in token.headers
    access = resolve_opaque_access_token(bundle["access_token"])
    assert access and set(access["scopes"]) == selected

    # Even possession of the internal backing token cannot create offline
    # authority through the OAuth or legacy refresh entrypoints.
    refreshed = client.post(
        "/oauth/token",
        data={
            "grant_type": "refresh_token",
            "client_id": "xcelsior-connector",
            "refresh_token": access["session_token"],
        },
    )
    assert refreshed.status_code == 400, refreshed.text
    assert refreshed.json()["error"] == "invalid_grant"
    client.cookies.clear()
    legacy_refresh = client.post(
        "/api/auth/refresh", headers={"Authorization": f"Bearer {bundle['access_token']}"}
    )
    assert legacy_refresh.status_code == 400, legacy_refresh.text
    assert legacy_refresh.json()["error"] == "invalid_grant"
    principal = client.get(
        "/api/auth/introspect", headers={"Authorization": f"Bearer {bundle['access_token']}"}
    )
    assert principal.status_code == 200, principal.text
    assert set(principal.json()["scopes"]) == selected


def test_first_party_refresh_does_not_require_the_optional_connector_permission():
    from oauth_service import issue_user_tokens, rotate_refresh_token

    login_token = _register_user("first-party-no-offline@xcelsior.ca")
    principal = client.get(
        "/api/auth/introspect", headers={"Authorization": f"Bearer {login_token}"}
    )
    assert principal.status_code == 200, principal.text
    bundle = issue_user_tokens(principal.json(), None, client_id="xcelsior-web", scopes=["profile"])
    assert {"refresh_token", "session_token", "refresh_expires_in"} <= bundle.keys()
    first_party_profile = client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {bundle['access_token']}"}
    )
    assert first_party_profile.status_code == 200, first_party_profile.text
    assert first_party_profile.json()["user"]["email"] == "first-party-no-offline@xcelsior.ca"
    assert {"name", "role", "country", "province"} <= first_party_profile.json()["user"].keys()
    first_party_identity = client.get(
        "/api/auth/introspect", headers={"Authorization": f"Bearer {bundle['access_token']}"}
    )
    assert first_party_identity.json()["email"] == "first-party-no-offline@xcelsior.ca"
    refreshed = rotate_refresh_token(bundle["refresh_token"], None)
    assert refreshed["refresh_token"] != bundle["refresh_token"]
    assert refreshed["scope"] == "profile"


@pytest.mark.parametrize("identity_scopes", [{"profile"}, {"email"}, {"profile", "email"}])
def test_connector_identity_fields_follow_each_identity_scope(identity_scopes):
    from oauth_service import issue_user_tokens, resolve_opaque_access_token

    email = "connector-identity-scope@xcelsior.ca"
    login_token = _register_user(email)
    user = resolve_opaque_access_token(login_token)
    assert user
    original_email = user["email"]
    original_scopes = list(user["scopes"])
    bundle = issue_user_tokens(
        user,
        None,
        client_id="xcelsior-connector",
        scopes=["instances:read", *sorted(identity_scopes)],
        audience=MCP_RESOURCE_AUDIENCE,
    )
    import json
    from oauth_service import _base64url_decode

    claims = json.loads(_base64url_decode(bundle["access_token"].split(".")[1]))
    assert claims["sub"] == user["user_id"]
    assert ("name" in claims) == ("profile" in identity_scopes and bool(user.get("name")))
    assert ("email" in claims) == ("email" in identity_scopes)
    assert ("preferred_username" in claims) == ("email" in identity_scopes)
    if "email" in identity_scopes:
        assert claims["preferred_username"] == email
    assert not {"session_token", "role", "is_admin", "customer_id"} & claims.keys()
    headers = {"Authorization": f"Bearer {bundle['access_token']}"}
    profile = client.get("/api/auth/me", headers=headers)
    introspection = client.get("/api/auth/introspect", headers=headers)
    assert introspection.status_code == 200, introspection.text
    assert introspection.json()["user_id"] == user["user_id"]
    assert ("email" in introspection.json()) == ("email" in identity_scopes)
    if "profile" in identity_scopes:
        assert profile.status_code == 200, profile.text
        visible = profile.json()["user"]
        assert visible["user_id"] == user["user_id"]
        assert ("email" in visible) == ("email" in identity_scopes)
        assert {"name", "role"} <= visible.keys()
    else:
        assert profile.status_code == 403, profile.text
    if "email" in identity_scopes:
        assert introspection.json()["email"] == email
        if "profile" in identity_scopes:
            assert visible["email"] == email
    # Redacting serialized output must not change internal authorization data.
    internal = resolve_opaque_access_token(bundle["access_token"])
    assert internal["email"] == email
    assert set(internal["scopes"]) == {"instances:read", *identity_scopes}
    assert user["email"] == original_email
    assert user["scopes"] == original_scopes
