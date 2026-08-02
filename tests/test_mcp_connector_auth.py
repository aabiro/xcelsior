"""Connector auth front door: resource identity, redirects, CIMD, DCR.

Gate GX0 (docs/mcp-enterprise-adoption-plan.md §6). These are the assertions
that can be made without a live server; the end-to-end run against a public
endpoint lives in `scripts/gx0_conformance.py`.

Every test here guards a failure that is invisible from our side: a connector
that cannot discover where to authenticate, a token bound to the wrong
identifier, a loopback callback rejected because its port changed, or a
self-registered client quietly holding operator authority.
"""

from __future__ import annotations

import time

import pytest

import oauth_registration
import oauth_service
from oauth_service import (
    CONNECTOR_REDIRECT_URIS,
    OAuthGrantError,
    canonical_mcp_resource,
    is_loopback_redirect,
    normalize_resource_indicator,
    redirect_uri_matches,
)


# ── Canonical resource identifier (BLOCKER 1, X0.2) ───────────────────────


def test_canonical_resource_is_the_connector_url_not_the_origin():
    """The identifier is what a user pastes, path included."""
    assert oauth_service.MCP_RESOURCE_AUDIENCE.endswith("/mcp"), (
        "the canonical resource must be the exact connector URL; Claude "
        "requires `resource` to match the URL the user entered"
    )


def test_legacy_origin_resource_canonicalises_during_the_window(monkeypatch):
    monkeypatch.setattr(oauth_service, "MCP_LEGACY_AUDIENCE_SUNSET", "2999-01-01T00:00:00Z")
    assert canonical_mcp_resource(oauth_service.MCP_LEGACY_RESOURCE_AUDIENCE) == (
        oauth_service.MCP_RESOURCE_AUDIENCE
    )
    # A client that cached the old metadata gets a token bound to the URL the
    # server actually serves, not to the string the client happened to hold.
    assert normalize_resource_indicator(oauth_service.MCP_LEGACY_RESOURCE_AUDIENCE) == (
        oauth_service.MCP_RESOURCE_AUDIENCE
    )


def test_legacy_origin_resource_is_rejected_after_sunset(monkeypatch):
    monkeypatch.setattr(oauth_service, "MCP_LEGACY_AUDIENCE_SUNSET", "2000-01-01T00:00:00Z")
    assert oauth_service.legacy_mcp_audience_accepted() is False
    assert oauth_service.accepted_mcp_audiences() == (oauth_service.MCP_RESOURCE_AUDIENCE,)
    assert canonical_mcp_resource(oauth_service.MCP_LEGACY_RESOURCE_AUDIENCE) is None


def test_trailing_slash_does_not_create_a_second_identifier():
    assert canonical_mcp_resource(f"{oauth_service.MCP_RESOURCE_AUDIENCE}/") == (
        oauth_service.MCP_RESOURCE_AUDIENCE
    )


def test_unrelated_resource_indicator_is_rejected():
    with pytest.raises(OAuthGrantError) as excinfo:
        normalize_resource_indicator("https://attacker.example.test/mcp")
    assert excinfo.value.error == "invalid_target"


def test_absent_resource_indicator_falls_back_to_the_api_audience():
    assert normalize_resource_indicator("") == oauth_service.OAUTH_AUDIENCE
    assert normalize_resource_indicator(None, default="") == ""


# ── Loopback redirect matching (RFC 8252 §7.3, X0.3) ──────────────────────


def test_loopback_matches_on_two_different_random_ports():
    """The GX0 assertion, in unit form: the port must not be the gate."""
    registered = "http://127.0.0.1:8976/callback"
    assert redirect_uri_matches(registered, "http://127.0.0.1:51423/callback")
    assert redirect_uri_matches(registered, "http://127.0.0.1:62197/callback")


def test_loopback_hosts_are_interchangeable():
    assert redirect_uri_matches("http://localhost:1/cb", "http://127.0.0.1:44444/cb")
    assert redirect_uri_matches("http://127.0.0.1:1/cb", "http://localhost:44444/cb")


def test_wildcard_port_registration_accepts_any_loopback_path():
    assert redirect_uri_matches("http://127.0.0.1:*", "http://127.0.0.1:39211/oauth/callback")
    assert redirect_uri_matches("http://localhost:*", "http://localhost:39211/cb")


def test_loopback_path_still_has_to_match_when_registered():
    assert not redirect_uri_matches(
        "http://127.0.0.1:8976/callback", "http://127.0.0.1:8976/other"
    )


def test_port_widening_never_applies_to_a_routable_host():
    """A wildcard port on a real host would hand codes to anyone who binds it."""
    assert not redirect_uri_matches("https://claude.ai:*", "https://claude.ai:8443/cb")
    assert not redirect_uri_matches("https://claude.ai/cb", "https://claude.ai:9999/cb")
    assert not redirect_uri_matches("http://evil.example:*", "http://evil.example:80/cb")


def test_redirect_with_fragment_or_userinfo_never_matches():
    assert not redirect_uri_matches("http://127.0.0.1:*", "http://user:pw@127.0.0.1:80/cb")
    assert not redirect_uri_matches("http://127.0.0.1:*", "http://127.0.0.1:80/cb#frag")


def test_exact_match_still_wins_for_provider_callbacks():
    claude = "https://claude.ai/api/mcp/auth_callback"
    assert claude in CONNECTOR_REDIRECT_URIS, (
        "Claude posts authorization codes here; an unregistered callback makes "
        "the directory connector unusable"
    )
    assert redirect_uri_matches(claude, claude)
    assert not redirect_uri_matches(claude, "https://claude.ai/api/mcp/other")


def test_is_loopback_redirect_classifies_https_as_routable():
    assert is_loopback_redirect("http://127.0.0.1:5000/cb")
    assert not is_loopback_redirect("https://127.0.0.1:5000/cb")
    assert not is_loopback_redirect("https://claude.ai/cb")


# ── Connector scope policy (X0.7) ─────────────────────────────────────────


def test_dynamic_clients_can_never_request_operator_scopes():
    for scope in ("hosts:evict", "hosts:operate", "control_plane:operate"):
        with pytest.raises(OAuthGrantError) as excinfo:
            oauth_registration.normalize_requested_scopes(scope)
        assert excinfo.value.error == "invalid_scope"


def test_dynamic_clients_can_never_request_the_blanket_api_scope():
    """`api` short-circuits every per-tool scope check in the MCP gateway."""
    assert "api" not in oauth_registration.CONNECTOR_ALLOWED_SCOPES
    with pytest.raises(OAuthGrantError):
        oauth_registration.normalize_requested_scopes("api")


def test_default_scopes_are_read_biased():
    defaults = set(oauth_registration.CONNECTOR_DEFAULT_SCOPES)
    assert "instances:write" not in defaults
    assert "inference:write" not in defaults
    assert "billing:write" not in defaults
    assert {"instances:read", "billing:read", "gpu:read"} <= defaults


def test_refresh_capability_is_always_granted():
    """Without offline_access a connector is forced back through the browser."""
    assert "offline_access" in oauth_registration.normalize_requested_scopes("instances:read")


# ── Redirect allowlist (X0.7) ─────────────────────────────────────────────


def test_registration_rejects_an_off_allowlist_redirect_host():
    with pytest.raises(OAuthGrantError) as excinfo:
        oauth_registration.validate_redirect_uris(["https://evil.example.test/cb"])
    assert excinfo.value.error == "invalid_redirect_uri"


def test_registration_accepts_provider_callbacks_and_loopback():
    accepted = oauth_registration.validate_redirect_uris(
        ["https://claude.ai/api/mcp/auth_callback", "http://127.0.0.1:7777/cb"]
    )
    assert len(accepted) == 2


def test_registration_rejects_plaintext_http_on_a_routable_host():
    with pytest.raises(OAuthGrantError):
        oauth_registration.validate_redirect_uris(["http://claude.ai/api/mcp/auth_callback"])


def test_same_origin_redirect_is_accepted_for_a_metadata_document_client():
    """Domain control is what a metadata-document client id actually proves."""
    accepted = oauth_registration.validate_redirect_uris(
        ["https://connector.example.test/cb"], same_origin_host="connector.example.test"
    )
    assert accepted == ["https://connector.example.test/cb"]
    with pytest.raises(OAuthGrantError):
        oauth_registration.validate_redirect_uris(
            ["https://elsewhere.example.test/cb"], same_origin_host="connector.example.test"
        )


# ── CIMD (X0.6) ───────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_cimd_cache():
    oauth_registration.reset_cimd_cache()
    yield
    oauth_registration.reset_cimd_cache()


def test_cimd_client_id_shape():
    assert oauth_registration.is_cimd_client_id("https://ai.example.test/mcp/client.json")
    # A bare origin would make every domain one implicit client.
    assert not oauth_registration.is_cimd_client_id("https://ai.example.test/")
    assert not oauth_registration.is_cimd_client_id("https://ai.example.test")
    assert not oauth_registration.is_cimd_client_id("http://ai.example.test/client.json")
    assert not oauth_registration.is_cimd_client_id("https://a.test/c.json#f")
    assert not oauth_registration.is_cimd_client_id("xoa_deadbeef")


def _document(**overrides):
    document = {
        "client_id": "https://ai.example.test/mcp/client.json",
        "client_name": "Example Assistant",
        "redirect_uris": ["https://ai.example.test/oauth/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_method": "none",
        "scope": "instances:read billing:read",
    }
    document.update(overrides)
    return document


def test_cimd_resolves_to_a_pinned_public_client(monkeypatch):
    monkeypatch.setattr(oauth_registration, "_fetch_cimd_document", lambda _: _document())
    client = oauth_registration.resolve_cimd_client("https://ai.example.test/mcp/client.json")
    assert client["client_type"] == "public"
    assert client["registration_source"] == "cimd"
    # Pinned: a metadata-document client cannot mint a general API token.
    assert client["resource_audience"] == oauth_service.MCP_RESOURCE_AUDIENCE
    assert "authorization_code" in client["grant_types"]
    assert "offline_access" in client["scopes"]


def test_cimd_rejects_a_document_that_claims_another_client_id(monkeypatch):
    """Otherwise one hosted document could impersonate every other client."""
    monkeypatch.setattr(
        oauth_registration,
        "_fetch_cimd_document",
        lambda _: _document(client_id="https://other.example.test/client.json"),
    )
    with pytest.raises(OAuthGrantError) as excinfo:
        oauth_registration.resolve_cimd_client("https://ai.example.test/mcp/client.json")
    assert excinfo.value.error == "invalid_client_metadata"


def test_cimd_rejects_a_cross_origin_redirect(monkeypatch):
    monkeypatch.setattr(
        oauth_registration,
        "_fetch_cimd_document",
        lambda _: _document(redirect_uris=["https://unrelated.example.test/cb"]),
    )
    with pytest.raises(OAuthGrantError):
        oauth_registration.resolve_cimd_client("https://ai.example.test/mcp/client.json")


def test_cimd_rejects_a_confidential_client(monkeypatch):
    monkeypatch.setattr(
        oauth_registration,
        "_fetch_cimd_document",
        lambda _: _document(token_endpoint_auth_method="client_secret_post"),
    )
    with pytest.raises(OAuthGrantError):
        oauth_registration.resolve_cimd_client("https://ai.example.test/mcp/client.json")


def test_cimd_rejects_operator_scopes_in_the_document(monkeypatch):
    monkeypatch.setattr(
        oauth_registration,
        "_fetch_cimd_document",
        lambda _: _document(scope="instances:read hosts:evict"),
    )
    with pytest.raises(OAuthGrantError) as excinfo:
        oauth_registration.resolve_cimd_client("https://ai.example.test/mcp/client.json")
    assert excinfo.value.error == "invalid_scope"


def test_cimd_document_is_cached_so_authorize_is_not_a_fetch_per_request(monkeypatch):
    calls = {"n": 0}

    def _counting(_client_id):
        calls["n"] += 1
        return _document()

    monkeypatch.setattr(oauth_registration, "_fetch_cimd_document", _counting)
    for _ in range(3):
        oauth_registration.resolve_cimd_client("https://ai.example.test/mcp/client.json")
    assert calls["n"] == 1


def test_cimd_refuses_to_fetch_from_a_private_address():
    """`client_id` must not become a server-side request forgery primitive."""
    with pytest.raises(OAuthGrantError) as excinfo:
        oauth_registration._reject_private_address("localhost")
    assert excinfo.value.error == "invalid_client_metadata"


# ── Token lifetimes (X0.5) ────────────────────────────────────────────────


def test_mcp_access_tokens_live_about_an_hour():
    assert 1800 <= oauth_service.MCP_ACCESS_TOKEN_TTL_SEC <= 7200
    assert oauth_service.MCP_ACCESS_TOKEN_TTL_SEC > oauth_service.ACCESS_TOKEN_TTL_SEC


def test_refresh_tokens_live_about_thirty_days():
    assert oauth_service.REFRESH_TOKEN_TTL_SEC == 86400 * 30


def test_mcp_audience_selects_the_connector_ttl(monkeypatch):
    issued: dict[str, dict] = {}
    monkeypatch.setattr(
        oauth_service,
        "_cache_set_json",
        lambda namespace, key, value, ttl, **_: issued.__setitem__(key, dict(value)) is None,
    )
    bundle = oauth_service._issue_opaque_access_token(
        {"email": "u@example.test", "user_id": "u1"},
        client_id="some-connector",
        scopes=["instances:read"],
        session_token="rt",
        session_type="browser",
        audience=oauth_service.MCP_RESOURCE_AUDIENCE,
    )
    assert bundle["expires_in"] == oauth_service.MCP_ACCESS_TOKEN_TTL_SEC
    stored = issued[bundle["access_token"]]
    assert stored["expires_at"] - stored["issued_at"] == pytest.approx(
        oauth_service.MCP_ACCESS_TOKEN_TTL_SEC, abs=2
    )


def test_non_mcp_agent_tokens_keep_the_short_ttl(monkeypatch):
    monkeypatch.setattr(oauth_service, "_cache_set_json", lambda *a, **k: True)
    bundle = oauth_service._issue_opaque_access_token(
        {"email": "u@example.test", "user_id": "u1"},
        client_id="some-agent",
        scopes=["api"],
        session_token="rt",
        session_type="browser",
        audience=oauth_service.OAUTH_AUDIENCE,
    )
    assert bundle["expires_in"] == oauth_service.ACCESS_TOKEN_TTL_SEC


# ── Resource pinning (X0.6/X0.7 containment) ──────────────────────────────


def test_a_pinned_client_cannot_be_authorized_for_the_general_api(monkeypatch):
    monkeypatch.setattr(oauth_service, "_cache_set_json", lambda *a, **k: True)
    pinned_client = {
        "client_id": "xoa_pinned",
        "resource_audience": oauth_service.MCP_RESOURCE_AUDIENCE,
        "scopes": ["instances:read"],
    }
    with pytest.raises(OAuthGrantError) as excinfo:
        oauth_service.issue_authorization_code(
            client=pinned_client,
            user={"email": "u@example.test", "user_id": "u1"},
            redirect_uri="https://claude.ai/api/mcp/auth_callback",
            code_challenge="c" * 43,
            code_challenge_method="S256",
            scopes=["instances:read"],
            audience=oauth_service.OAUTH_AUDIENCE,
        )
    assert excinfo.value.error == "invalid_target"


def test_a_pinned_client_is_authorized_for_its_own_resource(monkeypatch):
    monkeypatch.setattr(oauth_service, "_cache_set_json", lambda *a, **k: True)
    code = oauth_service.issue_authorization_code(
        client={
            "client_id": "xoa_pinned",
            "resource_audience": oauth_service.MCP_RESOURCE_AUDIENCE,
            "scopes": ["instances:read"],
        },
        user={"email": "u@example.test", "user_id": "u1"},
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        code_challenge="c" * 43,
        code_challenge_method="S256",
        scopes=["instances:read"],
        audience=oauth_service.MCP_RESOURCE_AUDIENCE,
    )
    assert code


# ── Consent bookkeeping (BLOCKER 2, "…our login page → approve") ──────────


def test_consent_request_is_single_use(monkeypatch):
    store: dict[tuple[str, str], dict] = {}
    monkeypatch.setattr(
        oauth_service,
        "_cache_set_json",
        lambda ns, key, value, ttl, **_: store.__setitem__((ns, key), dict(value)) is None,
    )
    monkeypatch.setattr(
        oauth_service, "_cache_getdel_json", lambda ns, key: store.pop((ns, key), None)
    )
    key = oauth_service.stash_consent_request({"user_id": "u1", "scopes": ["instances:read"]})
    assert oauth_service.take_consent_request(key)["user_id"] == "u1"
    # A decision that could be replayed is not a decision.
    assert oauth_service.take_consent_request(key) is None


def test_consent_must_cover_every_requested_scope(monkeypatch):
    import db

    monkeypatch.setattr(
        db.ConsentStore, "get", staticmethod(lambda *_: {"scopes": ["instances:read"]})
    )
    assert oauth_service.consent_covers("u1", "c1", "", ["instances:read"])
    # A previously read-only approval must not silently authorize spending.
    assert not oauth_service.consent_covers("u1", "c1", "", ["instances:read", "instances:write"])


def test_every_connector_scope_has_a_plain_language_description():
    """A consent screen showing raw scope strings is not informed consent."""
    for scope in oauth_registration.CONNECTOR_ALLOWED_SCOPES:
        assert oauth_service.describe_scope(scope) != scope, f"{scope} has no description"


def test_registration_expiry_is_bounded_and_renewable():
    expiry = oauth_registration.registration_expiry()
    assert expiry.timestamp() > time.time()
    assert oauth_registration.DCR_UNUSED_TTL_DAYS >= 1


# ── Surface attribution (X7.34) ───────────────────────────────────────────


def test_surface_is_attributed_from_the_provider_callback():
    from oauth_registration import classify_surface

    assert classify_surface("https://claude.ai/api/mcp/auth_callback") == "claude"
    assert classify_surface("https://chatgpt.com/connector_platform_oauth_redirect") == "chatgpt"
    assert classify_surface("https://grok.com/api/mcp/auth_callback") == "grok"
    assert classify_surface("https://copilotstudio.microsoft.com/oauth/callback") == "microsoft"
    assert classify_surface("https://cursor.com/oauth/cb") == "cursor"


def test_a_loopback_callback_is_local_not_guessed():
    from oauth_registration import classify_surface

    # A native client binds 127.0.0.1; its identity genuinely is not in the
    # redirect, and inventing one would make the attribution report fiction.
    assert classify_surface("http://127.0.0.1:51423/callback") == "local"
    assert classify_surface("http://localhost:8976/cb") == "local"


def test_a_metadata_document_client_id_attributes_even_over_loopback():
    from oauth_registration import classify_surface

    assert classify_surface(
        "http://127.0.0.1:51423/callback", "https://claude.ai/mcp/client.json"
    ) == "claude"


def test_an_unrecognised_callback_is_unknown_rather_than_mislabelled():
    from oauth_registration import classify_surface

    assert classify_surface("https://someone-elses-agent.example/cb") == "unknown"
    assert classify_surface("") == "unknown"
