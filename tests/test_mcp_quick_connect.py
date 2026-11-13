"""Functional coverage for the /dashboard/mcp quick-connect endpoint.

Verifies the always-there copy-paste token flow: find-or-create a system-managed
MCP client, reveal its key once, rotate on regenerate, and never surface
the client in the user-facing OAuth client list.
"""

import os
import uuid

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
    assert body["access_token"].startswith("xcel_ai_")
    # Agent keys are revoked, not expired. Reporting an expiry would make
    # clients schedule a refresh that has nothing to refresh.
    assert body["expires_in"] is None
    assert body["in_use"] is False
    assert body["mcp_url"].endswith("/mcp")
    assert "gpu:read" in body["scopes"]


def test_quick_connect_is_idempotent():
    token = _register_and_get_token("qc-idem@xcelsior.ca")
    first = client.get("/api/mcp/quick-connect", headers=_auth(token)).json()
    second = client.get("/api/mcp/quick-connect", headers=_auth(token)).json()
    # Reloads must preserve even a copied-but-not-yet-used credential, and
    # show it again in full rather than masked (which pushed people to
    # regenerate, revoking the key their agent was using).
    assert first["client_id"] == second["client_id"]
    assert first["access_token"] and second["access_token"] == first["access_token"]
    assert first["key_id"] == second["key_id"]
    assert second["in_use"] is False
    assert _authenticates(first["access_token"])


def test_a_key_minted_before_keys_were_kept_encrypted_stays_masked():
    from db import auth_connection

    token = _register_and_get_token("qc-legacy@xcelsior.ca")
    first = client.get("/api/mcp/quick-connect", headers=_auth(token)).json()
    with auth_connection() as conn:
        conn.execute(
            "UPDATE agent_api_keys SET key_ciphertext = NULL WHERE key_id = %s",
            (first["key_id"],),
        )
    again = client.get("/api/mcp/quick-connect", headers=_auth(token)).json()
    assert again["access_token"] is None
    assert again["key_prefix"] == first["key_prefix"]


def test_the_stored_copy_never_leaves_through_the_key_list():
    token = _register_and_get_token("qc-list@xcelsior.ca")
    client.get("/api/mcp/quick-connect", headers=_auth(token))
    listed = client.get("/api/agent-keys", headers=_auth(token))
    assert listed.status_code == 200
    assert "key_ciphertext" not in listed.text
    assert "xcel_ai_" not in listed.text.replace("xcel_ai_…", "")


def test_concurrent_first_load_only_mints_one_key():
    from concurrent.futures import ThreadPoolExecutor

    token = _register_and_get_token("qc-concurrent@xcelsior.ca")

    def load():
        with TestClient(app) as isolated:
            response = isolated.get("/api/mcp/quick-connect?surface=cli", headers=_auth(token))
            assert response.status_code == 200, response.text
            return response.json()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: load(), range(2)))
    assert results[0]["client_id"] == results[1]["client_id"]
    assert results[0]["key_id"] == results[1]["key_id"]
    # One key minted, and both loads show that same key.
    assert results[0]["access_token"] == results[1]["access_token"]
    assert _authenticates(results[0]["access_token"])


def test_failed_rotation_preserves_previous_credential(monkeypatch):
    import oauth_service

    token = _register_and_get_token("qc-atomic@xcelsior.ca")
    first = client.get("/api/mcp/quick-connect", headers=_auth(token)).json()

    def fail(**kwargs):
        raise RuntimeError("Key storage unavailable")

    monkeypatch.setattr(oauth_service, "issue_agent_api_key", fail)
    with pytest.raises(RuntimeError, match="Key storage unavailable"):
        client.get("/api/mcp/quick-connect?regenerate=true", headers=_auth(token))
    assert _authenticates(first["access_token"])
    again = client.get("/api/mcp/quick-connect", headers=_auth(token)).json()
    assert again["client_id"] == first["client_id"]
    assert again["key_id"] == first["key_id"]


def test_regenerate_rotates_the_client():
    token = _register_and_get_token("qc-regen@xcelsior.ca")
    first = client.get("/api/mcp/quick-connect", headers=_auth(token)).json()
    rotated = client.get("/api/mcp/quick-connect?regenerate=true", headers=_auth(token)).json()
    assert rotated["client_id"] != first["client_id"]


def _authenticates(key: str) -> bool:
    return client.get("/api/auth/me", headers=_auth(key)).status_code == 200


def test_regenerate_revokes_the_previous_key():
    """Rotation that leaves the old key working is not rotation.

    Deleting the OAuth client did nothing to the agent keys bound to it, and
    key validation never looks the client up, so "Regenerate" used to hand out
    a second live credential while the one in the old config kept working.
    """
    token = _register_and_get_token("qc-revoke@xcelsior.ca")
    first = client.get("/api/mcp/quick-connect", headers=_auth(token)).json()["access_token"]
    assert _authenticates(first)
    second = client.get("/api/mcp/quick-connect?regenerate=true", headers=_auth(token)).json()[
        "access_token"
    ]
    assert second and second != first
    assert not _authenticates(first)
    assert _authenticates(second)


def test_cli_surface_is_its_own_credential():
    """The CLI tab's key is not the MCP key, and rotating one leaves the other."""
    token = _register_and_get_token("qc-cli@xcelsior.ca")
    mcp = client.get("/api/mcp/quick-connect?surface=mcp", headers=_auth(token)).json()
    cli = client.get("/api/mcp/quick-connect?surface=cli", headers=_auth(token)).json()
    assert mcp["client_id"] != cli["client_id"]
    assert mcp["access_token"] != cli["access_token"]
    # The Agent Skill drives the REST API through the SDK.
    assert _authenticates(cli["access_token"])

    rotated = client.get(
        "/api/mcp/quick-connect?surface=cli&regenerate=true", headers=_auth(token)
    ).json()["access_token"]
    assert not _authenticates(cli["access_token"])
    assert _authenticates(rotated)
    assert _authenticates(mcp["access_token"])


def test_unknown_surface_is_rejected():
    token = _register_and_get_token("qc-surface@xcelsior.ca")
    r = client.get("/api/mcp/quick-connect?surface=desktop", headers=_auth(token))
    assert r.status_code == 422


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


# ── The connector URL follows the environment, not a literal ──────────


def _quick_connect_api_url(monkeypatch, **env) -> str:
    """One Quick Connect call, with the URL environment set as given."""
    import routes.auth as auth_routes

    for name in ("XCELSIOR_PUBLIC_URL", "XCELSIOR_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    token = _register_and_get_token(f"qc-url-{uuid.uuid4().hex[:10]}@example.com")
    response = client.get("/api/mcp/quick-connect", headers=_auth(token))
    assert response.status_code == 200, response.text
    assert auth_routes  # the route module is the thing under test
    return response.json()["api_url"]


def test_the_connector_url_follows_the_plumbed_base_url(monkeypatch):
    """The staging case, and the reason this was wrong.

    `XCELSIOR_PUBLIC_URL` was read here and mapped in no compose file, so the
    container never received it and the literal fallback always won. On
    production that is invisible — the fallback *is* production — so the first
    place it would have shown up is staging, handing a staging user a config
    pointing at prod.
    """
    assert (
        _quick_connect_api_url(monkeypatch, XCELSIOR_BASE_URL="https://staging.xcelsior.ca")
        == "https://staging.xcelsior.ca"
    )


def test_an_explicit_public_url_still_wins(monkeypatch):
    """Any deployment already setting it keeps working."""
    assert (
        _quick_connect_api_url(
            monkeypatch,
            XCELSIOR_PUBLIC_URL="https://explicit.example",
            XCELSIOR_BASE_URL="https://staging.xcelsior.ca",
        )
        == "https://explicit.example"
    )


def test_a_trailing_slash_does_not_produce_a_double_slash(monkeypatch):
    """`{api_url}/api/...` is what a client builds from this."""
    assert (
        _quick_connect_api_url(monkeypatch, XCELSIOR_BASE_URL="https://staging.xcelsior.ca/")
        == "https://staging.xcelsior.ca"
    )


def test_with_nothing_set_it_falls_back_to_production(monkeypatch):
    """Unchanged behaviour where neither name is present."""
    assert _quick_connect_api_url(monkeypatch) == "https://xcelsior.ca"


# ── Permissions: Read, Write, Custom ──────────────────────────────────────


def _set_permissions(token: str, **body):
    return client.put("/api/mcp/quick-connect/permissions", headers=_auth(token), json=body)


def _write_class(scopes: list[str]) -> list[str]:
    return [s for s in scopes if not s.endswith(":read")]


def test_a_key_nobody_narrowed_is_the_write_preset():
    from oauth_service import MCP_QUICK_CONNECT_SCOPES

    token = _register_and_get_token("qc-perm-default@xcelsior.ca")
    body = client.get("/api/mcp/quick-connect", headers=_auth(token)).json()
    assert body["permission_preset"] == "write"
    assert body["scopes"] == list(MCP_QUICK_CONNECT_SCOPES)
    assert body["permission_presets"]["write"] == list(MCP_QUICK_CONNECT_SCOPES)
    assert body["permission_presets"]["read"]
    assert not _write_class(body["permission_presets"]["read"])


def test_read_holds_no_scope_that_changes_anything():
    """The point of the Read tab: it can look at instances and billing, and
    cannot launch, terminate, open a shell, add a key or touch a volume."""
    token = _register_and_get_token("qc-perm-read@xcelsior.ca")
    first = client.get("/api/mcp/quick-connect", headers=_auth(token)).json()

    r = _set_permissions(token, preset="read")
    assert r.status_code == 200, r.text
    scopes = r.json()["scopes"]
    assert "instances:read" in scopes and "billing:read" in scopes
    assert not _write_class(scopes), scopes
    assert r.json()["keys_updated"] == 1

    after = client.get("/api/mcp/quick-connect", headers=_auth(token)).json()
    assert after["permission_preset"] == "read"
    assert after["scopes"] == scopes
    # Narrowing must not cost the user their working setup.
    assert after["access_token"] == first["access_token"]
    assert after["key_id"] == first["key_id"]


def test_the_existing_key_is_what_gets_narrowed():
    """Stored scopes that the live key ignores would be a picker that lies.

    Driven through a real route with the real key: a write the key could make
    before is refused after, and a read still answers.
    """
    token = _register_and_get_token("qc-perm-live@xcelsior.ca")
    key = client.get("/api/mcp/quick-connect", headers=_auth(token)).json()["access_token"]

    def delete_endpoint():
        return client.delete("/api/v2/serverless/endpoints/does-not-exist", headers=_auth(key))

    assert delete_endpoint().status_code != 403
    assert _set_permissions(token, preset="read").status_code == 200
    refused = delete_endpoint()
    assert refused.status_code == 403, refused.text
    assert "inference:write" in refused.text
    assert client.get("/api/v2/serverless/endpoints", headers=_auth(key)).status_code == 200

    assert _set_permissions(token, preset="write").status_code == 200
    assert delete_endpoint().status_code != 403


def test_checking_write_brings_read_with_it():
    token = _register_and_get_token("qc-perm-implied@xcelsior.ca")
    r = _set_permissions(token, preset="custom", scopes=["volumes:write", "instances:connect"])
    assert r.status_code == 200, r.text
    assert r.json()["scopes"] == ["instances:read", "instances:connect", "volumes:read", "volumes:write"]
    assert r.json()["permission_preset"] == "custom"


def test_custom_cannot_reach_past_the_quick_connect_key():
    """Custom narrows. `billing:write` and `ssh:read` are real scopes that are
    deliberately not on this key, and a checkbox must not be a way to add them."""
    token = _register_and_get_token("qc-perm-ceiling@xcelsior.ca")
    before = client.get("/api/mcp/quick-connect", headers=_auth(token)).json()["scopes"]
    for scope in ("billing:write", "ssh:read", "hosts:evict", "api", "made:up"):
        r = _set_permissions(token, preset="custom", scopes=["gpu:read", scope])
        assert r.status_code == 400, (scope, r.text)
    assert _set_permissions(token, preset="custom", scopes=[]).status_code == 400
    assert _set_permissions(token, preset="everything").status_code == 422
    assert client.get("/api/mcp/quick-connect", headers=_auth(token)).json()["scopes"] == before


def test_a_key_cannot_change_its_own_permissions():
    token = _register_and_get_token("qc-perm-self@xcelsior.ca")
    key = client.get("/api/mcp/quick-connect", headers=_auth(token)).json()["access_token"]
    assert _set_permissions(token, preset="read").status_code == 200
    assert _set_permissions(key, preset="write").status_code == 403
    assert client.get("/api/mcp/quick-connect", headers=_auth(token)).json()["permission_preset"] == "read"


def test_regenerating_keeps_the_chosen_permissions():
    token = _register_and_get_token("qc-perm-regen@xcelsior.ca")
    client.get("/api/mcp/quick-connect", headers=_auth(token))
    chosen = _set_permissions(token, preset="custom", scopes=["gpu:read", "instances:write"]).json()
    rotated = client.get("/api/mcp/quick-connect?regenerate=true", headers=_auth(token)).json()
    assert rotated["permission_preset"] == "custom"
    assert rotated["scopes"] == chosen["scopes"] == ["instances:read", "instances:write", "gpu:read"]

    _set_permissions(token, preset="read")
    rotated = client.get("/api/mcp/quick-connect?regenerate=true", headers=_auth(token)).json()
    assert rotated["permission_preset"] == "read"
    assert not _write_class(rotated["scopes"])
    assert _authenticates(rotated["access_token"])


def test_each_surface_keeps_its_own_permissions():
    token = _register_and_get_token("qc-perm-surface@xcelsior.ca")
    client.get("/api/mcp/quick-connect?surface=cli", headers=_auth(token))
    assert _set_permissions(token, preset="read", surface="mcp").status_code == 200
    cli = client.get("/api/mcp/quick-connect?surface=cli", headers=_auth(token)).json()
    mcp = client.get("/api/mcp/quick-connect?surface=mcp", headers=_auth(token)).json()
    assert cli["permission_preset"] == "write" and _write_class(cli["scopes"])
    assert mcp["permission_preset"] == "read" and not _write_class(mcp["scopes"])
