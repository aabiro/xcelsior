import os
import pytest
from fastapi.testclient import TestClient

import scheduler

os.environ.setdefault("XCELSIOR_API_TOKEN", "testtoken")
os.environ.setdefault("XCELSIOR_ENV", "test")

from api import app

client = TestClient(app)


def _register_and_get_token(email: str, password: str = "testpass123") -> str:
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": password},
    )
    assert reg.status_code == 200
    reg_body = reg.json()
    token = reg_body.get("access_token")
    if token:
        return token

    if reg_body.get("email_verification_required"):
        import routes._deps as _deps_mod
        from db import auth_connection

        verification_token = None
        if _deps_mod._USE_PERSISTENT_AUTH:
            with auth_connection() as conn:
                row = conn.execute(
                    "SELECT email_verification_token FROM users WHERE email = %s",
                    (email,),
                ).fetchone()
            if row:
                verification_token = row["email_verification_token"]
        else:
            verification_token = _deps_mod._users_db.get(email, {}).get("email_verification_token")

        assert verification_token, f"missing verification token for {email}"
        verified = client.post("/api/auth/verify-email", json={"token": verification_token})
        assert verified.status_code == 200, verified.text
        verified_body = verified.json()
        token = verified_body.get("access_token")
        if token:
            return token

    login = client.post(
        "/api/auth/login",
        json={"email": email, "password": password},
    )
    assert login.status_code == 200, login.text
    return login.json()["access_token"]


@pytest.fixture(autouse=True)
def clean_oauth_migration_state():
    import routes._deps as _deps_mod
    from routes.agent import _host_telemetry
    from oauth_service import reset_auth_cache_for_tests
    from db import auth_connection

    with scheduler._atomic_mutation() as conn:
        conn.execute("DELETE FROM state")

    with auth_connection() as conn:
        conn.execute("DELETE FROM oauth_refresh_tokens")
        conn.execute("DELETE FROM oauth_clients")
        conn.execute("DELETE FROM api_keys")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM users")

    reset_auth_cache_for_tests()
    client.cookies.clear()
    _host_telemetry.clear()
    _deps_mod._RATE_BUCKETS.clear()
    _deps_mod._AUTH_RATE_BUCKETS.clear()
    _deps_mod._users_db.clear()
    _deps_mod._sessions.clear()
    _deps_mod._api_keys.clear()
    yield


class TestOAuthMigrationSecurity:
    def _machine_token(self, email: str, scopes: list[str]) -> str:
        """A client-credentials token holding exactly *scopes*."""
        token = _register_and_get_token(email)
        created = client.post(
            "/api/oauth/clients",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "client_name": f"machine-{email}",
                "client_type": "confidential",
                "redirect_uris": [],
                "grant_types": ["client_credentials"],
                "scopes": scopes,
            },
        )
        assert created.status_code == 200, created.text[:300]
        payload = created.json()["client"]
        token_resp = client.post(
            "/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": payload["client_id"],
                "client_secret": payload["client_secret"],
                "scope": " ".join(scopes),
            },
        )
        assert token_resp.status_code == 200, token_resp.text[:300]
        return token_resp.json()["access_token"]

    def test_the_api_wildcard_scope_cannot_be_registered(self):
        """`api` short-circuited every per-tool check. It is gone from the
        vocabulary, so asking for it now fails where the mistake is made.

        These security tests used to register clients with `scopes: ["api"]` and
        assert the resulting token was still refused. That framing outlived the
        wildcard: `api` is no longer a scope any client can hold.
        """
        token = _register_and_get_token("wildcard-machine@xcelsior.ca")
        refused = client.post(
            "/api/oauth/clients",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "client_name": "Wildcard Machine",
                "client_type": "confidential",
                "redirect_uris": [],
                "grant_types": ["client_credentials"],
                "scopes": ["api"],
            },
        )
        assert refused.status_code == 400, refused.text[:300]
        assert "api" in refused.text

    def test_machine_client_cannot_access_mfa(self):
        """Account security is not delegable to a machine credential."""
        machine_token = self._machine_token(
            "mfa-machine@xcelsior.ca", ["instances:read"]
        )
        r = client.get(
            "/api/auth/mfa/methods", headers={"Authorization": f"Bearer {machine_token}"}
        )
        assert r.status_code == 403, r.text[:200]

    def test_machine_client_cannot_access_ssh_keys_by_default(self):
        """`ssh:read` is now grantable — but only when actually granted.

        Before this was a scope no client could hold, so the endpoint was
        unreachable by construction. It is reachable now, which makes this the
        real test: a machine credential without `ssh:read` is still refused.
        """
        machine_token = self._machine_token(
            "ssh-machine@xcelsior.ca", ["instances:read"]
        )
        r = client.get("/api/ssh/keys", headers={"Authorization": f"Bearer {machine_token}"})
        assert r.status_code == 403, r.text[:200]

    def test_ssh_keys_admit_an_oauth_machine_token_holding_the_scope(self):
        """Superseded deliberately, and rewritten rather than deleted.

        This test previously asserted the opposite: that `routes/ssh.py` refused
        `client_credentials` even with `ssh:read`, because `_require_user_grant`
        rejects machine tokens outright. That was the real behaviour and worth
        pinning at the time.

        It was changed on purpose. Gate P2 requires launch → wait → connect →
        run → terminate using **only tool calls**, and an OAuth-connected agent
        that cannot register its public key cannot complete it. Registering a
        *public* key is not the same act as changing a password: the agent
        already holds the private half, so the platform is accepting an
        assertion about a key the caller controls.

        The narrowing is confined to the SSH key routes via
        `_require_user_or_scoped_machine`; the account-security surface keeps
        `_require_user_grant`, which
        `tests/test_ssh_key_registration_credentials.py` asserts separately.
        """
        machine_token = self._machine_token(
            "ssh-granted@xcelsior.ca", ["instances:read", "ssh:read"]
        )
        r = client.get("/api/ssh/keys", headers={"Authorization": f"Bearer {machine_token}"})
        assert r.status_code == 200, r.text[:200]

    def test_ssh_keys_still_refuse_a_machine_token_without_the_scope(self):
        """The refusal that keeps the change narrow.

        Admitting machine credentials is only safe because the scope is
        actually required. Without this, the supersession above would read as
        "machine tokens may now reach SSH keys".
        """
        machine_token = self._machine_token(
            "ssh-unscoped@xcelsior.ca", ["instances:read"]
        )
        r = client.get("/api/ssh/keys", headers={"Authorization": f"Bearer {machine_token}"})
        assert r.status_code == 403, r.text[:200]

    def test_api_keys_permanently_rejected(self, monkeypatch):
        """API keys are permanently disabled — auth with one must return 401."""
        import routes._deps as _deps_mod

        token = _register_and_get_token("rejected-key@xcelsior.ca")

        # /api/keys/generate should return 410 Gone
        key_resp = client.post(
            "/api/keys/generate",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "test-key", "scope": "full-access"},
        )
        assert key_resp.status_code == 410

        # /api/keys list should return 410 Gone
        list_resp = client.get(
            "/api/keys",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert list_resp.status_code == 410
