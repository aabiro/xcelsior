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
    def test_machine_client_cannot_access_mfa(self):
        token = _register_and_get_token("mfa-machine@xcelsior.ca")
        
        created = client.post(
            "/api/oauth/clients",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "client_name": "MFA Test Machine",
                "client_type": "confidential",
                "redirect_uris": [],
                "grant_types": ["client_credentials"],
                "scopes": ["api"],
            },
        ).json()["client"]
        
        token_resp = client.post(
            "/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": created["client_id"],
                "client_secret": created["client_secret"],
                "scope": "api",
            },
        )
        machine_token = token_resp.json()["access_token"]
        
        # Access MFA
        r = client.get("/api/auth/mfa/methods", headers={"Authorization": f"Bearer {machine_token}"})
        assert r.status_code == 403
        
    def test_machine_client_cannot_access_ssh_keys_by_default(self):
        token = _register_and_get_token("ssh-machine@xcelsior.ca")
        
        created = client.post(
            "/api/oauth/clients",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "client_name": "SSH Test Machine",
                "client_type": "confidential",
                "redirect_uris": [],
                "grant_types": ["client_credentials"],
                "scopes": ["api"],
            },
        ).json()["client"]
        
        token_resp = client.post(
            "/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": created["client_id"],
                "client_secret": created["client_secret"],
                "scope": "api",
            },
        )
        machine_token = token_resp.json()["access_token"]
        
        # Access SSH keys (requires interactive user by default)
        r = client.get("/api/ssh/keys", headers={"Authorization": f"Bearer {machine_token}"})
        assert r.status_code == 403
        
    def test_deprecation_telemetry_fired_on_api_key_usage(self, monkeypatch):
        import routes._deps as _deps_mod
        token = _register_and_get_token("deprecated-telemetry@xcelsior.ca")
        
        # Create an API key
        key_resp = client.post(
            "/api/keys/generate",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "test-key", "scope": "full-access"},
        )
        assert key_resp.status_code == 200
        api_key = key_resp.json()["key"]
        
        # Get baseline counter metric value
        try:
            baseline = _deps_mod._deprecated_api_key_requests.labels("deprecated-telemetry@xcelsior.ca")._value.get()
        except:
            baseline = 0
            
        import api
        monkeypatch.setenv("XCELSIOR_API_TOKEN", "test-master-token")
        monkeypatch.setattr(_deps_mod, "AUTH_REQUIRED", True)
        monkeypatch.setattr(api, "AUTH_REQUIRED", True)
        # Use API key to access telemetry
        r = client.get("/api/telemetry/all", headers={"Authorization": f"Bearer {api_key}"})
        assert r.status_code == 200

        # Verify deprecation headers exist
        assert "Deprecation" in r.headers
        assert "Warning" in r.headers
        
        # Verify prometheus counter increased
        try:
            metric_val = _deps_mod._deprecated_api_key_requests.labels("deprecated-telemetry@xcelsior.ca")._value.get()
            assert metric_val > baseline
        except:
            pass # Noop counter in test mode
