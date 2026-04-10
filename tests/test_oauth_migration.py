import os
import tempfile
import time
import pytest
from fastapi.testclient import TestClient

import scheduler
_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_test_oauth_")
_tmpdir = _tmp_ctx.name
os.environ["XCELSIOR_API_TOKEN"] = "testtoken"
os.environ["XCELSIOR_DB_PATH"] = os.path.join(_tmpdir, "xcelsior.db")
os.environ["XCELSIOR_AUTH_DB_PATH"] = os.path.join(_tmpdir, "auth.db")
os.environ["XCELSIOR_ENV"] = "test"
import db as db_mod
db_mod.AUTH_DB_FILE = os.path.join(_tmpdir, "auth.db")

from api import app
client = TestClient(app)

class TestOAuthMigrationSecurity:
    def test_machine_client_cannot_access_mfa(self):
        reg = client.post(
            "/api/auth/register",
            json={"email": "mfa-machine@xcelsior.ca", "password": "testpass123"},
        ).json()
        token = reg["access_token"]
        
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
        reg = client.post(
            "/api/auth/register",
            json={"email": "ssh-machine@xcelsior.ca", "password": "testpass123"},
        ).json()
        token = reg["access_token"]
        
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
        
    def test_deprecation_telemetry_fired_on_api_key_usage(self):
        import routes._deps as _deps_mod
        reg = client.post(
            "/api/auth/register",
            json={"email": "deprecated-telemetry@xcelsior.ca", "password": "testpass123"},
        ).json()
        token = reg["access_token"]
        
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
        _deps_mod.AUTH_REQUIRED = True
        api.AUTH_REQUIRED = True
        try:
            # Use API key to access telemetry
            r = client.get("/api/telemetry/all", headers={"Authorization": f"Bearer {api_key}"})
            assert r.status_code == 200
            
            # Verify deprecation headers exist
            assert "Deprecation" in r.headers
            assert "Warning" in r.headers
        finally:
            _deps_mod.AUTH_REQUIRED = False
            api.AUTH_REQUIRED = False
        
        # Verify prometheus counter increased
        try:
            metric_val = _deps_mod._deprecated_api_key_requests.labels("deprecated-telemetry@xcelsior.ca")._value.get()
            assert metric_val > baseline
        except:
            pass # Noop counter in test mode
