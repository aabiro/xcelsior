import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_API_TOKEN", "testtoken")
os.environ.setdefault("XCELSIOR_ENV", "test")

from api import app
from db import MfaStore, UserStore, auth_connection


@pytest.fixture(autouse=True)
def clean_mfa_flow_state():
    with auth_connection() as conn:
        conn.execute("DELETE FROM mfa_backup_codes")
        conn.execute("DELETE FROM mfa_methods")
        conn.execute("DELETE FROM mfa_challenges")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM users WHERE email LIKE 'mfa-flow-%'")
    yield
    with auth_connection() as conn:
        conn.execute("DELETE FROM mfa_backup_codes")
        conn.execute("DELETE FROM mfa_methods")
        conn.execute("DELETE FROM mfa_challenges")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM users WHERE email LIKE 'mfa-flow-%'")


class TestMfaFlow:
    def test_login_requires_mfa_when_enabled_method_exists_even_if_flag_is_stale(self):
        client = TestClient(app)
        email = f"mfa-flow-login-{uuid.uuid4().hex[:8]}@xcelsior.ca"
        password = "StaleLogin123!"

        reg = client.post(
            "/api/auth/register",
            json={"email": email, "password": password, "name": "MFA Stale"},
        )
        assert reg.status_code == 200, reg.text

        MfaStore.create_method({
            "email": email,
            "method_type": "totp",
            "secret": "JBSWY3DPEHPK3PXP",
            "enabled": 1,
            "created_at": time.time(),
        })
        UserStore.update_user(email, {"mfa_enabled": 0})
        client.cookies.clear()

        login = client.post("/api/auth/login", json={"email": email, "password": password})
        assert login.status_code == 200, login.text
        body = login.json()
        assert body["mfa_required"] is True
        assert "totp" in body["methods"]
        assert bool(UserStore.get_user(email)["mfa_enabled"]) is True

    def test_mfa_methods_endpoint_uses_live_methods_when_flag_is_stale(self):
        client = TestClient(app)
        email = f"mfa-flow-methods-{uuid.uuid4().hex[:8]}@xcelsior.ca"
        password = "StaleMethods123!"

        reg = client.post(
            "/api/auth/register",
            json={"email": email, "password": password, "name": "MFA Methods"},
        )
        assert reg.status_code == 200, reg.text

        MfaStore.create_method({
            "email": email,
            "method_type": "passkey",
            "credential_id": "test-credential-id",
            "public_key": "test-public-key",
            "device_name": "Test Passkey",
            "enabled": 1,
            "created_at": time.time(),
        })
        UserStore.update_user(email, {"mfa_enabled": 0})

        res = client.get("/api/auth/mfa/methods")
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["mfa_enabled"] is True
        assert any(method["type"] == "passkey" and method["enabled"] for method in body["methods"])
        assert bool(UserStore.get_user(email)["mfa_enabled"]) is True
