"""Phase 14 — serverless feature flag (global env + owner allowlist)."""

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_PERSISTENT_AUTH"] = "true"
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")

from api import app
from db import UserStore

client = TestClient(app)


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    import api as api_mod
    import routes._deps as deps
    import routes.auth as auth

    monkeypatch.setattr(deps, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(auth, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)
    api_mod._RATE_BUCKETS.clear()


def _register() -> tuple[dict, str]:
    email = f"slff-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Flag"},
    )
    assert reg.status_code == 200
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    body = login.json()
    user = reg.json().get("user") or body.get("user") or {}
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    assert UserStore.get_user(email) is not None
    return headers, str(user["customer_id"])


class TestServerlessFeatureFlag:
    def test_enabled_probe_without_auth(self):
        r = client.get("/api/v2/serverless/enabled")
        assert r.status_code == 200
        body = r.json()
        assert body.get("ok") is True
        assert "enabled" in body
        assert "global_enabled" in body

    def test_create_blocked_when_globally_disabled(self, monkeypatch):
        monkeypatch.setenv("XCELSIOR_SERVERLESS_ENABLED", "false")
        headers, _ = _register()
        r = client.post(
            "/api/v2/serverless/endpoints",
            headers=headers,
            json={
                "name": "blocked",
                "mode": "custom",
                "docker_image": "xcelsior/serverless-base:cuda12.4-py3.12",
            },
        )
        assert r.status_code == 404

    def test_list_empty_when_not_on_allowlist(self, monkeypatch):
        headers, customer_id = _register()
        other = f"cust-{uuid.uuid4().hex[:12]}"
        monkeypatch.setenv("XCELSIOR_SERVERLESS_ENABLED", "true")
        monkeypatch.setenv("XCELSIOR_SERVERLESS_ALLOWLIST", other)
        r = client.get("/api/v2/serverless/endpoints", headers=headers)
        assert r.status_code == 200
        assert r.json().get("endpoints") == []

        monkeypatch.setenv("XCELSIOR_SERVERLESS_ALLOWLIST", customer_id)
        r2 = client.get("/api/v2/serverless/enabled", headers=headers)
        assert r2.status_code == 200
        assert r2.json().get("enabled") is True

    def test_allowlisted_owner_can_create(self, monkeypatch):
        headers, customer_id = _register()
        monkeypatch.setenv("XCELSIOR_SERVERLESS_ENABLED", "true")
        monkeypatch.setenv("XCELSIOR_SERVERLESS_ALLOWLIST", customer_id)
        deposit = client.post(
            f"/api/billing/wallet/{customer_id}/deposit",
            json={"amount_cad": 25.0},
            headers=headers,
        )
        assert deposit.status_code == 200
        r = client.post(
            "/api/v2/serverless/endpoints",
            headers=headers,
            json={
                "name": "allowed",
                "mode": "custom",
                "docker_image": "xcelsior/serverless-base:cuda12.4-py3.12",
                "min_workers": 0,
                "max_workers": 1,
            },
        )
        assert r.status_code == 200, r.text[:300]
        endpoint_id = r.json()["endpoint"]["endpoint_id"]
        client.delete(
            f"/api/v2/serverless/endpoints/{endpoint_id}",
            headers=headers,
        )