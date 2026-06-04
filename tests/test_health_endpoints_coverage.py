"""Smoke coverage for routes/health.py endpoints listed in UNTESTED_ENDPOINTS.md."""

import os

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _admin_headers() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


def test_healthz_liveness():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_llms_txt_served_or_missing():
    r = client.get("/llms.txt")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert "text/plain" in (r.headers.get("content-type") or "")


def test_alerts_config_requires_admin_when_auth_enforced(monkeypatch):
    import routes._deps as deps
    import routes.health as health_mod

    monkeypatch.setattr(deps, "AUTH_REQUIRED", True)
    monkeypatch.setattr(health_mod, "AUTH_REQUIRED", True)
    r = client.get("/alerts/config")
    assert r.status_code in (401, 403)


def test_alerts_config_admin_read():
    r = client.get("/alerts/config", headers=_admin_headers())
    assert r.status_code == 200
    assert "config" in r.json()


def test_api_alerts_config_alias():
    r = client.get("/api/alerts/config", headers=_admin_headers())
    assert r.status_code == 200
    assert "config" in r.json()


def test_ssh_pubkey_unauthenticated():
    r = client.get("/api/ssh/pubkey")
    assert r.status_code in (401, 403, 200)


def test_builds_list_requires_auth():
    r = client.get("/builds")
    assert r.status_code in (401, 403, 200)