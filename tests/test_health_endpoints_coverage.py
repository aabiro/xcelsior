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


def test_builds_list():
    r = client.get("/builds")
    assert r.status_code == 200
    assert "builds" in r.json()


def test_legacy_auth_verify_page():
    r = client.get("/_internal/legacy-auth/verify")
    assert r.status_code == 200
    assert "text/html" in (r.headers.get("content-type") or "")
    assert "Device" in r.text or "device" in r.text.lower()


def test_legacy_dashboard():
    r = client.get("/legacy/settings")
    assert r.status_code == 200
    assert "text/html" in (r.headers.get("content-type") or "")


def test_legacy_auth_device_flow():
    r = client.post("/_internal/legacy-auth/device")
    assert r.status_code == 200
    body = r.json()
    assert body.get("device_code")
    assert body.get("user_code")

    r_poll = client.post(
        "/_internal/legacy-auth/token",
        json={
            "device_code": body["device_code"],
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        },
    )
    assert r_poll.status_code in (428, 410)

    r_verify = client.post(
        "/_internal/legacy-auth/verify",
        json={"user_code": "ZZZZ-ZZZZ"},
    )
    assert r_verify.status_code in (404, 410)


def test_ssh_pubkey_aliases():
    for path in ("/ssh/pubkey", "/api/ssh/pubkey"):
        r = client.get(path)
        assert r.status_code == 200
        assert "public_key" in r.json()


def test_ssh_keygen_requires_auth():
    r = client.post("/ssh/keygen")
    assert r.status_code in (401, 403, 200)


def test_ssh_keygen_authenticated():
    import uuid

    email = f"healthcov-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Health"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    r = client.post("/ssh/keygen", headers=headers)
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert r.json().get("public_key")


def test_build_dockerfile_preview():
    r = client.post("/build/tiny-model/dockerfile")
    assert r.status_code == 200
    assert "dockerfile" in r.json()


def test_build_image_mocked(monkeypatch):
    import routes.health as health_mod

    monkeypatch.setattr(
        health_mod,
        "build_and_push",
        lambda *a, **k: {"built": True, "model": "mock-model", "image": "mock:latest"},
    )
    r = client.post(
        "/build",
        json={"model": "mock-model", "base_image": "python:3.11-slim", "push": False},
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_alerts_config_admin_put():
    r = client.put(
        "/alerts/config",
        headers=_admin_headers(),
        json={"email_enabled": False},
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_api_alerts_config_admin_put_alias():
    r = client.put(
        "/api/alerts/config",
        headers=_admin_headers(),
        json={"telegram_enabled": False},
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_slurm_instances_admin():
    r = client.get("/api/slurm/instances", headers=_admin_headers())
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "jobs" in r.json()


def test_metrics_snapshot_includes_volumes(monkeypatch):
    monkeypatch.setattr(
        "routes.health.get_metrics_snapshot",
        lambda: {
            "active_hosts": 1,
            "running_jobs": 0,
            "failed_jobs": 0,
            "billing_totals": {"total_revenue": 0, "records": 0},
            "volumes": {"total": 3, "error": 1, "nfs_reachable": 1},
            "notifications": {"retained_total": 0},
            "web_push": {},
        },
    )
    r = client.get("/metrics")
    assert r.status_code == 200
    volumes = r.json()["metrics"]["volumes"]
    assert volumes["total"] == 3
    assert volumes["error"] == 1


def test_metrics_prometheus_volume_gauges(monkeypatch):
    monkeypatch.setattr(
        "routes.health.get_metrics_snapshot",
        lambda: {
            "active_hosts": 0,
            "running_jobs": 0,
            "failed_jobs": 0,
            "billing_totals": {"total_revenue": 0, "records": 0},
            "volumes": {"total": 5, "error": 2, "nfs_reachable": 1},
            "notifications": {"retained_total": 0},
            "web_push": {},
        },
    )
    r = client.get("/metrics/prometheus")
    assert r.status_code == 200
    body = r.text
    assert "xcelsior_volumes_total 5" in body
    assert "xcelsior_volumes_error 2" in body
    assert "xcelsior_nfs_reachable 1" in body