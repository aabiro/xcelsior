"""Smoke coverage for routes/hosts.py (UNTESTED_ENDPOINTS.md)."""

import os
import uuid

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


@pytest.fixture(scope="module")
def user_headers():
    email = f"hostcov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Host Cov"},
    ).json()
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    assert login.status_code == 200
    return {
        "email": email,
        "user_id": reg["user"].get("user_id", email),
        "headers": {"Authorization": f"Bearer {login.json()['access_token']}"},
    }


def test_hosts_register_web(user_headers):
    r = client.post(
        "/api/hosts/register",
        json={
            "hostname": f"cov-{uuid.uuid4().hex[:6]}",
            "gpu_model": "RTX-4090",
            "vram_gb": 24,
            "cost_per_hour": 0.5,
            "country": "CA",
            "province": "ON",
        },
        headers=user_headers["headers"],
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    host = r.json().get("host") or {}
    assert host.get("host_id")
    assert host.get("gpu_model") == "RTX-4090"


def test_hosts_register_web_requires_auth():
    r = client.post(
        "/api/hosts/register",
        json={
            "hostname": "no-auth",
            "gpu_model": "RTX-4090",
            "vram_gb": 24,
        },
    )
    assert r.status_code == 401


def test_hosts_check_admin():
    r = client.post("/hosts/check", headers=_admin_headers())
    assert r.status_code == 200
    assert "results" in r.json()


def test_hosts_list_and_get(user_headers):
    # host_id must match operator identity for non-admin callers
    host_id = user_headers["user_id"]
    r = client.put(
        "/host",
        json={
            "host_id": host_id,
            "ip": "10.0.0.9",
            "gpu_model": "RTX-4090",
            "total_vram_gb": 24,
            "free_vram_gb": 24,
            "country": "CA",
            "province": "ON",
        },
        headers=user_headers["headers"],
    )
    assert r.status_code == 200

    r2 = client.get("/hosts?active_only=false", headers=user_headers["headers"])
    assert r2.status_code == 200
    assert any(h["host_id"] == host_id for h in r2.json().get("hosts", []))

    r3 = client.get(f"/host/{host_id}", headers=user_headers["headers"])
    assert r3.status_code == 200
    assert r3.json().get("host", {}).get("host_id") == host_id


def test_hosts_compute_scores():
    r = client.get("/compute-scores")
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert isinstance(r.json().get("scores"), dict)