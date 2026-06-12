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
            "gpu_model": "RTX 4090",
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
    assert host.get("gpu_model") == "RTX 4090"


def test_hosts_register_web_requires_auth():
    r = client.post(
        "/api/hosts/register",
        json={
            "hostname": "no-auth",
            "gpu_model": "RTX 4090",
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
            "gpu_model": "RTX 4090",
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


def _machine_token_for_user(user_headers: dict, *, scopes: str = "api") -> str:
    created = client.post(
        "/api/oauth/clients",
        headers=user_headers["headers"],
        json={
            "client_name": "Worker Host Agent",
            "client_type": "confidential",
            "redirect_uris": [],
            "grant_types": ["client_credentials"],
            "scopes": ["api", "hosts:read", "hosts:write"],
        },
    )
    assert created.status_code == 200, created.text
    oauth_client = created.json()["client"]
    token_resp = client.post(
        "/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": oauth_client["client_id"],
            "client_secret": oauth_client["client_secret"],
            "scope": scopes,
        },
    )
    assert token_resp.status_code == 200, token_resp.text
    return token_resp.json()["access_token"]


def test_worker_oauth_heartbeat_for_dashboard_host(user_headers):
    """Dashboard-registered hosts use custom host_id; worker OAuth must match owner."""
    reg = client.post(
        "/api/hosts/register",
        json={
            "hostname": f"worker-{uuid.uuid4().hex[:6]}",
            "gpu_model": "RTX 4090",
            "vram_gb": 24,
            "cost_per_hour": 0.35,
            "country": "CA",
            "province": "ON",
        },
        headers=user_headers["headers"],
    )
    assert reg.status_code == 200, reg.text
    host_id = reg.json()["host"]["host_id"]

    machine_token = _machine_token_for_user(user_headers)
    heartbeat = client.put(
        "/host",
        json={
            "host_id": host_id,
            "ip": "192.168.1.50",
            "gpu_model": "RTX 4090",
            "total_vram_gb": 24,
            "free_vram_gb": 24,
            "country": "CA",
            "province": "ON",
        },
        headers={"Authorization": f"Bearer {machine_token}"},
    )
    assert heartbeat.status_code == 200, heartbeat.text


def test_worker_oauth_heartbeat_rejects_foreign_host(user_headers):
    owner = user_headers
    other_email = f"other-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": other_email, "password": "StrongPass123!", "name": "Other"},
    )
    other_login = client.post(
        "/api/auth/login", json={"email": other_email, "password": "StrongPass123!"}
    )
    assert other_login.status_code == 200
    other_headers = {"Authorization": f"Bearer {other_login.json()['access_token']}"}

    reg = client.post(
        "/api/hosts/register",
        json={
            "hostname": f"owned-{uuid.uuid4().hex[:6]}",
            "gpu_model": "RTX 4090",
            "vram_gb": 24,
            "cost_per_hour": 0.5,
            "country": "CA",
            "province": "ON",
        },
        headers=owner["headers"],
    )
    assert reg.status_code == 200, reg.text
    host_id = reg.json()["host"]["host_id"]

    machine_token = _machine_token_for_user({"headers": other_headers})
    heartbeat = client.put(
        "/host",
        json={
            "host_id": host_id,
            "ip": "10.0.0.2",
            "gpu_model": "RTX 4090",
            "total_vram_gb": 24,
            "free_vram_gb": 24,
            "country": "CA",
            "province": "ON",
        },
        headers={"Authorization": f"Bearer {machine_token}"},
    )
    assert heartbeat.status_code == 403


def test_worker_oauth_registers_custom_host_id(user_headers):
    """First heartbeat may claim a custom host_id for the OAuth client creator."""
    host_id = f"gpu-2060-{uuid.uuid4().hex[:8]}"
    machine_token = _machine_token_for_user(user_headers)
    heartbeat = client.put(
        "/host",
        json={
            "host_id": host_id,
            "ip": "192.168.1.60",
            "gpu_model": "RTX 2060",
            "total_vram_gb": 6,
            "free_vram_gb": 6,
            "country": "CA",
            "province": "ON",
        },
        headers={"Authorization": f"Bearer {machine_token}"},
    )
    assert heartbeat.status_code == 200, heartbeat.text

    listed = client.get("/hosts?active_only=false", headers=user_headers["headers"])
    host = next(h for h in listed.json().get("hosts", []) if h["host_id"] == host_id)
    assert host.get("owner") in {user_headers["user_id"], user_headers["email"]}