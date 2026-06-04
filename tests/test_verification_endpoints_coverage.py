"""Smoke coverage for routes/verification.py (UNTESTED_ENDPOINTS.md)."""

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
    email = f"vercov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Verify Cov"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    assert login.status_code == 200
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def test_verification_verified_hosts():
    r = client.get("/api/verified-hosts")
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert isinstance(r.json().get("hosts"), list)


def test_verification_status_unverified():
    host_id = f"host-{uuid.uuid4().hex[:8]}"
    r = client.get(f"/api/verify/{host_id}/status")
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert r.json().get("status") == "unverified"


def test_verification_run_requires_auth():
    host_id = f"host-{uuid.uuid4().hex[:8]}"
    r = client.post(f"/api/verify/{host_id}", json={"gpu_info": {}, "network_info": {}})
    assert r.status_code == 401


def test_verification_run_authenticated(user_headers):
    host_id = f"host-{uuid.uuid4().hex[:8]}"
    r = client.post(
        f"/api/verify/{host_id}",
        json={"gpu_info": {"model": "test-gpu"}, "network_info": {}},
        headers=user_headers,
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "verification" in r.json()


def test_verification_admin_approve_and_reject():
    host_id = f"host-{uuid.uuid4().hex[:8]}"
    r = client.post(
        f"/api/verify/{host_id}/approve",
        params={"notes": "coverage"},
        headers=_admin_headers(),
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert r.json().get("status") == "verified"

    r2 = client.get(f"/api/verify/{host_id}/status")
    assert r2.status_code == 200
    assert r2.json().get("verification") is not None

    r3 = client.post(
        f"/api/verify/{host_id}/reject",
        params={"reason": "coverage reject"},
        headers=_admin_headers(),
    )
    assert r3.status_code == 200
    assert r3.json().get("ok") is True
    assert r3.json().get("status") == "deverified"


def test_agent_verify():
    host_id = f"host-{uuid.uuid4().hex[:8]}"
    r = client.post(
        "/agent/verify",
        json={
            "host_id": host_id,
            "report": {"gpu_model": "RTX 4090", "vram_gb": 24},
        },
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert r.json().get("host_id") == host_id