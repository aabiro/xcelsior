"""Smoke coverage for routes/autoscale.py (UNTESTED_ENDPOINTS.md)."""

import os

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _admin_headers() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def user_headers():
    import uuid

    email = f"ascov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Ascov"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def test_autoscale_cycle_admin():
    r = client.post("/autoscale/cycle", headers=_admin_headers())
    assert r.status_code == 200
    body = r.json()
    assert "provisioned" in body
    assert "assigned" in body
    assert "deprovisioned" in body


def test_autoscale_up_admin():
    r = client.post("/autoscale/up", headers=_admin_headers())
    assert r.status_code == 200
    assert "provisioned" in r.json()


def test_autoscale_down_admin():
    r = client.post("/autoscale/down", headers=_admin_headers())
    assert r.status_code == 200
    assert "deprovisioned" in r.json()


def test_autoscale_pool_read_admin():
    r = client.get("/autoscale/pool", headers=_admin_headers())
    assert r.status_code == 200
    assert "pool" in r.json()


def test_autoscale_cycle_forbidden_non_admin(user_headers):
    r = client.post("/autoscale/cycle", headers=user_headers)
    assert r.status_code == 403