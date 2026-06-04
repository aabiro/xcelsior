"""Smoke coverage for routes/marketplace.py (UNTESTED_ENDPOINTS.md)."""

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
    email = f"mkcov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Market Cov"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def test_marketplace_spot_prices():
    r = client.get("/api/v2/marketplace/spot-prices")
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "spot_prices" in r.json()


def test_marketplace_spot_history():
    r = client.get("/api/v2/marketplace/spot-prices/RTX-4090/history")
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "history" in r.json()


def test_marketplace_stats_v2():
    r = client.get("/api/v2/marketplace/stats")
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_marketplace_search():
    r = client.post(
        "/api/v2/marketplace/search",
        json={"gpu_model": "", "limit": 10},
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "offers" in r.json()


def test_marketplace_release(user_headers):
    r = client.post(
        "/api/v2/marketplace/release/nonexistent-allocation-id",
        headers=user_headers,
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_marketplace_bill_missing_job():
    r = client.post(
        "/marketplace/bill/nonexistent-job-id",
        headers=_admin_headers(),
    )
    assert r.status_code == 400
    assert r.status_code != 500