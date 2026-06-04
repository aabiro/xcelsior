"""Smoke coverage for routes/reputation.py — GET /api/reputation/me."""

import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


@pytest.fixture(scope="module")
def user_headers():
    email = f"repcov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Reputation Cov"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def test_reputation_me(user_headers):
    r = client.get("/api/reputation/me", headers=user_headers)
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    assert "tier" in body or "score" in body