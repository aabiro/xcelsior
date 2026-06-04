"""Smoke coverage for routes/cloudburst.py — GET /api/v2/burst/status."""

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
    email = f"burstcov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Burst Cov"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def test_cloudburst_status(user_headers):
    r = client.get("/api/v2/burst/status", headers=user_headers)
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_cloudburst_status_requires_auth():
    r = client.get("/api/v2/burst/status")
    assert r.status_code == 401