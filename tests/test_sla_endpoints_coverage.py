"""Smoke coverage for routes/sla.py — GET /api/sla/violations/{host_id}."""

import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


@pytest.fixture(scope="module")
def user_headers():
    email = f"slacov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "SLA Cov"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def test_sla_violations(user_headers):
    host_id = f"sla-host-{uuid.uuid4().hex[:8]}"
    r = client.get(f"/api/sla/violations/{host_id}", headers=user_headers)
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert r.json().get("host_id") == host_id
    assert isinstance(r.json().get("violations"), list)