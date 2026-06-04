"""Smoke coverage for routes/agent.py — POST /agent/ssh-status/{job_id}."""

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
def job_id():
    email = f"agentcov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Agent Cov"},
    ).json()
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    customer_id = reg["user"]["customer_id"]
    dep = client.post(
        f"/api/billing/wallet/{customer_id}/deposit",
        json={"amount_cad": 50.0, "description": "Agent cov"},
        headers=headers,
    )
    assert dep.status_code == 200
    inst = client.post(
        "/instance",
        json={"name": f"agt-{uuid.uuid4().hex[:6]}", "vram_needed_gb": 1},
        headers=headers,
    ).json()["instance"]
    return inst["job_id"]


def test_agent_ssh_status(job_id):
    r = client.post(
        f"/agent/ssh-status/{job_id}",
        json={
            "ok": True,
            "sshd_present": True,
            "sshd_started": True,
            "key_count": 1,
            "summary": "coverage smoke",
            "level": "info",
        },
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True