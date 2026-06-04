"""Smoke coverage for routes/events.py (UNTESTED_ENDPOINTS.md)."""

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


def _fund_wallet(customer_id: str, headers: dict) -> None:
    r = client.post(
        f"/api/billing/wallet/{customer_id}/deposit",
        json={"amount_cad": 50.0, "description": "Events cov"},
        headers=headers,
    )
    assert r.status_code == 200, r.text[:200]


@pytest.fixture(scope="module")
def job_ctx():
    email = f"evtcov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Events Cov"},
    ).json()
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    customer_id = reg["user"]["customer_id"]
    _fund_wallet(customer_id, headers)
    job = client.post(
        "/instance",
        json={"name": f"evt-{uuid.uuid4().hex[:6]}", "vram_needed_gb": 1},
        headers=headers,
    ).json()["instance"]
    job_id = job["job_id"]
    from events import get_state_machine

    sm = get_state_machine()
    sm.transition(job_id, "queued", "assigned")
    return {"headers": headers, "job_id": job_id}


def test_events_list_all_admin():
    r = client.get("/api/events", headers=_admin_headers())
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert isinstance(r.json().get("events"), list)


def test_events_verify_chain_admin():
    r = client.get("/api/audit/verify-chain", headers=_admin_headers())
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "chain_integrity" in r.json()


def test_events_entity_job(job_ctx):
    job_id = job_ctx["job_id"]
    r = client.get(
        f"/api/events/job/{job_id}",
        headers=job_ctx["headers"],
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert r.json().get("entity_id") == job_id
    assert isinstance(r.json().get("events"), list)


def test_events_lease_missing_or_present(job_ctx):
    job_id = job_ctx["job_id"]
    r = client.get(
        f"/api/events/leases/{job_id}",
        headers=job_ctx["headers"],
    )
    assert r.status_code in (200, 404), r.text[:300]
    if r.status_code == 200:
        assert r.json().get("ok") is True
        assert "lease" in r.json()


def test_events_instance_audit_trail(job_ctx):
    job_id = job_ctx["job_id"]
    r = client.get(
        f"/api/audit/instance/{job_id}",
        headers=job_ctx["headers"],
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert r.json().get("job_id") == job_id
    assert r.json().get("count") >= 1