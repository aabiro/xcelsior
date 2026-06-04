"""Smoke coverage for remaining jurisdiction queue routes."""

from __future__ import annotations

import os

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _admin_headers() -> dict[str, str]:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


def test_jurisdiction_hosts_filter():
    r = client.post(
        "/api/jurisdiction/hosts",
        json={"canada_only": True, "province": None, "trust_tier": None},
    )
    assert r.status_code != 500
    if r.status_code == 200:
        data = r.json()
        assert data.get("ok") is True
        assert "hosts" in data


def test_process_queue_ca():
    r = client.post("/queue/process/ca", headers=_admin_headers())
    assert r.status_code != 500
    if r.status_code == 200:
        assert "assigned" in r.json()


def test_process_queue_sovereign():
    r = client.post(
        "/api/queue/process-sovereign",
        json={"canada_only": True, "province": None, "trust_tier": None},
        headers=_admin_headers(),
    )
    assert r.status_code != 500
    if r.status_code == 200:
        data = r.json()
        assert data.get("ok") is True
        assert "jobs" in data