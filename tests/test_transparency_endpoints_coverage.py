"""Smoke coverage for routes/transparency.py (UNTESTED_ENDPOINTS.md)."""

import os

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_transparency_legal_request_and_respond():
    r = client.post(
        "/api/transparency/legal-request",
        json={
            "request_type": "subpoena",
            "jurisdiction": "CA",
            "authority": "Test Court",
            "scope": "coverage smoke",
            "notes": "pytest",
        },
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    request_id = r.json().get("request_id")
    assert request_id

    r2 = client.post(
        f"/api/transparency/legal-request/{request_id}/respond",
        params={"complied": False, "challenged": True, "notes": "challenged in test"},
    )
    assert r2.status_code == 200
    assert r2.json().get("ok") is True
    assert r2.json().get("request_id") == request_id


def test_transparency_report():
    r = client.get("/api/transparency/report?months=3")
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "summary" in r.json()
    assert r.json().get("period_months") == 3