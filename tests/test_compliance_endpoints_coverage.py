"""Smoke coverage for routes/compliance.py (UNTESTED_ENDPOINTS.md)."""

import os
import time
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


@pytest.fixture(scope="module")
def provider_ctx():
    email = f"compliance-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Compliance"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    provider_id = f"prov-{uuid.uuid4().hex[:8]}"
    reg = client.post(
        "/api/providers/register",
        json={
            "provider_id": provider_id,
            "email": email,
            "provider_type": "individual",
            "legal_name": "Compliance Prov",
            "province": "ON",
        },
        headers=headers,
    )
    if reg.status_code != 200:
        from db import UserStore
        from stripe_connect import get_stripe_manager

        now = time.time()
        mgr = get_stripe_manager()
        with mgr._conn() as conn:
            conn.execute(
                """INSERT INTO provider_accounts
                   (provider_id, provider_type, stripe_account_id, status,
                    corporation_name, business_number, gst_hst_number,
                    email, legal_name, country, province, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'CA', %s, %s)
                   ON CONFLICT (provider_id) DO NOTHING""",
                (
                    provider_id,
                    "individual",
                    "",
                    "onboarding",
                    "",
                    "",
                    "",
                    email,
                    "Compliance Prov",
                    "ON",
                    now,
                ),
            )
            conn.commit()
        UserStore.update_user(email, {"provider_id": provider_id, "role": "provider"})
    login2 = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    headers = {"Authorization": f"Bearer {login2.json()['access_token']}"}
    return {"provider_id": provider_id, "headers": headers}


def test_compliance_provinces():
    r = client.get("/api/compliance/provinces")
    assert r.status_code == 200
    assert "provinces" in r.json()
    assert "ON" in r.json()["provinces"]


def test_compliance_detect_province():
    r = client.get(
        "/api/compliance/detect-province",
        headers={"CF-IPCountry": "CA", "CF-Region": "ON"},
    )
    assert r.status_code == 200
    assert r.json().get("province") == "ON"


def test_compliance_tax_rates():
    r = client.get("/api/compliance/tax-rates")
    assert r.status_code == 200
    assert "rates" in r.json()
    assert "ON" in r.json()["rates"]


def test_compliance_trust_tier_requirements():
    r = client.get("/api/compliance/trust-tier-requirements")
    assert r.status_code == 200
    assert "tiers" in r.json()
    assert len(r.json()["tiers"]) >= 1


def test_compliance_quebec_pia_check():
    r = client.post(
        "/api/compliance/quebec-pia-check",
        json={
            "data_origin_province": "QC",
            "processing_province": "ON",
            "data_contains_pi": True,
        },
    )
    assert r.status_code == 200
    assert "pia_required" in r.json()


def test_provider_gst_threshold(provider_ctx):
    pid = provider_ctx["provider_id"]
    r = client.get(
        f"/api/billing/gst-threshold/{pid}",
        headers=provider_ctx["headers"],
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert r.json().get("provider_id") == pid


def test_provider_gst_threshold_forbidden():
    email = f"other-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Other"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    r = client.get(
        "/api/billing/gst-threshold/prov-not-owned",
        headers=headers,
    )
    assert r.status_code in (403, 404)