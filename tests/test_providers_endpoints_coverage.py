"""Smoke coverage for routes/providers.py (UNTESTED_ENDPOINTS.md)."""

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

OK_OR_HANDLED = {200, 400, 401, 403, 404, 422, 502, 503}


@pytest.fixture(scope="module")
def provider_user():
    email = f"provcov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Prov Cov"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    provider_id = f"prov-{uuid.uuid4().hex[:8]}"
    reg_resp = client.post(
        "/api/providers/register",
        json={
            "provider_id": provider_id,
            "email": email,
            "provider_type": "individual",
            "legal_name": "Coverage Provider",
            "province": "ON",
        },
        headers=headers,
    )
    if reg_resp.status_code != 200:
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
                    "Coverage Provider",
                    "ON",
                    now,
                ),
            )
            conn.commit()
        UserStore.update_user(email, {"provider_id": provider_id, "role": "provider"})

    # Session JWT may predate provider_id on the user row — refresh token.
    login2 = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    headers = {"Authorization": f"Bearer {login2.json()['access_token']}"}
    return {
        "email": email,
        "provider_id": provider_id,
        "headers": headers,
    }


def test_providers_list(provider_user):
    from db import UserStore

    r = client.get("/api/providers", headers=provider_user["headers"])
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert "count" in r.json()

    db_user = UserStore.get_user(provider_user["email"]) or {}
    if db_user.get("provider_id"):
        assert r.json()["count"] >= 1, (
            f"user row has provider_id={db_user.get('provider_id')!r} but list returned empty"
        )


def test_providers_get(provider_user):
    pid = provider_user["provider_id"]
    r = client.get(f"/api/providers/{pid}", headers=provider_user["headers"])
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert r.json()["provider"]["provider_id"] == pid


def test_providers_earnings(provider_user):
    pid = provider_user["provider_id"]
    r = client.get(
        f"/api/providers/{pid}/earnings",
        headers=provider_user["headers"],
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_providers_abandon_onboarding(provider_user):
    pid = provider_user["provider_id"]
    r = client.post(
        f"/api/providers/{pid}/abandon-onboarding",
        headers=provider_user["headers"],
    )
    assert r.status_code in OK_OR_HANDLED
    assert r.status_code != 500


def test_providers_resume_onboarding(provider_user):
    pid = provider_user["provider_id"]
    r = client.post(
        f"/api/providers/{pid}/resume-onboarding",
        headers=provider_user["headers"],
    )
    assert r.status_code in OK_OR_HANDLED
    assert r.status_code != 500


def test_providers_incorporation_missing_file(provider_user):
    pid = provider_user["provider_id"]
    r = client.post(
        f"/api/providers/{pid}/incorporation",
        headers=provider_user["headers"],
        json={"file_id": "nonexistent-file-id"},
    )
    assert r.status_code in OK_OR_HANDLED
    assert r.status_code != 500


def test_providers_payout_validation(provider_user):
    pid = provider_user["provider_id"]
    r = client.post(
        f"/api/providers/{pid}/payout",
        headers=provider_user["headers"],
    )
    assert r.status_code == 400


def test_providers_paypal_status(provider_user):
    pid = provider_user["provider_id"]
    r = client.get(
        f"/api/providers/{pid}/paypal",
        headers=provider_user["headers"],
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    assert "paypal_enabled" in body
    assert body["paypal"]["provider_id"] == pid


def test_providers_paypal_onboard_unconfigured(provider_user, monkeypatch):
    import paypal_connect
    monkeypatch.setattr(paypal_connect, "PAYPAL_ENABLED", False)
    pid = provider_user["provider_id"]
    r = client.post(
        f"/api/providers/{pid}/paypal/onboard",
        headers=provider_user["headers"],
    )
    assert r.status_code in (400, 502)
    assert r.status_code != 500



def test_providers_paypal_refresh(provider_user):
    pid = provider_user["provider_id"]
    r = client.post(
        f"/api/providers/{pid}/paypal/refresh",
        headers=provider_user["headers"],
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    assert body["paypal"]["provider_id"] == pid


def test_providers_payout_paypal_rail_unconfigured(provider_user):
    pid = provider_user["provider_id"]
    r = client.post(
        f"/api/providers/{pid}/payout",
        params={"job_id": "job-paypal-cov", "total_cad": 50.0, "payment_rail": "paypal"},
        headers=provider_user["headers"],
    )
    assert r.status_code in (400, 502)
    assert r.status_code != 500


def test_providers_webhook_no_signature():
    r = client.post(
        "/api/providers/webhook",
        content=b"{}",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code in OK_OR_HANDLED
    assert r.status_code != 500


def test_providers_register_idempotent_shape(provider_user):
    """Register path is exercised at fixture setup; assert route still handles repeat."""
    email = provider_user["email"]
    pid = f"prov-{uuid.uuid4().hex[:8]}"
    r = client.post(
        "/api/providers/register",
        json={
            "provider_id": pid,
            "email": email,
            "provider_type": "individual",
            "legal_name": "Second",
            "province": "ON",
        },
        headers=provider_user["headers"],
    )
    assert r.status_code in OK_OR_HANDLED
    assert r.status_code != 500