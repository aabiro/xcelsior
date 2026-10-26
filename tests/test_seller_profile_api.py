"""Collect seller identity before a new payout account, and never return the tax id.

The SIN is generated so it passes the Luhn check. It is not a person's number.
The business number is the nine digits 123456789, format only.
"""

from __future__ import annotations

import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest
from fastapi.testclient import TestClient

from api import app
from seller_identity import SellerProfileRequired, luhn_ok

client = TestClient(app)
pytestmark = pytest.mark.needs_db

PASSWORD = "StrongPass123!"


def _sin(prefix: str = "13069254") -> str:
    for digit in "0123456789":
        candidate = prefix + digit
        if luhn_ok(candidate):
            return candidate
    raise AssertionError(f"no Luhn check digit for {prefix}")


def _auth() -> tuple[dict, dict]:
    email = f"seller-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    registered = client.post(
        "/api/auth/register",
        json={"email": email, "password": PASSWORD, "name": "Seller"},
    )
    assert registered.status_code in (200, 201), registered.text[:300]
    login = client.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert login.status_code == 200, login.text[:300]
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    me = client.get("/api/auth/me", headers=headers)
    assert me.status_code == 200, me.text[:300]
    user = me.json()["user"]
    assert user["email"] == email
    return user, headers


def _profile(tax_id: str, **overrides: object) -> dict:
    body = {
        "legal_name": "Test Supplier",
        "address_line1": "1 Test Street",
        "city": "London",
        "province": "ON",
        "postal_code": "N6A 1A1",
        "country": "CA",
        "tax_id_kind": "sin",
        "tax_id": tax_id,
        "gst_hst_number": "",
        "accepted_terms": True,
    }
    body.update(overrides)
    return body


def _delete_provider(provider_id: str) -> None:
    from stripe_connect import get_stripe_manager

    with get_stripe_manager()._conn() as conn:
        conn.execute("DELETE FROM provider_accounts WHERE provider_id = %s", (provider_id,))


def test_checkout_business_details_are_optional_and_normalized():
    _user, headers = _auth()
    rejected = client.patch(
        "/api/auth/me",
        json={"gst_hst_number": "not-a-number"},
        headers=headers,
    )
    assert rejected.status_code == 400

    saved = client.patch(
        "/api/auth/me",
        json={"business_name": "  North   Labs ", "gst_hst_number": "123456789 RT 0001"},
        headers=headers,
    )
    assert saved.status_code == 200, saved.text[:300]
    me = client.get("/api/auth/me", headers=headers).json()["user"]
    assert me["business_name"] == "North Labs"
    assert me["gst_hst_number"] == "123456789RT0001"

    cleared = client.patch(
        "/api/auth/me",
        json={"business_name": "", "gst_hst_number": ""},
        headers=headers,
    )
    assert cleared.status_code == 200, cleared.text[:300]
    me = client.get("/api/auth/me", headers=headers).json()["user"]
    assert me["business_name"] == ""
    assert me["gst_hst_number"] == ""


def test_seller_profile_stores_a_sin_encrypted_and_omits_it_from_the_api():
    from security import decrypt_secret
    from stripe_connect import get_stripe_manager

    user, headers = _auth()
    sin = _sin()
    refused = client.put(
        "/api/providers/seller-profile",
        json=_profile(sin, accepted_terms=False),
        headers=headers,
    )
    assert refused.status_code == 400
    assert "independent supplier" in refused.text

    saved = client.put(
        "/api/providers/seller-profile",
        json=_profile(sin, gst_hst_number="123456789RT0001"),
        headers=headers,
    )
    assert saved.status_code == 200, saved.text[:400]
    body = saved.json()
    provider_id = body["provider_id"]
    try:
        profile = body["profile"]
        raw = saved.text
        assert sin not in raw
        assert "tax_id_ciphertext" not in profile
        assert profile.get("stripe_account_id") is None
        assert profile["tax_id_last4"] == sin[-4:]
        assert profile["seller_profile_complete"] is True
        assert profile["postal_code"] == "N6A1A1"
        assert profile["gst_hst_number"] == "123456789RT0001"

        with get_stripe_manager()._conn() as conn:
            row = conn.execute(
                """SELECT tax_id_ciphertext, tax_id_last4, business_number, tax_id_kind
                   FROM provider_accounts WHERE provider_id = %s""",
                (provider_id,),
            ).fetchone()
        assert row is not None
        ciphertext = row["tax_id_ciphertext"]
        assert ciphertext and ciphertext != sin and sin not in ciphertext
        assert decrypt_secret(ciphertext) == sin
        assert row["tax_id_last4"] == sin[-4:]
        assert row["business_number"] in ("", None)
        assert row["tax_id_kind"] == "sin"

        fetched = client.get("/api/providers/seller-profile", headers=headers)
        assert fetched.status_code == 200, fetched.text[:300]
        assert sin not in fetched.text
        assert fetched.json()["profile"]["seller_profile_complete"] is True

        public = client.get(f"/api/providers/{provider_id}", headers=headers)
        assert public.status_code == 200, public.text[:300]
        assert sin not in public.text
        assert "tax_id_ciphertext" not in public.json()["provider"]

        kept = client.put(
            "/api/providers/seller-profile",
            json=_profile(""),
            headers=headers,
        )
        assert kept.status_code == 200, kept.text[:300]
        assert kept.json()["profile"]["tax_id_last4"] == sin[-4:]

        changed = client.put(
            "/api/providers/seller-profile",
            json=_profile("", tax_id_kind="bn"),
            headers=headers,
        )
        assert changed.status_code == 400
        assert "change its type" in changed.text

        me = client.get("/api/auth/me", headers=headers).json()["user"]
        assert me["provider_id"] == provider_id
        assert me["role"] != "provider"
        assert user["email"]
    finally:
        _delete_provider(provider_id)


def test_register_without_a_seller_profile_does_not_open_a_payout_account():
    user, headers = _auth()
    provider_id = user.get("customer_id") or user["user_id"]
    registered = client.post(
        "/api/providers/register",
        json={
            "provider_id": "prov-not-used",
            "email": user["email"],
            "legal_name": "Test Supplier",
            "province": "ON",
        },
        headers=headers,
    )
    assert registered.status_code == 400, registered.text[:400]
    assert "tax identifier" in registered.text
    assert "acct_" not in registered.text

    from stripe_connect import get_stripe_manager

    with get_stripe_manager()._conn() as conn:
        row = conn.execute(
            "SELECT stripe_account_id FROM provider_accounts WHERE provider_id = %s",
            (provider_id,),
        ).fetchone()
    assert row is None


def test_paypal_onboarding_waits_for_the_seller_profile():
    user, headers = _auth()
    provider_id = user.get("customer_id") or user["user_id"]
    from stripe_connect import get_stripe_manager

    with get_stripe_manager()._conn() as conn:
        conn.execute(
            """INSERT INTO provider_accounts
               (provider_id, email, status, created_at)
               VALUES (%s, %s, 'pending', %s)""",
            (provider_id, user["email"], 1.0),
        )
    try:
        onboard = client.post(f"/api/providers/{provider_id}/paypal/onboard", headers=headers)
        assert onboard.status_code == 400, onboard.text[:400]
        assert "tax identifier" in onboard.text
    finally:
        _delete_provider(provider_id)


def test_profile_gate_runs_before_stripe_and_skips_an_existing_account(monkeypatch):
    import stripe_connect

    monkeypatch.setattr(stripe_connect, "STRIPE_ENABLED", False)
    monkeypatch.setattr(stripe_connect, "stripe", None)
    manager = stripe_connect.get_stripe_manager()
    missing = f"prov-{uuid.uuid4().hex[:12]}"
    with pytest.raises(SellerProfileRequired, match="tax identifier"):
        manager.create_provider_account(provider_id=missing, email="gate@xcelsior.ca")

    existing = f"prov-{uuid.uuid4().hex[:12]}"
    with manager._conn() as conn:
        conn.execute(
            """INSERT INTO provider_accounts
               (provider_id, email, stripe_account_id, status, created_at)
               VALUES (%s, %s, 'acct_existing', 'active', %s)""",
            (existing, "gate@xcelsior.ca", 1.0),
        )
    try:
        with pytest.raises(RuntimeError, match="Stripe Connect is not configured"):
            manager.create_provider_account(provider_id=existing, email="gate@xcelsior.ca")
    finally:
        _delete_provider(existing)
