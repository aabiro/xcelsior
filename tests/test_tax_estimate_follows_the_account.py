"""Usage quotes spend pretax credits; tax is collected on credit deposits.

Neither endpoint can invent an hourly tax charge. The deposit endpoint uses
the validated saved address and Stripe Tax for the actual payment instead.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_PERSISTENT_AUTH", "true")

from api import app

client = TestClient(app)

RATE_ENDPOINTS = ("/api/pricing/rates", "/api/pricing/spot-quote")


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    import api as api_mod
    import routes._deps as deps
    import routes.auth as auth

    monkeypatch.setattr(deps, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(auth, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)


@pytest.fixture
def bc_customer() -> dict:
    """A signed-in customer whose account says British Columbia."""
    email = f"tax-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Tax"},
    )
    login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    assert login.status_code == 200, login.text[:300]
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    r = client.put(
        "/api/auth/me/profile", json={"province": "BC", "country": "CA"}, headers=headers
    )
    assert r.status_code == 200, r.text[:300]
    return headers


@pytest.mark.parametrize("endpoint", RATE_ENDPOINTS)
def test_usage_quote_is_pretax_for_a_signed_in_account(endpoint, bc_customer):
    r = client.get(f"{endpoint}?gpu_model=RTX%204090", headers=bc_customer)
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["tax_collection"] == "credit_deposit"
    assert body["tax_rate"] is None and body["tax_amount"] is None
    assert body["total_with_tax"] is None
    assert body["total_per_hour"] > 0


@pytest.mark.parametrize("endpoint", RATE_ENDPOINTS)
def test_usage_quote_rejects_a_tax_province_override(endpoint, bc_customer):
    r = client.get(f"{endpoint}?gpu_model=RTX%204090&province=ON", headers=bc_customer)
    assert r.status_code == 422, r.text[:300]
    assert "saved billing address" in r.text


@pytest.mark.parametrize("endpoint", RATE_ENDPOINTS)
def test_no_account_does_not_fabricate_a_tax_rate_or_location(endpoint):
    r = client.get(f"{endpoint}?gpu_model=RTX%204090")
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert "province" not in body and "country" not in body
    assert body["tax_collection"] == "credit_deposit"
    assert body["tax_rate"] is None and body["tax_amount"] is None
    assert body["total_with_tax"] is None
    assert "saved billing address" in body["tax_description"]


@pytest.mark.parametrize("endpoint", RATE_ENDPOINTS)
@pytest.mark.parametrize(
    "location", [{"country": "US", "province": "CA"}, {"country": "GB", "province": "ENG"}]
)
def test_international_account_never_receives_canadian_gst(endpoint, location, bc_customer):
    update = client.put("/api/auth/me/profile", headers=bc_customer, json=location)
    assert update.status_code == 200, update.text
    result = client.get(f"{endpoint}?gpu_model=RTX%204090", headers=bc_customer)
    assert result.status_code == 200, result.text
    assert result.json()["tax_rate"] is None
    assert result.json()["tax_collection"] == "credit_deposit"
