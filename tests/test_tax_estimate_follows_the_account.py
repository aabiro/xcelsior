"""The quoted tax rate must come from the same place Stripe will charge from.

Stripe Tax is the merchant-of-record calculation for real money and it reads the
**account** country/province. `api_create_payment_intent` states this outright:

    Stripe Tax uses the **account** country/province (settings/profile) — no
    deposit province picker. IP is a fallback when the account has no location.

`/api/pricing/rates` and `/api/pricing/spot-quote` are the estimates the launch
UI shows, and they took `province` as a query parameter defaulting to `"ON"`.
The launch modal sent `province || "ON"`, where that value came from
`detectProvince()` — a call to `/api/compliance/detect-province`, a route that
does not exist. The 404 landed in `.catch(() => setProvince("ON"))`.

So every customer outside Ontario was quoted 13% HST and then charged their own
province's rate. The estimate was not merely approximate, it was systematically
wrong in a way no one could see, because the request that was supposed to set it
had been failing since the endpoint was removed.

Both endpoints now resolve the province the way Stripe does. An explicit query
value still wins — a caller pricing a location other than their own is a real
use — and with neither a value nor an account the rate falls back to
"GST 5% (province unknown)" rather than silently picking a province.
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
def test_the_quote_follows_the_account_province(endpoint, bc_customer):
    r = client.get(f"{endpoint}?gpu_model=RTX%204090", headers=bc_customer)
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["province"] == "BC", (
        f"quoted {body['province']!r} for a BC account — the estimate is not "
        "reading the field Stripe Tax charges from"
    )
    assert body["tax_rate"] == pytest.approx(0.12), body


@pytest.mark.parametrize("endpoint", RATE_ENDPOINTS)
def test_an_explicit_province_still_wins(endpoint, bc_customer):
    """Pricing a location other than your own stays possible."""
    r = client.get(f"{endpoint}?gpu_model=RTX%204090&province=ON", headers=bc_customer)
    assert r.status_code == 200, r.text[:300]
    assert r.json()["tax_rate"] == pytest.approx(0.13)


@pytest.mark.parametrize("endpoint", RATE_ENDPOINTS)
def test_no_account_and_no_value_says_unknown(endpoint):
    """Anonymous callers get an honest fallback, not a guessed province."""
    r = client.get(f"{endpoint}?gpu_model=RTX%204090")
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["province"] == ""
    assert "unknown" in body["tax_description"].lower(), body
    assert body["tax_rate"] == pytest.approx(0.05)
