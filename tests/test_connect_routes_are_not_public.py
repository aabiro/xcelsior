"""The Stripe Connect routes had no authentication at all.

`routes/stripe_connect_v2.py` is headed "Sample Integration", but it is mounted
in the live app and every call it makes uses the platform's real Stripe key.
Seven API routes carried no guard.

The clearest harm needed no Stripe key to exploit: `GET /api/connect/accounts`
reads `display_name, contact_email, stripe_account_id` for *every* connected
account out of Postgres and returns it, so it disclosed the whole list to any
anonymous caller whether or not Stripe was configured. Alongside it, anyone
could create connected accounts and platform products on our Stripe, read an
account's KYC requirements, and mint an onboarding link that completes
onboarding for an account they merely name.

Account and product *management* is a platform-operator action and is now
admin-gated. Two routes stay public deliberately — a storefront has to list
products and start a checkout for a buyer who is not signed in — and the
webhook stays open because a signature is the only thing Stripe can present.
That split is asserted here so it stays a decision rather than an accident.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")

from api import app

client = TestClient(app)

ADMIN_ONLY = [
    ("GET", "/api/connect/accounts", None),
    ("POST", "/api/connect/accounts", {"display_name": "Probe", "contact_email": "p@x.ca", "country": "CA"}),
    ("GET", "/api/connect/accounts/acct_probe/onboarding-link", None),
    ("GET", "/api/connect/accounts/acct_probe/status", None),
    ("POST", "/api/connect/products", {"name": "Probe", "price_cents": 100, "currency": "cad", "account_id": "acct_probe"}),
]

PUBLIC_BY_DESIGN = [
    ("GET", "/api/connect/products", None),
    ("POST", "/api/connect/webhooks", {}),
]


@pytest.fixture
def signed_in_non_admin():
    email = f"connect-probe-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Connect Probe"},
    )
    token = r.json().get("access_token")
    assert token, f"could not register: {r.status_code} {r.text[:200]}"
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize("method, path, body", ADMIN_ONLY, ids=[f"{m} {p}" for m, p, _ in ADMIN_ONLY])
def test_a_non_admin_cannot_manage_connect(signed_in_non_admin, method, path, body) -> None:
    kw = {"headers": signed_in_non_admin}
    if body is not None:
        kw["json"] = body
    r = client.request(method, path, **kw)
    assert r.status_code in (401, 403), (
        f"{method} {path} answered {r.status_code} to a signed-in non-admin; these "
        "create accounts and products on the platform's own Stripe"
    )


def test_the_account_listing_does_not_leak_without_a_stripe_key(signed_in_non_admin) -> None:
    """This one reads Postgres directly, so a missing Stripe key is no defence."""
    r = client.get("/api/connect/accounts", headers=signed_in_non_admin)
    assert r.status_code in (401, 403), r.text[:200]
    assert "contact_email" not in r.text, f"the listing leaked anyway: {r.text[:300]}"
    assert "stripe_account_id" not in r.text


@pytest.mark.parametrize("method, path, body", PUBLIC_BY_DESIGN, ids=[f"{m} {p}" for m, p, _ in PUBLIC_BY_DESIGN])
def test_the_storefront_and_webhook_stay_reachable(method, path, body) -> None:
    """Gating these would break a buyer who is not signed in, and break Stripe.

    Asserted so the split above reads as a decision. `< 500` rather than a
    specific code: without a configured Stripe key these answer 503, and an
    unsigned webhook is refused — both are the handler deciding, not a gate.
    """
    kw = {}
    if body is not None:
        kw["json"] = body
    r = client.request(method, path, **kw)
    assert r.status_code not in (401, 403), (
        f"{method} {path} now requires authentication; a storefront buyer and "
        "Stripe's own webhook both arrive without a session"
    )
