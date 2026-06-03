"""Coverage for previously-untested billing endpoints (see UNTESTED_ENDPOINTS.md).

These 21 routes had no test reference. This suite smoke-tests each one through
the real app + Postgres: the clean ones assert 200 + shape; the optional
subsystems (crypto/Lightning, disabled in test) must degrade gracefully (503,
never an unhandled 500); input validation must return 4xx, never 500.

Regression: DELETE /api/billing/payment-methods/{id} previously let Stripe's
InvalidRequestError propagate (→500) AND detached by id without an ownership
check (IDOR). It must now return a clean 404 for an unknown/foreign id.
"""

import os

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"
os.environ["XCELSIOR_AUTH_RATE_LIMIT_REQUESTS"] = "5000"

import uuid

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

# Statuses that mean "endpoint is reachable and handled the request" — i.e. not
# a crash. 5xx other than a deliberate 503 (feature disabled) is a failure.
OK_OR_HANDLED = {200, 400, 401, 402, 404, 409, 422, 503}


@pytest.fixture(scope="module")
def auth():
    email = f"billcov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Bill Cov"},
    )
    r = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    token = r.json()["access_token"]
    return email, {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def two_users():
    """Two distinct accounts for cross-customer access-control tests."""
    users = []
    for label in ("a", "b"):
        email = f"billcov-{label}-{uuid.uuid4().hex[:10]}@xcelsior.ca"
        reg = client.post(
            "/api/auth/register",
            json={"email": email, "password": "StrongPass123!", "name": f"Bill {label}"},
        ).json()
        login = client.post(
            "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
        ).json()
        token = login["access_token"]
        customer_id = reg["user"]["customer_id"]
        users.append(
            {
                "email": email,
                "customer_id": customer_id,
                "headers": {"Authorization": f"Bearer {token}"},
            }
        )
    return users[0], users[1]


# ── Public read endpoints (no auth) — must be 200 with a sane body ──────


def test_pricing_models():
    r = client.get("/api/pricing/models")
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_pricing_rates():
    r = client.get("/api/pricing/rates")
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_pricing_reservations():
    r = client.get("/api/pricing/reservations")
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_lightning_enabled_flag():
    r = client.get("/api/billing/lightning/enabled")
    assert r.status_code == 200
    assert "enabled" in r.json()


def test_paypal_enabled_flag():
    r = client.get("/api/billing/paypal/enabled")
    assert r.status_code == 200
    assert "enabled" in r.json()


def test_lightning_rate():
    # Lightning may be disabled in test → graceful 503, never a 500.
    r = client.get("/api/billing/lightning/rate")
    assert r.status_code in (200, 503)


# ── Optional crypto/Lightning subsystems: graceful when disabled ───────


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("GET", "/api/billing/crypto/rate", None),
        ("GET", "/api/billing/crypto/deposit/nonexistent-dep", None),
        ("GET", "/api/billing/lightning/deposit/nonexistent-dep", None),
        ("POST", "/api/billing/crypto/deposit", {"amount_cad": 10}),
        ("POST", "/api/billing/crypto/refresh/nonexistent-dep", {}),
        ("POST", "/api/billing/lightning/deposit", {"amount_cad": 10}),
    ],
)
def test_optional_payment_subsystems_degrade_gracefully(auth, method, path, body):
    _, headers = auth
    r = client.request(method, path, json=body, headers=headers)
    assert r.status_code in OK_OR_HANDLED, f"{method} {path} -> {r.status_code}: {r.text[:200]}"
    assert r.status_code < 500 or r.status_code == 503


# ── Authenticated billing endpoints ────────────────────────────────────


def test_list_payment_methods(auth):
    _, headers = auth
    r = client.get("/api/billing/payment-methods", headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json().get("payment_methods", r.json().get("methods", [])), list)


def test_auto_topup_get(auth):
    _, headers = auth
    r = client.get("/api/v2/billing/auto-topup", headers=headers)
    assert r.status_code == 200


def test_auto_topup_configure(auth):
    _, headers = auth
    r = client.post(
        "/api/v2/billing/auto-topup",
        json={"enabled": True, "threshold_cad": 5, "topup_cad": 20},
        headers=headers,
    )
    assert r.status_code in (200, 400, 422)


def test_usage_summary(auth):
    email, headers = auth
    r = client.get(f"/api/billing/usage/{email}", headers=headers)
    assert r.status_code == 200


def test_generate_invoice(auth):
    email, headers = auth
    r = client.get(f"/api/billing/invoice/{email}", headers=headers)
    assert r.status_code == 200


def test_download_invoice(auth):
    email, headers = auth
    r = client.get(f"/api/billing/invoice/{email}/download", headers=headers)
    assert r.status_code in (200, 404)


def test_setup_intent(auth):
    _, headers = auth
    r = client.post("/api/billing/setup-intent", json={}, headers=headers)
    assert r.status_code in (200, 402, 503)


def test_payment_intent(auth):
    _, headers = auth
    r = client.post("/api/billing/payment-intent", json={"amount_cad": 10}, headers=headers)
    # Reachable + handled; must not 500 (422 = validation, 402/503 = stripe state).
    assert r.status_code in OK_OR_HANDLED


# ── Regression: detach must verify ownership + handle Stripe errors ────


def test_detach_unknown_payment_method_returns_404_not_500(auth):
    """Unknown/foreign payment method id must yield a clean 404, not a 500.

    Previously the Stripe InvalidRequestError propagated (500) and there was no
    ownership check (a user could detach another customer's card by id).
    """
    _, headers = auth
    r = client.delete("/api/billing/payment-methods/pm_nonexistent_xyz", headers=headers)
    assert r.status_code == 404, f"expected 404, got {r.status_code}: {r.text[:200]}"


def test_detach_requires_auth():
    # Fresh client so no session cookie leaks in from the module-scoped `client`.
    fresh = TestClient(app)
    r = fresh.delete("/api/billing/payment-methods/pm_whatever")
    assert r.status_code == 401


# ── Regression: customer-scoped billing routes require auth + ownership ─


_CUSTOMER_SCOPED_GETS = [
    "/api/billing/wallet/{cid}",
    "/api/billing/wallet/{cid}/history",
    "/api/billing/wallet/{cid}/depletion",
    "/api/billing/usage/{cid}",
    "/api/billing/invoice/{cid}",
    "/api/billing/export/caf/{cid}",
    "/api/billing/invoices/{cid}",
    "/api/billing/invoice/{cid}/download",
]


@pytest.mark.parametrize("path_tpl", _CUSTOMER_SCOPED_GETS)
def test_customer_billing_get_requires_auth(path_tpl):
    fresh = TestClient(app)
    r = fresh.get(path_tpl.format(cid="victim@example.com"))
    assert r.status_code == 401


@pytest.mark.parametrize("path_tpl", _CUSTOMER_SCOPED_GETS)
def test_customer_billing_get_forbidden_cross_account(two_users, path_tpl):
    user_a, user_b = two_users
    r = client.get(path_tpl.format(cid=user_b["customer_id"]), headers=user_a["headers"])
    assert r.status_code == 403


def test_wallet_deposit_requires_auth():
    fresh = TestClient(app)
    r = fresh.post(
        "/api/billing/wallet/victim@example.com/deposit",
        json={"amount_cad": 10.0, "description": "probe"},
    )
    assert r.status_code == 401


def test_wallet_deposit_forbidden_cross_account(two_users):
    user_a, user_b = two_users
    r = client.post(
        f"/api/billing/wallet/{user_b['customer_id']}/deposit",
        json={"amount_cad": 10.0, "description": "probe"},
        headers=user_a["headers"],
    )
    assert r.status_code == 403
