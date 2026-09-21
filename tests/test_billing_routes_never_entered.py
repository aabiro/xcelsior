"""Five billing handlers the suite never reached.

`scripts/measure_route_execution.py` found these never entered, all scored
"covered" by `UNTESTED_ENDPOINTS.md` because other tests mention the
`/api/billing/` and `/api/v2/billing/` prefixes:

    GET  /api/billing/crypto/enabled
    GET  /api/v2/billing/pending-verification
    POST /api/billing/portal-session
    POST /api/v2/billing/auto-topup-plans/{plan_id}/execute
    POST /api/v2/billing/pending-verification/{stripe_intent_id}/resume

Three of them are on the SCA recovery path — the one that tells a customer a
top-up stopped because their bank is waiting for them. Its own docstring
explains that before it existed "a charge stopped dead pending verification
looked exactly like one in flight". A surface built to make a silent failure
visible is a poor thing to leave unexercised.

Stripe is not configured in the test environment, so the routes that need it
answer 503. That is still the handler running: the assertions below pin *which*
refusal, and that none of them is a 500.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")

from api import app

client = TestClient(app)


@pytest.fixture
def auth():
    email = f"billing-routes-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Billing Routes"},
    )
    token = r.json().get("access_token")
    assert token, f"could not register: {r.status_code} {r.text[:200]}"
    return {"Authorization": f"Bearer {token}"}


# ── Crypto availability ───────────────────────────────────────────────────


def test_crypto_enabled_always_answers() -> None:
    """A feature probe that errors is worse than one that says "off".

    The dashboard calls this to decide whether to render the Bitcoin option,
    so a 500 here is a broken billing page rather than a hidden button.
    """
    r = client.get("/api/billing/crypto/enabled")
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["ok"] is True
    assert "enabled" in body, f"no verdict in the response: {body}"


# ── SCA recovery ──────────────────────────────────────────────────────────


def test_pending_verification_lists_nothing_for_a_fresh_wallet(auth) -> None:
    r = client.get("/api/v2/billing/pending-verification", headers=auth)
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["ok"] is True
    assert body["pending"] == []
    assert body["count"] == 0
    assert "waiting" in body["message"].lower()


def test_pending_verification_never_returns_a_client_secret(auth) -> None:
    """The docstring commits to this, and it is the kind of promise that rots.

    A `client_secret` confirms a payment. The listing exists to *say* what is
    waiting; handing out confirmation credentials from a list endpoint is how
    one leaks into a log or a support screenshot.
    """
    r = client.get("/api/v2/billing/pending-verification", headers=auth)
    assert "client_secret" not in r.text, f"a confirmation credential leaked: {r.text[:300]}"


def test_pending_verification_requires_authentication() -> None:
    r = client.get("/api/v2/billing/pending-verification")
    assert r.status_code in (401, 403), r.text[:200]


def test_pending_verification_refuses_another_customers_scope(auth) -> None:
    """`customer_id` is caller-supplied, so the access check is the only thing
    standing between it and someone else's pending charges."""
    r = client.get(
        "/api/v2/billing/pending-verification",
        params={"customer_id": f"cust-{uuid.uuid4().hex[:12]}"},
        headers=auth,
    )
    assert r.status_code in (403, 404), (
        f"reading another customer's pending charges returned {r.status_code}: "
        f"{r.text[:200]}"
    )


def test_resuming_an_unknown_intent_is_not_a_fault(auth) -> None:
    r = client.post(
        f"/api/v2/billing/pending-verification/pi_{uuid.uuid4().hex[:20]}/resume",
        headers=auth,
    )
    assert r.status_code < 500, f"an unknown intent faulted: {r.status_code} {r.text[:300]}"
    assert r.status_code in (400, 402, 403, 404, 409, 503), r.text[:200]


def test_resuming_requires_authentication() -> None:
    r = client.post(f"/api/v2/billing/pending-verification/pi_{uuid.uuid4().hex[:20]}/resume")
    assert r.status_code in (401, 403), r.text[:200]


# ── Stripe customer portal ────────────────────────────────────────────────


def test_portal_session_without_stripe_is_a_503_not_a_500(auth) -> None:
    """The handler catches both RuntimeError and everything else as 503.

    A misconfigured processor is unavailability, not a server fault — the
    distinction decides whether the client retries or reports a bug.
    """
    r = client.post("/api/billing/portal-session", headers=auth)
    assert r.status_code != 500, f"portal session faulted: {r.text[:300]}"
    assert r.status_code in (200, 503), r.text[:300]


def test_portal_session_requires_authentication() -> None:
    r = client.post("/api/billing/portal-session")
    assert r.status_code in (401, 403), r.text[:200]


# ── Auto-top-up plan execution ────────────────────────────────────────────


def test_executing_an_unknown_plan_is_not_found(auth) -> None:
    r = client.post(
        f"/api/v2/billing/auto-topup-plans/{uuid.uuid4()}/execute", headers=auth
    )
    assert r.status_code < 500, f"an unknown plan faulted: {r.status_code} {r.text[:300]}"
    assert r.status_code in (400, 403, 404, 409), r.text[:200]


def test_executing_a_malformed_plan_id_is_a_client_error(auth) -> None:
    """`plan_id` reaches a `uuid` column; an unparseable one used to be a 500.

    Covered by the `psycopg.DataError` floor in `api.py`; asserted here because
    this is one of the routes that reaches it.
    """
    r = client.post("/api/v2/billing/auto-topup-plans/not-a-uuid/execute", headers=auth)
    assert r.status_code < 500, f"a malformed plan id faulted: {r.status_code} {r.text[:300]}"
    assert 400 <= r.status_code < 500, r.text[:200]


def test_executing_a_plan_requires_authentication() -> None:
    r = client.post(f"/api/v2/billing/auto-topup-plans/{uuid.uuid4()}/execute")
    assert r.status_code in (401, 403), r.text[:200]
