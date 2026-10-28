"""Credits reach a wallet only through a verified payment or an admin.

Every route that moves a wallet balance up is covered here from the customer's
side: the one thing a customer must never be able to do is end up with credit
that no payment stands behind.

- ``POST /api/billing/refund`` took the exit code and failure reason from the
  request body and refunded the job's cost on them, to any job owner, as many
  times as they asked. ``exit_code=255`` was a full refund every time.
- ``POST /api/billing/paypal/capture-order`` credited whichever wallet the body
  named, for whatever currency the order was in. PayPal's JS SDK creates orders
  with only the public client id, so the order's currency is the buyer's choice:
  10 000 JPY credited 10 000 CAD.
"""

import os
import time
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import pytest
from fastapi.testclient import TestClient

from api import app
from billing import get_billing_engine

client = TestClient(app)


def _admin_headers() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def customer():
    email = f"nopay-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "No Pay"},
    ).json()
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    ).json()
    return {
        "email": email,
        "customer_id": reg["user"]["customer_id"],
        "headers": {"Authorization": f"Bearer {login['access_token']}"},
    }


def _balance(cid: str) -> float:
    return get_billing_engine().get_wallet(cid)["balance_cad"]


def _metered_job(owner: str, hours: float = 1.0, rate: float = 4.0, job_id: str = "") -> str:
    """A finished job owned by ``owner`` that cost ``hours * rate`` CAD."""
    job_id = job_id or f"j-nopay-{uuid.uuid4().hex[:10]}"
    now = time.time()
    get_billing_engine().meter_job(
        {
            "job_id": job_id,
            "owner": owner,
            "started_at": now - hours * 3600,
            "completed_at": now,
        },
        {"host_id": f"h-{job_id}", "gpu_model": "A100", "cost_per_hour": rate},
    )
    return job_id


def _refund(job_id: str, headers: dict, exit_code: int = 255, reason: str = "hardware"):
    return client.post(
        "/api/billing/refund",
        json={"job_id": job_id, "exit_code": exit_code, "failure_reason": reason},
        headers=headers,
    )


# ── Refunds ───────────────────────────────────────────────────────────


def test_a_job_owner_cannot_refund_their_own_job(customer):
    cid = customer["customer_id"]
    # Launching needs a balance; the attack is on what comes after.
    get_billing_engine().deposit(cid, 10.0, "Test seed", idempotency_key=f"seed:{cid}")
    job = client.post(
        "/instance",
        json={"name": "refund-me", "vram_needed_gb": 1},
        headers=customer["headers"],
    )
    assert job.status_code == 200, job.text
    job_id = _metered_job(cid, job_id=job.json()["instance"]["job_id"])
    before = _balance(cid)

    for _ in range(3):
        r = _refund(job_id, customer["headers"])
        assert r.status_code == 403, r.text

    assert _balance(cid) == pytest.approx(before)


def test_an_admin_refund_credits_a_job_once(customer):
    cid = customer["customer_id"]
    job_id = _metered_job(cid)
    before = _balance(cid)

    first = _refund(job_id, _admin_headers())
    assert first.status_code == 200, first.text
    refunded = first.json()["refund_amount_cad"]
    assert refunded > 0
    assert _balance(cid) == pytest.approx(before + refunded)

    again = _refund(job_id, _admin_headers())
    assert again.status_code == 200, again.text
    assert again.json()["refund"] is False
    assert again.json()["already_refunded"] is True
    assert _balance(cid) == pytest.approx(before + refunded)


def test_a_second_classification_does_not_refund_the_same_job_again(customer):
    cid = customer["customer_id"]
    job_id = _metered_job(cid)
    be = get_billing_engine()
    before = _balance(cid)

    half = be.process_refund(job_id, exit_code=1, failure_reason="network timeout")
    assert half["refund"] is True
    full = be.process_refund(job_id, exit_code=255, failure_reason="hardware")
    assert full["refund"] is False

    assert _balance(cid) == pytest.approx(before + half["refund_amount_cad"])
    with be._conn() as conn:
        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM wallet_transactions WHERE idempotency_key = %s",
            (f"refund:{job_id}",),
        ).fetchone()
    assert rows["n"] == 1


# ── Direct deposit and admin credit ───────────────────────────────────


def test_direct_deposit_is_refused_outside_dev_and_test(customer, monkeypatch):
    import routes.billing as billing_mod

    monkeypatch.setattr(billing_mod.env_config, "is_relaxed_env", lambda *a, **k: False)
    cid = customer["customer_id"]
    before = _balance(cid)
    r = client.post(
        f"/api/billing/wallet/{cid}/deposit",
        json={"amount_cad": 500.0, "description": "free money"},
        headers=customer["headers"],
    )
    assert r.status_code == 403, r.text
    assert _balance(cid) == pytest.approx(before)


def test_admin_credit_is_refused_to_the_wallet_owner(customer):
    cid = customer["customer_id"]
    before = _balance(cid)
    r = client.post(
        f"/api/billing/wallet/{cid}/admin-credit",
        json={"amount_cad": 50.0, "reason": "please", "idempotency_key": uuid.uuid4().hex},
        headers=customer["headers"],
    )
    assert r.status_code == 403
    assert _balance(cid) == pytest.approx(before)


# ── PayPal ────────────────────────────────────────────────────────────


class _Resp:
    def __init__(self, body: dict, status_code: int = 201):
        self._body = body
        self.status_code = status_code
        self.text = str(body)

    def json(self):
        return self._body


@pytest.fixture()
def paypal(monkeypatch):
    """PayPal configured, with the order and capture answers under test control."""
    import routes.billing as billing_mod

    monkeypatch.setattr(billing_mod, "_PAYPAL_CLIENT_ID", "test-client")
    monkeypatch.setattr(billing_mod, "_PAYPAL_CLIENT_SECRET", "test-secret")
    monkeypatch.setattr(billing_mod, "_paypal_access_token", lambda: "tok")
    state: dict = {"order": None, "capture": None, "captured": [], "credited": []}

    monkeypatch.setattr(
        billing_mod, "_paypal_get_order", lambda order_id, token=None: state["order"]
    )

    def fake_post(url, **_kw):
        state["captured"].append(url)
        return _Resp(state["capture"])

    monkeypatch.setattr(billing_mod.httpx, "post", fake_post)

    def fake_credit(customer_id, order_id, amount_cad, capture_id):
        state["credited"].append((customer_id, order_id, amount_cad, capture_id))
        return {"balance_cad": amount_cad}

    monkeypatch.setattr(billing_mod, "_paypal_credit_capture", fake_credit)
    return state


def _approved(custom_id: str, currency: str = "CAD", value: str = "25.00") -> dict:
    return {
        "id": "ORDER-1",
        "status": "APPROVED",
        "purchase_units": [
            {"custom_id": custom_id, "amount": {"value": value, "currency_code": currency}}
        ],
    }


def _captured(status: str = "COMPLETED", currency: str = "CAD", value: str = "25.00") -> dict:
    return {
        "status": "COMPLETED",
        "purchase_units": [
            {
                "payments": {
                    "captures": [
                        {
                            "id": "CAP-1",
                            "status": status,
                            "amount": {"value": value, "currency_code": currency},
                        }
                    ]
                }
            }
        ],
    }


def _capture(customer: dict, order_id: str = "ORDER-1"):
    return client.post(
        "/api/billing/paypal/capture-order",
        json={"customer_id": customer["customer_id"], "order_id": order_id},
        headers=customer["headers"],
    )


def test_paypal_capture_credits_a_completed_cad_payment(customer, paypal):
    paypal["order"] = _approved(customer["customer_id"])
    paypal["capture"] = _captured()
    r = _capture(customer)
    assert r.status_code == 200, r.text
    assert paypal["credited"] == [(customer["customer_id"], "ORDER-1", 25.0, "CAP-1")]


def test_paypal_capture_refuses_an_order_made_for_another_wallet(customer, paypal):
    paypal["order"] = _approved("cust-someone-else")
    paypal["capture"] = _captured()
    r = _capture(customer)
    assert r.status_code == 403, r.text
    # Refused before capture: the other customer's payment is left untouched.
    assert paypal["captured"] == []
    assert paypal["credited"] == []


def test_paypal_does_not_capture_an_order_in_another_currency(customer, paypal):
    paypal["order"] = _approved(customer["customer_id"], currency="JPY", value="10000")
    paypal["capture"] = _captured(currency="JPY", value="10000")
    r = _capture(customer)
    assert r.status_code == 400, r.text
    # Refused before capture: no credit could follow, so the buyer is not charged.
    assert paypal["captured"] == []
    assert paypal["credited"] == []


def test_paypal_capture_does_not_credit_another_currency_as_cad(customer, paypal):
    paypal["order"] = _approved(customer["customer_id"])
    paypal["capture"] = _captured(currency="JPY", value="10000")
    r = _capture(customer)
    assert r.status_code == 400, r.text
    assert paypal["credited"] == []


def test_paypal_order_approved_webhook_does_not_capture_another_currency(paypal, monkeypatch):
    import routes.billing as billing_mod

    monkeypatch.setattr(billing_mod, "_paypal_verify_webhook_signature", lambda _r, _e: True)
    paypal["order"] = _approved("cust-x", currency="JPY", value="10000")
    paypal["capture"] = _captured(currency="JPY", value="10000")
    r = client.post(
        "/api/billing/paypal/webhook",
        json={"event_type": "CHECKOUT.ORDER.APPROVED", "resource": paypal["order"]},
    )
    assert r.status_code == 200, r.text
    assert paypal["captured"] == []
    assert paypal["credited"] == []


def test_paypal_capture_does_not_credit_a_pending_payment(customer, paypal):
    paypal["order"] = _approved(customer["customer_id"])
    paypal["capture"] = _captured(status="PENDING")
    r = _capture(customer)
    assert r.status_code == 409, r.text
    assert paypal["credited"] == []


def test_paypal_webhook_does_not_credit_another_currency(monkeypatch):
    import routes.billing as billing_mod

    monkeypatch.setattr(billing_mod, "_PAYPAL_CLIENT_ID", "test-client")
    monkeypatch.setattr(billing_mod, "_PAYPAL_CLIENT_SECRET", "test-secret")
    monkeypatch.setattr(billing_mod, "_paypal_verify_webhook_signature", lambda _r, _e: True)
    monkeypatch.setattr(
        billing_mod,
        "_paypal_get_order",
        lambda order_id, token=None: {"id": order_id, "purchase_units": [{"custom_id": "cust-x"}]},
    )
    credited: list = []
    monkeypatch.setattr(
        billing_mod,
        "_paypal_credit_capture",
        lambda *a: credited.append(a) or {"balance_cad": 0.0},
    )

    r = client.post(
        "/api/billing/paypal/webhook",
        json={
            "event_type": "PAYMENT.CAPTURE.COMPLETED",
            "resource": {
                "id": "CAP-JPY",
                "status": "COMPLETED",
                "amount": {"value": "10000", "currency_code": "JPY"},
                "supplementary_data": {"related_ids": {"order_id": "ORDER-JPY"}},
            },
        },
    )
    assert r.status_code == 200
    assert credited == []
