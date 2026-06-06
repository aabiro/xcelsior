"""PayPal webhook handler tests."""

import os

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_paypal_webhook_rejects_without_signature(monkeypatch):
    import routes.billing as billing_mod

    monkeypatch.setattr(billing_mod, "_PAYPAL_CLIENT_ID", "test-client")
    monkeypatch.setattr(billing_mod, "_PAYPAL_CLIENT_SECRET", "test-secret")
    monkeypatch.setattr(billing_mod, "_PAYPAL_WEBHOOK_ID", "WH-TEST")
    monkeypatch.setattr(billing_mod, "_paypal_verify_webhook_signature", lambda _r, _e: False)

    r = client.post(
        "/api/billing/paypal/webhook",
        json={"event_type": "PAYMENT.CAPTURE.COMPLETED", "resource": {}},
    )
    assert r.status_code == 400


def test_paypal_webhook_capture_completed_credits_wallet(monkeypatch):
    import routes.billing as billing_mod

    monkeypatch.setattr(billing_mod, "_PAYPAL_CLIENT_ID", "test-client")
    monkeypatch.setattr(billing_mod, "_PAYPAL_CLIENT_SECRET", "test-secret")
    monkeypatch.setattr(billing_mod, "_PAYPAL_WEBHOOK_ID", "WH-TEST")
    monkeypatch.setattr(billing_mod, "_paypal_verify_webhook_signature", lambda _r, _e: True)
    monkeypatch.setattr(
        billing_mod,
        "_paypal_get_order",
        lambda order_id: {
            "id": order_id,
            "purchase_units": [{"custom_id": "cust-paypal-wh"}],
        },
    )

    credited: list[tuple] = []

    def fake_credit(customer_id, order_id, amount_cad, capture_id):
        credited.append((customer_id, order_id, amount_cad, capture_id))
        return {"balance_cad": amount_cad, "tx_id": "TX-test"}

    monkeypatch.setattr(billing_mod, "_paypal_credit_capture", fake_credit)

    r = client.post(
        "/api/billing/paypal/webhook",
        json={
            "event_type": "PAYMENT.CAPTURE.COMPLETED",
            "resource": {
                "id": "CAP-123",
                "amount": {"value": "25.00", "currency_code": "CAD"},
                "supplementary_data": {"related_ids": {"order_id": "ORDER-9"}},
            },
        },
    )
    assert r.status_code == 200
    assert credited == [("cust-paypal-wh", "ORDER-9", 25.0, "CAP-123")]


def test_paypal_webhook_order_completed_credits_wallet(monkeypatch):
    import routes.billing as billing_mod

    monkeypatch.setattr(billing_mod, "_PAYPAL_CLIENT_ID", "test-client")
    monkeypatch.setattr(billing_mod, "_PAYPAL_CLIENT_SECRET", "test-secret")
    monkeypatch.setattr(billing_mod, "_PAYPAL_WEBHOOK_ID", "WH-TEST")
    monkeypatch.setattr(billing_mod, "_paypal_verify_webhook_signature", lambda _r, _e: True)

    credited: list[tuple] = []
    monkeypatch.setattr(
        billing_mod,
        "_paypal_credit_capture",
        lambda customer_id, order_id, amount_cad, capture_id: credited.append(
            (customer_id, order_id, amount_cad, capture_id)
        )
        or {"balance_cad": amount_cad},
    )

    r = client.post(
        "/api/billing/paypal/webhook",
        json={
            "event_type": "CHECKOUT.ORDER.COMPLETED",
            "resource": {
                "id": "ORDER-42",
                "purchase_units": [
                    {
                        "custom_id": "cust-42",
                        "payments": {
                            "captures": [
                                {
                                    "id": "CAP-42",
                                    "status": "COMPLETED",
                                    "amount": {"value": "10.00", "currency_code": "CAD"},
                                }
                            ]
                        },
                    }
                ],
            },
        },
    )
    assert r.status_code == 200
    assert credited == [("cust-42", "ORDER-42", 10.0, "CAP-42")]