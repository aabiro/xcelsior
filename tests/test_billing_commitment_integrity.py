"""A failed dependency cannot produce a successful financial operation."""

from unittest.mock import MagicMock
import uuid

import pytest
from fastapi.testclient import TestClient

from api import app
from billing import get_billing_engine


@pytest.fixture
def customer():
    with TestClient(app) as client:
        email = f"commitment-{uuid.uuid4().hex}@example.test"
        registered = client.post(
            "/api/auth/register",
            json={
                "email": email,
                "password": "StrongPass123!",
                "name": "Integrity Test",
            },
        )
        assert registered.status_code == 200, registered.text
        login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
        assert login.status_code == 200, login.text
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        profile = client.put(
            "/api/auth/me/profile",
            headers=headers,
            json={
                "account_type": "individual",
                "onboarding_intent": "rent",
                "country": "CA",
                "province": "ON",
                "address_line1": "1 Test Street",
                "city": "London",
                "postal_code": "N6A 1A1",
            },
        )
        assert profile.status_code == 200, profile.text
        yield client, headers, registered.json()["user"]["customer_id"]


def test_commitment_write_failure_returns_503_and_emits_no_success(customer, monkeypatch):
    import routes.billing as routes

    client, headers, customer_id = customer
    write = MagicMock(side_effect=RuntimeError("storage unavailable"))
    monkeypatch.setattr(get_billing_engine(), "record_reservation", write)
    event = MagicMock()
    monkeypatch.setattr(routes, "broadcast_sse", event)
    result = client.post("/api/pricing/reserve", headers=headers, json={"customer_id": customer_id})
    assert result.status_code == 503, result.text
    assert "could not be saved" in result.json()["error"]["message"]
    event.assert_not_called()
    assert get_billing_engine().list_reservations(customer_id) == []


def test_successful_commitment_is_durable_and_readable(customer):
    client, headers, customer_id = customer
    result = client.post("/api/pricing/reserve", headers=headers, json={"customer_id": customer_id})
    assert result.status_code == 200, result.text
    rows = get_billing_engine().list_reservations(customer_id)
    assert len(rows) == 1
    assert rows[0]["commitment_id"] == result.json()["commitment_id"]
    assert rows[0]["status"] == "active"
    assert rows[0]["province"] is None
    assert result.json()["tax_collection"] == "credit_deposit"
    assert result.json()["tax_rate"] is None


def test_payment_tax_failure_is_retriable_and_creates_no_intent(customer, monkeypatch):
    import stripe_connect
    from db import pg_connection

    client, headers, customer_id = customer
    sdk = MagicMock()
    sdk.tax.Calculation.create.side_effect = RuntimeError("private provider error")
    monkeypatch.setattr(stripe_connect, "STRIPE_ENABLED", True)
    monkeypatch.setattr(stripe_connect, "stripe", sdk)
    result = client.post(
        "/api/billing/payment-intent",
        headers=headers,
        json={"customer_id": customer_id, "amount_cad": 25},
    )
    assert result.status_code == 503, result.text
    assert "private provider error" not in result.text
    sdk.PaymentIntent.create.assert_not_called()
    with pg_connection() as conn:
        count = conn.execute(
            "SELECT count(*) FROM payment_intents WHERE customer_id=%s", (customer_id,)
        ).fetchone()[0]
    assert count == 0


@pytest.mark.parametrize("amount", [True, "25.00", 25.001, "NaN", "Infinity"])
def test_payment_amount_is_a_real_json_number_in_whole_cents(customer, monkeypatch, amount):
    import routes.billing as routes

    client, headers, customer_id = customer
    create = MagicMock()
    monkeypatch.setattr(routes.get_stripe_manager(), "create_credit_deposit", create)
    result = client.post(
        "/api/billing/payment-intent",
        headers=headers,
        json={"customer_id": customer_id, "amount_cad": amount},
    )
    assert result.status_code == 422, result.text
    create.assert_not_called()


@pytest.mark.parametrize("quantity", [True, "2", 1.5])
def test_commitment_quantity_cannot_coerce_other_data_types(customer, quantity):
    client, headers, customer_id = customer
    result = client.post(
        "/api/pricing/reserve",
        headers=headers,
        json={"customer_id": customer_id, "quantity": quantity},
    )
    assert result.status_code == 422, result.text
    assert get_billing_engine().list_reservations(customer_id) == []
