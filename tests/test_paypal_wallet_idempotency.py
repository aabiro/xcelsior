"""PayPal wallet credit idempotency — no double-credit on replay."""

from __future__ import annotations

import uuid

import pytest

from billing import get_billing_engine


@pytest.fixture
def customer_id():
    return f"paypal-idem-{uuid.uuid4().hex[:10]}@xcelsior.ca"


def test_paypal_deposit_idempotent_by_order_id(customer_id):
    """Same paypal-{order_id} key must not credit twice."""
    be = get_billing_engine()
    order_id = f"ORDER-{uuid.uuid4().hex[:8]}"
    key = f"paypal-{order_id}"

    first = be.deposit(customer_id, 25.0, "PayPal deposit (CAP-1)", idempotency_key=key)
    second = be.deposit(customer_id, 25.0, "PayPal deposit (CAP-1 replay)", idempotency_key=key)

    assert first["balance_cad"] == 25.0
    assert second["balance_cad"] == 25.0
    wallet = be.get_wallet(customer_id)
    assert wallet["balance_cad"] == 25.0