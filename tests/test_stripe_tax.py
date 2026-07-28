"""Stripe Tax helpers — account location, not deposit province UI."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_resolve_prefers_account_country_province():
    from stripe_tax import resolve_tax_customer_details

    details, source = resolve_tax_customer_details(
        address={"country": "CA", "province": "ON"},
        ip_address="8.8.8.8",
    )
    assert source == "account_address"
    assert details["address"]["country"] == "CA"
    assert details["address"]["state"] == "ON"
    assert "ip_address" not in details


def test_resolve_ip_fallback_when_no_account_location():
    from stripe_tax import resolve_tax_customer_details

    details, source = resolve_tax_customer_details(
        address=None,
        ip_address="99.226.1.2, 10.0.0.1",
    )
    assert source == "ip_address"
    assert details["ip_address"] == "99.226.1.2"


def test_calculate_uses_account_address_not_picker(monkeypatch):
    from stripe_tax import calculate_wallet_deposit_tax

    mock = MagicMock()
    calc = MagicMock()
    calc.id = "taxcalc_acct"
    calc.tax_amount_exclusive = 325
    calc.amount_total = 2825
    calc.tax_breakdown = []
    mock.tax.Calculation.create.return_value = calc

    with (
        patch("stripe_connect.STRIPE_ENABLED", True),
        patch("stripe_connect.stripe", mock),
    ):
        out = calculate_wallet_deposit_tax(
            amount_cents=2500,
            address={"country": "CA", "province": "ON"},
            ip_address="1.1.1.1",
            stripe_mod=mock,
        )

    assert out["tax_enabled"] is True
    assert out["tax_amount_cents"] == 325
    assert out["credit_amount_cents"] == 2500
    assert out["amount_total"] == 2825
    assert out["location_source"] == "account_address"
    kwargs = mock.tax.Calculation.create.call_args.kwargs
    assert kwargs["customer_details"]["address"]["state"] == "ON"
    assert kwargs["line_items"][0]["tax_behavior"] == "exclusive"


def test_create_credit_deposit_links_tax_calc_and_stores_pretax():
    """PI charges total with tax; local payment_intents stores pretax credits."""
    from stripe_connect import StripeConnectManager

    mgr = StripeConnectManager.__new__(StripeConnectManager)
    mock_stripe = MagicMock()
    mock_stripe.PaymentIntent.create.return_value = MagicMock(
        id="pi_tax_1",
        client_secret="pi_tax_1_secret",
    )

    tax_info = {
        "tax_calculation_id": "taxcalc_1",
        "amount_total": 2825,
        "tax_amount_cents": 325,
        "credit_amount_cents": 2500,
        "tax_enabled": True,
        "breakdown": [],
        "location_source": "account_address",
    }

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute = MagicMock()

    with (
        patch("stripe_connect.STRIPE_ENABLED", True),
        patch("stripe_connect.stripe", mock_stripe),
        patch("stripe_tax.calculate_wallet_deposit_tax", return_value=tax_info),
        patch.object(StripeConnectManager, "_conn", return_value=mock_conn),
        patch("billing.get_billing_engine") as mock_be,
    ):
        mock_be.return_value.ensure_stripe_customer.return_value = "cus_1"
        result = mgr.create_credit_deposit(
            "cust-1",
            25.0,
            address={"country": "CA", "province": "ON"},
            email="u@example.com",
        )

    assert result["credit_amount_cad"] == 25.0
    assert result["tax_amount_cad"] == 3.25
    assert result["charge_amount_cad"] == 28.25
    assert "Compute credits" in (result.get("description") or "")
    pi_kwargs = mock_stripe.PaymentIntent.create.call_args.kwargs
    assert pi_kwargs["amount"] == 2825
    assert pi_kwargs["customer"] == "cus_1"
    assert "Compute credits" in pi_kwargs["description"]
    assert pi_kwargs["hooks"]["inputs"]["tax"]["calculation"] == "taxcalc_1"
    # Local row stores pretax credit cents
    insert_args = mock_conn.execute.call_args[0][1]
    assert insert_args[2] == 2500  # credit_cents
