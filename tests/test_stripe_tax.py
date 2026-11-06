"""Stripe Tax helpers — account location, not deposit province UI."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _address():
    return {
        "country": "CA",
        "province": "ON",
        "line1": "1 Test Street",
        "line2": "",
        "city": "London",
        "postal_code": "N6A 1A1",
    }


def test_resolve_prefers_account_country_province():
    from stripe_tax import resolve_tax_customer_details

    details, source = resolve_tax_customer_details(
        address=_address(),
        ip_address="8.8.8.8",
    )
    assert source == "account_address"
    assert details["address"]["country"] == "CA"
    assert details["address"]["state"] == "ON"
    assert "ip_address" not in details


@pytest.mark.parametrize("ip", ["", "99.226.1.2", "99.226.1.2, 10.0.0.1"])
def test_tax_never_invents_a_country_or_province_when_profile_is_missing(ip):
    from stripe_tax import resolve_tax_customer_details

    with pytest.raises(ValueError, match="billing address"):
        resolve_tax_customer_details(address=None, ip_address=ip)


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
            address=_address(),
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
            address=_address(),
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


@pytest.mark.parametrize(
    "overrides",
    [
        {"country": "USA"},
        {"country": ""},
        {"province": "XX"},
        {"line1": ""},
        {"postal_code": "V6B 1A1"},
    ],
)
def test_tax_rejects_incomplete_or_conflicting_address_fields(overrides):
    from stripe_tax import normalize_customer_address

    with pytest.raises(ValueError):
        normalize_customer_address({**_address(), **overrides})


def test_tax_address_comes_from_current_saved_profile_not_token_claims(monkeypatch):
    from routes.billing import _account_tax_address
    from db import UserStore

    saved = {
        "email": "billing@example.test",
        "country": "CA",
        "province": "BC",
        "address_line1": "1 Test Street",
        "city": "Vancouver",
        "postal_code": "V6B 1A1",
    }
    monkeypatch.setattr(UserStore, "get_user", lambda email: saved)
    out = _account_tax_address({"email": saved["email"], "country": "CA", "province": "ON"})
    assert out["province"] == "BC"
    assert out["line1"] == "1 Test Street"
    assert out["city"] == "Vancouver"
    assert out["postal_code"] == "V6B 1A1"


def test_admin_billing_uses_requested_customers_profile(monkeypatch):
    from routes.billing import _billing_account_profile, _profile_tax_address
    from db import UserStore

    target = {
        "customer_id": "cust-target",
        "email": "target@example.test",
        "name": "Test Customer",
        "account_type": "individual",
        "onboarding_intent": "rent",
        "country": "US",
        "province": "NY",
        "address_line1": "1 Test Street",
        "city": "New York",
        "postal_code": "10001",
    }
    lookup = MagicMock(return_value=target)
    monkeypatch.setattr(UserStore, "get_user_by_customer_id", lookup)
    actual = _billing_account_profile(
        {"email": "admin@example.test", "country": "CA", "province": "ON"}, "cust-target"
    )
    lookup.assert_called_once_with("cust-target")
    assert actual["email"] == "target@example.test"
    assert _profile_tax_address(actual, required=True)["state"] == "NY"


def test_customer_sync_updates_existing_stripe_customer_from_saved_profile(monkeypatch):
    from billing import BillingEngine
    from db import UserStore
    import stripe_connect

    profile = {
        "customer_id": "cust-profile",
        "email": "saved@example.test",
        "name": "Test Customer",
        "account_type": "business",
        "business_name": "Test Business",
        "onboarding_intent": "rent",
        "country": "CA",
        "province": "ON",
        "address_line1": "1 Test Street",
        "address_line2": "",
        "city": "London",
        "postal_code": "N6A 1A1",
    }
    monkeypatch.setattr(UserStore, "get_user_by_customer_id", lambda cid: profile)
    sdk = MagicMock()
    monkeypatch.setattr(stripe_connect, "STRIPE_ENABLED", True)
    monkeypatch.setattr(stripe_connect, "stripe", sdk)
    engine = BillingEngine.__new__(BillingEngine)
    monkeypatch.setattr(engine, "get_wallet", lambda cid: {"stripe_customer_id": "cus-existing"})
    assert (
        engine.ensure_stripe_customer("cust-profile", email="stale@example.test") == "cus-existing"
    )
    sdk.Customer.create.assert_not_called()
    fields = sdk.Customer.modify.call_args.kwargs
    assert fields["email"] == "saved@example.test"
    assert fields["name"] == "Test Business"
    assert fields["address"] == {
        "country": "CA",
        "state": "ON",
        "line1": "1 Test Street",
        "line2": "",
        "city": "London",
        "postal_code": "N6A 1A1",
    }


def test_customer_sync_rejects_missing_profile_before_contacting_stripe(monkeypatch):
    from billing import BillingEngine
    from db import UserStore
    import stripe_connect

    monkeypatch.setattr(UserStore, "get_user_by_customer_id", lambda cid: None)
    sdk = MagicMock()
    monkeypatch.setattr(stripe_connect, "STRIPE_ENABLED", True)
    monkeypatch.setattr(stripe_connect, "stripe", sdk)
    engine = BillingEngine.__new__(BillingEngine)
    with pytest.raises(ValueError, match="account profile"):
        engine.ensure_stripe_customer("cust-missing")
    sdk.Customer.create.assert_not_called()
    sdk.Customer.modify.assert_not_called()
