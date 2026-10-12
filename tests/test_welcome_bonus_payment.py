"""A welcome bonus requires real card funding and is never cash revenue."""

import copy
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import stripe
from fastapi import HTTPException
from starlette.requests import Request

from billing import BillingEngine
import routes.billing as routes
import stripe_connect


@pytest.fixture
def account(monkeypatch):
    engine = BillingEngine()
    cid = f"bonus-{uuid.uuid4().hex}"
    engine.get_wallet(cid)
    with engine._conn() as conn:
        conn.execute(
            "UPDATE wallets SET stripe_customer_id = %s WHERE customer_id = %s",
            ("cus_bonus", cid),
        )
    monkeypatch.setattr(engine, "list_payment_methods", lambda _: [{"id": "pm_bonus"}])
    charge = {
        "id": "ch_bonus",
        "customer": "cus_bonus",
        "payment_intent": f"pi_{uuid.uuid4().hex}",
        "currency": "cad",
        "paid": True,
        "captured": True,
        "amount_captured": 565,
        "amount_refunded": 0,
        "disputed": False,
        "payment_method_details": {"type": "card"},
    }
    intent = {
        "id": charge["payment_intent"],
        "customer": "cus_bonus",
        "status": "succeeded",
        "currency": "cad",
        "amount_received": 565,
        "latest_charge": charge,
    }
    retrieve = Mock(
        side_effect=lambda *a, **kw: stripe.StripeObject.construct_from(copy.deepcopy(intent), None)
    )
    monkeypatch.setattr(stripe_connect, "STRIPE_ENABLED", True)
    monkeypatch.setattr(
        stripe_connect, "stripe", SimpleNamespace(PaymentIntent=SimpleNamespace(retrieve=retrieve))
    )
    return engine, cid, intent, retrieve


def register(account, *, amount=500, status="succeeded", currency="cad", owner=None):
    engine, cid, intent, _ = account
    with engine._conn() as conn:
        conn.execute(
            """INSERT INTO payment_intents
               (intent_id, customer_id, amount_cents, currency, status,
                stripe_intent_id, description, created_at)
               VALUES (%s, %s, %s, %s, %s, %s, 'test top-up', 1)""",
            (intent["id"], owner or cid, amount, currency, status, intent["id"]),
        )


def test_saved_card_and_confirmed_five_dollar_deposit_qualify(account):
    register(account)
    engine, cid, intent, retrieve = account
    assert engine.welcome_bonus_eligible(cid)
    retrieve.assert_called_once_with(intent["id"], expand=["latest_charge"])


@pytest.mark.parametrize(
    "amount,status,currency",
    [
        (499, "succeeded", "cad"),
        (500, "created", "cad"),
        (500, "requires_action", "cad"),
        (500, "failed", "cad"),
        (500, "succeeded", "usd"),
    ],
)
def test_insufficient_pending_failed_or_foreign_currency_does_not_qualify(
    account, amount, status, currency
):
    register(account, amount=amount, status=status, currency=currency)
    engine, cid, _, retrieve = account
    assert not engine.welcome_bonus_eligible(cid)
    retrieve.assert_not_called()


def test_card_without_a_payment_and_manual_deposits_do_not_qualify(account):
    engine, cid, _, retrieve = account
    engine.deposit(cid, 100)
    engine.grant_credit(cid, 10, "Old promotional credit")
    assert not engine.welcome_bonus_eligible(cid)
    retrieve.assert_not_called()


def test_deposit_without_a_saved_card_does_not_qualify(account, monkeypatch):
    register(account)
    engine, cid, _, retrieve = account
    monkeypatch.setattr(engine, "list_payment_methods", lambda _: [])
    assert not engine.welcome_bonus_eligible(cid)
    retrieve.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("customer", "cus_other"),
        ("status", "processing"),
        ("currency", "usd"),
        ("amount_received", 499),
    ],
)
def test_stripe_intent_must_confirm_the_payment(account, field, value):
    register(account)
    engine, cid, intent, _ = account
    intent[field] = value
    assert not engine.welcome_bonus_eligible(cid)


@pytest.mark.parametrize(
    "field,value",
    [
        ("customer", "cus_other"),
        ("payment_intent", "pi_other"),
        ("currency", "usd"),
        ("paid", False),
        ("captured", False),
        ("amount_captured", 499),
        ("amount_refunded", 1),
        ("disputed", True),
        ("payment_method_details", {"type": "us_bank_account"}),
    ],
)
def test_refunded_disputed_unsettled_or_noncard_charges_do_not_qualify(account, field, value):
    register(account)
    engine, cid, intent, _ = account
    intent["latest_charge"][field] = value
    assert not engine.welcome_bonus_eligible(cid)


def test_other_customers_deposit_does_not_qualify(account):
    register(account, owner="another-customer")
    engine, cid, _, retrieve = account
    assert not engine.welcome_bonus_eligible(cid)
    retrieve.assert_not_called()


@pytest.fixture
def claim(account, monkeypatch):
    engine, cid, _, _ = account
    user = {"email": "bonus@example.test", "customer_id": cid}
    monkeypatch.setattr(routes, "_get_current_user", lambda _: user)
    monkeypatch.setattr(routes, "_USE_PERSISTENT_AUTH", False)
    monkeypatch.setattr(routes, "_users_db", {user["email"]: user})
    monkeypatch.setattr(routes, "get_billing_engine", lambda: engine)
    return lambda owner=cid: routes.api_claim_free_credits(
        owner, Request({"type": "http", "headers": []})
    )


def test_claim_denies_unfunded_account_without_crediting(account, claim):
    with pytest.raises(HTTPException) as error:
        claim()
    assert error.value.status_code == 403
    assert "$5" in error.value.detail
    assert account[0].get_wallet(account[1])["balance_cad"] == 0


def test_claim_fails_closed_when_stripe_cannot_verify(account, claim):
    register(account)
    account[3].side_effect = RuntimeError("Stripe unavailable")
    with pytest.raises(HTTPException) as error:
        claim()
    assert error.value.status_code == 503
    assert account[0].get_wallet(account[1])["balance_cad"] == 0


def test_claim_refuses_other_account(account, claim):
    with pytest.raises(HTTPException) as error:
        claim("someone-else")
    assert error.value.status_code == 403
    account[3].assert_not_called()


def test_concurrent_claims_award_five_once_as_promotional_credit(account, claim):
    register(account)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: claim(), range(4)))
    assert sum(not r["already_claimed"] for r in results) == 1
    engine, cid, _, _ = account
    wallet = engine.get_wallet(cid)
    assert wallet["balance_cad"] == 5
    assert wallet["total_deposited_cad"] == 0
    with engine._conn() as conn:
        row = conn.execute(
            "SELECT tx_type, amount_micros FROM wallet_transactions WHERE customer_id = %s", (cid,)
        ).fetchone()
    assert row["tx_type"] == "credit"
    assert row["amount_micros"] == 5_000_000


def test_old_ten_dollar_bonus_is_preserved_and_cannot_be_claimed_again(account, claim):
    engine, cid, _, retrieve = account
    engine.deposit(
        cid, 10, "Welcome bonus — $10 free credits", idempotency_key=f"free-credits-{cid}"
    )
    result = claim()
    assert result == {"ok": True, "amount_cad": 10, "balance_cad": 10, "already_claimed": True}
    retrieve.assert_not_called()
