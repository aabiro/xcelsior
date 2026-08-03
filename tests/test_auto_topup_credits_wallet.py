"""Auto-top-up must actually credit the wallet it charged the card for.

`check_low_balance_and_topup` charges a saved card off-session and calls it a
success. Crediting the wallet is not its job — that happens when Stripe confirms
the charge, in `StripeConnectManager._handle_payment_succeeded`, which is
idempotent per event id. That division of labour is correct: the processor is
the only authority on whether money moved.

But that handler credits the wallet only if it can find the intent:

    row = conn.execute(
        "SELECT customer_id, amount_cents FROM payment_intents "
        "WHERE stripe_intent_id=%s", (si_id,)).fetchone()
    if row:
        engine.deposit(...)

and the only writer of `payment_intents` is
`StripeConnectManager.create_payment_intent` — the dashboard deposit path.
Auto-top-up calls `stripe.PaymentIntent.create` directly and registers nothing.
So the webhook arrives, matches no row, falls through the `if`, and the inbox
marks the event processed.

The customer is charged and receives no credits.

Nothing reports it. `topped_up` increments, the log line says the intent was
created, `auto_topup_failures` resets to zero, and the event inbox shows
`processed`. Every signal says the top-up worked. The wallet stays below its
threshold, so the next sweep charges the card again — and again — while jobs
keep failing for insufficient balance.

These tests assert the behaviour end to end: charge the card, deliver the
event Stripe would send, and require the balance to reflect it. They fail
against the code as written, which is the point.
"""

from __future__ import annotations

import os
import secrets
import time
import types

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

from billing import BillingEngine, cad_to_micros

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _conn:
        _conn.execute("SELECT idempotency_key FROM wallet_transactions LIMIT 0")
    _PG_AVAILABLE = True
except Exception:
    _PG_AVAILABLE = False

pg = pytest.mark.skipif(not _PG_AVAILABLE, reason="PostgreSQL not available")

TOPUP_CAD = 25.0
THRESHOLD_CAD = 10.0
STARTING_CAD = 5.0


@pytest.fixture(autouse=True)
def _clean():
    if not _PG_AVAILABLE:
        yield
        return
    pool = _get_pg_pool()
    for _ in range(2):
        with pool.connection() as conn:
            conn.execute("DELETE FROM wallet_transactions WHERE customer_id LIKE 'topup-test-%%'")
            conn.execute("DELETE FROM wallets WHERE customer_id LIKE 'topup-test-%%'")
            conn.execute(
                "DELETE FROM payment_intents WHERE customer_id LIKE 'topup-test-%%'"
            )
            conn.commit()
        yield
        return


def _arm_wallet(engine: BillingEngine, customer_id: str) -> None:
    """A wallet below its threshold with a saved card: the sweep will charge it."""
    engine.get_wallet(customer_id)
    engine.deposit(
        customer_id,
        STARTING_CAD,
        description="seed",
        idempotency_key=f"seed-{customer_id}",
    )
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.execute(
            """UPDATE wallets
                  SET auto_topup_enabled = true,
                      auto_topup_threshold_micros = %s,
                      auto_topup_amount_micros = %s,
                      stripe_payment_method_id = 'pm_test_saved_card',
                      stripe_customer_id = 'cus_test_customer',
                      auto_topup_failures = 0,
                      last_topup_attempt_at = 0,
                      status = 'active'
                WHERE customer_id = %s""",
            (cad_to_micros(THRESHOLD_CAD), cad_to_micros(TOPUP_CAD), customer_id),
        )
        conn.commit()


class _FakeIntent:
    """What Stripe returns for a successful off-session confirmation."""

    def __init__(self, intent_id: str):
        self.id = intent_id
        self.status = "succeeded"
        self.client_secret = f"{intent_id}_secret_never_leaves_the_server"


def _capture_stripe(monkeypatch, intent_id: str) -> dict:
    """Stand in for Stripe and record the charge arguments."""
    captured: dict = {}

    def _create(**kwargs):
        captured.update(kwargs)
        return _FakeIntent(intent_id)

    import stripe_connect

    fake = types.SimpleNamespace(PaymentIntent=types.SimpleNamespace(create=_create))
    monkeypatch.setattr(stripe_connect, "stripe", fake, raising=False)
    monkeypatch.setattr(stripe_connect, "STRIPE_ENABLED", True, raising=False)
    return captured


@pg
def test_charged_card_results_in_credited_wallet(monkeypatch):
    """The whole point of auto-top-up: card charged, balance restored.

    Charge the card, then deliver the `payment_intent.succeeded` event Stripe
    sends on confirmation. The balance must reflect the money taken.
    """
    engine = BillingEngine()
    customer_id = f"topup-test-{secrets.token_hex(4)}"
    intent_id = f"pi_{secrets.token_hex(8)}"
    _arm_wallet(engine, customer_id)
    charge = _capture_stripe(monkeypatch, intent_id)

    result = engine.check_low_balance_and_topup()
    assert result["topped_up"] == 1, f"the sweep did not charge the card: {result}"
    assert charge["off_session"] is True and charge["confirm"] is True, (
        "auto-top-up must be a merchant-initiated off-session charge; "
        f"got {charge!r}"
    )

    from stripe_connect import StripeConnectManager

    StripeConnectManager()._handle_payment_succeeded(
        {"id": intent_id}, f"evt_{secrets.token_hex(8)}"
    )

    balance = engine.get_wallet(customer_id)["balance_cad"]
    assert balance == pytest.approx(STARTING_CAD + TOPUP_CAD), (
        f"card charged ${TOPUP_CAD:.2f} but wallet holds ${balance:.2f}. "
        "The confirmation event could not find the intent, because auto-top-up "
        "never registered it in `payment_intents`. The customer paid and got "
        "nothing."
    )


@pg
def test_topup_intent_is_registered_before_the_card_is_charged(monkeypatch):
    """The intent must be recorded, or the confirmation has nothing to match.

    Separated from the test above so a failure says *why* the credit was lost
    rather than only that it was.
    """
    engine = BillingEngine()
    customer_id = f"topup-test-{secrets.token_hex(4)}"
    intent_id = f"pi_{secrets.token_hex(8)}"
    _arm_wallet(engine, customer_id)
    _capture_stripe(monkeypatch, intent_id)

    engine.check_low_balance_and_topup()

    pool = _get_pg_pool()
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT customer_id, amount_cents FROM payment_intents "
            "WHERE stripe_intent_id = %s",
            (intent_id,),
        ).fetchone()

    assert row is not None, (
        f"{intent_id} was charged but never written to `payment_intents`. "
        "`_handle_payment_succeeded` looks the intent up by that id and does "
        "nothing when it is absent, so the confirmation is silently discarded."
    )
    assert row[1] == int(TOPUP_CAD * 100), (
        f"registered amount {row[1]} cents does not match the "
        f"{int(TOPUP_CAD * 100)} cents charged; the wallet would be credited "
        "the wrong amount."
    )


@pg
def test_repeated_confirmation_credits_once(monkeypatch):
    """Stripe retries webhooks. A retry must not double-credit.

    This one passes today — `deposit` is keyed on the event id. It is here so
    that fixing the two failures above cannot be done by crediting inline at
    charge time, which would credit again on every delivery of the event.
    """
    engine = BillingEngine()
    customer_id = f"topup-test-{secrets.token_hex(4)}"
    intent_id = f"pi_{secrets.token_hex(8)}"
    event_id = f"evt_{secrets.token_hex(8)}"
    _arm_wallet(engine, customer_id)
    _capture_stripe(monkeypatch, intent_id)

    engine.check_low_balance_and_topup()

    from stripe_connect import StripeConnectManager

    mgr = StripeConnectManager()
    mgr._handle_payment_succeeded({"id": intent_id}, event_id)
    mgr._handle_payment_succeeded({"id": intent_id}, event_id)
    mgr._handle_payment_succeeded({"id": intent_id}, event_id)

    balance = engine.get_wallet(customer_id)["balance_cad"]
    expected = STARTING_CAD + TOPUP_CAD
    assert balance == pytest.approx(expected), (
        f"three deliveries of {event_id} produced ${balance:.2f}, expected "
        f"${expected:.2f}: "
        + (
            "the credit is not idempotent per event"
            if balance > expected
            else "nothing was credited at all — the same missing "
            "`payment_intents` row as the failures above"
        )
    )


@pg
def test_a_declined_charge_credits_nothing(monkeypatch):
    """The inverse guard: no charge, no credit, and the failure is counted.

    Without this, 'always credit on sweep' would satisfy the tests above while
    handing out free money whenever a card declines.
    """
    engine = BillingEngine()
    customer_id = f"topup-test-{secrets.token_hex(4)}"
    _arm_wallet(engine, customer_id)

    import stripe_connect

    def _decline(**kwargs):
        raise RuntimeError("Your card was declined.")

    monkeypatch.setattr(
        stripe_connect,
        "stripe",
        types.SimpleNamespace(PaymentIntent=types.SimpleNamespace(create=_decline)),
        raising=False,
    )
    monkeypatch.setattr(stripe_connect, "STRIPE_ENABLED", True, raising=False)

    result = engine.check_low_balance_and_topup()
    assert result["errors"] == 1 and result["topped_up"] == 0

    balance = engine.get_wallet(customer_id)["balance_cad"]
    assert balance == pytest.approx(STARTING_CAD), (
        f"a declined card credited the wallet anyway: ${balance:.2f}"
    )

    pool = _get_pg_pool()
    with pool.connection() as conn:
        failures = conn.execute(
            "SELECT auto_topup_failures FROM wallets WHERE customer_id = %s",
            (customer_id,),
        ).fetchone()[0]
    assert failures == 1, f"the decline was not recorded: failures={failures}"


@pg
def test_sweep_does_not_recharge_a_wallet_it_already_topped_up(monkeypatch):
    """Two sweeps before the event lands must not charge the card twice.

    The sweep selects on `balance_micros <= auto_topup_threshold_micros`, and
    the balance does not move until Stripe confirms. Between the charge and the
    confirmation the wallet still matches, so the next sweep charges again.
    `last_topup_attempt_at` is written but only consulted when
    `auto_topup_failures > 0`, so a *successful* charge sets no cooldown at all.
    """
    engine = BillingEngine()
    customer_id = f"topup-test-{secrets.token_hex(4)}"
    _arm_wallet(engine, customer_id)

    charges: list[str] = []

    def _create(**kwargs):
        intent_id = f"pi_{secrets.token_hex(8)}"
        charges.append(intent_id)
        return _FakeIntent(intent_id)

    import stripe_connect

    monkeypatch.setattr(
        stripe_connect,
        "stripe",
        types.SimpleNamespace(PaymentIntent=types.SimpleNamespace(create=_create)),
        raising=False,
    )
    monkeypatch.setattr(stripe_connect, "STRIPE_ENABLED", True, raising=False)

    engine.check_low_balance_and_topup()
    engine.check_low_balance_and_topup()

    assert len(charges) == 1, (
        f"the card was charged {len(charges)} times for one top-up: {charges}. "
        "A charge in flight must suppress the next sweep until it settles or "
        "expires."
    )
