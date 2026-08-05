"""Replaying a funding call charges once; asking twice charges twice.

Gate P1: *"Replaying any funding call with the same idempotency key produces
exactly one charge."* Both halves matter, and only the first was true.

**The half that worked.** `charge_saved_card` passes an idempotency key to
Stripe, and Stripe answers a repeated key with the *original* PaymentIntent. A
retried request cannot produce a second charge.

**The half that did not.** Stripe says nothing about having replayed. It returns
the original intent as if it had just created it, so the caller cannot tell "I
charged the card" from "I was handed the receipt for a charge I made earlier".
The route reported `charged: true` either way.

That mattered because of how the key was derived. With no `Idempotency-Key`
header the route built one from customer, amount, card and `time // 300` — a
five-minute bucket. Two *deliberate* $10 top-ups on the same card a minute apart
produced the same key, so the second returned the first charge and answered
`charged: true`. The user believed they had added $20 and had added $10. The
comment above that code claimed "a genuinely separate top-up a minute later gets
a new key", which is not what a five-minute bucket does.

The fix is in two places, and neither is sufficient alone:

* **`top_up_wallet` mints a key per invocation.** One tool call is one intended
  charge, so two deliberate top-ups carry two keys. It also re-enables client
  retries, which `api.ts` disables when no key is present — without it a timeout
  was a single blind attempt.
* **The replay is reported.** `_register_payment_intent` already had the signal
  and threw it away: `ON CONFLICT (stripe_intent_id) DO NOTHING` yields
  `rowcount == 0` when the intent is already on file, which happens exactly when
  Stripe replayed. It now returns that, `charge_saved_card` passes it up as
  `replayed`, and the route stops claiming a charge it did not make.

Generating keys without reporting replays would turn a retry into a double
charge wherever a key failed to reach Stripe. Reporting without generating would
leave deliberate repeats silently merged. Hence both.
"""

from __future__ import annotations

import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest  # noqa: E402


class _FakeIntent:
    def __init__(self, intent_id: str, status: str = "succeeded"):
        self.id = intent_id
        self.status = status


class _IdempotentStripe:
    """Stripe's real contract: a repeated key returns the original intent.

    A fake that minted a fresh intent per call would make every assertion here
    pass while testing nothing — the replay would never occur.
    """

    def __init__(self):
        self.by_key: dict[str, _FakeIntent] = {}
        self.create_calls = 0
        outer = self

        class PaymentIntent:
            @staticmethod
            def create(**kwargs):
                outer.create_calls += 1
                key = str(kwargs.get("idempotency_key") or "")
                if key and key in outer.by_key:
                    return outer.by_key[key]
                intent = _FakeIntent(f"pi_{uuid.uuid4().hex[:20]}")
                if key:
                    outer.by_key[key] = intent
                return intent

        self.PaymentIntent = PaymentIntent


@pytest.fixture
def engine():
    from billing import get_billing_engine

    return get_billing_engine()


@pytest.fixture
def stripe_fake(monkeypatch):
    import stripe_connect

    fake = _IdempotentStripe()
    monkeypatch.setattr(stripe_connect, "stripe", fake, raising=False)
    monkeypatch.setattr(stripe_connect, "STRIPE_ENABLED", True, raising=False)
    return fake


@pytest.fixture
def probe_customer():
    """A synthetic customer id, and its `payment_intents` rows removed after."""
    customer_id = f"replayprobe_{uuid.uuid4().hex[:12]}"
    yield customer_id
    from db import _get_pg_pool

    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.execute("DELETE FROM payment_intents WHERE customer_id = %s", (customer_id,))
        conn.commit()


def _charge(engine, customer_id: str, key: str, amount_micros: int = 10_000_000) -> dict:
    return engine.charge_saved_card(
        customer_id,
        amount_micros,
        stripe_customer_id="cus_replayprobe",
        payment_method_id="pm_replayprobe",
        idempotency_key=key,
        description="replay probe",
    )


# ── The mechanism ──────────────────────────────────────────────────────


def test_registering_an_intent_twice_reports_the_second_as_a_replay(engine, probe_customer):
    """The signal that was being discarded.

    `ON CONFLICT (stripe_intent_id) DO NOTHING` makes the repeat harmless, and
    `rowcount` is the only thing that distinguishes it from a first write.
    """
    stripe_intent_id = f"pi_{uuid.uuid4().hex[:20]}"
    common = dict(
        customer_id=probe_customer,
        amount_cents=1000,
        stripe_intent_id=stripe_intent_id,
        description="replay probe",
        created_at=0.0,
    )

    first = engine._register_payment_intent(intent_id=f"pi_a_{uuid.uuid4().hex[:8]}", **common)
    second = engine._register_payment_intent(intent_id=f"pi_b_{uuid.uuid4().hex[:8]}", **common)

    assert first is True, "the first registration of an intent must report as new"
    assert second is False, (
        "the second registration of the same Stripe intent reported as new; the "
        "caller cannot then tell a replayed charge from a fresh one"
    )


# ── Gate P1: the same key produces exactly one charge ──────────────────


def test_replaying_the_same_key_produces_one_charge_and_says_so(
    engine, stripe_fake, probe_customer
):
    """The gate, plus the part that makes it legible.

    Stripe returns the original intent, so no second charge exists. The second
    call must report `replayed` rather than presenting that intent as new work.
    """
    key = f"probe-{uuid.uuid4().hex[:16]}"

    first = _charge(engine, probe_customer, key)
    second = _charge(engine, probe_customer, key)

    assert first["stripe_intent_id"] == second["stripe_intent_id"], (
        "the fake did not honour the idempotency key, so this asserts nothing"
    )
    assert first["replayed"] is False
    assert second["replayed"] is True, (
        "a replayed charge was reported as a fresh one — the account holder is "
        "told they added funds that were never charged"
    )

    from db import _get_pg_pool

    pool = _get_pg_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT count(*) FROM payment_intents WHERE customer_id = %s",
            (probe_customer,),
        ).fetchone()[0]
    assert rows == 1, f"one charge should leave one intent row, found {rows}"


def test_two_deliberate_topups_with_different_keys_are_two_charges(
    engine, stripe_fake, probe_customer
):
    """The inverse, which is the half that was broken.

    Asking twice on purpose must charge twice. If distinct intents collapse into
    one key, this is where it shows up.
    """
    first = _charge(engine, probe_customer, f"probe-{uuid.uuid4().hex[:16]}")
    second = _charge(engine, probe_customer, f"probe-{uuid.uuid4().hex[:16]}")

    assert first["stripe_intent_id"] != second["stripe_intent_id"]
    assert first["replayed"] is False
    assert second["replayed"] is False, (
        "a second deliberate top-up was treated as a replay of the first"
    )


# ── The two halves of the fix, guarded structurally ────────────────────


def test_the_route_does_not_claim_a_charge_it_did_not_make():
    """`charged` must follow the replay, not the absence of an exception."""
    import inspect

    import routes.billing as billing_routes

    source = inspect.getsource(billing_routes.api_billing_manual_topup)
    assert 'charge.get("replayed")' in source, (
        "the route ignores the replay flag, so a repeated key answers "
        '"charged: true" for a charge that did not happen'
    )
    assert '"charged": not replayed' in source, (
        "`charged` is no longer derived from the replay flag"
    )


def test_the_tool_supplies_a_key_when_the_model_omits_one():
    """The other half: one invocation, one intent, one key.

    Without this the route falls back to bucketing by customer, amount and card
    in a five-minute window, and two deliberate top-ups inside that window
    become one charge. `api.ts` also disables retries when no key is present, so
    the omission cost retry safety as well.
    """
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parent.parent
        / "mcp"
        / "src"
        / "tools"
        / "billing.ts"
    ).read_text(encoding="utf-8")

    assert "randomUUID" in source, (
        "top_up_wallet no longer generates an idempotency key; a model that "
        "omits one gets the route's five-minute bucket instead"
    )
    assert "idempotency_key ?? `topup-" in source, (
        "the generated key no longer defers to a caller-supplied one, so a "
        "deliberate retry can no longer be expressed"
    )


# ── The other funding rail ─────────────────────────────────────────────


def test_the_wallet_deposit_rail_deduplicates_on_its_key(engine):
    """The crypto and PayPal rails credit through `deposit`, not through Stripe.

    They take the same gate: the same key must not credit twice. `deposit`
    dedupes on `wallet_transactions.idempotency_key`, which is a different
    mechanism from Stripe's, so it is asserted separately rather than assumed to
    behave the same way.
    """
    import inspect

    source = inspect.getsource(engine.deposit)
    assert "idempotency_key" in source, "deposit no longer accepts an idempotency key"
    assert "SELECT tx_id" in source and "wallet_transactions" in source, (
        "deposit no longer looks for an existing transaction with the same key, "
        "so a replayed deposit would credit the wallet a second time"
    )
