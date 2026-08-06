"""A charge stopped by an SCA challenge must be findable.

Gate P1 wants *"an SCA-pending state in the wallet UI, so a challenge that was
never completed is visible rather than silent"*. It could not be built: there
was nothing to read.

`_register_payment_intent` hardcoded `status = 'created'` for every row it
wrote, including the one the SCA branch registers when a charge is refused with
`authentication_required`. So a charge **stopped dead, waiting for a human**
looked identical to a charge **in flight, about to settle**. Same status, same
table. "Which of my payments need me to act?" had no answer, and the UI had
nothing to render a pending state from.

The value was already understood elsewhere in the same file:
`check_low_balance_and_topup` suppresses auto-top-up while an intent is
`('created', 'processing', 'requires_action')`. The vocabulary existed; the
writer never used it.

**This is a defect I introduced.** `9762d5d` added the SCA branch and reused
the registration helper without noticing it stamped every row `created`. The
recovery worked — the intent is registered, so the credit lands if the customer
ever completes the challenge — but nothing could *tell them to*.

**No `client_secret` here, deliberately.** Completing a challenge needs one.
A secret that can confirm a payment does not belong in a list endpoint that
returns twenty rows; it belongs in a single-intent resume call, scoped to
`billing:write`, which is still to be built. This endpoint answers "what is
waiting, and for how much" — which is what the wallet UI and an agent need to
say something true.
"""

from __future__ import annotations

import inspect
import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest  # noqa: E402


@pytest.fixture
def intent_probe():
    """A customer id whose `payment_intents` rows are removed afterwards."""
    from db import _get_pg_pool

    customer_id = f"cus_scaprobe_{uuid.uuid4().hex[:10]}"
    yield customer_id
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.execute("DELETE FROM payment_intents WHERE customer_id = %s", (customer_id,))
        conn.commit()


def test_an_sca_declined_charge_is_recorded_as_requiring_action(intent_probe):
    """The defect, at its source.

    Registered as `created`, an SCA-pending charge is indistinguishable from one
    that is simply in flight — and the difference is whether a human has to do
    something.
    """
    from billing import get_billing_engine
    from db import _get_pg_pool

    engine = get_billing_engine()
    stripe_intent_id = f"pi_{uuid.uuid4().hex[:20]}"
    engine._register_payment_intent(
        intent_id=f"pi_topup_{uuid.uuid4().hex[:16]}",
        customer_id=intent_probe,
        amount_cents=1000,
        stripe_intent_id=stripe_intent_id,
        description="probe (awaiting cardholder verification)",
        created_at=0.0,
        status="requires_action",
    )

    pool = _get_pg_pool()
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT status FROM payment_intents WHERE stripe_intent_id = %s",
            (stripe_intent_id,),
        ).fetchone()

    assert row is not None
    assert row[0] == "requires_action", (
        f"the intent was recorded as {row[0]!r}; an SCA-pending charge that "
        "reads as 'created' cannot be told apart from one in flight"
    )


def test_an_ordinary_charge_is_still_recorded_as_created(intent_probe):
    """The calibration control.

    A change that stamped *everything* `requires_action` would satisfy the test
    above and make the new endpoint list every charge ever made as needing the
    customer's attention.
    """
    from billing import get_billing_engine
    from db import _get_pg_pool

    engine = get_billing_engine()
    stripe_intent_id = f"pi_{uuid.uuid4().hex[:20]}"
    engine._register_payment_intent(
        intent_id=f"pi_topup_{uuid.uuid4().hex[:16]}",
        customer_id=intent_probe,
        amount_cents=2000,
        stripe_intent_id=stripe_intent_id,
        description="probe ordinary",
        created_at=0.0,
    )

    pool = _get_pg_pool()
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT status FROM payment_intents WHERE stripe_intent_id = %s",
            (stripe_intent_id,),
        ).fetchone()

    assert row[0] == "created", (
        "an ordinary charge is being recorded as requiring action; every "
        "payment would appear to need the cardholder"
    )


def test_the_sca_branch_marks_the_intent_it_registers():
    """Structural: the runtime tests above pass whatever the SCA path does.

    They call the helper directly. If `charge_saved_card`'s decline branch stops
    passing the status, SCA-pending charges silently become invisible again and
    nothing here would notice.
    """
    from billing import BillingEngine

    source = inspect.getsource(BillingEngine.charge_saved_card)
    assert 'status="requires_action"' in source, (
        "the SCA decline branch no longer records the intent as requiring "
        "action; a stopped charge would look like one in flight again"
    )


def test_the_listing_endpoint_returns_no_client_secret():
    """The one thing this endpoint must never do.

    Completing a challenge needs a `client_secret`, and it is tempting to
    return it here so the UI has everything in one call. A secret that confirms
    a payment, returned in a list of twenty, is a much larger surface than the
    single-intent resume it belongs to.
    """
    import routes.billing as billing_routes

    source = inspect.getsource(billing_routes.api_billing_pending_verification)
    assert "client_secret" not in source.replace("no `client_secret`", ""), (
        "the pending-verification listing returns a client_secret; that belongs "
        "in a single-intent resume call behind billing:write"
    )


def test_the_listing_endpoint_is_authorized_not_merely_authenticated():
    """Same rule as the eleven routes fixed in `3d46b0e`.

    This reads payment state, so a credential narrowed to exclude billing must
    not reach it.
    """
    import routes.billing as billing_routes

    source = inspect.getsource(billing_routes.api_billing_pending_verification)
    assert "_require_customer_access" in source, (
        "the pending-verification listing does not go through the guard that "
        "checks scope and ownership"
    )
