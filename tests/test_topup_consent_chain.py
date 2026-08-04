"""Topping up is gated by consent already given, not by a fresh approval.

`top_up_wallet` charges a real card without asking a human each time. That is a
deliberate decision, and it is only safe because three things are true at once.
Each is cheap to remove by accident while refactoring, and removing any one of
them turns "the user already agreed" into "an agent can charge a card".

1. **The card was added in the dashboard.** A first-party page is the only place
   card data is handled, and adding it is the payment mandate. The API cannot
   add a card; if none is saved, the top-up refuses with `no_saved_cards`.
2. **`billing:write` was granted deliberately.** The Quick Connect token this
   product tells users to paste carries `billing:read` and not `billing:write`,
   so the default credential cannot top up at all. Obtaining the capability
   means passing a consent screen that says what it does.
3. **Stripe enforces the rest** — Radar, issuer limits, insufficient funds, SCA.

What is *not* here, on purpose: a per-transaction approval, and a spend ceiling
of our own. The first re-decides what the user decided twice and would make
"never leave the terminal" false for the commonest action in the phase. The
second duplicates enforcement the processor does better, and two ceilings that
disagree is worse than one that works. `mcp_client_policies.per_action_max_micros`
is nullable and NULL already means "no ceiling" — a deployment that wants one
sets it.

The risk this design accepts, stated plainly: an agent holding `billing:write`
can move money **into** the user's own wallet from the user's own card. It
cannot move money out, to anywhere, ever. That asymmetry is what makes the
trade reasonable.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("XCELSIOR_ENV", "test")

import routes.billing as billing_routes  # noqa: E402


def test_the_topup_route_requires_billing_write():
    """Link 2. Without this the Quick Connect token could charge cards."""
    source = inspect.getsource(billing_routes.api_billing_manual_topup)
    assert '_require_scope(user, "billing:write")' in source, (
        "the top-up route no longer requires billing:write — the default "
        "connector token could charge a saved card"
    )


def test_the_topup_route_refuses_an_anonymous_caller():
    """Link 2, lower bound."""
    source = inspect.getsource(billing_routes.api_billing_manual_topup)
    assert 'raise HTTPException(401, "Not authenticated")' in source


def test_a_card_must_already_be_saved():
    """Link 1. The API cannot add a card, so there is nothing to charge.

    `resolve_payment_method` returning `no_saved_cards` is what makes "cards are
    added in the dashboard" enforceable rather than merely intended.
    """
    from billing import get_billing_engine

    engine = get_billing_engine()
    source = inspect.getsource(engine.resolve_payment_method)
    assert "no_saved_cards" in source, (
        "resolve_payment_method no longer distinguishes 'no cards saved'; a "
        "top-up against an account with no card would fail obscurely instead of "
        "telling the user to add one in the dashboard"
    )


def test_no_route_can_add_a_card():
    """Link 1, the part that matters most.

    If the API ever grows a card-creation endpoint, PAN handling comes with it
    and the consent story above stops being true. `setup-intent` is the
    permitted shape: it returns a client_secret for the *dashboard* to complete,
    and never accepts a card number.
    """
    source = inspect.getsource(billing_routes)
    for forbidden in ("card_number", '"number"', "cvc", "exp_month="):
        assert forbidden not in source, (
            f"routes/billing.py mentions {forbidden!r} — if the API has started "
            "accepting card details, card data is no longer confined to the "
            "dashboard and this consent model does not hold"
        )


def test_the_amount_is_bounded():
    """Not a policy ceiling — a typo guard.

    Distinct from a spend limit: this stops `10000` becoming `1000000` in a
    malformed tool call. Stripe would decline it anyway; failing before the
    network is cheaper and clearer.
    """
    field = billing_routes.ManualTopupRequest.model_fields["amount_cad"]
    meta = str(field.metadata)
    assert "gt=0" in meta or "Gt(gt=0)" in meta, "a top-up may not be zero or negative"
    assert "10000" in meta, "the sanity bound on amount_cad is gone"


def test_the_charge_is_idempotency_keyed():
    """A retried call must not charge twice.

    Stripe returns the original intent for a repeated key, which is the only
    thing standing between a network blip and a double charge.
    """
    source = inspect.getsource(billing_routes.api_billing_manual_topup)
    assert "idempotency_key" in source
    assert "Idempotency-Key" in source, (
        "the route no longer honours a client-supplied Idempotency-Key, so an "
        "agent retrying a timed-out call cannot make it safe"
    )
