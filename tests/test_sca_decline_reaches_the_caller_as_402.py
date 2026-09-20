"""An SCA decline must reach the caller as "not charged", not as an error.

Gate P1 clause 3 requires this be asserted "by forcing the decline with a
Stripe test card, not by mocking it". **This test does not satisfy that**, and
says so plainly: forcing a real 3DS decline needs a live server and Stripe test
mode, which is exactly what gate P1's headline clause also needs and is the
reason the phase is not done.

What it replaces is weaker still. The only assertion covering this clause was

    assert "authentication_required" in inspect.getsource(fn)

which passes as long as that literal appears anywhere in the function — the
route could map the decline to a 500 and still be green. This asserts the
contract the agent actually sees: HTTP 402, `charged: False`, a resume URL that
names the pending intent, and a message saying the card was not charged.

The processor is faked at its own boundary (`charge_saved_card`), raising a
real `stripe.CardError`. The shape of that error is itself pinned against the
live library by `tests/test_sca_decline_is_recoverable.py`, so this is not
inventing an error shape and then matching it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
import stripe  # noqa: E402

import routes.billing as rb  # noqa: E402

PENDING_INTENT_ID = "pi_test_needs_confirmation"


def _sca_error() -> stripe.error.CardError:
    """A real CardError carrying the decline and the intent, as Stripe sends it."""
    err = stripe.error.CardError(
        message="Your card was declined.",
        param="payment_method",
        code="card_declined",
        json_body={
            "error": {
                "code": "card_declined",
                "decline_code": "authentication_required",
                "payment_intent": {"id": PENDING_INTENT_ID},
            }
        },
    )
    err.payment_intent = {"id": PENDING_INTENT_ID}
    return err


def test_the_route_maps_an_sca_decline_to_402_not_charged(monkeypatch) -> None:
    """Call the route. Assert the contract an agent actually receives."""
    import fastapi

    engine = rb.get_billing_engine()
    monkeypatch.setattr(
        type(engine), "resolve_payment_method",
        # The route checks `ok` and then reads `payment_method` — a stub shaped
        # any other way lands in the 409 "which card?" branch and never reaches
        # the charge at all. The first version of this test did exactly that and
        # asserted against a 409 it had caused itself.
        lambda self, customer_id, **kw: {
            "ok": True,
            "payment_method": {"id": "pm_saved", "last4": "3155", "brand": "visa"},
        },
        raising=True,
    )
    monkeypatch.setattr(
        type(engine), "get_wallet",
        lambda self, customer_id: {"stripe_customer_id": "cus_test", "balance_micros": 0},
        raising=True,
    )

    def refuse(self, *_a, **_k):
        raise _sca_error()

    monkeypatch.setattr(type(engine), "charge_saved_card", refuse, raising=True)
    monkeypatch.setattr(
        rb, "_get_current_user",
        lambda _request: {"user_id": "u-test", "email": "t@x.ca", "auth_type": "session"},
        raising=True,
    )
    monkeypatch.setattr(rb, "_require_scope", lambda *_a, **_k: None, raising=True)

    class _Req:
        headers: dict = {}
        cookies: dict = {}

    with pytest.raises(fastapi.HTTPException) as excinfo:
        rb.api_billing_manual_topup(
            rb.ManualTopupRequest(amount_cad=25.0, payment_method_id="pm_saved"),
            _Req(),
        )

    exc = excinfo.value
    assert exc.status_code == 402, (
        f"an SCA decline surfaced as {exc.status_code}; the agent cannot tell a "
        "challenge from a failure, and 402 is what says 'not charged, resumable'"
    )
    detail = exc.detail
    assert isinstance(detail, dict), f"the 402 detail is not structured: {detail!r}"
    assert detail.get("charged") is False, (
        f"the result does not state the card was NOT charged: {detail!r}"
    )
    assert detail.get("reason") == "authentication_required"
    assert PENDING_INTENT_ID in str(detail.get("resume_url", "")), (
        "the resume link does not name the pending intent, so the dashboard "
        f"cannot resume *this* payment: {detail.get('resume_url')!r}"
    )
    assert "not charged" in str(detail.get("message", "")).lower()


def test_every_authentication_decline_is_recognised_by_the_routes_predicate() -> None:
    """The predicate must cover all three, not just the obvious one."""
    assert rb._AUTHENTICATION_DECLINES, "the decline set is empty"
    for decline in ("authentication_required", "authentication_not_handled",
                    "mobile_device_authentication_required"):
        assert decline in rb._AUTHENTICATION_DECLINES, (
            f"{decline} is not treated as recoverable; a user who simply has not "
            "confirmed yet is told something went wrong"
        )


def test_an_ordinary_decline_is_not_treated_as_recoverable() -> None:
    """If everything looked like SCA, a real decline would offer a useless resume link."""
    for ordinary in ("insufficient_funds", "lost_card", "generic_decline"):
        assert ordinary not in rb._AUTHENTICATION_DECLINES
