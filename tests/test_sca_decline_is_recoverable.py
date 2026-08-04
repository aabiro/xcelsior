"""An SCA decline must leave the charge recoverable, not lost.

Built against a **real `stripe.CardError`** constructed from the JSON body
Stripe actually returns, not a mock with the attributes this code hopes for.
The distinction caught a defect: the branch originally matched on
`error.code == "authentication_required"`, and Stripe puts that value in
`decline_code` while `code` carries `card_declined`. A hand-rolled mock with
`code="authentication_required"` would have passed and shipped the miss.

Stripe's documented behaviour, which these tests encode:

* the request fails with HTTP 402 and the PaymentIntent status becomes
  `requires_payment_method`
* the declined PaymentIntent is attached to the error
* recovery is the cardholder confirming *that* intent

Why it matters here: `_handle_payment_succeeded` credits a wallet only when it
can match the Stripe intent id to a `payment_intents` row. Before this, an SCA
decline registered nothing — so a customer who completed the challenge was
charged for real and credited nothing. That is `601cb05`'s defect arriving by
the SCA path.

SCA itself should be rare: it is a European (PSD2) requirement, this platform
bills in CAD, and `create_setup_intent` already passes `usage="off_session"`,
which is what lets Stripe claim the merchant-initiated exemption. Rare is not
the same as handled — the failure is silent, and a silent rare failure is worse
than a loud one.
"""

from __future__ import annotations

import os

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest  # noqa: E402

PENDING_INTENT_ID = "pi_3SCAprobe0000000000000"


def _real_card_error() -> Exception:
    """A genuine `stripe.CardError`, built from Stripe's documented body.

    `StripeError.error` is constructed from `json_body["error"]`, so passing the
    real shape exercises the same attribute path production takes.
    """
    import stripe

    return stripe.CardError(
        message="Your card was declined. This transaction requires authentication.",
        param="payment_method",
        code="card_declined",
        http_status=402,
        json_body={
            "error": {
                "type": "card_error",
                "code": "card_declined",
                # The value Stripe actually uses for SCA. Matching only on
                # `code` above would miss it.
                "decline_code": "authentication_required",
                "message": "This transaction requires authentication.",
                "payment_intent": {
                    "id": PENDING_INTENT_ID,
                    "status": "requires_payment_method",
                    "client_secret": f"{PENDING_INTENT_ID}_secret_probe",
                },
            }
        },
    )


def test_the_error_shape_this_code_depends_on_is_real():
    """Pin the accessor, so a library upgrade that moves it fails here.

    Every assertion below rests on `err.error.payment_intent` and
    `err.error.decline_code` existing. If a future `stripe` release renames
    either, this fails with the reason rather than the SCA branch silently
    never firing.
    """
    err = _real_card_error()
    error_object = getattr(err, "error", None)
    assert error_object is not None, "stripe.CardError no longer exposes .error"
    assert getattr(error_object, "decline_code", None) == "authentication_required"
    intent = getattr(error_object, "payment_intent", None)
    assert intent is not None, "stripe no longer attaches the declined PaymentIntent"
    got = intent.get("id") if hasattr(intent, "get") else getattr(intent, "id", "")
    assert got == PENDING_INTENT_ID


def test_all_three_authentication_declines_are_treated_as_recoverable():
    """Stripe has three, not one, and they arrive in sequence.

    A user who ignores the first prompt and lets the agent retry gets
    `authentication_not_handled` — "you tried to proceed without performing the
    required authentication, so the issuer declined again". Handling only
    `authentication_required` sends that second attempt down the generic
    failure path, telling someone who simply has not confirmed yet that
    something broke.

    Read from the route's own constant rather than restated here, so the set
    cannot drift from what the code checks.
    """
    from routes.billing import _AUTHENTICATION_DECLINES

    assert "authentication_required" in _AUTHENTICATION_DECLINES
    assert "authentication_not_handled" in _AUTHENTICATION_DECLINES, (
        "a retry without completing the challenge would be reported as a "
        "failure rather than as 'still needs confirming'"
    )
    assert "mobile_device_authentication_required" in _AUTHENTICATION_DECLINES

    # And the control: an ordinary decline is not in the set, or every failed
    # charge would tell the user to go and authenticate something.
    for ordinary in ("insufficient_funds", "expired_card", "generic_decline"):
        assert ordinary not in _AUTHENTICATION_DECLINES


def test_matching_only_on_code_would_miss_the_sca_decline():
    """The defect the real error shape exposed, kept as a guard.

    `code` is `card_declined`; the SCA signal is in `decline_code`. A branch
    that reads only `code` compiles, runs, and never fires — the charge lands in
    the generic failure path and the customer is told something went wrong
    rather than that their bank wants a confirmation.
    """
    err = _real_card_error()
    assert getattr(err.error, "code", "") == "card_declined"
    assert getattr(err.error, "decline_code", "") == "authentication_required"


def test_the_declined_intent_is_registered_so_the_credit_can_land(monkeypatch):
    """The load-bearing behaviour.

    Stripe raises; `charge_saved_card` must still record the pending intent
    before re-raising, because that row is the only thing connecting the later
    `payment_intent.succeeded` webhook to the wallet it should credit.
    """
    from billing import get_billing_engine

    engine = get_billing_engine()

    class _Raising:
        class PaymentIntent:
            @staticmethod
            def create(**kwargs):
                raise _real_card_error()

    import stripe_connect

    monkeypatch.setattr(stripe_connect, "STRIPE_ENABLED", True, raising=False)
    monkeypatch.setattr(stripe_connect, "stripe", _Raising, raising=False)

    registered: list[dict] = []
    monkeypatch.setattr(
        engine,
        "_register_payment_intent",
        lambda **kw: registered.append(kw),
    )

    with pytest.raises(Exception):
        engine.charge_saved_card(
            "cust_probe",
            25_000_000,
            stripe_customer_id="cus_probe",
            payment_method_id="pm_probe",
            idempotency_key="probe-key",
            description="Wallet top-up",
        )

    assert registered, (
        "an SCA decline registered no payment_intents row; a customer who "
        "completes the challenge would be charged and never credited"
    )
    assert registered[0]["stripe_intent_id"] == PENDING_INTENT_ID, (
        "the registered row does not carry the declined intent id, so the "
        "confirmation webhook cannot match it"
    )
    assert registered[0]["customer_id"] == "cust_probe"


def test_an_ordinary_decline_registers_nothing(monkeypatch):
    """The calibration control.

    A plain `card_declined` with no attached intent must not write a row —
    registering a phantom intent would leave `payment_intents` holding rows no
    webhook will ever confirm, and every one of them looks like money in
    flight.
    """
    import stripe

    from billing import get_billing_engine

    engine = get_billing_engine()

    class _Raising:
        class PaymentIntent:
            @staticmethod
            def create(**kwargs):
                raise stripe.CardError(
                    message="Your card was declined.",
                    param="payment_method",
                    code="card_declined",
                    http_status=402,
                    json_body={
                        "error": {
                            "type": "card_error",
                            "code": "card_declined",
                            "decline_code": "insufficient_funds",
                            "message": "Your card was declined.",
                        }
                    },
                )

    import stripe_connect

    monkeypatch.setattr(stripe_connect, "STRIPE_ENABLED", True, raising=False)
    monkeypatch.setattr(stripe_connect, "stripe", _Raising, raising=False)

    registered: list[dict] = []
    monkeypatch.setattr(
        engine, "_register_payment_intent", lambda **kw: registered.append(kw)
    )

    with pytest.raises(Exception):
        engine.charge_saved_card(
            "cust_probe",
            25_000_000,
            stripe_customer_id="cus_probe",
            payment_method_id="pm_probe",
            idempotency_key="probe-key-2",
            description="Wallet top-up",
        )

    assert not registered, (
        "an ordinary decline registered a payment_intents row; nothing will "
        "ever confirm it and it will read as money in flight forever"
    )
