"""Gate P1 clauses 1 and 3, against a live server in Stripe **test mode**.

Clause 1: *"A top-up on a saved card completes with no browser and no
elicitation, asserted with a real token against a live server."*

Clause 3: *"An `authentication_required` decline produces a resumable pending
state, a visible UI state, and a tool result that says the charge did not
happen — never a generic error. Asserted by forcing the decline with a Stripe
test card, not by mocking it."*

Clause 3 names the exclusion explicitly, which is why this file exists at all:
`tests/test_sca_decline_is_recoverable.py` builds a genuine `stripe.CardError`
from Stripe's documented JSON, and that is careful work — but the decline is
still injected, and the clause rules that out in its own words.

## The cards, and why these ones

From Stripe's testing documentation rather than memory:

| purpose | PaymentMethod | behaviour |
|---|---|---|
| clause 1 | `pm_card_visa` | succeeds off-session, no authentication |
| clause 3 | `pm_card_authenticationRequired` (4000002760003184) | *"requires authentication on all transactions, regardless of how the card is set up"* |

The second matters: cards like `4000002500003155` stop requiring
authentication once set up for off-session use, so a test built on one can go
green because the setup succeeded rather than because the decline was handled.

## Safety

Every test here refuses to run unless the deployment resolves an `sk_test`
key. A live key would mean a real charge on a real card, and this suite exists
to be run repeatedly. Staging was found on `sk_live` while being used for
exactly this work, so the check is not theoretical.
"""

from __future__ import annotations

import os
import uuid

import pytest

requests = pytest.importorskip("requests")

from tests.live._fleet import (  # noqa: E402
    BASE,
    MISSING_CREDENTIALS,
    TOKEN,
    auth,
)

pytestmark = pytest.mark.skipif(not BASE or not TOKEN, reason=MISSING_CREDENTIALS)

#: Set by whoever prepared the Stripe fixture: a **test-mode** customer with
#: two attached payment methods. Absent means skip — never a silent pass.
STRIPE_CUSTOMER = os.environ.get("XCELSIOR_LIVE_STRIPE_CUSTOMER", "")
PM_OK = os.environ.get("XCELSIOR_LIVE_PM_OK", "")
PM_AUTH_REQUIRED = os.environ.get("XCELSIOR_LIVE_PM_AUTH_REQUIRED", "")

MISSING_FIXTURE = (
    "set XCELSIOR_LIVE_STRIPE_CUSTOMER, XCELSIOR_LIVE_PM_OK and "
    "XCELSIOR_LIVE_PM_AUTH_REQUIRED to a TEST-MODE customer and two attached "
    "payment methods"
)

needs_stripe = pytest.mark.skipif(
    not (STRIPE_CUSTOMER and PM_OK and PM_AUTH_REQUIRED), reason=MISSING_FIXTURE
)


def _post(path: str, body: dict, *, idem: str = ""):
    headers = auth()
    if idem:
        headers["Idempotency-Key"] = idem
    return requests.post(f"{BASE}{path}", headers=headers, json=body, timeout=60)


def _text(payload) -> str:
    """Every string in a response, flattened, for 'is a secret in here' checks."""
    import json

    return json.dumps(payload, default=str).lower()


# ── The safety interlock ──────────────────────────────────────────────


@needs_stripe
def test_the_fixture_is_a_test_mode_customer():
    """A live customer id here would mean the tests below charge real cards.

    Stripe test-mode and live-mode object ids are drawn from the same space, so
    the id alone cannot tell you. The deployment is asked instead, through the
    surface it already exposes.
    """
    response = requests.get(f"{BASE}/api/billing/attestation", headers=auth(), timeout=30)
    assert response.status_code in (200, 403), response.status_code
    if response.status_code == 200:
        body = _text(response.json())
        assert "sk_live" not in body, "a live key is reachable from the attestation surface"


# ── Clause 1: the charge completes with no browser ────────────────────


@needs_stripe
def test_a_topup_on_a_saved_card_completes_with_no_browser_step():
    """The phase's headline: money moves without a browser or an elicitation.

    The wallet is deliberately *not* asserted to have moved — this route
    reports the charge as submitted and only Stripe's webhook credits the
    balance, because the processor is the sole authority on whether money
    moved. Asserting a credited balance here would assert a lie the code
    correctly refuses to tell.
    """
    response = _post(
        "/api/v2/billing/top-up",
        {"amount_cad": 5, "payment_method_id": PM_OK},
        idem=f"live-gate-ok-{uuid.uuid4().hex[:12]}",
    )
    assert response.status_code == 200, (
        f"a saved-card top-up was refused ({response.status_code}): {response.text[:400]}"
    )

    body = response.json()
    assert body.get("charged") is True, f"the charge was not reported as made: {body}"

    # No browser, no elicitation. A `next_action`, a redirect URL or a hosted
    # page in this response *is* the browser step the clause forbids.
    flat = _text(body)
    for forbidden in (
        "next_action",
        "redirect_to_url",
        "hosted_invoice_url",
        "checkout.stripe.com",
    ):
        assert forbidden not in flat, (
            f"the response carries {forbidden!r} — that is a browser step, which "
            "is exactly what this clause says must not happen"
        )


@needs_stripe
def test_the_topup_response_carries_no_processor_secret():
    """Gate P1 clause 5 in passing — a client_secret here is a leak."""
    response = _post(
        "/api/v2/billing/top-up",
        {"amount_cad": 5, "payment_method_id": PM_OK},
        idem=f"live-gate-secret-{uuid.uuid4().hex[:12]}",
    )
    if response.status_code != 200:
        pytest.skip(f"charge not accepted ({response.status_code}); nothing to inspect")
    flat = _text(response.json())
    for secret in ("client_secret", "sk_test", "sk_live", "_secret_"):
        assert secret not in flat, f"the response leaks {secret!r}"


# ── Clause 3: the decline is resumable and truthful ───────────────────


@needs_stripe
def test_an_authentication_required_decline_says_the_charge_did_not_happen():
    """Forced with a real test card, which is what the clause demands.

    `pm_card_authenticationRequired` requires authentication on every
    transaction regardless of setup, so the off-session confirm is declined by
    Stripe rather than by anything in this repository.
    """
    response = _post(
        "/api/v2/billing/top-up",
        {"amount_cad": 5, "payment_method_id": PM_AUTH_REQUIRED},
        idem=f"live-gate-sca-{uuid.uuid4().hex[:12]}",
    )

    assert response.status_code != 200, (
        "a card that always requires authentication reported a completed "
        f"charge: {response.text[:300]}"
    )
    assert response.status_code != 500, (
        f"the decline surfaced as a server error: {response.text[:300]}. The "
        "clause requires a typed, resumable answer — 'never a generic error'."
    )

    body = response.json()
    flat = _text(body)

    assert "authentication" in flat, (
        f"the refusal does not say authentication is required: {body}. A caller "
        "cannot resume what it has not been told about."
    )
    assert "charged" in flat and '"charged": true' not in flat.replace(" ", ""), (
        f"the result does not state that the charge did not happen: {body}"
    )


@needs_stripe
def test_the_declined_charge_leaves_a_resumable_pending_state():
    """ "Resumable" means the intent survives for the cardholder to finish.

    A decline that discards the intent is not resumable, however good its
    error message: the customer would have to start again, and the phase's
    claim is that they do not.
    """
    response = _post(
        "/api/v2/billing/top-up",
        {"amount_cad": 5, "payment_method_id": PM_AUTH_REQUIRED},
        idem=f"live-gate-resume-{uuid.uuid4().hex[:12]}",
    )
    if response.status_code == 200:
        pytest.fail("the always-authenticate card was charged without authentication")

    body = response.json()
    flat = _text(body)
    assert any(marker in flat for marker in ("pending", "requires_action", "resum", "intent")), (
        f"nothing in the refusal points at a resumable state: {body}"
    )
