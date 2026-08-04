"""A path exempt from bearer auth must authenticate by something else.

`PUBLIC_PATHS` is how this codebase declares "this endpoint does not take a
bearer". For a webhook that is correct — Stripe and PayPal send a signature, not a
token — but the exemption is only safe while the handler actually verifies that
signature. Nothing checked that, so the exemption rested on a claim.

It matters concretely. On 2026-08-04, `/api/connect/webhooks` was **not** exempt,
so every Stripe delivery was answered 401 by the middleware before the handler
ran. That masked a second defect underneath it: the signing secret was set in
`.env` and mapped into no container, so had the request arrived it would have got
503. Two failures stacked, neither visible, and the endpoint unreachable for as
long as it had existed.

Adding the exemption fixes the 401. This file is what stops the exemption becoming
an open endpoint: for every exempt webhook path, an unsigned request must be
refused.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")

from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402
from routes._deps import PUBLIC_PATHS  # noqa: E402

client = TestClient(app)

#: Exempt paths that are webhooks, and the header that carries their credential.
#: Listed explicitly rather than pattern-matched on "webhook": a path that stops
#: being a webhook, or a new one that arrives, should require an edit here with a
#: decision attached.
WEBHOOK_EXEMPTIONS = {
    "/api/connect/webhooks": "stripe-signature",
    "/api/billing/paypal/webhook": None,  # verified against PayPal's API, no header
    "/api/providers/webhook": None,  # shared-secret body field
}


def test_every_webhook_path_we_exempt_is_one_we_know_about():
    """A new exemption has to come here and say how it authenticates."""
    exempt_webhooks = {p for p in PUBLIC_PATHS if "webhook" in p}
    unaccounted = sorted(exempt_webhooks - set(WEBHOOK_EXEMPTIONS))
    assert not unaccounted, (
        f"{unaccounted} are exempt from bearer auth and undocumented here. State "
        "what authenticates them, or remove the exemption — an exempt path with no "
        "recorded second credential is an open endpoint."
    )


def test_the_stripe_webhook_is_exempt_at_all():
    """The 401 defect, asserted directly.

    Without this, restoring the old behaviour is a one-line deletion that no test
    notices, and Stripe deliveries silently start failing again.
    """
    assert "/api/connect/webhooks" in PUBLIC_PATHS, (
        "/api/connect/webhooks is not exempt from bearer auth, so every Stripe "
        "delivery is answered 401 before the handler runs"
    )


def test_the_stripe_webhook_refuses_an_unsigned_request():
    """The exemption is not an open door.

    A 400 or 503 is a refusal and both are correct: 400 for the missing signature,
    503 if no signing secret is configured in this environment. What must never
    appear is 2xx.
    """
    r = client.post("/api/connect/webhooks", json={"id": "evt_probe", "type": "probe"})
    assert r.status_code != 200, (
        "an unsigned webhook was ACCEPTED. The exemption from bearer auth is only "
        f"safe because the signature is the credential. Body: {r.text[:200]!r}"
    )
    assert r.status_code in (400, 503), (
        f"expected a refusal naming the missing signature, got {r.status_code}: "
        f"{r.text[:200]!r}"
    )


def test_the_stripe_webhook_refuses_a_forged_signature():
    """A present-but-wrong signature must fail verification, not pass on presence."""
    r = client.post(
        "/api/connect/webhooks",
        json={"id": "evt_probe", "type": "probe"},
        headers={"stripe-signature": "t=1,v1=deadbeef"},
    )
    assert r.status_code != 200, (
        "a forged signature was ACCEPTED — the handler is checking that the header "
        f"exists rather than that it verifies. Body: {r.text[:200]!r}"
    )


@pytest.mark.parametrize("path", sorted(WEBHOOK_EXEMPTIONS))
def test_no_exempt_webhook_acts_on_an_unauthenticated_post(path):
    """The safety property is that nothing was *acted on*, not the status code.

    `/api/providers/webhook` answers **200** to an unverified event, with
    `handled: false` in the body — it verified, refused, and reported delivery.
    Nothing is trusted and no state changes, so the exemption is safe; but 200
    tells Stripe the event was delivered, so Stripe stops retrying an event that
    was never processed. That is a separate defect from this one and is filed
    rather than changed here, because altering the status code changes retry
    behaviour on a live money path and deserves to be deliberate.

    So this asserts what actually matters: an unauthenticated post is not handled.
    """
    r = client.post(path, json={})
    if r.status_code != 200:
        return  # an outright refusal is the cleanest form of not-handled
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    assert body.get("handled") is not True, (
        f"{path} is exempt from bearer auth and reported HANDLING an "
        f"unauthenticated event. Body: {r.text[:300]!r}"
    )
