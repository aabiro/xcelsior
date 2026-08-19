"""A provider asking "why am I not paid?" must get an answer they can act on.

`get_provider_account` shipped describing itself as returning "what payouts are
enabled, and what is still outstanding before earnings can be paid out". It
returned neither. `provider_accounts` has no such columns, and
`StripeManager.get_provider` fetched `charges_enabled`, `payouts_enabled` and
`requirements.disabled_reason` from Stripe, used them to derive a one-word
`status`, and **threw them away**.

So the surface promised an itemised answer and delivered `restricted` — one word
that collapses "we need your bank account" and "we need a photo of your ID",
neither of which the provider can guess. The data was in the same API response
the whole time.

## The two properties

**The shape is always there.** `payouts` is present on every provider dict, from
`get_provider` and from the admin `list_providers`, so a caller never has to
distinguish "no requirements" from "no such field".

**`checked_live` is load-bearing.** Stripe is not consulted when it is
unconfigured or the account has no id, and the retrieve can fail — it is wrapped
in a `try` that logs and continues. Without the flag, an empty `currently_due`
reads as *"nothing outstanding, you are done"* in precisely the case where
nothing was asked. That is the false-green shape this suite keeps finding, on
the surface that tells someone why they have not been paid.

## What is deliberately not here

The Stripe onboarding link. `POST /api/providers/{id}/resume-onboarding` mints an
AccountLink, and an AccountLink can set the external bank account — it is a
payout-destination-change capability, and Stripe's own guidance is not to
distribute the URL. It stays a browser action behind `providers:write`, for the
same reason `list_pending_verifications` returns the list route and not the
`client_secret` one: a tool response goes into a model's context and into audit
records.

Requirement entries are Stripe's field *names* — `external_account`,
`individual.id_number`. Names, never values. No document, bank number or date of
birth passes through here, which is what makes the block safe to hand a model.
"""

from __future__ import annotations

import json
import pathlib
import re
import time
import uuid

import pytest

from stripe_connect import get_stripe_manager

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Exactly the keys the block carries. Asserted as a set in both directions: an
#: added key must be considered here (is it a value rather than a field name?),
#: and a removed one breaks a description that promises it.
PAYOUT_KEYS = {
    "charges_enabled",
    "payouts_enabled",
    "currently_due",
    "past_due",
    "disabled_reason",
    "checked_live",
}


@pytest.fixture
def provider_row():
    """A provider with no Stripe account id, so nothing live is consulted."""
    provider_id = f"blockers-{uuid.uuid4().hex[:8]}"
    mgr = get_stripe_manager()
    with mgr._conn() as conn:
        conn.execute(
            """INSERT INTO provider_accounts
               (provider_id, provider_type, stripe_account_id, status, email,
                legal_name, country, province, created_at)
               VALUES (%s, 'individual', '', 'onboarding', %s, 'Blockers', 'CA', 'ON', %s)""",
            (provider_id, f"{provider_id}@xcelsior.ca", time.time()),
        )
    yield provider_id
    with mgr._conn() as conn:
        conn.execute("DELETE FROM provider_accounts WHERE provider_id=%s", (provider_id,))


# ── The shape ─────────────────────────────────────────────────────────


def test_the_block_is_present_even_when_stripe_is_never_asked(provider_row):
    provider = get_stripe_manager().get_provider(provider_row)
    assert provider is not None
    assert "payouts" in provider, "a caller cannot tell 'no requirements' from 'no field'"
    assert set(provider["payouts"]) == PAYOUT_KEYS


def test_an_unchecked_account_does_not_read_as_nothing_outstanding(provider_row):
    """The false-green this exists to prevent."""
    payouts = get_stripe_manager().get_provider(provider_row)["payouts"]
    assert payouts["checked_live"] is False
    assert payouts["currently_due"] == []
    assert payouts["payouts_enabled"] is False


def test_the_admin_listing_carries_the_same_shape(provider_row):
    """Two callers, one shape. The listing never goes live — see the comment
    there: one page would become hundreds of Stripe calls."""
    rows = get_stripe_manager().list_providers()
    mine = [r for r in rows if r["provider_id"] == provider_row]
    assert mine, "the fixture's provider is not in the listing"
    assert set(mine[0]["payouts"]) == PAYOUT_KEYS
    assert mine[0]["payouts"]["checked_live"] is False


# ── The live path ─────────────────────────────────────────────────────


def test_requirements_from_stripe_reach_the_caller(provider_row, monkeypatch):
    """The whole point: the itemised answer, not the one-word status."""
    import stripe_connect

    class _Acct:
        def __str__(self):
            return json.dumps(
                {
                    "charges_enabled": False,
                    "payouts_enabled": False,
                    "requirements": {
                        "disabled_reason": "requirements.past_due",
                        "currently_due": ["external_account", "individual.id_number"],
                        "past_due": ["external_account"],
                    },
                }
            )

    class _StripeStub:
        class Account:
            @staticmethod
            def retrieve(_account_id):
                return _Acct()

    mgr = get_stripe_manager()
    with mgr._conn() as conn:
        conn.execute(
            "UPDATE provider_accounts SET stripe_account_id='acct_probe' WHERE provider_id=%s",
            (provider_row,),
        )
    monkeypatch.setattr(stripe_connect, "STRIPE_ENABLED", True)
    monkeypatch.setattr(stripe_connect, "stripe", _StripeStub)

    payouts = mgr.get_provider(provider_row)["payouts"]
    assert payouts["checked_live"] is True
    assert payouts["currently_due"] == ["external_account", "individual.id_number"]
    assert payouts["past_due"] == ["external_account"]
    assert payouts["disabled_reason"] == "requirements.past_due"


def test_a_stripe_failure_reports_unchecked_rather_than_clear(provider_row, monkeypatch):
    """The retrieve is wrapped in a `try` that logs and continues. That is the
    right behaviour — a Stripe outage must not fail the provider's own account
    page — but it must not report a clean bill of health either."""
    import stripe_connect

    class _Exploding:
        class Account:
            @staticmethod
            def retrieve(_account_id):
                raise RuntimeError("stripe is down")

    mgr = get_stripe_manager()
    with mgr._conn() as conn:
        conn.execute(
            "UPDATE provider_accounts SET stripe_account_id='acct_probe' WHERE provider_id=%s",
            (provider_row,),
        )
    monkeypatch.setattr(stripe_connect, "STRIPE_ENABLED", True)
    monkeypatch.setattr(stripe_connect, "stripe", _Exploding)

    payouts = mgr.get_provider(provider_row)["payouts"]
    assert payouts["checked_live"] is False, (
        "a failed Stripe call left checked_live true, so an empty currently_due "
        "now reads as 'nothing outstanding' to whoever asked why they are unpaid"
    )


# ── The description must not promise more than the route returns ──────


def test_the_tool_description_promises_only_fields_that_exist(provider_row):
    """The defect that produced this file, as a guard.

    `get_provider_account`'s description named `payouts_enabled` and "what is
    still outstanding" when neither was in the response. TypeScript cannot catch
    it — the description is prose and the route is Python — so the two are
    compared here, in both directions: a field named in the text must exist in
    the response, and a field in the block must be named in the text.
    """
    text = (ROOT / "mcp" / "src" / "tools" / "descriptions.ts").read_text(encoding="utf-8")
    match = re.search(r"\n  get_provider_account:\n(.*?)\n\n", text, re.S)
    assert match, "get_provider_account's description block was not found"
    description = match.group(1)

    payouts = get_stripe_manager().get_provider(provider_row)["payouts"]
    named = {key for key in PAYOUT_KEYS if key in description}

    missing_from_response = named - set(payouts)
    assert not missing_from_response, (
        f"the description promises {sorted(missing_from_response)}, which the "
        "route does not return — a model will tell a provider it can name what "
        "is outstanding and then have nothing to name"
    )
    unmentioned = set(payouts) - named
    assert not unmentioned, (
        f"{sorted(unmentioned)} is returned and the description never mentions "
        "it, so no model reading tools/list knows to look at it"
    )


# ── The other payout destination ──────────────────────────────────────


def test_the_paypal_route_redacts_what_its_siblings_redact(provider_row):
    """`GET /api/providers/{id}/paypal` returned the identifiers, alone among
    the provider reads.

    The other four `pop()` `stripe_account_id`, `paypal_merchant_id`,
    `paypal_payer_id` and `paypal_tracking_id`. This one returned the PayPal
    three under shorter names — `merchant_id`, `payer_id`, `tracking_id` — so the
    redaction boundary had a hole shaped exactly like the fields it was drawn
    around. Nothing rendered them: the dashboard's own response type does not
    declare two of the three, and no component reads any. It mattered once the
    route went behind `get_paypal_status`, because a tool response lands in model
    context and in audit records.

    The onboarding *state* is what a caller needs. The identifiers are what an
    integration would need to act on the account.
    """
    from paypal_connect import get_paypal_manager

    profile = get_paypal_manager().get_paypal_profile(provider_row)
    assert profile is not None
    # The manager still returns them — redaction is the route's job, so this
    # documents that the sensitive data exists and the boundary is where it
    # stops, rather than implying the store was changed.
    assert "merchant_id" in profile

    import routes.providers as providers_routes

    source = pathlib.Path(providers_routes.__file__).read_text(encoding="utf-8")
    block = source.split("def api_provider_paypal_status", 1)[1].split("\n@router", 1)[0]
    for identifier in ("merchant_id", "payer_id", "tracking_id"):
        assert f'"{identifier}"' in block, (
            f"api_provider_paypal_status no longer redacts {identifier}, so the "
            "PayPal payout destination is readable through get_paypal_status"
        )
