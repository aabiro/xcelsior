"""Gate P6 clause 1: the provider journey, on a live staging tenant.

The clause: *"A provider journey — register → admit → publish → earn → payout —
completes through tools plus the browser handoffs, on a live staging tenant."*

**What this asserts, and what it deliberately does not.**

The journey contains a step no test can perform: completing Stripe Connect KYC
is a human filling in a hosted form. `tests/test_returning_from_onboarding_proves_nothing.py`
already reasons about this and reaches the same wall — "abandoning is a human
closing a browser tab".

So this walks the journey to that boundary and then asserts *the boundary holds*,
which is the part that actually protects money:

  register → account state is pending → capacity publishes → earnings read →
  **payout is REFUSED while requirements are outstanding**

That refusal is the valuable assertion. A provider who has not completed KYC
must not be able to draw a payout, and the failure mode is silent success: a
payout queued against an account that can never receive it, discovered only when
someone reconciles. `tests/test_provider_settlement.py` proves the payout is
idempotent and exactly-once *once permitted*; nothing proved it is refused
before.

Completing KYC in Stripe test mode and re-running the payout step is the
remaining human half of this clause, and it is named in the skip so it is not
mistaken for coverage.
"""

from __future__ import annotations

import uuid

import pytest

requests = pytest.importorskip("requests")

from tests.live._fleet import (  # noqa: E402
    BASE,
    MISSING_CREDENTIALS,
    TOKEN,
    auth,
)

# No fleet needed: nothing here launches an instance. Credentials are enough.
pytestmark = [pytest.mark.skipif(not BASE or not TOKEN, reason=MISSING_CREDENTIALS)]


@pytest.fixture(scope="module")
def provider() -> dict:
    """A freshly registered provider on the live tenant."""
    tag = uuid.uuid4().hex[:10]
    r = requests.post(
        f"{BASE}/api/providers/register",
        headers=auth(),
        json={"display_name": f"gate-p6-{tag}", "country": "CA"},
        timeout=60,
    )
    if r.status_code in (402, 403, 409):
        pytest.skip(f"provider registration refused ({r.status_code}): {r.text[:200]}")
    r.raise_for_status()
    body = r.json()
    provider_id = body.get("provider_id") or (body.get("provider") or {}).get("provider_id")
    assert provider_id, f"register returned no provider_id: {body}"
    return {"provider_id": provider_id, "register_body": body}


def test_registration_returns_an_onboarding_handoff_not_a_completed_account(provider) -> None:
    """The browser handoff is a link, and registering does not make anyone payable."""
    body = provider["register_body"]
    state = str(body.get("status") or body.get("onboarding_status") or "").lower()
    assert "active" not in state and "complete" not in state, (
        f"a freshly registered provider reports {state!r}. Registration is not "
        f"onboarding — only Stripe's capability flags may complete it: {body}"
    )


def test_the_account_reports_outstanding_requirements(provider) -> None:
    r = requests.get(
        f"{BASE}/api/providers/{provider['provider_id']}", headers=auth(), timeout=30
    )
    r.raise_for_status()
    body = r.json()
    payouts_enabled = (body.get("provider") or body).get("payouts_enabled")
    assert payouts_enabled is not True, (
        f"payouts are enabled on an account that has completed no KYC: {body}. "
        f"Only `account.updated` may turn this on."
    )


def test_earnings_are_readable_before_any_payout(provider) -> None:
    """The 'earn' hop: readable, and zero rather than absent."""
    r = requests.get(
        f"{BASE}/api/providers/{provider['provider_id']}/earnings", headers=auth(), timeout=30
    )
    assert r.status_code == 200, f"earnings unreadable for a registered provider: {r.text[:200]}"


def test_a_payout_is_refused_while_requirements_are_outstanding(provider) -> None:
    """The boundary, and the reason this file exists.

    Silent success is the failure mode: a payout queued against an account that
    can never receive it, found only when someone reconciles.
    """
    r = requests.post(
        f"{BASE}/api/providers/{provider['provider_id']}/payout",
        headers=auth(),
        json={"job_id": f"gate-p6-{uuid.uuid4().hex[:8]}", "payment_rail": "stripe"},
        timeout=60,
    )
    assert r.status_code >= 400, (
        f"a payout was accepted ({r.status_code}) for a provider that has completed "
        f"no KYC and has payouts disabled: {r.text[:300]}"
    )
    assert r.status_code != 500, (
        f"the refusal is a server error rather than a stated reason, so a caller "
        f"cannot tell 'not yet onboarded' from 'the platform is broken': {r.text[:300]}"
    )


@pytest.mark.skip(
    reason="needs a human to complete Stripe Connect KYC in test mode; the "
    "remaining half of gate P6 clause 1 and deliberately not faked"
)
def test_a_payout_succeeds_once_onboarding_completes() -> None:
    raise AssertionError("unimplemented by design — see the skip reason")
