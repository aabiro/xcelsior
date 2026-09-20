"""Raising an unattended spend cap needs approval; lowering one does not.

Gate P1, clause 6. It was the one clause of that gate with no assertion at all
— the behaviour is implemented in `routes/billing.py` and was simply never
exercised, so nothing would have noticed it regressing.

`_auto_topup_widens` is the decision, and the case worth naming is the third:
raising the *threshold* does not raise any single charge, but it fires the
charge sooner and therefore more often. A check that only compared `amount_cad`
would pass its own review and leave the lever open.

This asserts the predicate directly rather than over HTTP. The route also
requires a wallet, a Stripe customer and a saved payment method, none of which
exist without a live processor — and gate P1's headline ("a top-up completes
with no browser, against a live server") is explicitly the thing that cannot be
asserted here. Pretending otherwise by mocking the processor would produce a
green test for the one claim this phase is not entitled to make yet. What is
asserted here is exactly the raise-vs-lower decision, which needs none of that.
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

from routes._deps import _MACHINE_AUTH_TYPES, _is_interactive_human  # noqa: E402
from routes.billing import AutoTopupConfig, _auto_topup_widens  # noqa: E402

BASE = {"enabled": True, "amount_cad": 50.0, "threshold_cad": 10.0}


def _cfg(**over) -> AutoTopupConfig:
    merged = {**BASE, "stripe_payment_method_id": "pm_test", **over}
    return AutoTopupConfig(**merged)


# ── widening: must require approval ──────────────────────────────────

@pytest.mark.parametrize(
    "change, why",
    [
        ({"amount_cad": 75.0}, "a larger single charge"),
        ({"threshold_cad": 25.0}, "fires sooner, so charges more often"),
        ({"amount_cad": 75.0, "threshold_cad": 25.0}, "both at once"),
    ],
)
def test_raising_a_cap_is_a_widening(change, why) -> None:
    assert _auto_topup_widens(BASE, _cfg(**change)) is True, (
        f"{change} was not treated as widening ({why}); an agent could raise "
        "unattended spending with no approval"
    )


def test_enabling_from_off_is_a_widening_whatever_the_amounts() -> None:
    """Off to on widens even when the numbers shrink: 0 unattended becomes some."""
    previous = {**BASE, "enabled": False, "amount_cad": 500.0, "threshold_cad": 400.0}
    assert _auto_topup_widens(previous, _cfg(amount_cad=1.0, threshold_cad=1.0)) is True


# ── narrowing: must not ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "change",
    [
        {"amount_cad": 25.0},
        {"threshold_cad": 5.0},
        {"amount_cad": 25.0, "threshold_cad": 5.0},
        {},  # unchanged is not a widening either
    ],
)
def test_lowering_or_holding_a_cap_is_not_a_widening(change) -> None:
    assert _auto_topup_widens(BASE, _cfg(**change)) is False, (
        f"{change} was treated as widening; lowering a cap would demand an "
        "approval the gate says it must not"
    )


def test_disabling_never_widens_however_large_the_numbers() -> None:
    """`enabled` is tested first for a reason: off charges nothing."""
    assert _auto_topup_widens(BASE, _cfg(enabled=False, amount_cad=10_000.0)) is False


# ── who the approval requirement applies to ──────────────────────────

def test_a_machine_caller_is_not_treated_as_its_own_approval() -> None:
    """The 409 only fires for non-humans; if every caller looked human it never would.

    Iterates `_MACHINE_AUTH_TYPES` rather than listing values, so a machine auth
    type added later is covered here without anyone remembering to. Writing the
    literals out is how this test would quietly stop covering the newest way in
    — the first draft did exactly that and used `agent_key`, which is not a
    value this system produces.
    """
    assert _MACHINE_AUTH_TYPES, "the machine auth-type set is empty; nothing is covered"

    assert _is_interactive_human({"grant_type": "client_credentials"}) is False, (
        "an OAuth client-credentials token counts as a person at the keyboard, "
        "so widening would skip approval entirely"
    )
    for auth_type in sorted(_MACHINE_AUTH_TYPES):
        assert _is_interactive_human({"auth_type": auth_type}) is False, (
            f"auth_type={auth_type!r} counts as a person present at the "
            "keyboard, so widening would skip approval entirely"
        )


def test_a_dashboard_session_is_its_own_approval() -> None:
    """A human clicking save IS the approval — demanding another is ceremony."""
    assert _is_interactive_human({"auth_type": "session"}) is True
