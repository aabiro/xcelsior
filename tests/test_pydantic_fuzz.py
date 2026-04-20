"""Property-based fuzz tests for hardened Pydantic input models.

Targets: the request models that were tightened with ``Field(gt=0, le=10000)``,
``max_length``, and regex ``pattern`` constraints in the input-validation
hardening sweep (routes/billing.py and routes/teams.py).

Strategy: generate arbitrary JSON-ish dicts and assert each model EITHER:
  a) raises ``ValidationError`` / ``TypeError`` (garbage correctly rejected), OR
  b) succeeds — in which case every hardened constraint is verified to hold
     on the resulting instance. Garbage must NEVER be silently accepted.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies as st
from pydantic import ValidationError

from routes.billing import (
    DepositRequest,
    PaymentIntentRequest,
    PayPalCreateOrderRequest,
    PayPalCaptureRequest,
)
from routes.teams import (
    AddTeamMemberRequest,
    CreateTeamRequest,
    UpdateTeamMemberRoleRequest,
)


# ── Garbage strategy: arbitrary dict with mixed value types ─────────

_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(min_size=0, max_size=80),
)

garbage_dict = st.dictionaries(
    keys=st.text(min_size=0, max_size=30),
    values=st.one_of(_scalar, st.lists(_scalar, max_size=3)),
    max_size=10,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _assert_finite_positive_bounded(x: float, hi: float) -> None:
    assert not math.isnan(x), f"NaN leaked past validation: {x!r}"
    assert not math.isinf(x), f"Inf leaked past validation: {x!r}"
    assert 0 < x <= hi, f"out of bounds ({hi}): {x!r}"


# ── Fuzz tests — one per hardened model ─────────────────────────────


@given(payload=garbage_dict)
@settings(deadline=None, max_examples=200)
def test_deposit_request_rejects_or_validates(payload):
    try:
        obj = DepositRequest(**payload)
    except (ValidationError, TypeError):
        return
    _assert_finite_positive_bounded(obj.amount_cad, 10000)
    assert isinstance(obj.description, str)
    assert len(obj.description) <= 500


@given(payload=garbage_dict)
@settings(deadline=None, max_examples=200)
def test_payment_intent_request_rejects_or_validates(payload):
    try:
        obj = PaymentIntentRequest(**payload)
    except (ValidationError, TypeError):
        return
    _assert_finite_positive_bounded(obj.amount_cad, 10000)
    assert 1 <= len(obj.customer_id) <= 128
    assert len(obj.description) <= 500


@given(payload=garbage_dict)
@settings(deadline=None, max_examples=200)
def test_paypal_create_order_rejects_or_validates(payload):
    try:
        obj = PayPalCreateOrderRequest(**payload)
    except (ValidationError, TypeError):
        return
    _assert_finite_positive_bounded(obj.amount_cad, 10000)
    assert 1 <= len(obj.customer_id) <= 128


@given(payload=garbage_dict)
@settings(deadline=None, max_examples=200)
def test_paypal_capture_rejects_or_validates(payload):
    try:
        obj = PayPalCaptureRequest(**payload)
    except (ValidationError, TypeError):
        return
    assert 1 <= len(obj.customer_id) <= 128
    assert 1 <= len(obj.order_id) <= 128


@given(payload=garbage_dict)
@settings(deadline=None, max_examples=200)
def test_create_team_rejects_or_validates(payload):
    try:
        obj = CreateTeamRequest(**payload)
    except (ValidationError, TypeError):
        return
    assert 1 <= len(obj.name) <= 128
    assert obj.plan in {"free", "pro", "enterprise"}


@given(payload=garbage_dict)
@settings(deadline=None, max_examples=200)
def test_add_team_member_rejects_or_validates(payload):
    try:
        obj = AddTeamMemberRequest(**payload)
    except (ValidationError, TypeError):
        return
    assert 3 <= len(obj.email) <= 254
    assert obj.role in {"admin", "member", "viewer"}


@given(payload=garbage_dict)
@settings(deadline=None, max_examples=200)
def test_update_team_member_role_rejects_or_validates(payload):
    try:
        obj = UpdateTeamMemberRoleRequest(**payload)
    except (ValidationError, TypeError):
        return
    assert obj.role in {"admin", "member", "viewer"}


# ── Explicit known-bad cases (must ALWAYS raise) ────────────────────


@pytest.mark.parametrize("bad_amount", [
    0, -1, -0.01, 10_000.01, 10_001, 1e9, float("nan"), float("inf"), -float("inf"),
])
def test_deposit_request_known_bad_amounts_always_rejected(bad_amount):
    with pytest.raises((ValidationError, TypeError)):
        DepositRequest(amount_cad=bad_amount)


@pytest.mark.parametrize("bad_plan", ["", "admin", "Free", "FREE", "basic", " ", "pro "])
def test_create_team_bad_plan_always_rejected(bad_plan):
    with pytest.raises((ValidationError, TypeError)):
        CreateTeamRequest(name="x", plan=bad_plan)


@pytest.mark.parametrize("bad_role", ["", "owner", "Admin", "ADMIN", "guest", " member"])
def test_update_team_member_bad_role_always_rejected(bad_role):
    with pytest.raises((ValidationError, TypeError)):
        UpdateTeamMemberRoleRequest(role=bad_role)


def test_create_team_long_name_rejected():
    with pytest.raises((ValidationError, TypeError)):
        CreateTeamRequest(name="x" * 129, plan="free")


def test_create_team_empty_name_rejected():
    with pytest.raises((ValidationError, TypeError)):
        CreateTeamRequest(name="", plan="free")


def test_payment_intent_long_description_rejected():
    with pytest.raises((ValidationError, TypeError)):
        PaymentIntentRequest(
            customer_id="c",
            amount_cad=100,
            description="x" * 501,
        )
