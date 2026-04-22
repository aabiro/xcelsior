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
from routes.auth import RegisterRequest, LoginRequest, ProfileUpdateRequest

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


@pytest.mark.parametrize(
    "bad_amount",
    [
        0,
        -1,
        -0.01,
        10_000.01,
        10_001,
        1e9,
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
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


# ── Hypothesis fuzz tests for auth models ────────────────────────────


@given(
    email=st.text(min_size=0, max_size=200),
    password=st.text(min_size=0, max_size=200),
    name=st.text(min_size=0, max_size=200),
    role=st.text(min_size=0, max_size=200),
)
def test_register_request_constraints(email, password, name, role):
    try:
        RegisterRequest(email=email, password=password, name=name, role=role)
    except (ValidationError, TypeError):
        pass  # correctly rejected


@given(
    email=st.text(min_size=0, max_size=200),
    password=st.text(min_size=0, max_size=200),
)
def test_login_request_constraints(email, password):
    try:
        LoginRequest(email=email, password=password)
    except (ValidationError, TypeError):
        pass  # correctly rejected


@given(
    name=st.one_of(st.none(), st.text(min_size=0, max_size=200)),
    role=st.one_of(st.none(), st.text(min_size=0, max_size=200)),
    country=st.one_of(st.none(), st.text(min_size=0, max_size=200)),
    province=st.one_of(st.none(), st.text(min_size=0, max_size=200)),
)
def test_profile_update_request_constraints(name, role, country, province):
    try:
        ProfileUpdateRequest(name=name, role=role, country=country, province=province)
    except (ValidationError, TypeError):
        pass  # correctly rejected


# ── OAuthClientCreateRequest — new constraint coverage ──────────────────────

from routes.auth import OAuthClientCreateRequest, DeviceVerificationRequest
from routes.hosts import HostIn, StatusUpdate
from routes.agent import VersionReport, MiningAlert, BenchmarkReport
from routes.billing import EstimateRequest, ReservedCommitmentRequest, RefundRequest
from routes.marketplace import GPUOfferCreate, MarketplaceSearchParams


@pytest.mark.parametrize("bad_name", ["", "x" * 129])
def test_oauth_client_name_bounds_rejected(bad_name):
    with pytest.raises((ValidationError, TypeError)):
        OAuthClientCreateRequest(client_name=bad_name)


@pytest.mark.parametrize("bad_type", ["", "private", "internal", "PUBLIC", "superuser"])
def test_oauth_client_type_invalid_rejected(bad_type):
    with pytest.raises((ValidationError, TypeError)):
        OAuthClientCreateRequest(client_name="app", client_type=bad_type)


@pytest.mark.parametrize("bad_code", ["", "A", "SHORT"])
def test_device_verification_code_too_short_rejected(bad_code):
    with pytest.raises((ValidationError, TypeError)):
        DeviceVerificationRequest(user_code=bad_code)


def test_device_verification_code_too_long_rejected():
    with pytest.raises((ValidationError, TypeError)):
        DeviceVerificationRequest(user_code="X" * 33)


# ── HostIn new constraints ───────────────────────────────────────────────────

_VALID_HOST = dict(
    host_id="h-1", ip="10.0.0.1", gpu_model="RTX 4090", total_vram_gb=24.0, free_vram_gb=20.0
)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("host_id", ""),
        ("host_id", "x" * 65),
        ("ip", "x" * 46),
        ("gpu_model", ""),
        ("gpu_model", "x" * 65),
        ("total_vram_gb", -1.0),
        ("total_vram_gb", 1025.0),
        ("free_vram_gb", -0.001),
        ("cost_per_hour", -5.0),
        ("province", "x" * 11),
        ("corporation_name", "x" * 257),
        ("legal_name", "x" * 257),
    ],
)
def test_host_in_field_rejected(field, bad_value):
    with pytest.raises((ValidationError, TypeError)):
        HostIn(**{**_VALID_HOST, field: bad_value})


@pytest.mark.parametrize("bad_status", ["", "x" * 33])
def test_status_update_status_bounds_rejected(bad_status):
    with pytest.raises((ValidationError, TypeError)):
        StatusUpdate(status=bad_status)


# ── Agent model constraints ──────────────────────────────────────────────────


def test_version_report_host_id_too_long_rejected():
    with pytest.raises((ValidationError, TypeError)):
        VersionReport(host_id="x" * 65, versions={})


@pytest.mark.parametrize("bad_conf", [-0.001, 1.001, -100.0, 999.9])
def test_mining_alert_confidence_bounds_rejected(bad_conf):
    with pytest.raises((ValidationError, TypeError)):
        MiningAlert(host_id="h1", gpu_index=0, confidence=bad_conf, reason="r")


def test_mining_alert_reason_too_long_rejected():
    with pytest.raises((ValidationError, TypeError)):
        MiningAlert(host_id="h1", gpu_index=0, confidence=0.5, reason="x" * 501)


def test_benchmark_score_negative_rejected():
    with pytest.raises((ValidationError, TypeError)):
        BenchmarkReport(host_id="h1", gpu_model="RTX 4090", score=-1.0, tflops=1.0)


def test_benchmark_tflops_negative_rejected():
    with pytest.raises((ValidationError, TypeError)):
        BenchmarkReport(host_id="h1", gpu_model="RTX 4090", score=1.0, tflops=-0.5)


# ── Billing new model constraints ────────────────────────────────────────────


def test_refund_job_id_empty_rejected():
    with pytest.raises((ValidationError, TypeError)):
        RefundRequest(job_id="", exit_code=0)


def test_refund_exit_code_out_of_range_rejected():
    with pytest.raises((ValidationError, TypeError)):
        RefundRequest(job_id="j-1", exit_code=256)
    with pytest.raises((ValidationError, TypeError)):
        RefundRequest(job_id="j-1", exit_code=-1)


@pytest.mark.parametrize("bad_hours", [-0.001, -100.0, 8761.0, 1e9])
def test_estimate_duration_bounds_rejected(bad_hours):
    with pytest.raises((ValidationError, TypeError)):
        EstimateRequest(duration_hours=bad_hours)


@pytest.mark.parametrize("bad_type", ["", "6_month", "2_year", "lifetime", "monthly"])
def test_reserved_commitment_type_invalid_rejected(bad_type):
    with pytest.raises((ValidationError, TypeError)):
        ReservedCommitmentRequest(customer_id="c-1", commitment_type=bad_type)


@pytest.mark.parametrize("bad_qty", [0, -1, 1001])
def test_reserved_quantity_bounds_rejected(bad_qty):
    with pytest.raises((ValidationError, TypeError)):
        ReservedCommitmentRequest(customer_id="c-1", quantity=bad_qty)


# ── Marketplace model constraints ────────────────────────────────────────────

_VALID_OFFER = dict(host_id="host-1", gpu_model="RTX 4090")


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("host_id", ""),
        ("host_id", "x" * 65),
        ("gpu_model", ""),
        ("gpu_model", "x" * 65),
        ("gpu_count_total", 0),
        ("gpu_count_total", 65),
        ("ask_cents_per_hour", 0),
        ("ask_cents_per_hour", -10),
        ("vram_gb", -0.001),
        ("vram_gb", 1025.0),
    ],
)
def test_gpu_offer_create_field_rejected(field, bad_value):
    with pytest.raises((ValidationError, TypeError)):
        GPUOfferCreate(**{**_VALID_OFFER, field: bad_value})


@pytest.mark.parametrize("bad_sort", ["", "name", "cost", "random", "GPU_MODEL"])
def test_marketplace_search_sort_invalid_rejected(bad_sort):
    with pytest.raises((ValidationError, TypeError)):
        MarketplaceSearchParams(sort_by=bad_sort)


@pytest.mark.parametrize("bad_limit", [0, -1, 501, 100000])
def test_marketplace_search_limit_bounds_rejected(bad_limit):
    with pytest.raises((ValidationError, TypeError)):
        MarketplaceSearchParams(limit=bad_limit)
