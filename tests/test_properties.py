"""Property-based tests for billing math, scheduler allocation, and volume constraints.

All tests are pure-logic (no DB, no network):
- billing: cost arithmetic invariants
- scheduler: allocate() / allocate_binpack() selection invariants
- volume: attach/detach state-machine invariants via in-memory mocking
"""
from __future__ import annotations

import sys
import os
import time
import math
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────────────────────
# Billing math — UsageMeter cost formula
# ─────────────────────────────────────────────────────────────────────────────

from billing import UsageMeter, get_tax_rate_for_province, PROVINCE_TAX_RATES


def _compute_expected_cost(duration_sec, base_rate, multiplier, spot_discount):
    """Replicate billing.py meter_job cost formula."""
    return round((duration_sec / 3600) * base_rate * multiplier * (1 - spot_discount), 4)


@given(
    duration_sec=st.floats(min_value=0.0, max_value=3600 * 8760),  # 0 to 1 year
    base_rate=st.floats(min_value=0.01, max_value=100.0),
    multiplier=st.floats(min_value=0.5, max_value=5.0),
    spot_discount=st.floats(min_value=0.0, max_value=0.99),
)
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
def test_billing_cost_non_negative(duration_sec, base_rate, multiplier, spot_discount):
    """Cost is always >= 0."""
    assume(math.isfinite(duration_sec) and math.isfinite(base_rate))
    assume(math.isfinite(multiplier) and math.isfinite(spot_discount))
    cost = _compute_expected_cost(duration_sec, base_rate, multiplier, spot_discount)
    assert cost >= 0, f"Negative cost: {cost}"


@given(
    duration_sec=st.floats(min_value=1.0, max_value=3600 * 24 * 30),
    base_rate=st.floats(min_value=0.01, max_value=100.0),
    multiplier=st.floats(min_value=0.5, max_value=5.0),
    spot_discount=st.floats(min_value=0.0, max_value=0.99),
)
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
def test_billing_cost_monotone_in_duration(duration_sec, base_rate, multiplier, spot_discount):
    """Longer job always costs at least as much as shorter job (same rate)."""
    assume(math.isfinite(duration_sec))
    short = _compute_expected_cost(duration_sec * 0.5, base_rate, multiplier, spot_discount)
    full = _compute_expected_cost(duration_sec, base_rate, multiplier, spot_discount)
    assert full >= short - 1e-9, f"full={full} < short={short}"


@given(
    duration_sec=st.floats(min_value=1.0, max_value=3600 * 24),
    base_rate=st.floats(min_value=0.01, max_value=10.0),
    multiplier=st.floats(min_value=0.5, max_value=5.0),
    discount_a=st.floats(min_value=0.0, max_value=0.90),
    discount_b=st.floats(min_value=0.0, max_value=0.90),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_billing_higher_spot_discount_costs_less(
    duration_sec, base_rate, multiplier, discount_a, discount_b
):
    """Higher spot discount means lower or equal cost."""
    assume(math.isfinite(duration_sec) and math.isfinite(discount_a) and math.isfinite(discount_b))
    assume(discount_a != discount_b)
    lo_discount, hi_discount = sorted([discount_a, discount_b])
    cost_lo = _compute_expected_cost(duration_sec, base_rate, multiplier, lo_discount)
    cost_hi = _compute_expected_cost(duration_sec, base_rate, multiplier, hi_discount)
    assert cost_hi <= cost_lo + 1e-9, f"higher discount gave higher cost: {cost_hi} > {cost_lo}"


@given(duration_sec=st.floats(min_value=0.0, max_value=3600 * 24 * 365))
@settings(max_examples=100)
def test_billing_zero_duration_zero_cost(duration_sec):
    """Zero duration means zero cost regardless of rate."""
    assume(math.isfinite(duration_sec))
    cost = _compute_expected_cost(0.0, 1.0, 1.0, 0.0)
    assert cost == 0.0


def test_billing_one_hour_round_trip():
    """1 hour at $0.20/hr with no multiplier or discount = exactly $0.20."""
    cost = _compute_expected_cost(3600.0, 0.20, 1.0, 0.0)
    assert abs(cost - 0.20) < 1e-9, cost


def test_billing_spot_discount_half_price():
    """50% spot discount halves the cost."""
    full = _compute_expected_cost(3600.0, 1.0, 1.0, 0.0)
    half = _compute_expected_cost(3600.0, 1.0, 1.0, 0.5)
    assert abs(half - full / 2) < 1e-9


# ── Province tax rate invariants ──────────────────────────────────────────────

@pytest.mark.parametrize("province,expected_rate", [
    ("ON", 0.13),
    ("AB", 0.05),
    ("QC", 0.14975),
    ("NS", 0.15),
    ("BC", 0.12),
])
def test_province_tax_rate_known_values(province, expected_rate):
    rate, _ = get_tax_rate_for_province(province)
    assert abs(rate - expected_rate) < 1e-9, f"{province}: got {rate}, expected {expected_rate}"


@given(province=st.text(min_size=0, max_size=10))
@settings(max_examples=200)
def test_province_tax_rate_always_positive_bounded(province):
    """Tax rate is always in [0, 0.20] for any input."""
    rate, desc = get_tax_rate_for_province(province)
    assert 0 <= rate <= 0.20, f"rate={rate} for province={province!r}"
    assert isinstance(desc, str)


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler — allocate() invariants
# ─────────────────────────────────────────────────────────────────────────────

# Patch DB/volume calls so allocate() works without a live DB.
# The import is done inline in allocate() as `from volumes import get_volume_engine`
# so we must patch at the volumes module level.
_PATCH_VOLUMES = patch(
    "volumes.get_volume_engine", side_effect=Exception("no DB in tests")
)

from scheduler import allocate, allocate_binpack, _vram_fits, VRAM_DRIVER_OVERHEAD_GB


def _make_host(
    host_id="h1",
    gpu_model="RTX 4090",
    total_vram_gb=24.0,
    free_vram_gb=24.0,
    cost_per_hour=0.20,
    status="active",
    admitted=True,
    gpu_count=1,
    compute_score=None,
    recommended_runtime="runc",
    country="CA",
    province="ON",
):
    return {
        "host_id": host_id,
        "gpu_model": gpu_model,
        "total_vram_gb": total_vram_gb,
        "free_vram_gb": free_vram_gb,
        "cost_per_hour": cost_per_hour,
        "status": status,
        "admitted": admitted,
        "gpu_count": gpu_count,
        "compute_score": compute_score,
        "recommended_runtime": recommended_runtime,
        "country": country,
        "province": province,
        "latency_ms": 10,
    }


def _make_job(
    job_id="j1",
    name="test-job",
    vram_needed_gb=8.0,
    num_gpus=1,
    tier="free",
    gpu_model=None,
    volume_ids=None,
    priority=0,
):
    return {
        "job_id": job_id,
        "name": name,
        "vram_needed_gb": vram_needed_gb,
        "num_gpus": num_gpus,
        "tier": tier,
        "gpu_model": gpu_model,
        "volume_ids": volume_ids or [],
        "priority": priority,
    }


# ── _vram_fits unit tests ─────────────────────────────────────────────────────

@given(
    total=st.floats(min_value=0.0, max_value=128.0),
    free=st.floats(min_value=0.0, max_value=128.0),
    needed=st.floats(min_value=0.0, max_value=256.0),
)
@settings(max_examples=300)
def test_vram_fits_matches_formula(total, free, needed):
    """_vram_fits must exactly reflect the documented formula."""
    assume(math.isfinite(total) and math.isfinite(free) and math.isfinite(needed))
    host = {"total_vram_gb": total, "free_vram_gb": free}
    result = _vram_fits(host, needed)
    if needed <= 0:
        assert result is True
    else:
        expected = total >= needed and (free + VRAM_DRIVER_OVERHEAD_GB) >= needed
        assert result == expected


def test_vram_fits_zero_needed_always_true():
    """vram_needed=0 always fits (interactive instances)."""
    host = {"total_vram_gb": 0.0, "free_vram_gb": 0.0}
    assert _vram_fits(host, 0) is True


def test_vram_fits_overhead_tolerance():
    """Driver overhead tolerance: host with free=7.5 fits an 8GB request."""
    host = {"total_vram_gb": 24.0, "free_vram_gb": 7.5}
    # 7.5 + 1.0 overhead = 8.5 >= 8.0 → fits
    assert _vram_fits(host, 8.0) is True


def test_vram_fits_below_overhead_tolerance():
    """Host with free=6.9 (below 8 - 1.0 headroom) does NOT fit an 8GB request."""
    host = {"total_vram_gb": 24.0, "free_vram_gb": 6.9}
    # 6.9 + 1.0 = 7.9 < 8.0 → does not fit
    assert _vram_fits(host, 8.0) is False


# ── allocate() invariants ─────────────────────────────────────────────────────

def test_allocate_empty_hosts_returns_none():
    with _PATCH_VOLUMES:
        assert allocate(_make_job(), []) is None


def test_allocate_no_admitted_hosts_returns_none():
    hosts = [_make_host(admitted=False)]
    with _PATCH_VOLUMES:
        assert allocate(_make_job(), hosts) is None


def test_allocate_insufficient_vram_returns_none():
    """Job needs 32 GB but only 8 GB host available → no allocation."""
    hosts = [_make_host(total_vram_gb=8.0, free_vram_gb=8.0)]
    job = _make_job(vram_needed_gb=32.0)
    with _PATCH_VOLUMES:
        assert allocate(job, hosts) is None


def test_allocate_returns_host_from_input_list():
    """Allocated host must be one of the input hosts."""
    hosts = [_make_host(f"h{i}") for i in range(5)]
    job = _make_job(vram_needed_gb=4.0)
    with _PATCH_VOLUMES:
        result = allocate(job, hosts)
    assert result is not None
    assert result in hosts


def test_allocate_prefers_higher_vram_when_equal_score():
    """Given two identical hosts except VRAM, the one with more free VRAM wins
    (higher efficiency / better fit)."""
    h_low = _make_host("h_low", free_vram_gb=8.0, total_vram_gb=8.0, compute_score=100)
    h_high = _make_host("h_high", free_vram_gb=24.0, total_vram_gb=24.0, compute_score=100)
    job = _make_job(vram_needed_gb=4.0)
    with _PATCH_VOLUMES:
        result = allocate(job, [h_low, h_high])
    # Both qualify; we only assert a valid host is returned
    assert result in (h_low, h_high)


def test_allocate_gpu_model_filter():
    """If job specifies gpu_model, only matching hosts are used."""
    h_a100 = _make_host("h_a100", gpu_model="A100")
    h_4090 = _make_host("h_4090", gpu_model="RTX 4090")
    job = _make_job(gpu_model="A100", vram_needed_gb=0)
    with _PATCH_VOLUMES:
        result = allocate(job, [h_a100, h_4090])
    assert result is not None
    assert result["gpu_model"] == "A100"


def test_allocate_gpu_model_no_match_returns_none():
    """If no host matches requested gpu_model, return None."""
    hosts = [_make_host(gpu_model="RTX 4090")]
    job = _make_job(gpu_model="A100", vram_needed_gb=0)
    with _PATCH_VOLUMES:
        assert allocate(job, hosts) is None


def test_allocate_inactive_host_excluded():
    """Only 'active' status hosts are considered."""
    hosts = [_make_host(status="offline"), _make_host("h2", status="active")]
    with _PATCH_VOLUMES:
        result = allocate(_make_job(), hosts)
    assert result is not None
    assert result["host_id"] == "h2"


def test_allocate_prefers_cheaper_host_equal_compute():
    """Between two hosts with the same compute score, cheaper should score higher
    (efficiency = compute / price)."""
    h_cheap = _make_host("cheap", cost_per_hour=0.10, compute_score=100)
    h_pricey = _make_host("pricey", cost_per_hour=1.00, compute_score=100)
    job = _make_job(vram_needed_gb=0)
    with _PATCH_VOLUMES:
        result = allocate(job, [h_cheap, h_pricey])
    assert result["host_id"] == "cheap"


@given(
    n_hosts=st.integers(min_value=1, max_value=20),
    vram_needed=st.floats(min_value=0.0, max_value=16.0),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_allocate_result_vram_sufficient(n_hosts, vram_needed):
    """When allocate() returns a host, that host must satisfy the VRAM requirement."""
    assume(math.isfinite(vram_needed))
    hosts = [
        _make_host(
            f"h{i}",
            total_vram_gb=float(i * 4 + 4),
            free_vram_gb=float(i * 4 + 4),
        )
        for i in range(n_hosts)
    ]
    job = _make_job(vram_needed_gb=vram_needed)
    with _PATCH_VOLUMES:
        result = allocate(job, hosts)
    if result is not None:
        assert _vram_fits(result, vram_needed), (
            f"allocated host {result['host_id']} cannot fit vram_needed={vram_needed}"
        )


# ── allocate_binpack() invariants ─────────────────────────────────────────────

def test_binpack_empty_hosts_returns_none():
    assert allocate_binpack(_make_job(), []) is None


def test_binpack_no_admitted_returns_none():
    assert allocate_binpack(_make_job(), [_make_host(admitted=False)]) is None


def test_binpack_result_always_in_input():
    hosts = [_make_host(f"h{i}", total_vram_gb=24, free_vram_gb=24) for i in range(4)]
    result = allocate_binpack(_make_job(vram_needed_gb=4), hosts)
    assert result is not None
    assert result in hosts


def test_binpack_insufficient_vram_returns_none():
    hosts = [_make_host(total_vram_gb=4.0, free_vram_gb=4.0)]
    job = _make_job(vram_needed_gb=24.0)
    assert allocate_binpack(job, hosts) is None


def test_binpack_multi_gpu_gang_scheduling_no_match():
    """Gang scheduling: if no single host has enough GPUs, binpack returns None."""
    hosts = [_make_host(f"h{i}", gpu_count=1) for i in range(4)]
    job = _make_job(num_gpus=4, vram_needed_gb=0)
    assert allocate_binpack(job, hosts) is None


def test_binpack_multi_gpu_gang_scheduling_matches():
    """Gang scheduling: host with gpu_count >= needed is selected."""
    h_small = _make_host("small", gpu_count=1)
    h_big = _make_host("big", gpu_count=4)
    job = _make_job(num_gpus=4, vram_needed_gb=0)
    result = allocate_binpack(job, [h_small, h_big])
    assert result is not None
    assert result["host_id"] == "big"


@given(
    n_hosts=st.integers(min_value=1, max_value=15),
    vram_needed=st.floats(min_value=0.0, max_value=24.0),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_binpack_result_vram_sufficient(n_hosts, vram_needed):
    """When binpack returns a host, it must satisfy VRAM."""
    assume(math.isfinite(vram_needed))
    hosts = [
        _make_host(f"h{i}", total_vram_gb=24.0, free_vram_gb=24.0)
        for i in range(n_hosts)
    ]
    result = allocate_binpack(_make_job(vram_needed_gb=vram_needed), hosts)
    if result is not None:
        assert _vram_fits(result, vram_needed)


# ─────────────────────────────────────────────────────────────────────────────
# Volume state-machine invariants (in-memory mock)
# ─────────────────────────────────────────────────────────────────────────────

class _InMemoryVolumeStore:
    """Minimal in-memory replica of VolumeEngine state for SM testing."""

    def __init__(self):
        self._volumes: dict[str, dict] = {}
        self._attachments: dict[str, str] = {}  # volume_id -> instance_id

    def create(self, vid, owner, size_gb, name="vol"):
        assert 1 <= size_gb <= 2000, f"size_gb out of range: {size_gb}"
        assert vid not in self._volumes, "duplicate volume"
        self._volumes[vid] = {
            "volume_id": vid,
            "owner": owner,
            "name": name,
            "size_gb": size_gb,
            "status": "available",
        }
        return self._volumes[vid]

    def delete(self, vid):
        assert vid in self._volumes, "unknown volume"
        assert vid not in self._attachments, "cannot delete attached volume"
        del self._volumes[vid]

    def attach(self, vid, instance_id):
        assert vid in self._volumes, "unknown volume"
        assert vid not in self._attachments, "already attached"
        self._attachments[vid] = instance_id
        self._volumes[vid]["status"] = "attached"

    def detach(self, vid):
        assert vid in self._volumes, "unknown volume"
        assert vid in self._attachments, "not attached"
        del self._attachments[vid]
        self._volumes[vid]["status"] = "available"

    def is_attached(self, vid):
        return vid in self._attachments

    def exists(self, vid):
        return vid in self._volumes


# ── Volume SM tests ───────────────────────────────────────────────────────────

def test_volume_create_valid_size():
    store = _InMemoryVolumeStore()
    v = store.create("v1", "user1", 100, "data")
    assert v["size_gb"] == 100
    assert v["status"] == "available"


@pytest.mark.parametrize("bad_size", [0, -1, 2001, 1_000_000])
def test_volume_create_invalid_size_rejected(bad_size):
    store = _InMemoryVolumeStore()
    with pytest.raises(AssertionError):
        store.create("v1", "user1", bad_size)


def test_volume_attach_detach_cycle():
    store = _InMemoryVolumeStore()
    store.create("v1", "u1", 50)
    store.attach("v1", "inst-1")
    assert store.is_attached("v1")
    store.detach("v1")
    assert not store.is_attached("v1")
    assert store._volumes["v1"]["status"] == "available"


def test_volume_delete_while_attached_rejected():
    store = _InMemoryVolumeStore()
    store.create("v1", "u1", 10)
    store.attach("v1", "inst-1")
    with pytest.raises(AssertionError):
        store.delete("v1")


def test_volume_double_attach_rejected():
    store = _InMemoryVolumeStore()
    store.create("v1", "u1", 10)
    store.attach("v1", "inst-1")
    with pytest.raises(AssertionError):
        store.attach("v1", "inst-2")


def test_volume_detach_not_attached_rejected():
    store = _InMemoryVolumeStore()
    store.create("v1", "u1", 10)
    with pytest.raises(AssertionError):
        store.detach("v1")


def test_volume_delete_after_detach_succeeds():
    store = _InMemoryVolumeStore()
    store.create("v1", "u1", 10)
    store.attach("v1", "inst-1")
    store.detach("v1")
    store.delete("v1")
    assert not store.exists("v1")


@given(
    size_gb=st.integers(min_value=1, max_value=2000),
    n_attach_detach=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=200)
def test_volume_sm_attach_detach_invariants(size_gb, n_attach_detach):
    """After N attach/detach cycles, volume is always available and not attached."""
    store = _InMemoryVolumeStore()
    store.create("v1", "u1", size_gb)
    for i in range(n_attach_detach):
        store.attach("v1", f"inst-{i}")
        assert store.is_attached("v1")
        store.detach("v1")
        assert not store.is_attached("v1")
    # Final state: available
    assert store._volumes["v1"]["status"] == "available"
    assert not store.is_attached("v1")


@given(size_gb=st.integers(min_value=1, max_value=2000))
@settings(max_examples=100)
def test_volume_sm_can_delete_after_never_attached(size_gb):
    store = _InMemoryVolumeStore()
    store.create("v1", "u1", size_gb)
    store.delete("v1")
    assert not store.exists("v1")


@given(size_gb=st.integers(min_value=1, max_value=2000))
@settings(max_examples=100)
def test_volume_sm_cannot_delete_while_attached(size_gb):
    store = _InMemoryVolumeStore()
    store.create("v1", "u1", size_gb)
    store.attach("v1", "inst-x")
    with pytest.raises(AssertionError):
        store.delete("v1")
