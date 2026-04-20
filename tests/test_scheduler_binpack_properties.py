"""Property-based tests for ``scheduler.allocate_binpack``.

Target: ``scheduler.allocate_binpack(job, hosts, user_province, volume_host_ids)``
at scheduler.py L592.

Invariants asserted (from the code):
  1. Empty hosts → None.
  2. Result, if any, is one of the input hosts.
  3. Result status == "active" and admitted is truthy.
  4. Result satisfies ``_vram_fits``.
  5. Result.gpu_model matches when specified.
  6. Multi-GPU is HARD — ``num_gpus_needed > 1`` requires single host with
     enough GPUs (vs. ``allocate`` which has a fallback).
  7. Determinism.
"""

from __future__ import annotations

import copy

from hypothesis import HealthCheck, given, settings, strategies as st

import scheduler


GPU_MODELS = ["RTX 3060", "RTX 4090", "A100", "H100", "L4"]

host_strategy = st.fixed_dictionaries({
    "host_id": st.text(alphabet="abcdef0123456789", min_size=3, max_size=8),
    "status": st.sampled_from(["active", "active", "active", "offline"]),
    "admitted": st.booleans(),
    "free_vram_gb": st.floats(min_value=0.0, max_value=80.0, allow_nan=False, allow_infinity=False),
    "total_vram_gb": st.floats(min_value=0.0, max_value=80.0, allow_nan=False, allow_infinity=False),
    "gpu_model": st.sampled_from(GPU_MODELS),
    "gpu_count": st.integers(min_value=1, max_value=8),
    "cost_per_hour": st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    "compute_score": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    "reputation_score": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    "country": st.sampled_from(["CA", "US"]),
    "province": st.sampled_from(["ON", "QC", "BC", ""]),
    "recommended_runtime": st.sampled_from(["runc", "gvisor", "kata"]),
})

job_strategy = st.fixed_dictionaries({
    "job_id": st.text(min_size=4, max_size=12, alphabet="abcdef0123456789"),
    "name": st.text(min_size=0, max_size=32),
    "num_gpus": st.integers(min_value=1, max_value=4),
    "gpu_model": st.one_of(st.just(""), st.sampled_from(GPU_MODELS)),
    "vram_needed_gb": st.floats(min_value=0.0, max_value=40.0, allow_nan=False, allow_infinity=False),
    "tier": st.sampled_from(["free", "community", "pro", "sovereign", "regulated", "secure"]),
})


@given(job=job_strategy)
@settings(deadline=None, max_examples=100)
def test_empty_hosts_returns_none(job):
    assert scheduler.allocate_binpack(job, []) is None


@given(job=job_strategy, hosts=st.lists(host_strategy, max_size=8))
@settings(deadline=None, max_examples=100)
def test_result_is_one_of_the_input_hosts(job, hosts):
    result = scheduler.allocate_binpack(job, hosts)
    if result is not None:
        assert result in hosts


@given(job=job_strategy, hosts=st.lists(host_strategy, max_size=8))
@settings(deadline=None, max_examples=100)
def test_result_is_active_and_admitted(job, hosts):
    result = scheduler.allocate_binpack(job, hosts)
    if result is not None:
        assert result.get("status", "active") == "active"
        assert bool(result.get("admitted", False))


@given(job=job_strategy, hosts=st.lists(host_strategy, max_size=8))
@settings(deadline=None, max_examples=100)
def test_result_satisfies_vram_fits(job, hosts):
    result = scheduler.allocate_binpack(job, hosts)
    if result is not None:
        assert scheduler._vram_fits(result, job["vram_needed_gb"])


@given(job=job_strategy, hosts=st.lists(host_strategy, max_size=8))
@settings(deadline=None, max_examples=100)
def test_result_gpu_model_matches_when_specified(job, hosts):
    result = scheduler.allocate_binpack(job, hosts)
    requested = (job["gpu_model"] or "").strip().lower()
    if result is not None and requested:
        got = (result.get("gpu_model") or "").strip().lower()
        assert got == requested


@given(job=job_strategy, hosts=st.lists(host_strategy, max_size=8))
@settings(deadline=None, max_examples=100)
def test_multi_gpu_is_strictly_satisfied(job, hosts):
    """Unlike ``allocate``, binpack hard-requires a single host with enough GPUs."""
    result = scheduler.allocate_binpack(job, hosts)
    if result is not None and job["num_gpus"] > 1:
        assert result.get("gpu_count", 1) >= job["num_gpus"]


@given(job=job_strategy, hosts=st.lists(host_strategy, max_size=8))
@settings(deadline=None, max_examples=100)
def test_determinism(job, hosts):
    r1 = scheduler.allocate_binpack(copy.deepcopy(job), copy.deepcopy(hosts))
    r2 = scheduler.allocate_binpack(copy.deepcopy(job), copy.deepcopy(hosts))
    if r1 is None:
        assert r2 is None
    else:
        assert r2 is not None
        assert r1.get("host_id") == r2.get("host_id")


@given(
    job=job_strategy,
    hosts=st.lists(host_strategy, max_size=8),
    province=st.sampled_from(["ON", "QC", "BC", None]),
)
@settings(
    deadline=None, max_examples=100,
    suppress_health_check=[HealthCheck.filter_too_much],
)
def test_locality_hint_does_not_break_feasibility(job, hosts, province):
    """Passing a locality hint must never violate VRAM or admission."""
    result = scheduler.allocate_binpack(job, hosts, user_province=province)
    if result is not None:
        assert bool(result.get("admitted", False))
        assert scheduler._vram_fits(result, job["vram_needed_gb"])
