"""Property-based tests for ``scheduler.allocate``.

Target: ``scheduler.allocate(job, hosts)`` (scheduler.py L457).

Strategy: generate random but realistic host/job dicts and assert the
invariants that the function DOES enforce (per its docstring + code):

  1. Empty hosts → None.
  2. Result, if any, is one of the input hosts.
  3. Result has status == "active" (filter in step 1).
  4. Result satisfies ``_vram_fits`` (VRAM filter in step 1).
  5. Result.gpu_model matches job.gpu_model when specified (filter in step 1b).
  6. Result.admitted is truthy (filter in step 2).
  7. allocate is deterministic for the same inputs.

NOT asserted (would be false):
  * Multi-GPU is not a hard requirement — the code explicitly falls back
    to best-available when no single host has enough GPUs.
  * Isolation tier — same, falls back with a warning.
"""

from __future__ import annotations

import copy

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

import scheduler


# ── Monkeypatch ``volumes.get_volume_engine`` so allocate never hits DB ─

@pytest.fixture(autouse=True)
def _stub_volume_engine(monkeypatch):
    """Stub get_volume_engine so the data-gravity branch is deterministic."""

    class _Stub:
        def get_volume_host_ids(self, _volume_ids):
            return set()

    # allocate imports from `volumes` lazily inside the function body; patch
    # both the module symbol and the already-imported attribute to be safe.
    import volumes as _volumes_mod
    monkeypatch.setattr(_volumes_mod, "get_volume_engine", lambda: _Stub())


# ── Strategies ───────────────────────────────────────────────────────

GPU_MODELS = ["RTX 3060", "RTX 4090", "A100", "H100", "L4", "A10"]

host_strategy = st.fixed_dictionaries({
    "host_id": st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
        min_size=3,
        max_size=8,
    ),
    "status": st.sampled_from(["active", "active", "active", "offline", "maintenance"]),
    "admitted": st.booleans(),
    "free_vram_gb": st.floats(min_value=0.0, max_value=80.0, allow_nan=False, allow_infinity=False),
    "total_vram_gb": st.floats(min_value=0.0, max_value=80.0, allow_nan=False, allow_infinity=False),
    "gpu_model": st.sampled_from(GPU_MODELS),
    "gpu_count": st.integers(min_value=1, max_value=8),
    "cost_per_hour": st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    "compute_score": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    "latency_ms": st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    "recommended_runtime": st.sampled_from(["runc", "gvisor", "kata"]),
})

job_strategy = st.fixed_dictionaries({
    "job_id": st.text(min_size=4, max_size=12, alphabet="abcdef0123456789"),
    "name": st.text(min_size=0, max_size=32),
    "num_gpus": st.integers(min_value=1, max_value=4),
    "gpu_model": st.one_of(st.just(""), st.sampled_from(GPU_MODELS)),
    "vram_needed_gb": st.floats(min_value=0.0, max_value=40.0, allow_nan=False, allow_infinity=False),
    "tier": st.sampled_from(["free", "community", "pro", "sovereign", "regulated", "secure"]),
    "volume_ids": st.just([]),
})


# ── Trivial invariants ──────────────────────────────────────────────


@given(job=job_strategy)
@settings(
    deadline=None, max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_empty_hosts_returns_none(job):
    assert scheduler.allocate(job, []) is None


# ── Functional invariants ───────────────────────────────────────────


@given(job=job_strategy, hosts=st.lists(host_strategy, max_size=8))
@settings(
    deadline=None, max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_result_is_one_of_the_input_hosts(job, hosts):
    result = scheduler.allocate(job, hosts)
    if result is not None:
        assert any(result is h for h in hosts) or result in hosts


@given(job=job_strategy, hosts=st.lists(host_strategy, max_size=8))
@settings(
    deadline=None, max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_result_status_is_active(job, hosts):
    result = scheduler.allocate(job, hosts)
    if result is not None:
        assert result.get("status", "active") == "active"


@given(job=job_strategy, hosts=st.lists(host_strategy, max_size=8))
@settings(
    deadline=None, max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_result_satisfies_vram_fits(job, hosts):
    result = scheduler.allocate(job, hosts)
    if result is not None:
        assert scheduler._vram_fits(result, job["vram_needed_gb"])


@given(job=job_strategy, hosts=st.lists(host_strategy, max_size=8))
@settings(
    deadline=None, max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_result_gpu_model_matches_when_specified(job, hosts):
    result = scheduler.allocate(job, hosts)
    requested = (job["gpu_model"] or "").strip().lower()
    if result is not None and requested:
        got = (result.get("gpu_model") or "").strip().lower()
        assert got == requested


@given(job=job_strategy, hosts=st.lists(host_strategy, max_size=8))
@settings(
    deadline=None, max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_result_is_admitted(job, hosts):
    result = scheduler.allocate(job, hosts)
    if result is not None:
        assert bool(result.get("admitted", False))


@given(job=job_strategy, hosts=st.lists(host_strategy, max_size=8))
@settings(
    deadline=None, max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_determinism(job, hosts):
    """Two identical calls must return the same host."""
    r1 = scheduler.allocate(copy.deepcopy(job), copy.deepcopy(hosts))
    r2 = scheduler.allocate(copy.deepcopy(job), copy.deepcopy(hosts))
    if r1 is None:
        assert r2 is None
    else:
        assert r2 is not None
        assert r1.get("host_id") == r2.get("host_id")


# ── Negative filter: no admitted → None ─────────────────────────────


@given(job=job_strategy, hosts=st.lists(host_strategy, max_size=8))
@settings(
    deadline=None, max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_returns_none_when_no_admitted_host_fits(job, hosts):
    """If every feasible host is unadmitted, allocate must return None."""
    # Force all to unadmitted.
    mutated = [dict(h, admitted=False) for h in hosts]
    assert scheduler.allocate(job, mutated) is None
