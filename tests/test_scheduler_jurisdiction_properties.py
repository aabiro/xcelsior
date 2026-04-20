"""Property-based tests for ``scheduler.allocate_jurisdiction_aware``.

Target: ``scheduler.allocate_jurisdiction_aware(job, hosts, constraint)``
at scheduler.py L4096.

Covers both the ``constraint=None`` fast path and the ``constraint!=None``
jurisdiction-filter path (the latter regressed historically due to an
arg-count mismatch with ``filter_hosts_by_jurisdiction`` — see the
``test_constraint_non_none_…`` test at the bottom of this file).

Invariants asserted:
  1. Empty hosts → None.
  2. Result is one of the input hosts.
  3. Result satisfies ``_vram_fits``.
  4. Determinism.
  5. When ``constraint`` restricts to Canada-only, the returned host
     has ``country == "CA"``.
"""

from __future__ import annotations

import copy

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

import scheduler


@pytest.fixture(autouse=True)
def _stub_verification_engine(monkeypatch):
    """Stub verification engine so the function is deterministic w.r.t. state."""

    class _Stub:
        def get_verified_hosts(self):
            return []

    monkeypatch.setattr(scheduler, "get_verification_engine", lambda: _Stub())

    # Stub reputation engine too (used inside score_host).
    class _Rep:
        search_boost = 1.0

    class _RepEngine:
        def compute_score(self, _host_id):
            return _Rep()

    monkeypatch.setattr(scheduler, "get_reputation_engine", lambda: _RepEngine())


GPU_MODELS = ["RTX 3060", "RTX 4090", "A100", "H100"]

host_strategy = st.fixed_dictionaries({
    "host_id": st.text(alphabet="abcdef0123456789", min_size=3, max_size=8),
    "status": st.just("active"),
    "admitted": st.booleans(),
    "free_vram_gb": st.floats(min_value=0.0, max_value=80.0, allow_nan=False, allow_infinity=False),
    "total_vram_gb": st.floats(min_value=0.0, max_value=80.0, allow_nan=False, allow_infinity=False),
    "gpu_model": st.sampled_from(GPU_MODELS),
    "gpu_count": st.integers(min_value=1, max_value=8),
    "cost_per_hour": st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    "compute_score": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    "country": st.sampled_from(["CA", "US"]),
})

job_strategy = st.fixed_dictionaries({
    "name": st.text(min_size=0, max_size=32),
    "vram_needed_gb": st.floats(min_value=0.0, max_value=40.0, allow_nan=False, allow_infinity=False),
})


@given(job=job_strategy)
@settings(
    deadline=None, max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_empty_hosts_returns_none(job):
    assert scheduler.allocate_jurisdiction_aware(job, [], constraint=None) is None


@given(job=job_strategy, hosts=st.lists(host_strategy, max_size=8))
@settings(
    deadline=None, max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_result_is_one_of_the_input_hosts(job, hosts):
    result = scheduler.allocate_jurisdiction_aware(job, hosts, constraint=None)
    if result is not None:
        assert result in hosts


@given(job=job_strategy, hosts=st.lists(host_strategy, max_size=8))
@settings(
    deadline=None, max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_result_satisfies_vram_fits(job, hosts):
    result = scheduler.allocate_jurisdiction_aware(job, hosts, constraint=None)
    if result is not None:
        assert scheduler._vram_fits(result, job["vram_needed_gb"])


@given(job=job_strategy, hosts=st.lists(host_strategy, max_size=8))
@settings(
    deadline=None, max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_determinism(job, hosts):
    r1 = scheduler.allocate_jurisdiction_aware(
        copy.deepcopy(job), copy.deepcopy(hosts), constraint=None
    )
    r2 = scheduler.allocate_jurisdiction_aware(
        copy.deepcopy(job), copy.deepcopy(hosts), constraint=None
    )
    if r1 is None:
        assert r2 is None
    else:
        assert r2 is not None
        assert r1.get("host_id") == r2.get("host_id")


# ── Non-None constraint path (regression for scheduler.py:4110 arg-count bug) ─

from jurisdiction import JurisdictionConstraint


@given(job=job_strategy, hosts=st.lists(host_strategy, max_size=8))
@settings(
    deadline=None, max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_constraint_non_none_does_not_crash_and_respects_country(job, hosts):
    """With an explicit country constraint, no non-matching host may be returned.

    Before the scheduler.py:4110 fix, this call raised ``TypeError`` because
    ``filter_hosts_by_jurisdiction`` was called with 2 args instead of 3.
    """
    constraint = JurisdictionConstraint(canada_only=True)
    result = scheduler.allocate_jurisdiction_aware(job, hosts, constraint=constraint)
    if result is not None:
        assert result.get("country") == "CA", (
            f"canada_only=True but got host with country={result.get('country')!r}"
        )
        assert scheduler._vram_fits(result, job["vram_needed_gb"])
