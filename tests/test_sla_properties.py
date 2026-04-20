"""Property-based tests for ``sla.compute_credit_pct`` and ``HostSLARecord.uptime_pct``.

Target: pure functions in ``sla.py``.

Invariants asserted for ``compute_credit_pct(uptime_pct)``:
  1. Always returns a value in {0.0, 10.0, 25.0, 100.0} (known tiers).
  2. Monotonic non-increasing in uptime_pct (more uptime ⇒ equal-or-less credit).
  3. Canonical boundary values (89.9, 90.0, 94.9, 95.0, 98.9, 99.0, 100.0).

Invariants asserted for ``HostSLARecord.uptime_pct``:
  4. Result always in [0, 100].
  5. No downtime ⇒ 100%.
  6. Downtime ≥ total ⇒ 0%.
  7. Linear in downtime for a fixed total.
"""

from hypothesis import given, settings, strategies as st

from sla import compute_credit_pct, HostSLARecord


VALID_CREDITS = {0.0, 10.0, 25.0, 100.0}


# ── compute_credit_pct ───────────────────────────────────────────────


@given(uptime=st.floats(min_value=-100, max_value=200, allow_nan=False, allow_infinity=False))
@settings(max_examples=200, deadline=None)
def test_credit_pct_always_valid_tier(uptime):
    """Result is one of the 4 known tiers."""
    assert compute_credit_pct(uptime) in VALID_CREDITS


@given(
    a=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300, deadline=None)
def test_credit_pct_monotonic_non_increasing(a, b):
    """Higher uptime cannot yield higher credit."""
    if a <= b:
        assert compute_credit_pct(a) >= compute_credit_pct(b)


def test_credit_pct_boundaries():
    """Canonical threshold values."""
    assert compute_credit_pct(100.0) == 0.0     # perfect uptime
    assert compute_credit_pct(99.5) == 0.0      # ≥99% → no credit
    assert compute_credit_pct(99.0) == 0.0      # exact 99.0 → no credit (tier is < 99.0)
    assert compute_credit_pct(98.9) == 10.0     # just below 99 → 10%
    assert compute_credit_pct(95.0) == 10.0     # exact 95 → 10%
    assert compute_credit_pct(94.99) == 25.0    # just below 95 → 25%
    assert compute_credit_pct(90.0) == 25.0     # exact 90 → 25%
    assert compute_credit_pct(89.9) == 100.0    # below 90 → full credit
    assert compute_credit_pct(0.0) == 100.0     # total outage


# ── HostSLARecord.uptime_pct ─────────────────────────────────────────


@given(
    total=st.floats(min_value=0.01, max_value=1e9, allow_nan=False, allow_infinity=False),
    downtime=st.floats(min_value=0, max_value=1e9, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, deadline=None)
def test_uptime_pct_bounded(total, downtime):
    """Result is always in [0, 100] regardless of input proportions."""
    rec = HostSLARecord(
        host_id="h", tier="standard", month="2026-04",
        total_seconds=total, downtime_seconds=downtime,
    )
    assert 0.0 <= rec.uptime_pct <= 100.0


@given(total=st.floats(min_value=0.01, max_value=1e9, allow_nan=False, allow_infinity=False))
@settings(max_examples=100, deadline=None)
def test_uptime_pct_no_downtime_is_100(total):
    """Zero downtime against any positive total ⇒ 100.0%."""
    rec = HostSLARecord(
        host_id="h", tier="standard", month="2026-04",
        total_seconds=total, downtime_seconds=0.0,
    )
    assert rec.uptime_pct == 100.0


@given(total=st.floats(min_value=0.01, max_value=1e9, allow_nan=False, allow_infinity=False))
@settings(max_examples=100, deadline=None)
def test_uptime_pct_full_downtime_is_0(total):
    """Downtime ≥ total ⇒ 0.0%."""
    rec = HostSLARecord(
        host_id="h", tier="standard", month="2026-04",
        total_seconds=total, downtime_seconds=total * 2,
    )
    assert rec.uptime_pct == 0.0


def test_uptime_pct_zero_total_returns_100():
    """Edge case: no data recorded yet ⇒ treated as 100% (no violations)."""
    rec = HostSLARecord(host_id="h", tier="standard", month="2026-04")
    assert rec.uptime_pct == 100.0


@given(
    total=st.floats(min_value=1, max_value=1e6, allow_nan=False, allow_infinity=False),
    pct_down=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, deadline=None)
def test_uptime_pct_linear_in_downtime(total, pct_down):
    """uptime_pct == 100 * (1 - downtime/total) for downtime ≤ total."""
    downtime = total * pct_down
    rec = HostSLARecord(
        host_id="h", tier="standard", month="2026-04",
        total_seconds=total, downtime_seconds=downtime,
    )
    expected = 100.0 * (1.0 - pct_down)
    assert abs(rec.uptime_pct - expected) < 1e-6
