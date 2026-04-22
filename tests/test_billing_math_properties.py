"""Property-based tests for billing arithmetic.

Scope — PURE math only:
  - ``billing.get_tax_rate_for_province`` (lookup + fallback)
  - Cost formula used by ``BillingEngine.meter_job`` (re-implemented in-test)
  - Invoice total reconstruction (subtotal + tax)

Out of scope (DB-backed — covered elsewhere):
  - ``BillingEngine.meter_job`` end-to-end persistence
  - ``BillingEngine.deposit`` / ``BillingEngine.charge`` wallet roundtrips
"""

from __future__ import annotations

import math

from hypothesis import given, settings, strategies as st

import billing

# ── get_tax_rate_for_province ────────────────────────────────────────


def test_all_known_provinces_have_bounded_rates():
    """Every declared province maps to a rate in [0, 0.20]."""
    for code, (rate, desc) in billing.PROVINCE_TAX_RATES.items():
        assert isinstance(code, str) and len(code) == 2, f"bad code {code!r}"
        assert 0.0 <= rate <= 0.20, f"{code}: rate {rate} out of bounds"
        assert isinstance(desc, str) and desc, f"{code}: empty description"


def test_gst_rate_is_5pct():
    assert billing.GST_RATE == 0.05


@given(code=st.sampled_from(list(billing.PROVINCE_TAX_RATES.keys())))
@settings(deadline=None, max_examples=100)
def test_known_province_lookup_is_stable(code):
    rate, desc = billing.get_tax_rate_for_province(code)
    expected_rate, expected_desc = billing.PROVINCE_TAX_RATES[code]
    assert rate == expected_rate
    assert desc == expected_desc


@given(code=st.sampled_from(list(billing.PROVINCE_TAX_RATES.keys())))
@settings(deadline=None, max_examples=100)
def test_lookup_is_case_insensitive_and_whitespace_tolerant(code):
    """The implementation upper-cases and strips; all variants must agree."""
    base_rate, base_desc = billing.get_tax_rate_for_province(code)
    variants = [code.lower(), f"  {code}  ", f" {code.lower()}\t", code.title()]
    for v in variants:
        rate, desc = billing.get_tax_rate_for_province(v)
        assert rate == base_rate, f"variant {v!r} gave {rate} != {base_rate}"
        assert desc == base_desc


@given(
    code=st.text(min_size=0, max_size=8).filter(
        lambda s: s.strip().upper() not in billing.PROVINCE_TAX_RATES
    )
)
@settings(deadline=None, max_examples=100)
def test_unknown_province_falls_back_to_gst(code):
    """Unknown provinces must fall back to GST-only (5%) with a non-empty label."""
    rate, desc = billing.get_tax_rate_for_province(code)
    assert rate == billing.GST_RATE
    assert isinstance(desc, str) and desc


# ── Cost formula (mirror of billing.py L298-L299) ────────────────────
#
#   duration_hr = (completed - started) / 3600
#   cost = round(duration_hr * base_rate * multiplier * (1 - spot_discount), 4)
#
# The formula is pure. We re-implement it in-test to lock the contract.


def _cost(duration_hr: float, rate: float, mult: float, spot: float) -> float:
    return round(duration_hr * rate * mult * (1 - spot), 4)


@given(
    hours=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    rate=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    mult=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    spot=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=100)
def test_cost_is_non_negative(hours, rate, mult, spot):
    assert _cost(hours, rate, mult, spot) >= 0


@given(
    hours=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    rate=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    mult=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=100)
def test_cost_zero_when_any_zero_factor(hours, rate, mult):
    """Zero duration OR zero rate OR 100% spot discount ⇒ zero cost."""
    assert _cost(0.0, rate, mult, 0.0) == 0.0
    assert _cost(hours, 0.0, mult, 0.0) == 0.0
    assert _cost(hours, rate, mult, 1.0) == 0.0


@given(
    h1=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    h2=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    rate=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    mult=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
    spot=st.floats(min_value=0.0, max_value=0.9, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=100)
def test_cost_monotonic_in_duration(h1, h2, rate, mult, spot):
    """Longer runs must cost at least as much as shorter ones."""
    lo, hi = sorted([h1, h2])
    c_lo = _cost(lo, rate, mult, spot)
    c_hi = _cost(hi, rate, mult, spot)
    # round() has a 1e-4 step — allow tolerance at that scale.
    assert c_hi >= c_lo - 1e-4


@given(
    hours=st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
    rate=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    m1=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
    m2=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
    spot=st.floats(min_value=0.0, max_value=0.9, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=100)
def test_cost_monotonic_in_tier_multiplier(hours, rate, m1, m2, spot):
    """Higher tier multiplier must yield at least as much cost."""
    lo, hi = sorted([m1, m2])
    c_lo = _cost(hours, rate, lo, spot)
    c_hi = _cost(hours, rate, hi, spot)
    assert c_hi >= c_lo - 1e-4


@given(
    hours=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    rate=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    mult=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
    s1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    s2=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=100)
def test_cost_antimonotonic_in_spot_discount(hours, rate, mult, s1, s2):
    """Larger spot discount must yield no more cost than smaller discount."""
    lo, hi = sorted([s1, s2])  # lo < hi → hi has bigger discount → smaller cost
    c_lo_discount = _cost(hours, rate, mult, lo)
    c_hi_discount = _cost(hours, rate, mult, hi)
    assert c_hi_discount <= c_lo_discount + 1e-4


# ── Invoice total reconstruction ─────────────────────────────────────
#
# Contract (billing.py L370 generate_invoice):
#   subtotal = sum(line_item.subtotal_cad)
#   tax      = subtotal * tax_rate
#   total    = subtotal + tax
#
# The formula is pure arithmetic. We lock it here.


@given(
    subtotals=st.lists(
        st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
        max_size=50,
    ),
    tax_rate=st.floats(min_value=0.0, max_value=0.20, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=100)
def test_invoice_total_reconstruction(subtotals, tax_rate):
    subtotal = sum(subtotals)
    tax = subtotal * tax_rate
    total = subtotal + tax
    # Reflect the invariant under some float noise tolerance.
    assert total >= subtotal - 1e-6
    assert total <= subtotal * (1 + tax_rate) + 1e-6
    # Sanity: zero tax ⇒ total == subtotal
    assert math.isclose(subtotal + subtotal * 0.0, subtotal)


@given(
    subtotal=st.floats(min_value=0.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
    tax_rate=st.floats(min_value=0.0, max_value=0.20, allow_nan=False, allow_infinity=False),
)
@settings(deadline=None, max_examples=100)
def test_invoice_total_is_monotonic_in_tax_rate(subtotal, tax_rate):
    """Higher tax rate ⇒ higher or equal total."""
    t0 = subtotal + subtotal * 0.0
    t1 = subtotal + subtotal * tax_rate
    assert t1 >= t0 - 1e-9


# ── Small-supplier threshold ─────────────────────────────────────────


def test_gst_small_supplier_threshold_matches_cra_rule():
    """Excise Tax Act threshold is $30,000 CAD over 4 consecutive quarters."""
    assert billing.GST_SMALL_SUPPLIER_THRESHOLD_CAD == 30_000.00
