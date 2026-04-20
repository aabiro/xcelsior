"""Property-based tests for reputation scoring.

Targets pure functions in ``reputation.py``:
  * ``score_to_tier(score)``
  * ``TIER_PLATFORM_COMMISSION`` / ``TIER_SEARCH_BOOST`` / ``TIER_PRICING_PREMIUM`` maps

Invariants asserted:
  1. ``score_to_tier`` is total (never raises) for any finite score.
  2. ``score_to_tier`` is monotonic: higher score ⇒ equal-or-higher tier rank.
  3. Known boundary scores map to the expected tier.
  4. ``TIER_PLATFORM_COMMISSION`` is monotonically non-increasing by tier rank
     (higher tier ⇒ lower commission — DIAMOND pays less than NEW_USER).
  5. ``TIER_SEARCH_BOOST`` is monotonically non-decreasing by tier rank.
  6. ``TIER_PRICING_PREMIUM`` is monotonically non-decreasing by tier rank.
"""

import math
from hypothesis import given, settings, strategies as st

from reputation import (
    ReputationTier,
    score_to_tier,
    TIER_THRESHOLDS,
    TIER_PLATFORM_COMMISSION,
    TIER_SEARCH_BOOST,
    TIER_PRICING_PREMIUM,
)


# Canonical tier ordering (by threshold ascending)
TIER_ORDER = [
    ReputationTier.NEW_USER,
    ReputationTier.BRONZE,
    ReputationTier.SILVER,
    ReputationTier.GOLD,
    ReputationTier.PLATINUM,
    ReputationTier.DIAMOND,
]
TIER_RANK = {t: i for i, t in enumerate(TIER_ORDER)}


# ── score_to_tier ────────────────────────────────────────────────────


@given(score=st.floats(min_value=-10_000, max_value=1_000_000, allow_nan=False, allow_infinity=False))
@settings(max_examples=200, deadline=None)
def test_score_to_tier_total(score):
    """Never raises, always returns a known tier."""
    tier = score_to_tier(score)
    assert tier in TIER_RANK


@given(
    a=st.floats(min_value=-100, max_value=2000, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=-100, max_value=2000, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300, deadline=None)
def test_score_to_tier_monotonic(a, b):
    """Higher score cannot produce a lower-ranked tier."""
    if a <= b:
        assert TIER_RANK[score_to_tier(a)] <= TIER_RANK[score_to_tier(b)]


def test_score_to_tier_boundaries():
    """Canonical threshold values land on the correct tier."""
    assert score_to_tier(0) == ReputationTier.NEW_USER
    assert score_to_tier(99.99) == ReputationTier.NEW_USER
    assert score_to_tier(100) == ReputationTier.BRONZE
    assert score_to_tier(249.99) == ReputationTier.BRONZE
    assert score_to_tier(250) == ReputationTier.SILVER
    assert score_to_tier(449.99) == ReputationTier.SILVER
    assert score_to_tier(450) == ReputationTier.GOLD
    assert score_to_tier(649.99) == ReputationTier.GOLD
    assert score_to_tier(650) == ReputationTier.PLATINUM
    assert score_to_tier(849.99) == ReputationTier.PLATINUM
    assert score_to_tier(850) == ReputationTier.DIAMOND
    assert score_to_tier(1_000_000) == ReputationTier.DIAMOND


@given(score=st.floats(min_value=-1000, max_value=2000, allow_nan=False, allow_infinity=False))
@settings(max_examples=100, deadline=None)
def test_score_to_tier_matches_thresholds(score):
    """The returned tier's threshold is the largest threshold ≤ score."""
    tier = score_to_tier(score)
    thresh = TIER_THRESHOLDS[tier]
    # Score must be ≥ this tier's threshold
    assert score >= thresh or tier == ReputationTier.NEW_USER
    # No higher tier should also qualify
    rank = TIER_RANK[tier]
    for higher in TIER_ORDER[rank + 1 :]:
        assert score < TIER_THRESHOLDS[higher]


# ── Tier lookup table monotonicity ──────────────────────────────────


def test_platform_commission_monotone_non_increasing():
    """Higher tier pays equal-or-lower commission (reward for quality)."""
    vals = [TIER_PLATFORM_COMMISSION[t] for t in TIER_ORDER]
    for a, b in zip(vals, vals[1:]):
        assert b <= a, f"commission must not increase: {vals}"


def test_search_boost_monotone_non_decreasing():
    """Higher tier gets equal-or-more marketplace visibility."""
    vals = [TIER_SEARCH_BOOST[t] for t in TIER_ORDER]
    for a, b in zip(vals, vals[1:]):
        assert b >= a, f"search boost must not decrease: {vals}"


def test_pricing_premium_monotone_non_decreasing():
    """Higher tier can charge equal-or-more premium."""
    vals = [TIER_PRICING_PREMIUM[t] for t in TIER_ORDER]
    for a, b in zip(vals, vals[1:]):
        assert b >= a, f"pricing premium must not decrease: {vals}"


def test_all_tiers_have_entries_in_lookup_tables():
    """No tier silently missing from a lookup — catches future enum additions."""
    for t in TIER_ORDER:
        assert t in TIER_PLATFORM_COMMISSION
        assert t in TIER_SEARCH_BOOST
        assert t in TIER_PRICING_PREMIUM
        assert t in TIER_THRESHOLDS


def test_commission_within_sane_bounds():
    """Commission is a fraction in [0, 1]."""
    for t, v in TIER_PLATFORM_COMMISSION.items():
        assert 0 <= v <= 1, f"{t}: commission {v} outside [0,1]"
