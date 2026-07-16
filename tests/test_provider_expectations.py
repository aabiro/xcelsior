"""Guards the provider-facing messaging against drift from the real engine.

provider_expectations.py is the single source of truth behind the signup email,
the inbox notification, the Earn docs page, and the onboarding wizard. If any
number here stops matching reputation.py, we are making promises to providers the
code doesn't keep — so these tests fail loudly instead.
"""

import os
import tempfile

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_provexp_test_")
os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

import provider_expectations as pe
from reputation import (
    PENALTY_POINTS,
    RELIABILITY_WEIGHTS,
    TIER_PLATFORM_COMMISSION,
    TIER_PRICING_PREMIUM,
    TIER_SEARCH_BOOST,
    TIER_THRESHOLDS,
    PenaltyType,
    ReputationTier,
)


def test_tier_table_matches_engine_exactly():
    """Every row in the human-facing table must equal the engine constants."""
    rows = {r["tier"]: r for r in pe.tier_reward_table()}
    # All six tiers present.
    assert set(rows) == {t.value for t in ReputationTier}
    for tier in ReputationTier:
        row = rows[tier.value]
        assert row["min_score"] == TIER_THRESHOLDS[tier]
        assert row["commission_pct"] == round(TIER_PLATFORM_COMMISSION[tier] * 100, 1)
        assert row["keep_pct"] == round((1 - TIER_PLATFORM_COMMISSION[tier]) * 100, 1)
        assert row["pricing_premium_pct"] == round(TIER_PRICING_PREMIUM[tier] * 100, 1)
        assert row["search_boost"] == TIER_SEARCH_BOOST[tier]


def test_keep_pct_is_complement_of_commission():
    for row in pe.tier_reward_table():
        assert abs(row["keep_pct"] + row["commission_pct"] - 100) < 1e-6


def test_reliability_breakdown_sums_to_100_and_matches_weights():
    breakdown = {c["key"]: c["weight_pct"] for c in pe.reliability_breakdown()}
    assert breakdown == {k: round(v * 100, 1) for k, v in RELIABILITY_WEIGHTS.items()}
    assert abs(sum(breakdown.values()) - 100.0) < 1e-6


def test_key_penalties_match_engine_and_are_negative():
    penalties = {p["type"]: p["points"] for p in pe.key_penalties()}
    assert penalties[PenaltyType.SLA_BREACH.value] == PENALTY_POINTS[PenaltyType.SLA_BREACH]
    for pts in penalties.values():
        assert pts < 0


def test_weekly_goals_are_ordered_and_recommended_is_present():
    hours = [g["hours_per_week"] for g in pe.WEEKLY_UPTIME_GOALS]
    assert hours == sorted(hours)  # ascending: casual -> pro
    assert pe.MINIMUM_WEEKLY_HOURS == hours[0]
    assert pe.RECOMMENDED_WEEKLY_HOURS in hours
    # Recommended must be less than a 24/7 week — we never ask for always-on.
    assert pe.RECOMMENDED_WEEKLY_HOURS < 168


def test_email_copy_never_overstates_the_reward():
    """The headline 'keep N% more' claim must be the true Gold-vs-entry delta."""
    body = pe.provider_welcome_email_text("Dana")
    gold_delta = pe._gold_keep_delta()
    diamond_keep = pe._diamond_keep()
    assert "Dana" in body
    assert f"{gold_delta}% more" in body
    assert f"{diamond_keep:.0f}%" in body
    # The reliability warning must state the real SLA penalty magnitude.
    assert str(abs(PENALTY_POINTS[PenaltyType.SLA_BREACH])) in body
    # It must reference the recommended weekly target.
    assert str(pe.RECOMMENDED_WEEKLY_HOURS) in body


def test_summary_is_serializable_and_complete():
    import json

    summary = pe.provider_expectations_summary()
    json.dumps(summary)  # must be JSON-safe for any endpoint/frontend use
    for key in (
        "recommended_weekly_hours",
        "weekly_goals",
        "tier_rewards",
        "reliability_components",
        "key_penalties",
    ):
        assert key in summary


def test_notification_copy_is_provider_oriented():
    title, body = pe.provider_welcome_notification()
    assert "earn" in title.lower()
    assert str(pe.RECOMMENDED_WEEKLY_HOURS) in body
    assert "mid-job" in body.lower()
