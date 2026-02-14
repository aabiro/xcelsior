"""Tests for Xcelsior reputation engine — scoring, tiers, decay, marketplace boost."""

import json
import os
import tempfile
import time

import pytest

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_rep_test_")
_tmpdir = _tmp_ctx.name
os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

from reputation import (
    MAX_ACTIVITY_POINTS,
    MAX_VERIFICATION_POINTS,
    PENALTY_POINTS,
    POINTS_PER_COMPLETED_JOB,
    PenaltyType,
    ReputationEngine,
    ReputationScore,
    ReputationStore,
    ReputationTier,
    TIER_PRICING_PREMIUM,
    TIER_SEARCH_BOOST,
    TIER_THRESHOLDS,
    VERIFICATION_POINTS,
    VerificationType,
    score_to_tier,
)


def _engine() -> ReputationEngine:
    store = ReputationStore(
        db_path=os.path.join(_tmpdir, f"rep_{os.urandom(4).hex()}.db")
    )
    return ReputationEngine(store=store)


# ── score_to_tier ─────────────────────────────────────────────────────


class TestScoreToTier:
    """Tier classification by score — 6-level model per REPORT_FEATURE_1.md."""

    def test_zero_is_new_user(self):
        assert score_to_tier(0) == ReputationTier.NEW_USER

    def test_99_is_new_user(self):
        assert score_to_tier(99) == ReputationTier.NEW_USER

    def test_100_is_bronze(self):
        assert score_to_tier(100) == ReputationTier.BRONZE

    def test_250_is_silver(self):
        assert score_to_tier(250) == ReputationTier.SILVER

    def test_450_is_gold(self):
        assert score_to_tier(450) == ReputationTier.GOLD

    def test_650_is_platinum(self):
        assert score_to_tier(650) == ReputationTier.PLATINUM

    def test_850_is_diamond(self):
        assert score_to_tier(850) == ReputationTier.DIAMOND

    def test_1500_is_diamond(self):
        assert score_to_tier(1500) == ReputationTier.DIAMOND


# ── Verification Points ──────────────────────────────────────────────


class TestVerificationPoints:
    """Verification adds points up to cap."""

    def test_email_verification_adds_points(self):
        eng = _engine()
        score = eng.add_verification("host-v1", VerificationType.EMAIL)
        assert score.verification_points == VERIFICATION_POINTS[VerificationType.EMAIL]

    def test_duplicate_verification_no_double_count(self):
        eng = _engine()
        eng.add_verification("host-v2", VerificationType.EMAIL)
        score = eng.add_verification("host-v2", VerificationType.EMAIL)
        assert score.verification_points == VERIFICATION_POINTS[VerificationType.EMAIL]

    def test_multiple_types_stack(self):
        eng = _engine()
        eng.add_verification("host-v3", VerificationType.EMAIL)
        score = eng.add_verification("host-v3", VerificationType.PHONE)
        expected = VERIFICATION_POINTS[VerificationType.EMAIL] + VERIFICATION_POINTS[VerificationType.PHONE]
        assert score.verification_points == min(expected, MAX_VERIFICATION_POINTS)

    def test_capped_at_max(self):
        eng = _engine()
        for vt in VerificationType:
            eng.add_verification("host-v4", vt)
        score = eng.compute_score("host-v4")
        assert score.verification_points <= MAX_VERIFICATION_POINTS


# ── Activity Scoring ──────────────────────────────────────────────────


class TestActivityScoring:
    """Job completion adds activity points."""

    def test_completed_job_adds_points(self):
        eng = _engine()
        score = eng.record_job_completed("host-act-1")
        assert score.activity_points >= POINTS_PER_COMPLETED_JOB
        assert score.jobs_completed >= 1

    def test_multiple_completions_accumulate(self):
        eng = _engine()
        eng.record_job_completed("host-act-2")
        score = eng.record_job_completed("host-act-2")
        assert score.activity_points >= 2 * POINTS_PER_COMPLETED_JOB

    def test_activity_capped(self):
        eng = _engine()
        # Directly set high activity to test cap
        existing = eng.store.get_score("host-act-cap")
        if not existing:
            eng._ensure_entity("host-act-cap")
        with eng.store._conn() as conn:
            conn.execute(
                "UPDATE reputation_scores SET activity_points = ? WHERE entity_id = ?",
                (MAX_ACTIVITY_POINTS + 100, "host-act-cap"),
            )
        score = eng.compute_score("host-act-cap")
        assert score.activity_points <= MAX_ACTIVITY_POINTS


# ── Penalties ─────────────────────────────────────────────────────────


class TestPenalties:
    """Penalty application reduces score."""

    def test_host_failure_penalty(self):
        eng = _engine()
        score = eng.record_job_failure("host-pen-1", is_host_fault=True)
        assert score.penalty_points < 0
        assert score.jobs_failed_host >= 1

    def test_user_failure_no_penalty(self):
        eng = _engine()
        score = eng.record_job_failure("host-pen-2", is_host_fault=False)
        assert score.penalty_points == 0
        assert score.jobs_failed_user >= 1

    def test_manual_penalty(self):
        eng = _engine()
        score = eng.apply_penalty("host-pen-3", PenaltyType.CHARGEBACK, reason="test")
        expected = PENALTY_POINTS[PenaltyType.CHARGEBACK]
        assert score.penalty_points <= expected  # negative

    def test_fraud_flag_severe(self):
        eng = _engine()
        score = eng.apply_penalty("host-pen-4", PenaltyType.FRAUD_FLAG)
        assert score.penalty_points <= PENALTY_POINTS[PenaltyType.FRAUD_FLAG]


# ── Reliability ───────────────────────────────────────────────────────


class TestReliability:
    """Reliability multiplier from measured metrics."""

    def test_perfect_reliability(self):
        eng = _engine()
        score = eng.update_reliability("host-rel-1", uptime_pct=1.0,
                                       job_success_rate=1.0, network_stability=1.0)
        assert score.reliability_score == pytest.approx(1.0, abs=0.01)

    def test_zero_reliability_zeroes_score(self):
        eng = _engine()
        eng.add_verification("host-rel-2", VerificationType.EMAIL)
        score = eng.update_reliability("host-rel-2", uptime_pct=0, job_success_rate=0,
                                       network_stability=0)
        assert score.reliability_score == 0.0
        assert score.final_score == 0


# ── Marketplace Boost ─────────────────────────────────────────────────


class TestMarketplaceBoost:
    """Search boost and pricing premium by tier."""

    def test_bronze_boost_is_1(self):
        assert TIER_SEARCH_BOOST[ReputationTier.BRONZE] == 1.0

    def test_platinum_boost_highest(self):
        assert TIER_SEARCH_BOOST[ReputationTier.PLATINUM] > TIER_SEARCH_BOOST[ReputationTier.GOLD]

    def test_gold_pricing_premium(self):
        assert TIER_PRICING_PREMIUM[ReputationTier.GOLD] > 0

    def test_bronze_no_premium(self):
        assert TIER_PRICING_PREMIUM[ReputationTier.BRONZE] == 0.0


# ── Event History ─────────────────────────────────────────────────────


class TestEventHistory:
    """Reputation event audit trail."""

    def test_events_recorded(self):
        eng = _engine()
        eng.record_job_completed("host-hist-1")
        eng.record_job_failure("host-hist-1", is_host_fault=True)
        history = eng.store.get_event_history("host-hist-1")
        assert len(history) >= 2

    def test_event_has_points_delta(self):
        eng = _engine()
        eng.apply_penalty("host-hist-2", PenaltyType.SLA_BREACH)
        history = eng.store.get_event_history("host-hist-2")
        assert any(ev["points_delta"] < 0 for ev in history)
