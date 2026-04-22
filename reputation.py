# Xcelsior Reputation Engine — Multi-factor trust scoring
# Implements REPORT_FEATURE_1.md scoring algorithm + REPORT_FEATURE_FINAL.md trust ladder
#
# Score Components (from REPORT_FEATURE_1.md):
#   Verifications:  +250 points (phone, email, gov ID, hardware audit)
#   Activity:       +10 points per completed job (diminishing returns after 50)
#   Reliability:    0.0–1.0 multiplier (measured uptime & stability)
#   Penalties:      -50 to -250 (damaged items, chargebacks, fraud flags)
#   Decay:          Inactive users slowly lose activity points
#
# Six-Level Display Tiers (REPORT_FEATURE_1.md §Reputation Model):
#   New User:   0–99
#   Bronze:     100–249
#   Silver:     250–449
#   Gold:       450–649
#   Platinum:   650–849
#   Diamond:    850+

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger("xcelsior")


# ── Score Thresholds (REPORT_FEATURE_1.md §Six-Level Reputation Model) ──


class ReputationTier(str, Enum):
    NEW_USER = "new_user"  # 0–99: Baseline access
    BRONZE = "bronze"  # 100–249: Search visibility
    SILVER = "silver"  # 250–449: Priority payout status
    GOLD = "gold"  # 450–649: Verified Badge status
    PLATINUM = "platinum"  # 650–849: Featured listing placement
    DIAMOND = "diamond"  # 850+: Reduced platform commission


TIER_THRESHOLDS = {
    ReputationTier.NEW_USER: 0,
    ReputationTier.BRONZE: 100,
    ReputationTier.SILVER: 250,
    ReputationTier.GOLD: 450,
    ReputationTier.PLATINUM: 650,
    ReputationTier.DIAMOND: 850,
}


# Reverse lookup: which tier for a given score
def score_to_tier(score: float) -> ReputationTier:
    if score >= 850:
        return ReputationTier.DIAMOND
    elif score >= 650:
        return ReputationTier.PLATINUM
    elif score >= 450:
        return ReputationTier.GOLD
    elif score >= 250:
        return ReputationTier.SILVER
    elif score >= 100:
        return ReputationTier.BRONZE
    return ReputationTier.NEW_USER


# ── Verification Types ────────────────────────────────────────────────


class VerificationType(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    GOV_ID = "gov_id"
    HARDWARE_AUDIT = "hardware_audit"
    INCORPORATION = "incorporation"  # For sovereignty tier
    DATA_CENTER = "data_center"  # Tier 3/4 DC verification


VERIFICATION_POINTS = {
    VerificationType.EMAIL: 50,
    VerificationType.PHONE: 50,
    VerificationType.GOV_ID: 75,
    VerificationType.HARDWARE_AUDIT: 75,
    VerificationType.INCORPORATION: 100,
    VerificationType.DATA_CENTER: 100,
}

MAX_VERIFICATION_POINTS = 250  # Cap per REPORT_FEATURE_1.md


# ── Penalty Types ─────────────────────────────────────────────────────


class PenaltyType(str, Enum):
    JOB_FAILURE_HOST = "job_failure_host"  # Host-side failure
    CHARGEBACK = "chargeback"  # Payment dispute
    FRAUD_FLAG = "fraud_flag"  # Suspicious activity
    SLA_BREACH = "sla_breach"  # Uptime SLA violation
    SECURITY_INCIDENT = "security_incident"  # Container escape / compromise
    HARDWARE_DAMAGE = "hardware_damage"  # Damaged rented hardware
    TERMS_VIOLATION = "terms_violation"  # TOS breach


PENALTY_POINTS = {
    PenaltyType.JOB_FAILURE_HOST: -50,
    PenaltyType.CHARGEBACK: -100,
    PenaltyType.FRAUD_FLAG: -250,
    PenaltyType.SLA_BREACH: -75,
    PenaltyType.SECURITY_INCIDENT: -200,
    PenaltyType.HARDWARE_DAMAGE: -150,
    PenaltyType.TERMS_VIOLATION: -100,
}


# ── Activity Scoring ──────────────────────────────────────────────────

POINTS_PER_COMPLETED_JOB = 10
POINTS_PER_HOSTING_DAY = 2  # Reward continuous availability

# Diminishing returns on job volume (REPORT_FEATURE_1.md §Volume Gaming)
# Per report: "focus on total value and complexity of successfully completed
# workloads" instead of rewarding volume equally.
# After DIMINISHING_RETURNS_THRESHOLD completed jobs, each additional job
# earns progressively fewer points.
DIMINISHING_RETURNS_THRESHOLD = 50  # Full points for first 50 jobs
DIMINISHING_RETURNS_FACTOR = 0.5  # Each subsequent job earns half base

# Decay: lose 1 point per day of inactivity (after 7 day grace)
DECAY_GRACE_DAYS = 7
DECAY_POINTS_PER_DAY = 1
MAX_ACTIVITY_POINTS = 500  # Cap to prevent gaming


# ── Reliability Score ─────────────────────────────────────────────────
# 0.0–1.0 multiplier applied to the final score
# Derived from: uptime percentage, job success rate, network stability

RELIABILITY_WEIGHTS = {
    "uptime_pct": 0.40,  # Measured uptime / expected uptime
    "job_success_rate": 0.35,  # Completed / (completed + host-failed)
    "network_stability": 0.25,  # 1.0 - (jitter_pct + packet_loss_pct)
}


# ── Score Record ──────────────────────────────────────────────────────


@dataclass
class ReputationScore:
    """Current reputation state for an entity (host or user)."""

    entity_id: str = ""
    entity_type: str = "host"  # "host" or "user"

    # Component scores
    verification_points: float = 0.0
    activity_points: float = 0.0
    penalty_points: float = 0.0  # Negative value
    reliability_score: float = 1.0  # 0.0–1.0 multiplier

    # Derived
    raw_score: float = 0.0
    final_score: float = 0.0
    tier: str = ReputationTier.NEW_USER

    # Stats
    jobs_completed: int = 0
    jobs_failed_host: int = 0
    jobs_failed_user: int = 0
    days_active: int = 0
    last_activity_at: float = 0.0
    verifications: str = "[]"  # JSON list of verification types
    uptime_pct: float = 0.0  # Raw 0.0–1.0 measured uptime

    # Marketplace impact
    search_boost: float = 1.0  # Higher tier = more visibility
    pricing_premium_pct: float = 0.0  # Gold/Platinum can charge more

    def to_dict(self) -> dict:
        return asdict(self)


# ── Marketplace Visibility Boosts ─────────────────────────────────────
# From REPORT_FEATURE_1.md: reputation influences marketplace visibility
# and pricing. Gold hosts can charge premium rates.

TIER_SEARCH_BOOST = {
    ReputationTier.NEW_USER: 0.8,  # Lower visibility until established
    ReputationTier.BRONZE: 1.0,
    ReputationTier.SILVER: 1.1,
    ReputationTier.GOLD: 1.25,
    ReputationTier.PLATINUM: 1.5,
    ReputationTier.DIAMOND: 2.0,  # Maximum marketplace visibility
}

# From REPORT_MARKETING_1.md: Verified "Gold" hosts at $0.70/hr vs $0.50/hr
# Diamond hosts get reduced platform commission instead of higher pricing
TIER_PRICING_PREMIUM = {
    ReputationTier.NEW_USER: 0.0,
    ReputationTier.BRONZE: 0.0,
    ReputationTier.SILVER: 0.05,  # 5% premium allowed
    ReputationTier.GOLD: 0.20,  # 20% premium (matches $0.50→$0.60)
    ReputationTier.PLATINUM: 0.40,  # 40% premium ($0.50→$0.70)
    ReputationTier.DIAMOND: 0.50,  # 50% premium ($0.50→$0.75)
}

# Diamond tier: reduced platform commission (per REPORT_FEATURE_1.md)
TIER_PLATFORM_COMMISSION = {
    ReputationTier.NEW_USER: 0.15,  # Standard 15% cut
    ReputationTier.BRONZE: 0.15,
    ReputationTier.SILVER: 0.15,
    ReputationTier.GOLD: 0.12,  # 12% for Gold
    ReputationTier.PLATINUM: 0.10,  # 10% for Platinum
    ReputationTier.DIAMOND: 0.08,  # 8% — "Reduced platform commission"
}


# ── Reputation Store ──────────────────────────────────────────────────


class ReputationStore:
    """PostgreSQL-backed reputation persistence."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path  # Legacy compat

    @contextmanager
    def _conn(self):
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def get_score(self, entity_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM reputation_scores WHERE entity_id = %s",
                (entity_id,),
            ).fetchone()
            return dict(row) if row else None

    def save_score(self, score: ReputationScore):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO reputation_scores
                   (entity_id, entity_type, verification_points, activity_points,
                    penalty_points, reliability_score, raw_score, final_score,
                    tier, jobs_completed, jobs_failed_host, jobs_failed_user,
                    days_active, last_activity_at, verifications,
                    search_boost, pricing_premium_pct, uptime_pct, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (entity_id) DO UPDATE SET
                     entity_type = EXCLUDED.entity_type, verification_points = EXCLUDED.verification_points,
                     activity_points = EXCLUDED.activity_points, penalty_points = EXCLUDED.penalty_points,
                     reliability_score = EXCLUDED.reliability_score, raw_score = EXCLUDED.raw_score,
                     final_score = EXCLUDED.final_score, tier = EXCLUDED.tier,
                     jobs_completed = EXCLUDED.jobs_completed, jobs_failed_host = EXCLUDED.jobs_failed_host,
                     jobs_failed_user = EXCLUDED.jobs_failed_user, days_active = EXCLUDED.days_active,
                     last_activity_at = EXCLUDED.last_activity_at, verifications = EXCLUDED.verifications,
                     search_boost = EXCLUDED.search_boost, pricing_premium_pct = EXCLUDED.pricing_premium_pct,
                     uptime_pct = EXCLUDED.uptime_pct, updated_at = EXCLUDED.updated_at""",
                (
                    score.entity_id,
                    score.entity_type,
                    score.verification_points,
                    score.activity_points,
                    score.penalty_points,
                    score.reliability_score,
                    score.raw_score,
                    score.final_score,
                    score.tier,
                    score.jobs_completed,
                    score.jobs_failed_host,
                    score.jobs_failed_user,
                    score.days_active,
                    score.last_activity_at,
                    (
                        score.verifications
                        if isinstance(score.verifications, str)
                        else json.dumps(score.verifications)
                    ),
                    score.search_boost,
                    score.pricing_premium_pct,
                    score.uptime_pct,
                    time.time(),
                ),
            )

    def record_event(
        self,
        entity_id: str,
        event_type: str,
        points_delta: float,
        reason: str = "",
        metadata: str = "{}",
    ):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO reputation_events
                   (entity_id, event_type, points_delta, reason, metadata, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (entity_id, event_type, points_delta, reason, metadata, time.time()),
            )

    def get_event_history(self, entity_id: str, limit: int = 50) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM reputation_events
                   WHERE entity_id = %s
                   ORDER BY created_at DESC LIMIT %s""",
                (entity_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]


# ── Reputation Engine ─────────────────────────────────────────────────


class ReputationEngine:
    """Computes and manages reputation scores.

    Implements the multi-factor scoring algorithm from REPORT_FEATURE_1.md
    with the trust ladder tiers from REPORT_MARKETING_FINAL.md.
    """

    def __init__(self, store: Optional[ReputationStore] = None):
        self.store = store or ReputationStore()

    def compute_score(self, entity_id: str) -> ReputationScore:
        """Recompute the full reputation score for an entity."""
        existing = self.store.get_score(entity_id)
        if not existing:
            score = ReputationScore(entity_id=entity_id)
            self.store.save_score(score)
            return score

        # Apply activity decay
        activity = float(existing["activity_points"])
        last_active = float(existing["last_activity_at"])
        if last_active > 0:
            inactive_days = (time.time() - last_active) / 86400
            if inactive_days > DECAY_GRACE_DAYS:
                decay = (inactive_days - DECAY_GRACE_DAYS) * DECAY_POINTS_PER_DAY
                activity = max(0, activity - decay)

        # Cap components
        verification = min(float(existing["verification_points"]), MAX_VERIFICATION_POINTS)
        activity = min(activity, MAX_ACTIVITY_POINTS)
        penalties = float(existing["penalty_points"])  # Already negative
        reliability = max(0.0, min(1.0, float(existing["reliability_score"])))

        # Raw = verification + activity + penalties
        raw = verification + activity + penalties
        # Final = raw * reliability multiplier
        final = max(0, round(raw * reliability, 1))

        tier = score_to_tier(final)

        score = ReputationScore(
            entity_id=entity_id,
            entity_type=existing["entity_type"],
            verification_points=verification,
            activity_points=round(activity, 1),
            penalty_points=penalties,
            reliability_score=reliability,
            raw_score=round(raw, 1),
            final_score=final,
            tier=tier,
            jobs_completed=existing["jobs_completed"],
            jobs_failed_host=existing["jobs_failed_host"],
            jobs_failed_user=existing["jobs_failed_user"],
            days_active=existing["days_active"],
            last_activity_at=existing["last_activity_at"],
            verifications=(
                existing["verifications"]
                if isinstance(existing["verifications"], str)
                else json.dumps(existing["verifications"])
            ),
            search_boost=TIER_SEARCH_BOOST.get(tier, 1.0),
            pricing_premium_pct=TIER_PRICING_PREMIUM.get(tier, 0.0),
            uptime_pct=float(existing.get("uptime_pct", 0.0)),
        )

        self.store.save_score(score)
        return score

    def add_verification(self, entity_id: str, vtype: VerificationType) -> ReputationScore:
        """Grant verification points (capped at MAX_VERIFICATION_POINTS)."""
        existing = self.store.get_score(entity_id)
        if not existing:
            self._ensure_entity(entity_id)
            existing = self.store.get_score(entity_id)

        raw_v = existing.get("verifications", "[]")
        if isinstance(raw_v, list):
            verifications = raw_v
        elif isinstance(raw_v, dict):
            verifications = []
        else:
            verifications = json.loads(raw_v)
        if vtype.value in verifications:
            return self.compute_score(entity_id)

        points = VERIFICATION_POINTS.get(vtype, 0)
        current = float(existing["verification_points"])
        new_total = min(current + points, MAX_VERIFICATION_POINTS)
        delta = new_total - current

        verifications.append(vtype.value)

        with self.store._conn() as conn:
            conn.execute(
                """UPDATE reputation_scores
                   SET verification_points = %s, verifications = %s, updated_at = %s
                   WHERE entity_id = %s""",
                (new_total, json.dumps(verifications), time.time(), entity_id),
            )

        self.store.record_event(
            entity_id,
            f"verification_{vtype.value}",
            delta,
            reason=f"Completed {vtype.value} verification",
        )
        log.info(
            "REPUTATION %s +%.0f verification (%s) total=%.0f",
            entity_id,
            delta,
            vtype.value,
            new_total,
        )
        return self.compute_score(entity_id)

    def record_job_completed(self, entity_id: str) -> ReputationScore:
        """Award activity points for a completed job.

        Implements diminishing returns per REPORT_FEATURE_1.md:
        "focus on total value and complexity of successfully completed
        workloads" — flat points for first N jobs, then diminishing.
        """
        existing = self.store.get_score(entity_id)
        if not existing:
            self._ensure_entity(entity_id)
            existing = self.store.get_score(entity_id)

        jobs_done = existing.get("jobs_completed", 0) if existing else 0

        # Diminishing returns: after threshold, each job earns less
        if jobs_done < DIMINISHING_RETURNS_THRESHOLD:
            points = POINTS_PER_COMPLETED_JOB
        else:
            excess = jobs_done - DIMINISHING_RETURNS_THRESHOLD
            # Halve every 50 jobs beyond threshold (min 1 point)
            reduction = DIMINISHING_RETURNS_FACTOR ** (excess // DIMINISHING_RETURNS_THRESHOLD + 1)
            points = max(1, round(POINTS_PER_COMPLETED_JOB * reduction, 1))

        activity = min(
            float(existing["activity_points"]) + points,
            MAX_ACTIVITY_POINTS,
        )

        with self.store._conn() as conn:
            conn.execute(
                """UPDATE reputation_scores
                   SET activity_points = %s, jobs_completed = jobs_completed + 1,
                       last_activity_at = %s, updated_at = %s
                   WHERE entity_id = %s""",
                (activity, time.time(), time.time(), entity_id),
            )

        self.store.record_event(
            entity_id,
            "job_completed",
            points,
            reason=f"Job completed successfully (+{points} points, job #{jobs_done + 1})",
        )
        return self.compute_score(entity_id)

    def record_job_failure(self, entity_id: str, is_host_fault: bool = True) -> ReputationScore:
        """Record a job failure. Host-side faults incur penalties."""
        existing = self.store.get_score(entity_id)
        if not existing:
            self._ensure_entity(entity_id)
            existing = self.store.get_score(entity_id)

        if is_host_fault:
            penalty = PENALTY_POINTS[PenaltyType.JOB_FAILURE_HOST]
            with self.store._conn() as conn:
                conn.execute(
                    """UPDATE reputation_scores
                       SET penalty_points = penalty_points + %s,
                           jobs_failed_host = jobs_failed_host + 1,
                           last_activity_at = %s, updated_at = %s
                       WHERE entity_id = %s""",
                    (penalty, time.time(), time.time(), entity_id),
                )
            self.store.record_event(
                entity_id,
                "job_failure_host",
                penalty,
                reason="Job failed due to host-side error",
            )
            log.warning("REPUTATION %s %.0f penalty (host failure)", entity_id, penalty)
        else:
            with self.store._conn() as conn:
                conn.execute(
                    """UPDATE reputation_scores
                       SET jobs_failed_user = jobs_failed_user + 1,
                           last_activity_at = %s, updated_at = %s
                       WHERE entity_id = %s""",
                    (time.time(), time.time(), entity_id),
                )
            self.store.record_event(
                entity_id,
                "job_failure_user",
                0,
                reason="Job failed due to user-side error (no penalty)",
            )

        return self.compute_score(entity_id)

    def apply_penalty(
        self, entity_id: str, ptype: PenaltyType, reason: str = ""
    ) -> ReputationScore:
        """Apply a manual penalty."""
        points = PENALTY_POINTS.get(ptype, -50)

        existing = self.store.get_score(entity_id)
        if not existing:
            self._ensure_entity(entity_id)

        with self.store._conn() as conn:
            conn.execute(
                """UPDATE reputation_scores
                   SET penalty_points = penalty_points + %s, updated_at = %s
                   WHERE entity_id = %s""",
                (points, time.time(), entity_id),
            )

        self.store.record_event(entity_id, f"penalty_{ptype.value}", points, reason=reason)
        log.warning("REPUTATION %s %.0f penalty (%s): %s", entity_id, points, ptype.value, reason)
        return self.compute_score(entity_id)

    def update_reliability(
        self,
        entity_id: str,
        uptime_pct: float = 1.0,
        job_success_rate: float = 1.0,
        network_stability: float = 1.0,
    ) -> ReputationScore:
        """Update the reliability multiplier from measured metrics."""
        reliability = (
            uptime_pct * RELIABILITY_WEIGHTS["uptime_pct"]
            + job_success_rate * RELIABILITY_WEIGHTS["job_success_rate"]
            + network_stability * RELIABILITY_WEIGHTS["network_stability"]
        )
        reliability = max(0.0, min(1.0, round(reliability, 4)))

        with self.store._conn() as conn:
            conn.execute(
                """UPDATE reputation_scores
                   SET reliability_score = %s, uptime_pct = %s, updated_at = %s
                   WHERE entity_id = %s""",
                (reliability, uptime_pct, time.time(), entity_id),
            )

        return self.compute_score(entity_id)

    def get_leaderboard(self, entity_type: str = "host", limit: int = 20) -> list:
        """Top-N hosts/users by reputation score."""
        with self.store._conn() as conn:
            rows = conn.execute(
                """SELECT r.entity_id, r.final_score AS score, r.tier,
                          r.jobs_completed, r.reliability_score,
                          r.search_boost, r.pricing_premium_pct,
                          h.payload->>'gpu_model' AS gpu_model
                   FROM reputation_scores r
                   LEFT JOIN hosts h ON h.host_id = r.entity_id
                   WHERE r.entity_type = %s
                   ORDER BY r.final_score DESC
                   LIMIT %s""",
                (entity_type, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def _ensure_entity(self, entity_id: str, entity_type: str = "host"):
        """Create a default score entry if one doesn't exist."""
        score = ReputationScore(entity_id=entity_id, entity_type=entity_type)
        self.store.save_score(score)

    # ── Bayesian Reputation Model ─────────────────────────────────────
    # Bayesian average: score = (alpha * prior) + ((1-alpha) * observed)
    # Alpha decays over time so mature providers rely more on actual data.
    # Cold start: new providers get network-average prior (not zero).

    def compute_bayesian_score(self, entity_id: str) -> dict:
        """Compute Bayesian reputation score with cold-start prior.

        Formula: bayesian = alpha * prior + (1 - alpha) * current_performance
        Alpha decays as the provider gains more observations:
          alpha = 1 / (1 + observed_jobs / prior_weight)

        Returns dict with bayesian_score, alpha, prior, observed, confidence.
        """
        existing = self.store.get_score(entity_id)
        if not existing:
            self._ensure_entity(entity_id)
            existing = self.store.get_score(entity_id)

        jobs_completed = int(existing.get("jobs_completed", 0))
        jobs_failed = int(existing.get("jobs_failed_host", 0))
        total_jobs = jobs_completed + jobs_failed

        # Network-wide prior (average across all providers)
        prior = self._get_network_prior()

        # Observed performance (0-1 scale)
        if total_jobs > 0:
            success_rate = jobs_completed / total_jobs
        else:
            success_rate = prior  # No data → default to prior

        # Alpha decay: starts at 1.0 (full prior), decays as jobs accumulate
        prior_weight = 10  # Equivalent to 10 "virtual" prior observations
        alpha = prior_weight / (prior_weight + total_jobs)

        # Bayesian average
        bayesian = alpha * prior + (1 - alpha) * success_rate

        # Time decay: events older than 90 days weighted at 0.5x
        last_active = float(existing.get("last_activity_at", 0))
        days_since_active = (time.time() - last_active) / 86400 if last_active > 0 else 999
        if days_since_active > 90:
            time_decay = 0.5 + 0.5 * max(0, 1 - (days_since_active - 90) / 180)
            bayesian *= time_decay

        # Confidence interval (wider with fewer observations)
        confidence = min(1.0, total_jobs / 50)  # Full confidence at 50+ jobs

        # Reliability multiplier
        reliability = float(existing.get("reliability_score", 0.5))
        weighted_score = round(bayesian * reliability * 1000, 1)  # Scale to 0-1000

        return {
            "entity_id": entity_id,
            "bayesian_score": round(bayesian, 4),
            "weighted_score": weighted_score,
            "alpha": round(alpha, 4),
            "prior": round(prior, 4),
            "observed": round(success_rate, 4),
            "total_jobs": total_jobs,
            "confidence": round(confidence, 4),
            "time_decay_applied": days_since_active > 90,
        }

    def _get_network_prior(self) -> float:
        """Compute network-wide average success rate as Bayesian prior."""
        try:
            with self.store._conn() as conn:
                row = conn.execute("""SELECT
                        COALESCE(SUM(jobs_completed), 0) as total_completed,
                        COALESCE(SUM(jobs_failed_host), 0) as total_failed
                       FROM reputation_scores
                       WHERE entity_type = 'host'""").fetchone()
                completed = int(row["total_completed"]) if row else 0
                failed = int(row["total_failed"]) if row else 0
                total = completed + failed
                if total >= 10:
                    return completed / total
        except Exception:
            pass
        return 0.7  # Default prior: 70% success rate

    # ── Fraud Detection ───────────────────────────────────────────────
    # Automated Sybil detection, benchmark spoofing, and early termination
    # pattern analysis per REPORT_FEATURE_FINAL.md.

    def detect_sybil(self, entity_id: str) -> dict:
        """Detect Sybil attacks: same IP/hardware registering multiple hosts.

        Flags accounts with identical hardware fingerprints or source IPs.
        """
        existing = self.store.get_score(entity_id)
        if not existing:
            return {"flagged": False, "reason": "Unknown entity"}

        try:
            from db import _get_pg_pool
            from psycopg.rows import dict_row

            pool = _get_pg_pool()
            with pool.connection() as conn:
                conn.row_factory = dict_row

                # Check for duplicate hardware fingerprints
                host_row = conn.execute(
                    "SELECT payload FROM hosts WHERE host_id = %s",
                    (entity_id,),
                ).fetchone()

                if not host_row:
                    return {"flagged": False, "reason": "Host not found in hosts table"}

                payload = host_row["payload"]
                if isinstance(payload, str):
                    payload = json.loads(payload)

                fingerprint = payload.get("hw_fingerprint") or payload.get("gpu_serial", "")
                source_ip = payload.get("source_ip") or payload.get("ip", "")

                suspects = []

                if fingerprint:
                    dupes = conn.execute(
                        """SELECT host_id FROM hosts
                           WHERE host_id != %s
                             AND (payload::jsonb->>'hw_fingerprint' = %s
                                  OR payload::jsonb->>'gpu_serial' = %s)""",
                        (entity_id, fingerprint, fingerprint),
                    ).fetchall()
                    suspects.extend([d["host_id"] for d in dupes])

                if source_ip and source_ip not in ("127.0.0.1", "::1"):
                    ip_dupes = conn.execute(
                        """SELECT host_id FROM hosts
                           WHERE host_id != %s
                             AND (payload::jsonb->>'source_ip' = %s
                                  OR payload::jsonb->>'ip' = %s)""",
                        (entity_id, source_ip, source_ip),
                    ).fetchall()
                    suspects.extend([d["host_id"] for d in ip_dupes])

                suspects = list(set(suspects))

                if suspects:
                    log.warning(
                        "SYBIL DETECTED: %s shares fingerprint/IP with %s",
                        entity_id,
                        suspects,
                    )
                    return {
                        "flagged": True,
                        "reason": f"Shares hardware fingerprint or IP with: {suspects}",
                        "suspect_hosts": suspects,
                        "entity_id": entity_id,
                    }

        except Exception as e:
            log.error("Sybil detection error for %s: %s", entity_id, e)

        return {"flagged": False, "entity_id": entity_id}

    def detect_benchmark_spoofing(
        self, entity_id: str, reported_perf: float, gpu_model: str
    ) -> dict:
        """Detect benchmark spoofing by comparing reported vs expected performance.

        Flags if deviation exceeds 20% from known GPU model baselines.
        """
        # Known baselines (TFLOPS FP16 — approximate)
        GPU_BASELINES = {
            "RTX 3090": 35.6,
            "RTX 4090": 82.6,
            "RTX 4080": 48.7,
            "RTX 4070": 29.1,
            "RTX 4060": 22.1,
            "A100": 77.9,
            "A40": 37.4,
            "H100": 267.0,
            "L40": 90.5,
        }

        baseline = None
        for model_name, perf in GPU_BASELINES.items():
            if model_name.lower() in gpu_model.lower():
                baseline = perf
                break

        if baseline is None:
            return {
                "flagged": False,
                "reason": f"No baseline for {gpu_model}",
                "entity_id": entity_id,
            }

        deviation = abs(reported_perf - baseline) / baseline

        if deviation > 0.20:
            severity = "high" if deviation > 0.50 else "medium"
            log.warning(
                "BENCHMARK SPOOFING SUSPECTED: %s reported %.1f TFLOPS for %s "
                "(expected ~%.1f, deviation %.0f%%)",
                entity_id,
                reported_perf,
                gpu_model,
                baseline,
                deviation * 100,
            )
            return {
                "flagged": True,
                "severity": severity,
                "reason": f"Reported {reported_perf:.1f} TFLOPS, expected ~{baseline:.1f} "
                f"for {gpu_model} (deviation: {deviation:.0%})",
                "entity_id": entity_id,
                "deviation_pct": round(deviation * 100, 1),
                "reported": reported_perf,
                "expected": baseline,
            }

        return {
            "flagged": False,
            "entity_id": entity_id,
            "deviation_pct": round(deviation * 100, 1),
        }

    def detect_early_termination_pattern(self, entity_id: str) -> dict:
        """Detect if a provider terminates an unusually high percentage of jobs early.

        Flags if > 5% of jobs are terminated by the provider before completion.
        """
        existing = self.store.get_score(entity_id)
        if not existing:
            return {"flagged": False, "reason": "Unknown entity"}

        completed = int(existing.get("jobs_completed", 0))
        failed_host = int(existing.get("jobs_failed_host", 0))
        total = completed + failed_host

        if total < 20:
            return {
                "flagged": False,
                "reason": "Insufficient data (< 20 jobs)",
                "entity_id": entity_id,
                "total_jobs": total,
            }

        termination_rate = failed_host / total

        if termination_rate > 0.05:
            log.warning(
                "EARLY TERMINATION PATTERN: %s has %.1f%% failure rate (%d/%d)",
                entity_id,
                termination_rate * 100,
                failed_host,
                total,
            )
            return {
                "flagged": True,
                "reason": f"Host-fault failure rate {termination_rate:.1%} exceeds 5% threshold",
                "entity_id": entity_id,
                "termination_rate": round(termination_rate, 4),
                "failed_host": failed_host,
                "total_jobs": total,
            }

        return {
            "flagged": False,
            "entity_id": entity_id,
            "termination_rate": round(termination_rate, 4),
        }

    def run_fraud_scan(
        self, entity_id: str, reported_perf: float = 0.0, gpu_model: str = ""
    ) -> dict:
        """Run all fraud detection checks on a provider.

        Returns summary with individual check results.
        """
        results = {
            "entity_id": entity_id,
            "checks": {},
            "any_flagged": False,
        }

        results["checks"]["sybil"] = self.detect_sybil(entity_id)
        results["checks"]["early_termination"] = self.detect_early_termination_pattern(entity_id)

        if reported_perf > 0 and gpu_model:
            results["checks"]["benchmark_spoofing"] = self.detect_benchmark_spoofing(
                entity_id,
                reported_perf,
                gpu_model,
            )

        results["any_flagged"] = any(c.get("flagged") for c in results["checks"].values())

        if results["any_flagged"]:
            log.warning("FRAUD SCAN ALERT: %s — %s", entity_id, results)

        return results


# ── GPU Reference Pricing ─────────────────────────────────────────────
# From REPORT_MARKETING_2.md — recommended starter rates in CAD
# These are *guidance* rates for providers; the marketplace allows
# self-pricing within bounds.

GPU_REFERENCE_PRICING_CAD = {
    # model: {base_rate, subsidized_rate, premium_rate (Gold/Platinum)}
    "RTX 3090": {
        "base_rate_cad": 0.35,
        "subsidized_starter_cad": 0.30,
        "premium_rate_cad": 0.45,
        "min_rate_cad": 0.20,
        "max_rate_cad": 0.80,
    },
    "RTX 4080": {
        "base_rate_cad": 0.38,
        "subsidized_starter_cad": 0.33,
        "premium_rate_cad": 0.55,
        "min_rate_cad": 0.25,
        "max_rate_cad": 1.00,
    },
    "RTX 4090": {
        "base_rate_cad": 0.45,
        "subsidized_starter_cad": 0.40,
        "premium_rate_cad": 0.70,
        "min_rate_cad": 0.30,
        "max_rate_cad": 1.20,
    },
    "RTX 5090": {
        "base_rate_cad": 0.55,
        "subsidized_starter_cad": 0.48,
        "premium_rate_cad": 0.85,
        "min_rate_cad": 0.35,
        "max_rate_cad": 1.50,
    },
    "A100 40GB": {
        "base_rate_cad": 1.30,
        "subsidized_starter_cad": 1.05,
        "premium_rate_cad": 1.75,
        "min_rate_cad": 0.80,
        "max_rate_cad": 3.00,
    },
    "A100 80GB": {
        "base_rate_cad": 1.70,
        "subsidized_starter_cad": 1.40,
        "premium_rate_cad": 2.25,
        "min_rate_cad": 1.00,
        "max_rate_cad": 3.80,
    },
    "A100": {
        "base_rate_cad": 1.50,
        "subsidized_starter_cad": 1.20,
        "premium_rate_cad": 2.00,
        "min_rate_cad": 0.90,
        "max_rate_cad": 3.50,
    },
    "L40S": {
        "base_rate_cad": 1.25,
        "subsidized_starter_cad": 1.05,
        "premium_rate_cad": 1.70,
        "min_rate_cad": 0.85,
        "max_rate_cad": 2.60,
    },
    "L40": {
        "base_rate_cad": 1.20,
        "subsidized_starter_cad": 1.00,
        "premium_rate_cad": 1.60,
        "min_rate_cad": 0.80,
        "max_rate_cad": 2.50,
    },
    "H100": {
        "base_rate_cad": 3.50,
        "subsidized_starter_cad": 3.00,
        "premium_rate_cad": 4.50,
        "min_rate_cad": 2.50,
        "max_rate_cad": 6.00,
    },
    "H200": {
        "base_rate_cad": 4.50,
        "subsidized_starter_cad": 3.80,
        "premium_rate_cad": 5.80,
        "min_rate_cad": 3.20,
        "max_rate_cad": 7.50,
    },
}

# From REPORT_MARKETING_2.md: sovereignty premium
SOVEREIGNTY_PREMIUM_PCT = 0.10  # 10% extra for Canada-only mode

# From REPORT_MARKETING_2.md: spot pricing at 70% of on-demand
SPOT_DISCOUNT_FACTOR = 0.30  # 30% discount for spot/preemptible


def get_reference_rate(
    gpu_model: str,
    tier: ReputationTier = ReputationTier.BRONZE,
    spot: bool = False,
    sovereignty: bool = False,
) -> float:
    """Get the reference rate in CAD/hr for a GPU model.

    Adjusts for:
    - Reputation tier (premium hosts can charge more)
    - Spot pricing (30% discount per REPORT_MARKETING_2.md)
    - Sovereignty premium (10% extra for Canada-only)
    """
    # Match GPU model: exact first, then longest-key-first substring to avoid
    # "A100" matching "A100 40GB" when only the generic entry is wanted.
    ref = GPU_REFERENCE_PRICING_CAD.get(gpu_model)
    if not ref:
        gpu_lower = gpu_model.lower()
        # Sort by key length descending so "A100 80GB" is tried before "A100"
        for model, pricing in sorted(
            GPU_REFERENCE_PRICING_CAD.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if model.lower() in gpu_lower:
                ref = pricing
                break

    if not ref:
        # Default to RTX 4090 pricing as baseline
        log.warning("PRICING: unknown GPU model %r, falling back to RTX 4090 rates", gpu_model)
        ref = GPU_REFERENCE_PRICING_CAD["RTX 4090"]

    # Base rate adjusted for tier
    premium_pct = TIER_PRICING_PREMIUM.get(tier, 0.0)
    rate = ref["base_rate_cad"] * (1 + premium_pct)

    # Spot discount
    if spot:
        rate *= 1 - SPOT_DISCOUNT_FACTOR

    # Sovereignty premium
    if sovereignty:
        rate *= 1 + SOVEREIGNTY_PREMIUM_PCT

    return round(rate, 4)


def estimate_job_cost(
    gpu_model: str,
    duration_hours: float,
    tier: ReputationTier = ReputationTier.BRONZE,
    spot: bool = False,
    sovereignty: bool = False,
    is_canadian: bool = True,
) -> dict:
    """Estimate job cost with AI Compute Access Fund rebate preview.

    From REPORT_FEATURE_2.md: `--estimate-rebate` / `simulate=true`
    """
    from jurisdiction import compute_fund_eligible_amount

    rate = get_reference_rate(gpu_model, tier, spot, sovereignty)
    gross_cost = round(rate * duration_hours, 4)

    fund = compute_fund_eligible_amount(gross_cost, is_canadian)

    return {
        "gpu_model": gpu_model,
        "duration_hours": duration_hours,
        "rate_cad_per_hour": rate,
        "gross_cost_cad": gross_cost,
        "currency": "CAD",
        "is_canadian_compute": is_canadian,
        "spot": spot,
        "sovereignty_premium": sovereignty,
        "host_tier": tier,
        # Fund breakdown
        "fund_eligible": fund["fund_eligible"],
        "fund_rate": fund["fund_label"],
        "fund_reimbursable_cad": fund["reimbursable_amount_cad"],
        "effective_cost_cad": round(gross_cost - fund["reimbursable_amount_cad"], 2),
        "savings_pct": (
            round(fund["reimbursable_amount_cad"] / gross_cost * 100, 1) if gross_cost > 0 else 0
        ),
    }


# ── Singletons ────────────────────────────────────────────────────────

_reputation_store: Optional[ReputationStore] = None
_reputation_engine: Optional[ReputationEngine] = None


def get_reputation_store() -> ReputationStore:
    global _reputation_store
    if _reputation_store is None:
        _reputation_store = ReputationStore()
    return _reputation_store


def get_reputation_engine() -> ReputationEngine:
    global _reputation_engine
    if _reputation_engine is None:
        _reputation_engine = ReputationEngine(get_reputation_store())
    return _reputation_engine
