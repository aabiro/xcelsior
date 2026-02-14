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

import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger("xcelsior")


# ── Score Thresholds (REPORT_FEATURE_1.md §Six-Level Reputation Model) ──

class ReputationTier(str, Enum):
    NEW_USER = "new_user"      # 0–99: Baseline access
    BRONZE = "bronze"          # 100–249: Search visibility
    SILVER = "silver"          # 250–449: Priority payout status
    GOLD = "gold"              # 450–649: Verified Badge status
    PLATINUM = "platinum"      # 650–849: Featured listing placement
    DIAMOND = "diamond"        # 850+: Reduced platform commission


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
    INCORPORATION = "incorporation"     # For sovereignty tier
    DATA_CENTER = "data_center"         # Tier 3/4 DC verification


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
    JOB_FAILURE_HOST = "job_failure_host"         # Host-side failure
    CHARGEBACK = "chargeback"                      # Payment dispute
    FRAUD_FLAG = "fraud_flag"                       # Suspicious activity
    SLA_BREACH = "sla_breach"                       # Uptime SLA violation
    SECURITY_INCIDENT = "security_incident"        # Container escape / compromise
    HARDWARE_DAMAGE = "hardware_damage"            # Damaged rented hardware
    TERMS_VIOLATION = "terms_violation"             # TOS breach


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
POINTS_PER_HOSTING_DAY = 2        # Reward continuous availability

# Diminishing returns on job volume (REPORT_FEATURE_1.md §Volume Gaming)
# Per report: "focus on total value and complexity of successfully completed
# workloads" instead of rewarding volume equally.
# After DIMINISHING_RETURNS_THRESHOLD completed jobs, each additional job
# earns progressively fewer points.
DIMINISHING_RETURNS_THRESHOLD = 50  # Full points for first 50 jobs
DIMINISHING_RETURNS_FACTOR = 0.5    # Each subsequent job earns half base

# Decay: lose 1 point per day of inactivity (after 7 day grace)
DECAY_GRACE_DAYS = 7
DECAY_POINTS_PER_DAY = 1
MAX_ACTIVITY_POINTS = 500         # Cap to prevent gaming


# ── Reliability Score ─────────────────────────────────────────────────
# 0.0–1.0 multiplier applied to the final score
# Derived from: uptime percentage, job success rate, network stability

RELIABILITY_WEIGHTS = {
    "uptime_pct": 0.40,           # Measured uptime / expected uptime
    "job_success_rate": 0.35,     # Completed / (completed + host-failed)
    "network_stability": 0.25,   # 1.0 - (jitter_pct + packet_loss_pct)
}


# ── Score Record ──────────────────────────────────────────────────────

@dataclass
class ReputationScore:
    """Current reputation state for an entity (host or user)."""
    entity_id: str = ""
    entity_type: str = "host"        # "host" or "user"

    # Component scores
    verification_points: float = 0.0
    activity_points: float = 0.0
    penalty_points: float = 0.0      # Negative value
    reliability_score: float = 1.0   # 0.0–1.0 multiplier

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
    verifications: str = "[]"        # JSON list of verification types

    # Marketplace impact
    search_boost: float = 1.0        # Higher tier = more visibility
    pricing_premium_pct: float = 0.0 # Gold/Platinum can charge more

    def to_dict(self) -> dict:
        return asdict(self)


# ── Marketplace Visibility Boosts ─────────────────────────────────────
# From REPORT_FEATURE_1.md: reputation influences marketplace visibility
# and pricing. Gold hosts can charge premium rates.

TIER_SEARCH_BOOST = {
    ReputationTier.NEW_USER: 0.8,     # Lower visibility until established
    ReputationTier.BRONZE: 1.0,
    ReputationTier.SILVER: 1.1,
    ReputationTier.GOLD: 1.25,
    ReputationTier.PLATINUM: 1.5,
    ReputationTier.DIAMOND: 2.0,      # Maximum marketplace visibility
}

# From REPORT_MARKETING_1.md: Verified "Gold" hosts at $0.70/hr vs $0.50/hr
# Diamond hosts get reduced platform commission instead of higher pricing
TIER_PRICING_PREMIUM = {
    ReputationTier.NEW_USER: 0.0,
    ReputationTier.BRONZE: 0.0,
    ReputationTier.SILVER: 0.05,    # 5% premium allowed
    ReputationTier.GOLD: 0.20,      # 20% premium (matches $0.50→$0.60)
    ReputationTier.PLATINUM: 0.40,  # 40% premium ($0.50→$0.70)
    ReputationTier.DIAMOND: 0.50,   # 50% premium ($0.50→$0.75)
}

# Diamond tier: reduced platform commission (per REPORT_FEATURE_1.md)
TIER_PLATFORM_COMMISSION = {
    ReputationTier.NEW_USER: 0.15,    # Standard 15% cut
    ReputationTier.BRONZE: 0.15,
    ReputationTier.SILVER: 0.15,
    ReputationTier.GOLD: 0.12,        # 12% for Gold
    ReputationTier.PLATINUM: 0.10,    # 10% for Platinum
    ReputationTier.DIAMOND: 0.08,     # 8% — "Reduced platform commission"
}


# ── Reputation Store ──────────────────────────────────────────────────

class ReputationStore:
    """SQLite-backed reputation persistence."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(
            os.path.dirname(__file__), "xcelsior_reputation.db"
        )
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS reputation_scores (
                    entity_id TEXT PRIMARY KEY,
                    entity_type TEXT DEFAULT 'host',
                    verification_points REAL DEFAULT 0,
                    activity_points REAL DEFAULT 0,
                    penalty_points REAL DEFAULT 0,
                    reliability_score REAL DEFAULT 1.0,
                    raw_score REAL DEFAULT 0,
                    final_score REAL DEFAULT 0,
                    tier TEXT DEFAULT 'bronze',
                    jobs_completed INTEGER DEFAULT 0,
                    jobs_failed_host INTEGER DEFAULT 0,
                    jobs_failed_user INTEGER DEFAULT 0,
                    days_active INTEGER DEFAULT 0,
                    last_activity_at REAL DEFAULT 0,
                    verifications TEXT DEFAULT '[]',
                    search_boost REAL DEFAULT 1.0,
                    pricing_premium_pct REAL DEFAULT 0,
                    updated_at REAL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS reputation_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    points_delta REAL DEFAULT 0,
                    reason TEXT DEFAULT '',
                    metadata TEXT DEFAULT '{}',
                    created_at REAL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_rep_events_entity
                    ON reputation_events(entity_id);
                CREATE INDEX IF NOT EXISTS idx_rep_events_time
                    ON reputation_events(created_at);
            """)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_score(self, entity_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM reputation_scores WHERE entity_id = ?",
                (entity_id,),
            ).fetchone()
            return dict(row) if row else None

    def save_score(self, score: ReputationScore):
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO reputation_scores
                   (entity_id, entity_type, verification_points, activity_points,
                    penalty_points, reliability_score, raw_score, final_score,
                    tier, jobs_completed, jobs_failed_host, jobs_failed_user,
                    days_active, last_activity_at, verifications,
                    search_boost, pricing_premium_pct, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    score.entity_id, score.entity_type,
                    score.verification_points, score.activity_points,
                    score.penalty_points, score.reliability_score,
                    score.raw_score, score.final_score, score.tier,
                    score.jobs_completed, score.jobs_failed_host,
                    score.jobs_failed_user, score.days_active,
                    score.last_activity_at, score.verifications,
                    score.search_boost, score.pricing_premium_pct,
                    time.time(),
                ),
            )

    def record_event(self, entity_id: str, event_type: str,
                     points_delta: float, reason: str = "",
                     metadata: str = "{}"):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO reputation_events
                   (entity_id, event_type, points_delta, reason, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (entity_id, event_type, points_delta, reason, metadata, time.time()),
            )

    def get_event_history(self, entity_id: str, limit: int = 50) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM reputation_events
                   WHERE entity_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
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
            verifications=existing["verifications"],
            search_boost=TIER_SEARCH_BOOST.get(tier, 1.0),
            pricing_premium_pct=TIER_PRICING_PREMIUM.get(tier, 0.0),
        )

        self.store.save_score(score)
        return score

    def add_verification(self, entity_id: str, vtype: VerificationType) -> ReputationScore:
        """Grant verification points (capped at MAX_VERIFICATION_POINTS)."""
        import json

        existing = self.store.get_score(entity_id)
        if not existing:
            self._ensure_entity(entity_id)
            existing = self.store.get_score(entity_id)

        verifications = json.loads(existing.get("verifications", "[]"))
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
                   SET verification_points = ?, verifications = ?, updated_at = ?
                   WHERE entity_id = ?""",
                (new_total, json.dumps(verifications), time.time(), entity_id),
            )

        self.store.record_event(
            entity_id, f"verification_{vtype.value}", delta,
            reason=f"Completed {vtype.value} verification",
        )
        log.info("REPUTATION %s +%.0f verification (%s) total=%.0f",
                 entity_id, delta, vtype.value, new_total)
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
                   SET activity_points = ?, jobs_completed = jobs_completed + 1,
                       last_activity_at = ?, updated_at = ?
                   WHERE entity_id = ?""",
                (activity, time.time(), time.time(), entity_id),
            )

        self.store.record_event(
            entity_id, "job_completed", points,
            reason=f"Job completed successfully (+{points} points, job #{jobs_done + 1})",
        )
        return self.compute_score(entity_id)

    def record_job_failure(self, entity_id: str,
                           is_host_fault: bool = True) -> ReputationScore:
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
                       SET penalty_points = penalty_points + ?,
                           jobs_failed_host = jobs_failed_host + 1,
                           last_activity_at = ?, updated_at = ?
                       WHERE entity_id = ?""",
                    (penalty, time.time(), time.time(), entity_id),
                )
            self.store.record_event(
                entity_id, "job_failure_host", penalty,
                reason="Job failed due to host-side error",
            )
            log.warning("REPUTATION %s %.0f penalty (host failure)", entity_id, penalty)
        else:
            with self.store._conn() as conn:
                conn.execute(
                    """UPDATE reputation_scores
                       SET jobs_failed_user = jobs_failed_user + 1,
                           last_activity_at = ?, updated_at = ?
                       WHERE entity_id = ?""",
                    (time.time(), time.time(), entity_id),
                )
            self.store.record_event(
                entity_id, "job_failure_user", 0,
                reason="Job failed due to user-side error (no penalty)",
            )

        return self.compute_score(entity_id)

    def apply_penalty(self, entity_id: str, ptype: PenaltyType,
                      reason: str = "") -> ReputationScore:
        """Apply a manual penalty."""
        points = PENALTY_POINTS.get(ptype, -50)

        existing = self.store.get_score(entity_id)
        if not existing:
            self._ensure_entity(entity_id)

        with self.store._conn() as conn:
            conn.execute(
                """UPDATE reputation_scores
                   SET penalty_points = penalty_points + ?, updated_at = ?
                   WHERE entity_id = ?""",
                (points, time.time(), entity_id),
            )

        self.store.record_event(entity_id, f"penalty_{ptype.value}", points, reason=reason)
        log.warning("REPUTATION %s %.0f penalty (%s): %s",
                    entity_id, points, ptype.value, reason)
        return self.compute_score(entity_id)

    def update_reliability(self, entity_id: str,
                           uptime_pct: float = 1.0,
                           job_success_rate: float = 1.0,
                           network_stability: float = 1.0) -> ReputationScore:
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
                   SET reliability_score = ?, updated_at = ?
                   WHERE entity_id = ?""",
                (reliability, time.time(), entity_id),
            )

        return self.compute_score(entity_id)

    def get_leaderboard(self, entity_type: str = "host",
                        limit: int = 20) -> list:
        """Top-N hosts/users by reputation score."""
        with self.store._conn() as conn:
            rows = conn.execute(
                """SELECT entity_id, final_score, tier, jobs_completed,
                          reliability_score, search_boost, pricing_premium_pct
                   FROM reputation_scores
                   WHERE entity_type = ?
                   ORDER BY final_score DESC
                   LIMIT ?""",
                (entity_type, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def _ensure_entity(self, entity_id: str, entity_type: str = "host"):
        """Create a default score entry if one doesn't exist."""
        score = ReputationScore(entity_id=entity_id, entity_type=entity_type)
        self.store.save_score(score)


# ── GPU Reference Pricing ─────────────────────────────────────────────
# From REPORT_MARKETING_2.md — recommended starter rates in CAD
# These are *guidance* rates for providers; the marketplace allows
# self-pricing within bounds.

GPU_REFERENCE_PRICING_CAD = {
    # model: {base_rate, subsidized_rate, premium_rate (Gold/Platinum)}
    "RTX 3090": {
        "base_rate_cad": 0.35,
        "subsidized_starter_cad": 0.30,
        "premium_rate_cad": 0.45,      # Gold-tier host
        "min_rate_cad": 0.20,
        "max_rate_cad": 0.80,
    },
    "RTX 4090": {
        "base_rate_cad": 0.45,
        "subsidized_starter_cad": 0.40,
        "premium_rate_cad": 0.70,      # Platinum-tier host per MARKETING_1
        "min_rate_cad": 0.30,
        "max_rate_cad": 1.20,
    },
    "RTX 4080": {
        "base_rate_cad": 0.38,
        "subsidized_starter_cad": 0.33,
        "premium_rate_cad": 0.55,
        "min_rate_cad": 0.25,
        "max_rate_cad": 1.00,
    },
    "A100": {
        "base_rate_cad": 1.50,
        "subsidized_starter_cad": 1.20,
        "premium_rate_cad": 2.00,
        "min_rate_cad": 0.90,
        "max_rate_cad": 3.50,
    },
    "H100": {
        "base_rate_cad": 3.50,
        "subsidized_starter_cad": 3.00,
        "premium_rate_cad": 4.50,
        "min_rate_cad": 2.50,
        "max_rate_cad": 6.00,
    },
    "L40": {
        "base_rate_cad": 1.20,
        "subsidized_starter_cad": 1.00,
        "premium_rate_cad": 1.60,
        "min_rate_cad": 0.80,
        "max_rate_cad": 2.50,
    },
}

# From REPORT_MARKETING_2.md: sovereignty premium
SOVEREIGNTY_PREMIUM_PCT = 0.10  # 10% extra for Canada-only mode

# From REPORT_MARKETING_2.md: spot pricing at 70% of on-demand
SPOT_DISCOUNT_FACTOR = 0.30     # 30% discount for spot/preemptible


def get_reference_rate(gpu_model: str, tier: ReputationTier = ReputationTier.BRONZE,
                       spot: bool = False, sovereignty: bool = False) -> float:
    """Get the reference rate in CAD/hr for a GPU model.

    Adjusts for:
    - Reputation tier (premium hosts can charge more)
    - Spot pricing (30% discount per REPORT_MARKETING_2.md)
    - Sovereignty premium (10% extra for Canada-only)
    """
    # Fuzzy match gpu model
    ref = None
    for model, pricing in GPU_REFERENCE_PRICING_CAD.items():
        if model.lower() in gpu_model.lower() or gpu_model.lower() in model.lower():
            ref = pricing
            break

    if not ref:
        # Default to RTX 4090 pricing as baseline
        ref = GPU_REFERENCE_PRICING_CAD["RTX 4090"]

    # Base rate adjusted for tier
    premium_pct = TIER_PRICING_PREMIUM.get(tier, 0.0)
    rate = ref["base_rate_cad"] * (1 + premium_pct)

    # Spot discount
    if spot:
        rate *= (1 - SPOT_DISCOUNT_FACTOR)

    # Sovereignty premium
    if sovereignty:
        rate *= (1 + SOVEREIGNTY_PREMIUM_PCT)

    return round(rate, 4)


def estimate_job_cost(gpu_model: str, duration_hours: float,
                      tier: ReputationTier = ReputationTier.BRONZE,
                      spot: bool = False, sovereignty: bool = False,
                      is_canadian: bool = True) -> dict:
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
        "savings_pct": round(fund["reimbursable_amount_cad"] / gross_cost * 100, 1)
            if gross_cost > 0 else 0,
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
