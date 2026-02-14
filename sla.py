# Xcelsior SLA Enforcement Module v2.0.0
# Automated SLA monitoring, credit calculation, and enforcement.
#
# Per REPORT_FEATURE_1.md (Report #1.B):
# - Tiered uptime targets: Community 99.0%, Secure 99.5%, Sovereign 99.9%
# - Automated credit calculation based on monthly uptime percentage
# - Downtime tracking via missed heartbeats and telemetry gaps
# - Host auto-demotion on persistent SLA breach

import os
import sqlite3
import time
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger("xcelsior.sla")

DB_PATH = os.environ.get("XCELSIOR_SLA_DB", "xcelsior_sla.db")

# ── SLA Tier Definitions ─────────────────────────────────────────────
# Per Report #1.B: "SLA Enforcement" section


class SLATier(str, Enum):
    COMMUNITY = "community"       # Consumer hardware, best-effort
    SECURE = "secure"             # Data-center grade, SOC 2 path
    SOVEREIGN = "sovereign"       # Canadian-owned, data residency guarantee


@dataclass
class SLATarget:
    """Performance and availability targets for a tier."""
    tier: SLATier
    availability_pct: float       # e.g. 99.0, 99.5, 99.9
    latency_ttft_ms: int          # Time-to-first-token max
    throughput_floor_pct: float   # % of peak benchmark score
    response_time_hours: int      # Support response SLA
    heartbeat_grace_sec: int      # Seconds before "down" flag
    max_thermal_c: int            # Thermal ceiling before degraded


SLA_TARGETS: dict[SLATier, SLATarget] = {
    SLATier.COMMUNITY: SLATarget(
        tier=SLATier.COMMUNITY,
        availability_pct=99.0,
        latency_ttft_ms=200,
        throughput_floor_pct=80.0,
        response_time_hours=4,
        heartbeat_grace_sec=90,   # 3 missed 30s heartbeats
        max_thermal_c=90,
    ),
    SLATier.SECURE: SLATarget(
        tier=SLATier.SECURE,
        availability_pct=99.5,
        latency_ttft_ms=100,
        throughput_floor_pct=90.0,
        response_time_hours=2,
        heartbeat_grace_sec=60,
        max_thermal_c=85,
    ),
    SLATier.SOVEREIGN: SLATarget(
        tier=SLATier.SOVEREIGN,
        availability_pct=99.9,
        latency_ttft_ms=50,
        throughput_floor_pct=95.0,
        response_time_hours=1,
        heartbeat_grace_sec=45,
        max_thermal_c=80,
    ),
}


# ── Credit Tiers ──────────────────────────────────────────────────────
# Per Report #1.B: "Automated Credit Calculation and Payouts" table
# Based on Google Cloud / Azure credit model.

CREDIT_TIERS: list[tuple[float, float, float]] = [
    # (min_uptime, max_uptime, credit_pct)
    (95.0, 99.0, 10.0),    # 95.0% - <99.0% → 10% credit
    (90.0, 95.0, 25.0),    # 90.0% - <95.0% → 25% credit
    (0.0,  90.0, 100.0),   # <90.0%          → 100% credit
]


def compute_credit_pct(uptime_pct: float) -> float:
    """Return the financial credit percentage owed based on uptime."""
    for min_up, max_up, credit in CREDIT_TIERS:
        if min_up <= uptime_pct < max_up:
            return credit
    return 0.0  # ≥99.0% uptime → no credit owed


# ── Data Models ───────────────────────────────────────────────────────

@dataclass
class DowntimePeriod:
    """A contiguous window where a host was unavailable or degraded."""
    host_id: str
    start_ts: float
    end_ts: float = 0.0
    reason: str = ""  # heartbeat_miss, thermal_throttle, gpu_error, manual
    resolved: bool = False

    @property
    def duration_sec(self) -> float:
        end = self.end_ts if self.end_ts else time.time()
        return max(0, end - self.start_ts)


@dataclass
class HostSLARecord:
    """Monthly SLA record for a host."""
    host_id: str
    tier: str
    month: str               # YYYY-MM
    total_seconds: float = 0.0
    downtime_seconds: float = 0.0
    incidents: int = 0
    credit_pct: float = 0.0
    credit_cad: float = 0.0
    enforced: bool = False

    @property
    def uptime_pct(self) -> float:
        if self.total_seconds <= 0:
            return 100.0
        return max(0, 100.0 * (1.0 - self.downtime_seconds / self.total_seconds))


@dataclass
class SLAViolation:
    """An individual SLA violation event."""
    host_id: str
    violation_type: str  # availability, latency, throughput, thermal
    severity: str        # warning, breach, critical
    metric_value: float
    threshold: float
    timestamp: float = 0.0
    details: str = ""


# ── SLA Engine ────────────────────────────────────────────────────────

class SLAEngine:
    """Tracks uptime, detects violations, calculates credits."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sla_downtime (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    host_id TEXT NOT NULL,
                    start_ts REAL NOT NULL,
                    end_ts REAL DEFAULT 0,
                    reason TEXT DEFAULT '',
                    resolved INTEGER DEFAULT 0,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                );
                CREATE TABLE IF NOT EXISTS sla_monthly (
                    host_id TEXT NOT NULL,
                    month TEXT NOT NULL,
                    tier TEXT NOT NULL DEFAULT 'community',
                    total_seconds REAL DEFAULT 0,
                    downtime_seconds REAL DEFAULT 0,
                    incidents INTEGER DEFAULT 0,
                    credit_pct REAL DEFAULT 0,
                    credit_cad REAL DEFAULT 0,
                    enforced INTEGER DEFAULT 0,
                    PRIMARY KEY (host_id, month)
                );
                CREATE TABLE IF NOT EXISTS sla_violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    host_id TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    metric_value REAL,
                    threshold REAL,
                    timestamp REAL,
                    details TEXT DEFAULT '',
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                );
                CREATE INDEX IF NOT EXISTS idx_downtime_host
                    ON sla_downtime(host_id, start_ts);
                CREATE INDEX IF NOT EXISTS idx_violations_host
                    ON sla_violations(host_id, timestamp);
            """)

    # ── Downtime Tracking ─────────────────────────────────────────────

    def record_downtime_start(self, host_id: str, reason: str = "heartbeat_miss") -> int:
        """Mark host as down. Returns downtime record ID."""
        with self._conn() as conn:
            # Check if there's already an open downtime for this host
            row = conn.execute(
                "SELECT id FROM sla_downtime WHERE host_id=? AND resolved=0 ORDER BY start_ts DESC LIMIT 1",
                (host_id,),
            ).fetchone()
            if row:
                return row["id"]  # Already tracking
            cur = conn.execute(
                "INSERT INTO sla_downtime (host_id, start_ts, reason) VALUES (?, ?, ?)",
                (host_id, time.time(), reason),
            )
            log.warning("SLA: host %s DOWN — reason=%s", host_id, reason)
            return cur.lastrowid

    def record_downtime_end(self, host_id: str) -> Optional[float]:
        """Mark host as recovered. Returns downtime duration in seconds."""
        now = time.time()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, start_ts FROM sla_downtime WHERE host_id=? AND resolved=0 ORDER BY start_ts DESC LIMIT 1",
                (host_id,),
            ).fetchone()
            if not row:
                return None
            duration = now - row["start_ts"]
            conn.execute(
                "UPDATE sla_downtime SET end_ts=?, resolved=1 WHERE id=?",
                (now, row["id"]),
            )
            log.info("SLA: host %s RECOVERED after %.0fs", host_id, duration)
            return duration

    # ── Violation Recording ───────────────────────────────────────────

    def record_violation(self, v: SLAViolation):
        """Record an SLA violation event."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO sla_violations
                   (host_id, violation_type, severity, metric_value, threshold, timestamp, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (v.host_id, v.violation_type, v.severity,
                 v.metric_value, v.threshold, v.timestamp or time.time(), v.details),
            )
        log.warning("SLA VIOLATION: host=%s type=%s severity=%s value=%.2f threshold=%.2f",
                     v.host_id, v.violation_type, v.severity, v.metric_value, v.threshold)

    # ── Telemetry-Based Checks ────────────────────────────────────────

    def check_telemetry(self, host_id: str, tier: str, metrics: dict) -> list[SLAViolation]:
        """Check live telemetry against SLA targets. Returns any violations found."""
        try:
            sla_tier = SLATier(tier)
        except ValueError:
            sla_tier = SLATier.COMMUNITY
        target = SLA_TARGETS[sla_tier]
        violations = []
        now = time.time()

        # Thermal check
        temp = metrics.get("temp", metrics.get("temperature", 0))
        if temp and temp > target.max_thermal_c:
            severity = "critical" if temp > target.max_thermal_c + 10 else "breach"
            v = SLAViolation(
                host_id=host_id, violation_type="thermal",
                severity=severity, metric_value=temp,
                threshold=target.max_thermal_c, timestamp=now,
                details=f"GPU temp {temp}°C exceeds {target.max_thermal_c}°C ceiling for {tier} tier",
            )
            violations.append(v)
            self.record_violation(v)

        # Throughput/utilization floor check
        util = metrics.get("utilization", metrics.get("gpu_util", 0))
        # Only flag if GPU is supposed to be doing work (has active jobs)
        active_jobs = metrics.get("active_jobs", 0)
        if active_jobs and active_jobs > 0 and util < target.throughput_floor_pct * 0.5:
            v = SLAViolation(
                host_id=host_id, violation_type="throughput",
                severity="warning", metric_value=util,
                threshold=target.throughput_floor_pct, timestamp=now,
                details=f"GPU utilization {util}% below floor with {active_jobs} active jobs",
            )
            violations.append(v)
            self.record_violation(v)

        # Memory error check
        mem_errors = metrics.get("memory_errors", 0)
        if mem_errors and mem_errors > 0:
            v = SLAViolation(
                host_id=host_id, violation_type="hardware",
                severity="critical", metric_value=mem_errors,
                threshold=0, timestamp=now,
                details=f"ECC uncorrectable memory errors detected: {mem_errors}",
            )
            violations.append(v)
            self.record_violation(v)

        return violations

    # ── Monthly Enforcement ───────────────────────────────────────────

    def enforce_monthly(self, host_id: str, tier: str, month: str,
                        monthly_spend_cad: float = 0.0) -> HostSLARecord:
        """Calculate monthly SLA record and credits owed.

        Args:
            host_id: The host to evaluate
            tier: SLA tier (community/secure/sovereign)
            month: YYYY-MM format
            monthly_spend_cad: The customer's total spend on this host for the month
        """
        import calendar
        from datetime import datetime

        # Calculate total seconds in month
        dt = datetime.strptime(month, "%Y-%m")
        days_in_month = calendar.monthrange(dt.year, dt.month)[1]
        total_sec = days_in_month * 86400

        # Sum downtime for this host in this month
        month_start = dt.timestamp()
        month_end = month_start + total_sec
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT start_ts, end_ts, resolved FROM sla_downtime
                   WHERE host_id=? AND start_ts < ? AND (end_ts > ? OR resolved=0)""",
                (host_id, month_end, month_start),
            ).fetchall()

        downtime_sec = 0.0
        incidents = 0
        for row in rows:
            start = max(row["start_ts"], month_start)
            end = row["end_ts"] if row["end_ts"] and row["resolved"] else min(time.time(), month_end)
            end = min(end, month_end)
            if end > start:
                downtime_sec += end - start
                incidents += 1

        record = HostSLARecord(
            host_id=host_id,
            tier=tier,
            month=month,
            total_seconds=total_sec,
            downtime_seconds=downtime_sec,
            incidents=incidents,
        )

        # Calculate credit
        record.credit_pct = compute_credit_pct(record.uptime_pct)
        record.credit_cad = round(monthly_spend_cad * record.credit_pct / 100.0, 2)
        record.enforced = True

        # Persist
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sla_monthly
                   (host_id, month, tier, total_seconds, downtime_seconds,
                    incidents, credit_pct, credit_cad, enforced)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (host_id, month, tier, total_sec, downtime_sec,
                 incidents, record.credit_pct, record.credit_cad, 1),
            )

        if record.credit_pct > 0:
            log.warning("SLA ENFORCEMENT: host=%s month=%s uptime=%.2f%% credit=%.1f%% ($%.2f CAD)",
                        host_id, month, record.uptime_pct, record.credit_pct, record.credit_cad)

        return record

    # ── Query Methods ─────────────────────────────────────────────────

    def get_host_sla(self, host_id: str, month: str) -> Optional[HostSLARecord]:
        """Get the SLA record for a host in a given month."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sla_monthly WHERE host_id=? AND month=?",
                (host_id, month),
            ).fetchone()
            if not row:
                return None
            return HostSLARecord(**dict(row))

    def get_violations(self, host_id: str, since: float = 0) -> list[dict]:
        """Get violation history for a host."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sla_violations WHERE host_id=? AND timestamp>=? ORDER BY timestamp DESC LIMIT 100",
                (host_id, since),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_active_downtimes(self) -> list[dict]:
        """Get all currently-open downtime periods."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sla_downtime WHERE resolved=0 ORDER BY start_ts ASC",
            ).fetchall()
            return [dict(r) for r in rows]

    def get_host_uptime_pct(self, host_id: str, window_sec: int = 2592000) -> float:
        """Calculate rolling uptime percentage over a window (default 30 days)."""
        now = time.time()
        cutoff = now - window_sec
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT start_ts, end_ts, resolved FROM sla_downtime
                   WHERE host_id=? AND (end_ts > ? OR resolved=0) AND start_ts < ?""",
                (host_id, cutoff, now),
            ).fetchall()
        downtime = 0.0
        for row in rows:
            start = max(row["start_ts"], cutoff)
            end = row["end_ts"] if row["end_ts"] and row["resolved"] else now
            end = min(end, now)
            if end > start:
                downtime += end - start
        return max(0, 100.0 * (1.0 - downtime / window_sec))


# ── Singleton ─────────────────────────────────────────────────────────

_sla_engine: Optional[SLAEngine] = None


def get_sla_engine() -> SLAEngine:
    global _sla_engine
    if _sla_engine is None:
        _sla_engine = SLAEngine()
    return _sla_engine
