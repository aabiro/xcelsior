# Xcelsior SLA Enforcement Module v2.0.0
# Automated SLA monitoring, credit calculation, and enforcement.
#
# Per REPORT_FEATURE_1.md (Report #1.B):
# - Tiered uptime targets: Community 99.0%, Secure 99.5%, Dedicated 99.9%
# - Automated credit calculation based on monthly uptime percentage
# - Downtime tracking via missed heartbeats and telemetry gaps
# - Host auto-demotion on persistent SLA breach

import calendar
import os
import time
import logging
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from money import micros_to_cad

log = logging.getLogger("xcelsior.sla")

DB_PATH = os.environ.get("XCELSIOR_SLA_DB", "xcelsior_sla.db")

# ── SLA Tier Definitions ─────────────────────────────────────────────
# Per Report #1.B: "SLA Enforcement" section


class SLATier(str, Enum):
    COMMUNITY = "community"  # Consumer hardware, best-effort
    SECURE = "secure"  # Data-center grade, SOC 2 path
    DEDICATED = "dedicated"  # Dedicated hardware, highest availability commitment


@dataclass
class SLATarget:
    """Performance and availability targets for a tier."""

    tier: SLATier
    availability_pct: float  # e.g. 99.0, 99.5, 99.9
    latency_ttft_ms: int  # Time-to-first-token max
    throughput_floor_pct: float  # % of peak benchmark score
    response_time_hours: int  # Support response SLA
    heartbeat_grace_sec: int  # Seconds before "down" flag
    max_thermal_c: int  # Thermal ceiling before degraded


SLA_TARGETS: dict[SLATier, SLATarget] = {
    SLATier.COMMUNITY: SLATarget(
        tier=SLATier.COMMUNITY,
        availability_pct=99.0,
        latency_ttft_ms=200,
        throughput_floor_pct=80.0,
        response_time_hours=4,
        heartbeat_grace_sec=90,  # 3 missed 30s heartbeats
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
    SLATier.DEDICATED: SLATarget(
        tier=SLATier.DEDICATED,
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


def _sla_tier(tier: str | SLATier) -> SLATier:
    try:
        return SLATier(tier)
    except ValueError:
        return SLATier.COMMUNITY


def credit_bands(tier: str | SLATier = SLATier.COMMUNITY) -> list[tuple[float, float, float]]:
    """``(min_uptime, max_uptime, credit_pct)`` bands for one tier.

    The first band starts at **the tier's own target**. It used to start at
    99.0% for every tier, so a Dedicated host sold at 99.9% that delivered
    99.2% breached its target and owed nothing.
    """
    target = SLA_TARGETS[_sla_tier(tier)].availability_pct
    return [
        (95.0, target, 10.0),  # 95.0% - <target → 10% credit
        (90.0, 95.0, 25.0),  # 90.0% - <95.0%  → 25% credit
        (0.0, 90.0, 100.0),  # <90.0%           → 100% credit
    ]


CREDIT_TIERS: list[tuple[float, float, float]] = credit_bands(SLATier.COMMUNITY)


def compute_credit_pct(uptime_pct: float, tier: str | SLATier = SLATier.COMMUNITY) -> float:
    """Return the financial credit percentage owed based on uptime."""
    for min_up, max_up, credit in credit_bands(tier):
        if min_up <= uptime_pct < max_up:
            return credit
    return 0.0  # at or above the tier's target → no credit owed


#: ``sla_downtime.reason`` of the zero-length row that marks when the health
#: check first observed a host. Time before it was never measured, so it is
#: neither uptime nor downtime: counting it as uptime would let a host that was
#: never watched read 100%.
MONITORING_STARTED = "monitoring_started"


def month_bounds(month: str) -> tuple[float, float]:
    """Epoch start and end of a ``YYYY-MM`` month.

    Local time on the writer, deliberately: placement's
    ``host_projection._month_bounds`` recomputes these bounds the same way to
    clamp the in-progress month, and the two must agree.
    """
    start = datetime.strptime(month, "%Y-%m")
    days = calendar.monthrange(start.year, start.month)[1]
    begin = start.timestamp()
    return begin, begin + days * 86400.0


def month_of(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m")


def _merged(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of intervals. Overlapping rows must not count the same second twice."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


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
    month: str  # YYYY-MM
    total_seconds: float = 0.0
    downtime_seconds: float = 0.0
    incidents: int = 0
    credit_pct: float = 0.0
    credit_cad: float = 0.0
    enforced: bool = False
    credit_issued_at: float = 0.0
    #: Integer micro-CAD owed to each affected customer. Derived, not stored.
    customer_credits: dict[str, int] = field(default_factory=dict)

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
    severity: str  # warning, breach, critical
    metric_value: float
    threshold: float
    timestamp: float = 0.0
    details: str = ""


# ── SLA Engine ────────────────────────────────────────────────────────


class SLAEngine:
    """Tracks uptime, detects violations, calculates credits."""

    def __init__(self, db_path: str = DB_PATH):
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

    # ── Downtime Tracking ─────────────────────────────────────────────

    def record_downtime_start(self, host_id: str, reason: str = "heartbeat_miss") -> int:
        """Mark host as down. Returns downtime record ID."""
        with self._conn() as conn:
            # Check if there's already an open downtime for this host
            row = conn.execute(
                "SELECT id FROM sla_downtime WHERE host_id=%s AND resolved=0 ORDER BY start_ts DESC LIMIT 1",
                (host_id,),
            ).fetchone()
            if row:
                return row["id"]  # Already tracking
            cur = conn.execute(
                "INSERT INTO sla_downtime (host_id, start_ts, reason) VALUES (%s, %s, %s) RETURNING id",
                (host_id, time.time(), reason),
            )
            log.warning("SLA: host %s DOWN — reason=%s", host_id, reason)
            return cur.fetchone()["id"]

    def record_downtime_end(self, host_id: str) -> Optional[float]:
        """Mark host as recovered. Returns downtime duration in seconds."""
        now = time.time()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, start_ts FROM sla_downtime WHERE host_id=%s AND resolved=0 ORDER BY start_ts DESC LIMIT 1",
                (host_id,),
            ).fetchone()
            if not row:
                return None
            duration = now - row["start_ts"]
            conn.execute(
                "UPDATE sla_downtime SET end_ts=%s, resolved=1 WHERE id=%s",
                (now, row["id"]),
            )
            log.info("SLA: host %s RECOVERED after %.0fs", host_id, duration)
            return duration

    def sync_host_liveness(
        self,
        *,
        observed: Iterable[str],
        down: Mapping[str, float],
        up: Iterable[str],
        now: float | None = None,
    ) -> None:
        """Bring the downtime log in line with one health check.

        ``observed`` is every host the check judged, ``down`` maps each host it
        has declared dead to when it first stopped answering, and ``up`` is
        every host it found alive. A host inside its grace window is observed
        but neither.

        Nothing called ``record_downtime_start`` before this, so every host
        measured 100% uptime. This is state, not transitions: a host still dead
        keeps its one open row, and a write that fails is repaired by the next
        check rather than leaving an outage that never opened or never closed.
        """
        now = time.time() if now is None else now
        observed, up = list(observed), list(up)
        with self._conn() as conn:
            if observed:
                conn.execute(
                    """INSERT INTO sla_downtime
                         (host_id, start_ts, end_ts, reason, resolved, created_at)
                       SELECT h, %s, %s, %s, 1, %s FROM unnest(%s::text[]) AS h
                        WHERE NOT EXISTS (SELECT 1 FROM sla_downtime d
                                           WHERE d.host_id = h AND d.reason = %s)""",
                    (now, now, MONITORING_STARTED, now, observed, MONITORING_STARTED),
                )
            if down:
                conn.execute(
                    """INSERT INTO sla_downtime (host_id, start_ts, reason, resolved, created_at)
                       SELECT h, s, 'heartbeat_miss', 0, %s
                         FROM unnest(%s::text[], %s::float8[]) AS d(h, s)
                        WHERE NOT EXISTS (SELECT 1 FROM sla_downtime o
                                           WHERE o.host_id = d.h AND o.resolved = 0)""",
                    (now, list(down), list(down.values())),
                )
            if up:
                conn.execute(
                    "UPDATE sla_downtime SET end_ts = %s, resolved = 1 "
                    "WHERE resolved = 0 AND host_id = ANY(%s)",
                    (now, up),
                )

    def monitored_since(self, host_id: str) -> Optional[float]:
        """When the health check first observed this host, or None if it never has."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MIN(start_ts) AS since FROM sla_downtime WHERE host_id = %s AND reason = %s",
                (host_id, MONITORING_STARTED),
            ).fetchone()
        return None if not row or row["since"] is None else float(row["since"])

    # ── Violation Recording ───────────────────────────────────────────

    def record_violation(self, v: SLAViolation):
        """Record an SLA violation event."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO sla_violations
                   (host_id, violation_type, severity, metric_value, threshold, timestamp, details)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    v.host_id,
                    v.violation_type,
                    v.severity,
                    v.metric_value,
                    v.threshold,
                    v.timestamp or time.time(),
                    v.details,
                ),
            )
        log.warning(
            "SLA VIOLATION: host=%s type=%s severity=%s value=%.2f threshold=%.2f",
            v.host_id,
            v.violation_type,
            v.severity,
            v.metric_value,
            v.threshold,
        )

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
                host_id=host_id,
                violation_type="thermal",
                severity=severity,
                metric_value=temp,
                threshold=target.max_thermal_c,
                timestamp=now,
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
                host_id=host_id,
                violation_type="throughput",
                severity="warning",
                metric_value=util,
                threshold=target.throughput_floor_pct,
                timestamp=now,
                details=f"GPU utilization {util}% below floor with {active_jobs} active jobs",
            )
            violations.append(v)
            self.record_violation(v)

        # Memory error check
        mem_errors = metrics.get("memory_errors", 0)
        if mem_errors and mem_errors > 0:
            v = SLAViolation(
                host_id=host_id,
                violation_type="hardware",
                severity="critical",
                metric_value=mem_errors,
                threshold=0,
                timestamp=now,
                details=f"ECC uncorrectable memory errors detected: {mem_errors}",
            )
            violations.append(v)
            self.record_violation(v)

        return violations

    # ── Monthly Enforcement ───────────────────────────────────────────

    def _outages(self, host_id: str, start: float, end: float) -> list[tuple[float, float]]:
        """Merged downtime of one host clipped to ``[start, end)``; open rows run to ``end``."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT start_ts, end_ts, resolved FROM sla_downtime
                   WHERE host_id = %s AND reason <> %s
                     AND start_ts < %s AND (end_ts > %s OR resolved = 0)""",
                (host_id, MONITORING_STARTED, end, start),
            ).fetchall()
        spans = []
        for row in rows:
            s = max(float(row["start_ts"]), start)
            e = min(float(row["end_ts"]) if row["resolved"] and row["end_ts"] else end, end)
            if e > s:
                spans.append((s, e))
        return _merged(spans)

    def _affected_spend(
        self,
        host_id: str,
        month_start: float,
        month_end: float,
        outages: list[tuple[float, float]],
    ) -> dict[str, int]:
        """Each affected customer's own spend on this host in the month, in micros.

        A customer is affected when they were billed on the host when it went
        down or while it was down. Billing runs every
        ``PERIODIC_BILLING_INTERVAL_SEC``, and a job requeued off a dead host
        stops being billed, so the cycle that was open when the host went quiet
        can end up to one interval before the outage began.

        This replaced paying the alphabetically first customer who had *ever*
        completed a job on the host, in any month, and nobody else.
        """
        from billing import PERIODIC_BILLING_INTERVAL_SEC

        with self._conn() as conn:
            cycles = conn.execute(
                """SELECT customer_id, period_start, period_end, amount_micros
                     FROM billing_cycles
                    WHERE host_id = %s AND status = 'charged' AND amount_micros > 0
                      AND period_start < %s AND period_end > %s""",
                (host_id, month_end, month_start - PERIODIC_BILLING_INTERVAL_SEC),
            ).fetchall()
        affected = {
            c["customer_id"]
            for c in cycles
            for s, e in outages
            if c["period_end"] >= s - PERIODIC_BILLING_INTERVAL_SEC and c["period_start"] <= e
        }
        spend: dict[str, int] = {}
        for c in cycles:
            if c["customer_id"] in affected and month_start <= c["period_start"] < month_end:
                spend[c["customer_id"]] = spend.get(c["customer_id"], 0) + int(c["amount_micros"])
        return spend

    def enforce_monthly(
        self, host_id: str, tier: str, month: str, *, now: float | None = None
    ) -> HostSLARecord:
        """Measure one host's month and the credit each affected customer is owed.

        Only monitored time counts: from when the health check first observed
        the host, up to the end of the month or ``now``, whichever is first.
        The in-progress month is measured over the time that has elapsed. The
        customers' spend comes from ``billing_cycles``; it used to be typed in
        by hand on the admin call, which made the credit whatever was typed.
        """
        now = time.time() if now is None else now
        month_start, month_end = month_bounds(month)
        since = self.monitored_since(host_id)
        window_start = max(month_start, since) if since is not None else month_end
        window_end = min(month_end, now)
        total_sec = max(0.0, window_end - window_start)

        outages = self._outages(host_id, window_start, window_end) if total_sec else []
        record = HostSLARecord(
            host_id=host_id,
            tier=tier,
            month=month,
            total_seconds=total_sec,
            downtime_seconds=sum(e - s for s, e in outages),
            incidents=len(outages),
        )
        record.credit_pct = compute_credit_pct(record.uptime_pct, tier) if total_sec else 0.0
        if record.credit_pct > 0:
            credit_bps = round(record.credit_pct * 100)
            record.customer_credits = {
                customer_id: spent * credit_bps // 10_000
                for customer_id, spent in self._affected_spend(
                    host_id, month_start, month_end, outages
                ).items()
                if spent * credit_bps // 10_000 > 0
            }
        credit_micros = sum(record.customer_credits.values())
        record.credit_cad = micros_to_cad(credit_micros)
        record.enforced = True

        # Persist
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO sla_monthly
                   (host_id, month, tier, total_seconds, downtime_seconds,
                    incidents, credit_pct, credit_micros, enforced)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (host_id, month) DO UPDATE SET
                     tier = EXCLUDED.tier, total_seconds = EXCLUDED.total_seconds,
                     downtime_seconds = EXCLUDED.downtime_seconds, incidents = EXCLUDED.incidents,
                     credit_pct = EXCLUDED.credit_pct, credit_micros = EXCLUDED.credit_micros,
                     enforced = EXCLUDED.enforced""",
                (
                    host_id,
                    month,
                    tier,
                    total_sec,
                    record.downtime_seconds,
                    record.incidents,
                    record.credit_pct,
                    # Integer micros in storage; the record keeps CAD for callers.
                    credit_micros,
                    1,
                ),
            )

        if record.credit_pct > 0:
            log.warning(
                "SLA ENFORCEMENT: host=%s month=%s uptime=%.2f%% credit=%.1f%% "
                "($%.2f CAD across %d customers)",
                host_id,
                month,
                record.uptime_pct,
                record.credit_pct,
                record.credit_cad,
                len(record.customer_credits),
            )

        return record

    # ── Query Methods ─────────────────────────────────────────────────

    def get_host_sla(self, host_id: str, month: str) -> Optional[HostSLARecord]:
        """Get the SLA record for a host in a given month.

        Columns are named rather than splatted: ``HostSLARecord(**row)`` raised
        on ``credit_micros``, ``credit_issued_at`` and ``credit_tx_id``, which
        the record has never had, so this returned nothing for any stored row.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sla_monthly WHERE host_id=%s AND month=%s",
                (host_id, month),
            ).fetchone()
        if not row:
            return None
        return HostSLARecord(
            host_id=row["host_id"],
            tier=row["tier"],
            month=row["month"],
            total_seconds=float(row["total_seconds"] or 0),
            downtime_seconds=float(row["downtime_seconds"] or 0),
            incidents=int(row["incidents"] or 0),
            credit_pct=float(row["credit_pct"] or 0),
            credit_cad=micros_to_cad(row["credit_micros"] or 0),
            enforced=bool(row["enforced"]),
            credit_issued_at=float(row.get("credit_issued_at") or 0),
        )

    def get_violations(self, host_id: str, since: float = 0) -> list[dict]:
        """Get violation history for a host."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sla_violations WHERE host_id=%s AND timestamp>=%s ORDER BY timestamp DESC LIMIT 100",
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
                   WHERE host_id=%s AND (end_ts > %s OR resolved=0) AND start_ts < %s""",
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

    # ── Auto-Credit Issuance ──────────────────────────────────────────

    def _hosts_with_tier(self, conn, host_ids: list[str]) -> list[tuple[str, str]]:
        rows = conn.execute(
            """SELECT h.id AS host_id, COALESCE(hosts.payload->>'sla_tier', 'community') AS tier
                 FROM unnest(%s::text[]) AS h(id)
                 LEFT JOIN hosts ON hosts.host_id = h.id
                ORDER BY h.id""",
            (host_ids,),
        ).fetchall()
        return [(r["host_id"], r["tier"]) for r in rows]

    def settle_month(self, month: str, *, now: float | None = None) -> list[dict]:
        """Credit every affected customer for a **closed** month, once.

        An open month is never settled: its uptime can still fall and its spend
        can still grow, and each customer's credit is keyed on the month, so an
        early partial credit could never be topped up.

        Each credit is idempotent on ``sla:{host}:{month}:{customer}``. The month
        is marked issued only when every customer's credit landed, so a failure
        is retried next run and a credit that already landed is not paid twice.
        """
        now = time.time() if now is None else now
        month_start, month_end = month_bounds(month)
        if month_end > now:
            return []
        with self._conn() as conn:
            down_hosts = [
                r["host_id"]
                for r in conn.execute(
                    """SELECT DISTINCT host_id FROM sla_downtime
                        WHERE reason <> %s AND start_ts < %s AND (end_ts > %s OR resolved = 0)""",
                    (MONITORING_STARTED, month_end, month_start),
                ).fetchall()
            ]
            hosts = self._hosts_with_tier(conn, down_hosts)

        from billing import get_billing_engine

        billing = get_billing_engine()
        issued: list[dict] = []
        for host_id, tier in hosts:
            existing = self.get_host_sla(host_id, month)
            if existing and existing.credit_issued_at:
                continue
            record = self.enforce_monthly(host_id, tier, month, now=now)
            tx_ids, settled = [], True
            for customer_id, micros in sorted(record.customer_credits.items()):
                try:
                    # A service credit, not money received: grant_credit records
                    # tx_type='credit', so revenue and FINTRAC totals that sum
                    # deposits never count it as cash paid in. The description
                    # is part of the idempotency check, so it holds nothing that
                    # could change between retries of a closed month.
                    tx = billing.grant_credit(
                        customer_id,
                        micros_to_cad(micros),
                        description=(
                            f"SLA credit: host {host_id}, {month}, "
                            f"{record.uptime_pct:.2f}% uptime, {record.credit_pct:g}% credit"
                        ),
                        idempotency_key=f"sla:{host_id}:{month}:{customer_id}",
                    )
                except Exception:
                    log.exception("SLA credit failed: host=%s customer=%s", host_id, customer_id)
                    settled = False
                    continue
                tx_ids.append(tx.get("tx_id", ""))
                if not tx.get("dedup"):
                    issued.append(
                        {
                            "host_id": host_id,
                            "month": month,
                            "customer_id": customer_id,
                            "credit_cad": micros_to_cad(micros),
                            "tx_id": tx.get("tx_id", ""),
                        }
                    )
                    log.info(
                        "SLA CREDIT ISSUED: host=%s month=%s customer=%s $%.2f CAD",
                        host_id,
                        month,
                        customer_id,
                        micros_to_cad(micros),
                    )
            if settled:
                with self._conn() as conn:
                    conn.execute(
                        "UPDATE sla_monthly SET credit_issued_at = %s, credit_tx_id = %s "
                        "WHERE host_id = %s AND month = %s",
                        (now, ",".join(tx_ids), host_id, month),
                    )
        return issued

    def run_cycle(self, *, now: float | None = None) -> list[dict]:
        """The hourly task: measure this month, settle last month.

        This month's records are what placement reads as measured uptime, so
        they are refreshed for every monitored host. Last month is settled once
        it has closed; ``settle_month`` makes every later run a no-op.
        """
        now = time.time() if now is None else now
        this_month = month_of(now)
        month_start, month_end = month_bounds(this_month)
        with self._conn() as conn:
            monitored = [
                r["host_id"]
                for r in conn.execute(
                    "SELECT DISTINCT host_id FROM sla_downtime WHERE reason = %s AND start_ts < %s",
                    (MONITORING_STARTED, month_end),
                ).fetchall()
            ]
            hosts = self._hosts_with_tier(conn, monitored)
        for host_id, tier in hosts:
            try:
                self.enforce_monthly(host_id, tier, this_month, now=now)
            except Exception:
                log.exception("SLA measurement failed: host=%s month=%s", host_id, this_month)
        return self.settle_month(month_of(month_start - 1), now=now)


# ── Singleton ─────────────────────────────────────────────────────────

_sla_engine: Optional[SLAEngine] = None


def get_sla_engine() -> SLAEngine:
    global _sla_engine
    if _sla_engine is None:
        _sla_engine = SLAEngine()
    return _sla_engine
