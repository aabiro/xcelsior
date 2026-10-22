"""Billing controller (§12.4, Track B B3.3) — meter-invariant reconciler.

Running attempts need an active meter; terminal attempts need settled usage.
The opt-in ``billing_active_meter_missing`` repair requires current fenced
running authority and a fresh observation. All three actions default to reporting.

Once an attempt has run, two terminal invariants must hold or money is wrong:

  * **exactly one usage_meter per attempt that ran** — a missing meter is a
    billing *leak* (the customer used a GPU that was never charged);
  * **no usage_meter left open** (``completed_at IS NULL``) after its attempt
    reached a terminal state — an orphaned meter that will never settle.

This controller surfaces violations as findings in the shared
``reconciliation_findings`` table. Per B0.3 rule 17 it is **report-only by
default**. The terminal finding types have defined remediations:
  - ``billing_missing_meter`` (``billing.meter_job`` — idempotent per attempt),
    opted in with ``XCELSIOR_RECONCILE_ACTION_BILLING_MISSING_METER=enforce``.
  - ``billing_orphaned_meter`` (safely closes orphaned meters using attempt
    terminal timestamps and micro-CAD cost calculation), opted in with
    ``XCELSIOR_RECONCILE_ACTION_BILLING_ORPHANED_METER=enforce``.

It reads the ledger and reports; it does not conceal a violation by silently
fixing it — created or closed meters are recorded and findings note the action.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

log = logging.getLogger("xcelsior")

RESOURCE_TYPE = "attempt"
FINDING_ACTIVE_METER = "billing_active_meter_missing"
FINDING_MISSING_METER = "billing_missing_meter"
FINDING_ORPHANED_METER = "billing_orphaned_meter"

# Terminal attempt states that *ran* and therefore must carry a meter. An
# attempt that never started (reserved/lease_offered) or never consumed GPU
# (cancelled/lost/fenced before start) is deliberately excluded.
_BILLABLE_TERMINAL = ("succeeded", "failed", "preempted")
_TERMINAL = ("succeeded", "failed", "cancelled", "preempted", "lost", "fenced")


@dataclass
class BillingReconcileResult:
    scanned: int = 0
    findings_opened: list[str] = field(default_factory=list)
    meters_created: list[str] = field(default_factory=list)
    meters_closed: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "findings_opened": self.findings_opened,
            "meters_created": self.meters_created,
            "meters_closed": self.meters_closed,
        }


def _enforce(finding_type: str) -> bool:
    from control_plane.reconcile import ActionPolicy, action_policy_for

    return action_policy_for(finding_type) == ActionPolicy.ENFORCE


def _open_finding(
    conn: Connection,
    *,
    resource_id: str,
    finding_type: str,
    severity: str,
    summary: str,
    tenant_id: str | None = None,
    observed: dict[str, Any] | None = None,
) -> bool:
    """Open one finding per (attempt, type); dedupe against an already-open one."""
    existing = conn.execute(
        """
        SELECT 1 FROM reconciliation_findings
         WHERE resource_type = %s AND resource_id = %s
           AND finding_type = %s AND resolved_at IS NULL
         LIMIT 1
        """,
        (RESOURCE_TYPE, resource_id, finding_type),
    ).fetchone()
    if existing is not None:
        return False
    conn.execute(
        """
        INSERT INTO reconciliation_findings
            (resource_type, resource_id, tenant_id, finding_type, severity,
             summary, observed, action_taken)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'report_only')
        """,
        (
            RESOURCE_TYPE,
            resource_id,
            tenant_id,
            finding_type,
            severity,
            summary,
            Jsonb(observed) if observed else None,
        ),
    )
    return True


def _ensure_attempt_metered(conn: Connection, attempt_id: str) -> str | None:
    """Enforce action for a missing meter: meter the attempt idempotently.

    Reconstructs the job/host dicts ``billing.meter_job`` expects (with an
    explicit ``attempt_id`` so it meters *this* attempt, not the job's current
    one) and calls it. ``meter_job`` collapses on the ``attempt_id`` unique
    index, so a concurrent or repeated enforce creates exactly one meter.
    """
    from psycopg.errors import LockNotAvailable
    from control_plane.attempt_billing import sync_attempt_billing

    try:
        with conn.transaction():
            # Same order as normal settlement; contention defers this repair.
            row = conn.execute(
                """SELECT j.job_id FROM jobs j JOIN job_attempts a USING(job_id)
                WHERE a.attempt_id=%s FOR UPDATE OF j NOWAIT""",
                (attempt_id,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "SELECT attempt_id FROM job_attempts WHERE attempt_id=%s FOR UPDATE NOWAIT",
                (attempt_id,),
            ).fetchone()
            sync_attempt_billing(conn, attempt_id)
            meter = conn.execute(
                "SELECT meter_id FROM usage_meters WHERE attempt_id=%s", (attempt_id,)
            ).fetchone()
            return str(meter[0]) if meter else None
    except LockNotAvailable:
        return None


def _close_orphaned_meter(conn: Connection, attempt_id: str, meter_id: str) -> str | None:
    """Enforce action for an orphaned open meter: close it idempotently.

    Uses the attempt's terminal timestamps to complete the meter, computing
    duration and total cost in micro-CAD. Idempotent: once completed_at is set,
    subsequent sweeps or concurrent calls are no-ops.
    """
    from billing import settle_attempt_meter

    meter = settle_attempt_meter(conn, attempt_id, meter_id)
    return meter.meter_id if meter else None


def _reconcile_active_meters(conn: Connection, result: BillingReconcileResult, limit: int) -> None:
    from psycopg.errors import LockNotAvailable
    from control_plane.active_meter import open_active_meter

    candidates = conn.execute(
        """SELECT a.attempt_id::text, a.job_id, j.tenant_id FROM job_attempts a
             JOIN jobs j ON j.active_attempt_id=a.attempt_id AND j.job_id=a.job_id
            WHERE a.status='running'
              AND NOT EXISTS (SELECT 1 FROM usage_meters m WHERE m.attempt_id=a.attempt_id::text
                              AND (m.completed_at IS NULL OR m.completed_at=0))
            ORDER BY a.started_at NULLS LAST, a.attempt_id LIMIT %s""",
        (limit,),
    ).fetchall()
    for aid, job_id, tenant_id in candidates:
        # Serialize finding creation across billing sweeps without waiting on
        # another replica. The attempt's locks separately fence lifecycle races.
        lock_row = conn.execute(
            "SELECT pg_try_advisory_xact_lock(hashtextextended(%s, 719))", (aid,)
        ).fetchone()
        if not lock_row or not lock_row[0]:
            continue
        if conn.execute(
            "SELECT 1 FROM usage_meters WHERE attempt_id=%s "
            "AND (completed_at IS NULL OR completed_at=0)",
            (aid,),
        ).fetchone():
            continue
        result.scanned += 1
        if _open_finding(
            conn,
            resource_id=aid,
            finding_type=FINDING_ACTIVE_METER,
            severity="warning",
            summary=f"running attempt {aid} has no active meter",
            tenant_id=tenant_id,
            observed={"job_id": job_id},
        ):
            result.findings_opened.append(aid)
        if not _enforce(FINDING_ACTIVE_METER):
            continue
        try:
            with conn.transaction():
                evidence = open_active_meter(conn, aid)
        except LockNotAvailable:
            evidence = {"reason": "authority_locked"}
        mid = evidence.get("meter_id")
        conn.execute(
            """UPDATE reconciliation_findings SET action_taken=%s, action_result=%s,
                      resolved_at=CASE WHEN %s THEN clock_timestamp() ELSE NULL END
                 WHERE resource_type='attempt' AND resource_id=%s
                   AND finding_type=%s AND resolved_at IS NULL""",
            (
                "open_active_meter" if mid else "active_meter_refused",
                Jsonb(evidence),
                bool(mid),
                aid,
                FINDING_ACTIVE_METER,
            ),
        )
        if mid:
            result.meters_created.append(mid)


def _record_meter_action(conn, attempt_id: str, finding_type: str, meter_id: str | None) -> None:
    conn.execute(
        """UPDATE reconciliation_findings SET action_taken=%s, action_result=%s,
                  resolved_at=CASE WHEN %s THEN clock_timestamp() ELSE NULL END
             WHERE resource_type='attempt' AND resource_id=%s AND finding_type=%s
               AND resolved_at IS NULL""",
        (
            "enforce" if meter_id else "meter_repair_refused",
            Jsonb(
                {"meter_id": meter_id, "reason": None if meter_id else "terminal_evidence_invalid"}
            ),
            bool(meter_id),
            attempt_id,
            finding_type,
        ),
    )


def reconcile_billing_meters(conn: Connection, *, limit: int = 500) -> BillingReconcileResult:
    """One sweep of the meter invariants. Report-only unless a type is enforced."""
    result = BillingReconcileResult()
    _reconcile_active_meters(conn, result, limit)

    # ── Invariant 1: every attempt that ran carries exactly one meter ──
    missing = conn.execute(
        """
        SELECT a.attempt_id, a.job_id
          FROM job_attempts a
          LEFT JOIN usage_meters m ON m.attempt_id = a.attempt_id::text
         WHERE a.status = ANY(%s)
           AND a.started_at IS NOT NULL
           AND a.ended_at IS NOT NULL
           AND m.meter_id IS NULL
         ORDER BY a.ended_at ASC
         LIMIT %s
        """,
        (list(_BILLABLE_TERMINAL), limit),
    ).fetchall()
    enforce_missing = _enforce(FINDING_MISSING_METER)
    for attempt_id, job_id in missing:
        result.scanned += 1
        aid = str(attempt_id)
        if _open_finding(
            conn,
            resource_id=aid,
            finding_type=FINDING_MISSING_METER,
            severity="error",
            summary=f"attempt {aid} (job {job_id}) ran to terminal but has no usage meter",
            observed={"job_id": str(job_id)},
        ):
            result.findings_opened.append(aid)
        if enforce_missing:
            meter_id = _ensure_attempt_metered(conn, aid)
            if meter_id:
                result.meters_created.append(str(meter_id))
            _record_meter_action(conn, aid, FINDING_MISSING_METER, meter_id)

    # ── Invariant 2: no meter left open after its attempt is terminal ──
    orphaned = conn.execute(
        """
        SELECT m.attempt_id, m.meter_id
          FROM usage_meters m
          JOIN job_attempts a ON a.attempt_id::text = m.attempt_id
         WHERE (m.completed_at IS NULL OR m.completed_at=0)
           AND a.status = ANY(%s)
         LIMIT %s
        """,
        (list(_TERMINAL), limit),
    ).fetchall()
    enforce_orphaned = _enforce(FINDING_ORPHANED_METER)
    for attempt_id, meter_id in orphaned:
        result.scanned += 1
        aid = str(attempt_id)
        mid = str(meter_id)
        if _open_finding(
            conn,
            resource_id=aid,
            finding_type=FINDING_ORPHANED_METER,
            severity="warning",
            summary=f"usage meter {mid} is open after attempt {aid} became terminal",
            observed={"meter_id": mid},
        ):
            result.findings_opened.append(aid)
        if enforce_orphaned:
            closed_id = _close_orphaned_meter(conn, aid, mid)
            if closed_id:
                result.meters_closed.append(str(closed_id))
            _record_meter_action(conn, aid, FINDING_ORPHANED_METER, closed_id)

    # A normal completion writer may have restored the invariant between sweeps.
    # Never resolve a completed-meter conflict on an attempt that is still running.
    conn.execute(
        """UPDATE reconciliation_findings f SET resolved_at=clock_timestamp()
            WHERE f.resource_type='attempt' AND f.finding_type=%s AND f.resolved_at IS NULL
              AND EXISTS (SELECT 1 FROM job_attempts a WHERE a.attempt_id::text=f.resource_id
                          AND (a.status=ANY(%s) OR EXISTS (
                              SELECT 1 FROM usage_meters m WHERE m.attempt_id=a.attempt_id::text
                              AND m.job_id=a.job_id AND m.host_id=a.host_id
                              AND (m.completed_at IS NULL OR m.completed_at=0))))""",
        (FINDING_ACTIVE_METER, list(_TERMINAL)),
    )
    # This controller owns its findings' resolution (the host reconciler no
    # longer clears them): resolve each once its invariant holds again.
    conn.execute(
        """UPDATE reconciliation_findings f SET resolved_at=clock_timestamp()
            WHERE f.resource_type='attempt' AND f.finding_type=%s AND f.resolved_at IS NULL
              AND NOT EXISTS (SELECT 1 FROM job_attempts a
                               WHERE a.attempt_id::text=f.resource_id
                                 AND a.status=ANY(%s)
                                 AND a.started_at IS NOT NULL AND a.ended_at IS NOT NULL
                                 AND NOT EXISTS (SELECT 1 FROM usage_meters m
                                                  WHERE m.attempt_id=a.attempt_id::text))""",
        (FINDING_MISSING_METER, list(_BILLABLE_TERMINAL)),
    )
    conn.execute(
        """UPDATE reconciliation_findings f SET resolved_at=clock_timestamp()
            WHERE f.resource_type='attempt' AND f.finding_type=%s AND f.resolved_at IS NULL
              AND NOT EXISTS (SELECT 1 FROM usage_meters m
                                JOIN job_attempts a ON a.attempt_id::text=m.attempt_id
                               WHERE m.attempt_id=f.resource_id
                                 AND (m.completed_at IS NULL OR m.completed_at=0)
                                 AND a.status=ANY(%s))""",
        (FINDING_ORPHANED_METER, list(_TERMINAL)),
    )

    return result


def reconcile_billing_meters_task() -> None:
    """Durable `scheduled_tasks` entry point (§12.4).

    Runs one meter-invariant sweep in its own control-plane transaction.
    Report-only by default; enforcement is per-type via the reconciler action
    env var. Logs only when something was found so a healthy sweep is quiet.
    """
    from control_plane.db import control_plane_transaction

    with control_plane_transaction() as conn:
        result = reconcile_billing_meters(conn)
    if result.findings_opened or result.meters_created or result.meters_closed:
        log.warning(
            "billing meter reconcile: scanned=%d findings=%d meters_created=%d meters_closed=%d",
            result.scanned,
            len(result.findings_opened),
            len(result.meters_created),
            len(result.meters_closed),
        )
