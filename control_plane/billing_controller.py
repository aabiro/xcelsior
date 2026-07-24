"""Billing controller (§12.4, Track B B3.3) — meter-invariant reconciler.

Once an attempt has run, two invariants must hold or money is wrong:

  * **exactly one usage_meter per attempt that ran** — a missing meter is a
    billing *leak* (the customer used a GPU that was never charged);
  * **no usage_meter left open** (``completed_at IS NULL``) after its attempt
    reached a terminal state — an orphaned meter that will never settle.

This controller surfaces both as findings in the shared
``reconciliation_findings`` table. Per B0.3 rule 17 it is **report-only by
default**; only ``billing_missing_meter`` has a defined, safe remediation
(``billing.meter_job`` — idempotent per attempt, so it can never double-charge),
opted in with ``XCELSIOR_RECONCILE_ACTION_BILLING_MISSING_METER=enforce``.
Orphaned meters are surfaced for a human, never auto-mutated.

It reads the ledger and reports; it does not conceal a violation by silently
fixing it — a created meter is recorded and the finding notes the action.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

log = logging.getLogger("xcelsior")

RESOURCE_TYPE = "attempt"
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "findings_opened": self.findings_opened,
            "meters_created": self.meters_created,
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
            RESOURCE_TYPE, resource_id, tenant_id, finding_type, severity,
            summary, Jsonb(observed) if observed else None,
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
    row = conn.execute(
        """
        SELECT a.job_id, a.host_id,
               EXTRACT(EPOCH FROM a.started_at)::float8,
               EXTRACT(EPOCH FROM a.ended_at)::float8,
               j.payload
          FROM job_attempts a
          JOIN jobs j ON j.job_id = a.job_id
         WHERE a.attempt_id = %s
        """,
        (attempt_id,),
    ).fetchone()
    if row is None:
        return None
    job_id, host_id, started, ended, payload = row
    payload = payload if isinstance(payload, dict) else {}
    job = {
        "job_id": str(job_id),
        "attempt_id": attempt_id,
        "owner": payload.get("owner") or "",
        "host_id": host_id,
        "started_at": float(started or 0),
        "completed_at": float(ended or 0),
        "vram_needed_gb": payload.get("vram_needed_gb", 0),
        "pricing_mode": payload.get("pricing_mode", "on_demand"),
        "gpu_model": payload.get("gpu_model"),
    }
    host: dict[str, Any] = {"host_id": host_id or ""}
    if host_id:
        hrow = conn.execute("SELECT payload FROM hosts WHERE host_id = %s", (host_id,)).fetchone()
        if hrow and isinstance(hrow[0], dict):
            host = {"host_id": host_id, **hrow[0]}

    from billing import get_billing_engine

    meter = get_billing_engine().meter_job(job, host)
    return getattr(meter, "meter_id", None)


def reconcile_billing_meters(conn: Connection, *, limit: int = 500) -> BillingReconcileResult:
    """One sweep of the meter invariants. Report-only unless a type is enforced."""
    result = BillingReconcileResult()

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
        if enforce_missing:
            meter_id = _ensure_attempt_metered(conn, aid)
            if meter_id:
                result.meters_created.append(str(meter_id))
        else:
            if _open_finding(
                conn,
                resource_id=aid,
                finding_type=FINDING_MISSING_METER,
                severity="error",
                summary=f"attempt {aid} (job {job_id}) ran to terminal but has no usage meter",
                observed={"job_id": str(job_id)},
            ):
                result.findings_opened.append(aid)

    # ── Invariant 2: no meter left open after its attempt is terminal ──
    orphaned = conn.execute(
        """
        SELECT m.attempt_id, m.meter_id
          FROM usage_meters m
          JOIN job_attempts a ON a.attempt_id::text = m.attempt_id
         WHERE m.completed_at IS NULL
           AND a.status = ANY(%s)
         LIMIT %s
        """,
        (list(_TERMINAL), limit),
    ).fetchall()
    for attempt_id, meter_id in orphaned:
        result.scanned += 1
        aid = str(attempt_id)
        # Report-only always: closing a meter recomputes cost and is never
        # auto-mutated here (not in _ENFORCEABLE) — a human settles it.
        if _open_finding(
            conn,
            resource_id=aid,
            finding_type=FINDING_ORPHANED_METER,
            severity="warning",
            summary=f"usage meter {meter_id} for attempt {aid} is open after the attempt is terminal",
            observed={"meter_id": str(meter_id)},
        ):
            result.findings_opened.append(aid)

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
    if result.findings_opened or result.meters_created:
        log.warning(
            "billing meter reconcile: scanned=%d findings=%d meters_created=%d",
            result.scanned, len(result.findings_opened), len(result.meters_created),
        )
