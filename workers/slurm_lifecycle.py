"""A Slurm-placed attempt's life, from what the reconciler saw (blueprint 2.4).

A host attempt learns its status from the worker's fenced reports. A Slurm
attempt has no worker on the other end: the reconciler's observation of its
mapping is the report. Each pass of this task (the ``slurm_lifecycle`` task,
every 30 s):

1. **Running.** An active attempt whose mapping is observed running becomes
   ``running`` from the mapping's start, and its job moves from
   assigned/scheduled to running. Before this the job read ``assigned`` for
   as long as Slurm ran it.
2. **Finished.** An attempt whose mapping has settled is closed:
   completed → succeeded, failed → failed, cancelled → cancelled, preempted →
   preempted, and a mapping closed ``unknown`` (its id reused by Slurm,
   migration 153) → lost.
3. **Charged,** only when the placement was made in a charging mode
   (``slurm_submissions.charge``, migration 163) and charging is on now.
   ``slurm_billing.settle_finished_attempt`` closes the attempt, debits the
   ledger's sum and marks its rows charged in one transaction. A shadow
   placement is closed and never charged, whatever the switch says later.

Each attempt is its own transaction, so one that can't settle (an unpriced
segment, a wallet that refuses) is logged and tried again next pass without
holding the rest back. Every transition appends the
``job.v1.attempt_status_changed`` event a host attempt appends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("xcelsior-slurm-lifecycle")

#: Attempts per pass; the rest wait their turn.
LIFECYCLE_BATCH = 50

_ACTIVE = (
    "reserved",
    "command_pending",
    "lease_offered",
    "lease_claimed",
    "starting",
    "running",
)
_BEFORE_RUNNING = tuple(s for s in _ACTIVE if s != "running")

#: Settled mapping state → (attempt status, job status).
_CLOSE = {
    "completed": ("succeeded", "completed"),
    "failed": ("failed", "failed"),
    "cancelled": ("cancelled", "cancelled"),
    "preempted": ("preempted", "preempted"),
    "unknown": ("lost", "failed"),
}


@dataclass
class LifecyclePass:
    running: list = field(default_factory=list)  # attempt ids moved to running
    closed: list = field(default_factory=list)  # (attempt id, status), not charged
    charged: list = field(default_factory=list)  # (attempt id, status, micros)
    errors: list = field(default_factory=list)  # (attempt id, message)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _due(conn, charging: bool, limit: int) -> list[dict]:
    from psycopg.rows import dict_row

    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            """
            SELECT s.attempt_id::text AS attempt_id, s.xcelsior_job_id AS job_id,
                   s.cluster_id, s.charge, a.status AS attempt_status, a.fencing_token,
                   m.observed_state, m.terminal_at IS NOT NULL AS settled,
                   m.started_at, m.ended_at
              FROM slurm_submissions s
              JOIN job_attempts a ON a.attempt_id = s.attempt_id
              JOIN slurm_job_mappings m ON m.xcelsior_attempt_id = s.attempt_id
             WHERE s.state = 'submitted'
               AND (
                    (a.status = ANY(%(active)s)
                     AND (m.terminal_at IS NOT NULL
                          OR (m.observed_state = 'running' AND a.status <> 'running')))
                 OR (%(charging)s AND s.charge AND m.terminal_at IS NOT NULL
                     AND m.observed_state <> 'unknown'
                     AND EXISTS (SELECT 1 FROM slurm_billing_segments b
                                  WHERE b.attempt_id = s.attempt_id AND NOT b.charged))
               )
             ORDER BY m.terminal_at NULLS FIRST, s.created_at
             LIMIT %(limit)s
            """,
            {"active": list(_ACTIVE), "charging": charging, "limit": limit},
        ).fetchall()


def _event(conn, row: dict, status: str) -> None:
    from control_plane.outbox import append_event

    append_event(
        conn,
        aggregate_type="job",
        aggregate_id=row["job_id"],
        event_type="job.v1.attempt_status_changed",
        payload={
            "attempt_id": row["attempt_id"],
            "host_id": None,
            "target": f"slurm:{row['cluster_id']}",
            "status": status,
            "fencing_token": row["fencing_token"],
        },
        idempotency_key=f"attempt_status:{row['attempt_id']}:{status}",
    )


def _mark_running(conn, row: dict) -> bool:
    from control_plane.state import job_phase

    moved = conn.execute(
        """
        UPDATE job_attempts
           SET status = 'running',
               started_at = COALESCE(started_at, %s, clock_timestamp())
         WHERE attempt_id = %s AND status = ANY(%s)
        """,
        (row["started_at"], row["attempt_id"], list(_BEFORE_RUNNING)),
    )
    if moved.rowcount != 1:
        return False
    conn.execute(
        """
        UPDATE jobs
           SET status = 'running', phase = %s, version = version + 1,
               updated_at = clock_timestamp()
         WHERE job_id = %s AND active_attempt_id = %s AND status = 'assigned'
        """,
        (job_phase("running"), row["job_id"], row["attempt_id"]),
    )
    _event(conn, row, "running")
    return True


def _close(conn, row: dict) -> str | None:
    """Close the attempt from its settled mapping, without charging."""
    from control_plane.state import job_phase

    attempt_status, job_status = _CLOSE.get(row["observed_state"], ("lost", "failed"))
    moved = conn.execute(
        """
        UPDATE job_attempts
           SET status = %s,
               started_at = COALESCE(started_at, %s),
               ended_at = COALESCE(ended_at, %s, clock_timestamp())
         WHERE attempt_id = %s AND status = ANY(%s)
        """,
        (attempt_status, row["started_at"], row["ended_at"], row["attempt_id"], list(_ACTIVE)),
    )
    if moved.rowcount != 1:
        return None
    conn.execute(
        """
        UPDATE jobs
           SET status = %s, phase = %s, version = version + 1,
               updated_at = clock_timestamp()
         WHERE job_id = %s AND active_attempt_id = %s AND status IN ('assigned', 'running')
        """,
        (job_status, job_phase(job_status), row["job_id"], row["attempt_id"]),
    )
    _event(conn, row, attempt_status)
    return attempt_status


def _advance_one(row: dict, charging: bool, result: LifecyclePass) -> None:
    from control_plane.db import run_transaction

    attempt_id = row["attempt_id"]
    if not row["settled"]:
        if run_transaction(lambda conn: _mark_running(conn, row), what="slurm_attempt_running"):
            result.running.append(attempt_id)
        return
    chargeable = charging and row["charge"] and row["observed_state"] != "unknown"
    if not chargeable:
        status = run_transaction(lambda conn: _close(conn, row), what="slurm_attempt_close")
        if status:
            result.closed.append((attempt_id, status))
        return

    from slurm_billing import settle_finished_attempt

    def settle(conn):
        settled = settle_finished_attempt(conn, attempt_id)
        if not settled.get("already_settled") and row["attempt_status"] in _ACTIVE:
            _event(conn, row, settled["attempt_status"])
        return settled

    settled = run_transaction(settle, what="slurm_attempt_settle")
    if not settled.get("already_settled"):
        result.charged.append((attempt_id, settled["attempt_status"], settled["charged_micros"]))


def advance_attempts(*, charging: bool | None = None, limit: int = LIFECYCLE_BATCH) -> dict:
    """One pass: running attempts marked, finished ones closed or settled.

    ``charging`` defaults to the XCELSIOR_SLURM_PLACEMENT mode's answer
    (``admin``, ``allowlist`` or ``all``). Off, nothing is charged, and an
    attempt whose placement was chargeable stays charge-pending: it is
    closed, and charged on the first pass after charging is on again.
    Closing runs in every mode, ``off`` included, so turning placement off
    never strands an attempt Slurm is still running.
    """
    from control_plane.scheduler.config import slurm_charging_enabled
    from db import pg_connection

    result = LifecyclePass()
    if charging is None:
        charging = slurm_charging_enabled()
    with pg_connection() as conn:
        rows = _due(conn, charging, limit)
    for row in rows:
        try:
            _advance_one(row, charging, result)
        except Exception as e:  # one attempt must not stop the rest
            result.errors.append((row["attempt_id"], f"{type(e).__name__}: {e}"[:300]))
            log.warning("slurm lifecycle %s: %s", row["attempt_id"], e)
    if result.running or result.closed or result.charged:
        log.info(
            "slurm lifecycle: %d running, %d closed, %d settled and charged",
            len(result.running),
            len(result.closed),
            len(result.charged),
        )
    return result.as_dict()
