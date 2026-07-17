"""Fenced attempt status transitions (§8.1 write gate, §9.2 state machine).

The worker reports what actually happened to its attempt —
``lease_claimed`` → ``starting`` → ``running`` → ``succeeded``/``failed``
— and every report must carry the full authority tuple
(job, attempt, host, fence). A report that fails the fence gate is
rejected *before* any state, billing, or routing is touched: a fenced-out
worker learns definitively that its authority is gone and must stop its
container (§11.5).

Terminal reports settle everything the reservation created, exactly once
and atomically: attempt terminal + timestamps, device allocations
released, lease released, job projection updated (active attempt
cleared), outbox event appended.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

from psycopg import Connection
from psycopg.types.json import Jsonb

from control_plane.leases import require_current_fence

log = logging.getLogger("xcelsior.control_plane.attempts")

# Statuses a worker may report, in causal order. Terminal pair shares a
# rank: whichever arrives first wins; the fence gate rejects the second
# (a terminal attempt is no longer the active authority).
_REPORT_ORDER: dict[str, int] = {
    "lease_claimed": 0,
    "starting": 1,
    "running": 2,
    "succeeded": 3,
    "failed": 3,
}

_TERMINAL = ("succeeded", "failed")

# Attempt status → legacy jobs.status projection (the 059 trigger derives
# phase/desired_state from this, so one write keeps everything coherent).
_JOB_PROJECTION = {
    "lease_claimed": "leased",
    "starting": "starting",
    "running": "running",
    "succeeded": "completed",
    "failed": "failed",
}


class AttemptStatusRejected(Exception):
    """Report refused for a reason other than fencing (bad/backward status)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AttemptStatusResult:
    job_id: str
    attempt_id: str
    status: str
    changed: bool
    terminal: bool


def _get(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return cast("dict[str, Any]", row)[key]
    return row[index]


def report_attempt_status(
    conn: Connection,
    *,
    job_id: str,
    attempt_id: str,
    host_id: str,
    fencing_token: int,
    status: str,
    failure_code: str | None = None,
    detail: dict[str, Any] | None = None,
) -> AttemptStatusResult:
    """Apply one fenced worker status report inside the caller's txn.

    Raises :class:`control_plane.leases.FencingViolation` when the tuple
    is not the job's current authority (the worker must stop), and
    :class:`AttemptStatusRejected` for malformed/backward reports.
    Same-status repeats are idempotent no-ops (at-least-once delivery).
    """
    if status not in _REPORT_ORDER:
        raise AttemptStatusRejected(
            "unsupported_status",
            f"status {status!r} is not reportable "
            f"(expected one of {sorted(_REPORT_ORDER)})",
        )

    # §8.1: authority first, before reading anything else.
    require_current_fence(
        conn,
        job_id=job_id,
        attempt_id=attempt_id,
        host_id=host_id,
        fencing_token=fencing_token,
    )

    row = conn.execute(
        "SELECT status FROM job_attempts WHERE attempt_id = %s FOR UPDATE",
        (attempt_id,),
    ).fetchone()
    if row is None:  # pragma: no cover - fence gate already verified it
        raise AttemptStatusRejected("attempt_missing", f"attempt {attempt_id} vanished")
    current = str(_get(row, "status", 0))

    if current == status:
        return AttemptStatusResult(job_id, attempt_id, status, False, status in _TERMINAL)
    current_rank = _REPORT_ORDER.get(current, -1)  # reserved etc. rank below all
    if _REPORT_ORDER[status] < current_rank:
        raise AttemptStatusRejected(
            "out_of_order",
            f"attempt is {current}; cannot move backward to {status}",
        )

    terminal = status in _TERMINAL
    conn.execute(
        """
        UPDATE job_attempts
           SET status = %(status)s,
               lease_claimed_at = CASE
                   WHEN %(status)s = 'lease_claimed'
                       THEN COALESCE(lease_claimed_at, clock_timestamp())
                   ELSE lease_claimed_at END,
               started_at = CASE
                   WHEN %(status)s = 'running'
                       THEN COALESCE(started_at, clock_timestamp())
                   ELSE started_at END,
               ended_at = CASE
                   WHEN %(terminal)s THEN clock_timestamp()
                   ELSE ended_at END,
               failure_code = CASE
                   WHEN %(status)s = 'failed' THEN %(failure_code)s
                   ELSE failure_code END,
               failure_details = CASE
                   WHEN %(status)s = 'failed'
                       THEN COALESCE(%(detail)s::jsonb, failure_details)
                   ELSE failure_details END
         WHERE attempt_id = %(attempt_id)s
        """,
        {
            "status": status,
            "terminal": terminal,
            "failure_code": failure_code or ("error" if status == "failed" else None),
            "detail": Jsonb(detail) if detail else None,
            "attempt_id": attempt_id,
        },
    )

    # Job projection: one legacy-status write; the 059 trigger keeps
    # phase/desired_state coherent with it.
    job_status = _JOB_PROJECTION[status]
    if terminal:
        conn.execute(
            """
            UPDATE jobs
               SET status = %(job_status)s,
                   active_attempt_id = NULL,
                   reason_code = CASE
                       WHEN %(job_status)s = 'failed' THEN %(failure_code)s
                       ELSE NULL END,
                   version = version + 1,
                   updated_at = clock_timestamp()
             WHERE job_id = %(job_id)s
               AND active_attempt_id = %(attempt_id)s
            """,
            {
                "job_status": job_status,
                "failure_code": failure_code or "error",
                "job_id": job_id,
                "attempt_id": attempt_id,
            },
        )
        # Settle the reservation's residue exactly once.
        conn.execute(
            """
            UPDATE gpu_device_allocations
               SET status = 'released',
                   released_at = clock_timestamp(),
                   release_reason = %s
             WHERE attempt_id = %s
               AND status = 'active'
            """,
            (f"attempt_{status}", attempt_id),
        )
        conn.execute(
            """
            UPDATE placement_leases
               SET status = 'released',
                   released_at = clock_timestamp()
             WHERE attempt_id = %s
               AND status IN ('offered', 'active')
            """,
            (attempt_id,),
        )
    else:
        conn.execute(
            """
            UPDATE jobs
               SET status = %(job_status)s,
                   version = version + 1,
                   updated_at = clock_timestamp()
             WHERE job_id = %(job_id)s
               AND active_attempt_id = %(attempt_id)s
            """,
            {"job_status": job_status, "job_id": job_id, "attempt_id": attempt_id},
        )

    conn.execute(
        """
        INSERT INTO outbox_events
            (aggregate_type, aggregate_id, event_type, payload,
             destination_class, idempotency_key)
        VALUES ('job', %s, 'job.v1.attempt_status_changed',
                jsonb_build_object(
                    'attempt_id', %s::text,
                    'host_id', %s::text,
                    'status', %s::text,
                    'fencing_token', %s::bigint
                ),
                'default', %s)
        ON CONFLICT (destination_class, idempotency_key) DO NOTHING
        """,
        (
            job_id, attempt_id, host_id, status, fencing_token,
            f"attempt_status:{attempt_id}:{status}",
        ),
    )
    log.info(
        "attempt %s (job %s): %s -> %s%s",
        attempt_id[:8], job_id, current, status,
        " [terminal]" if terminal else "",
    )
    return AttemptStatusResult(job_id, attempt_id, status, True, terminal)
