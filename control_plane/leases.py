"""Lease & fencing engine (blueprint §8.3, §11, ADR-005).

Authority lives in the tuple ``job_id + attempt_id + host_id + lease_id +
fencing_token``. Every worker-facing operation here is a compare-and-swap
against that exact tuple — a stale worker's request simply matches zero
rows and gets a typed rejection, never a partial effect.

Rules enforced (§8.3):
- claim requires the exact offered lease before its claim deadline;
- renewal cannot change host, lower the fence, or resurrect a
  non-active lease;
- released/expired/fenced leases never return to active;
- expiry sweeps close the attempt, release its allocations exactly once,
  and requeue the job for a fresh attempt with a *higher* fence — the old
  attempt is never rewound.

All functions run inside the caller's transaction
(``control_plane.db.run_transaction``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from psycopg import Connection

DEFAULT_EXPIRY_GRACE_SEC = 30.0


class LeaseError(Exception):
    """Base for typed lease-protocol rejections."""

    code = "lease_error"

    def __init__(self, message: str, **details: Any):
        super().__init__(message)
        self.details = details


class LeaseClaimRejected(LeaseError):
    """Wrong binding, expired offer, or already-claimed lease."""

    code = "lease_claim_rejected"


class LeaseRenewRejected(LeaseError):
    """Renewal denied: lease not active under this exact authority."""

    code = "lease_renew_rejected"


class FencingViolation(LeaseError):
    """A stale attempt/fence tried to act with authority it lost."""

    code = "fencing_violation"


@dataclass(frozen=True)
class LeaseGrant:
    lease_id: str
    job_id: str
    attempt_id: str
    host_id: str
    fencing_token: int
    expires_at: Any
    renewal_ttl_sec: int


def _get(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return cast("dict[str, Any]", row)[key]
    return row[index]


def claim_lease(
    conn: Connection,
    *,
    lease_id: str,
    job_id: str,
    attempt_id: str,
    host_id: str,
    fencing_token: int,
    worker_session_id: str,
) -> LeaseGrant:
    """Worker claims its offered lease — the §11.2 hard gate.

    The UPDATE matches only the exact offered lease with the exact
    authority tuple, before its claim deadline. Anything else — wrong
    host, wrong fence, expired offer, already active — matches zero rows
    and is rejected with a diagnosis, so the worker NACKs instead of
    starting a container it has no authority to run.
    """
    row = conn.execute(
        """
        UPDATE placement_leases
           SET status = 'active',
               claimed_at = clock_timestamp(),
               last_renewed_at = clock_timestamp(),
               expires_at = clock_timestamp()
                            + make_interval(secs => renewal_ttl_sec),
               last_worker_session_id = %(session)s
         WHERE lease_id = %(lease_id)s
           AND job_id = %(job_id)s
           AND attempt_id = %(attempt_id)s
           AND host_id = %(host_id)s
           AND fencing_token = %(fence)s
           AND status = 'offered'
           AND claim_deadline >= clock_timestamp()
        RETURNING expires_at, renewal_ttl_sec
        """,
        {
            "session": worker_session_id,
            "lease_id": lease_id,
            "job_id": job_id,
            "attempt_id": attempt_id,
            "host_id": host_id,
            "fence": fencing_token,
        },
    ).fetchone()
    if row is None:
        raise LeaseClaimRejected(
            _diagnose_claim_failure(conn, lease_id, host_id, fencing_token),
            lease_id=lease_id,
            attempt_id=attempt_id,
            host_id=host_id,
            fencing_token=fencing_token,
        )
    conn.execute(
        """
        UPDATE job_attempts
           SET status = 'lease_claimed',
               lease_claimed_at = clock_timestamp()
         WHERE attempt_id = %s
           AND status IN ('reserved', 'command_pending', 'lease_offered')
        """,
        (attempt_id,),
    )
    return LeaseGrant(
        lease_id=lease_id,
        job_id=job_id,
        attempt_id=attempt_id,
        host_id=host_id,
        fencing_token=fencing_token,
        expires_at=_get(row, "expires_at", 0),
        renewal_ttl_sec=int(_get(row, "renewal_ttl_sec", 1)),
    )


def _diagnose_claim_failure(
    conn: Connection, lease_id: str, host_id: str, fencing_token: int
) -> str:
    row = conn.execute(
        "SELECT status, host_id, fencing_token, claim_deadline < clock_timestamp() "
        "AS offer_expired FROM placement_leases WHERE lease_id = %s",
        (lease_id,),
    ).fetchone()
    if row is None:
        return "lease does not exist"
    status = _get(row, "status", 0)
    if _get(row, "host_id", 1) != host_id:
        return "lease is bound to a different host"
    if int(_get(row, "fencing_token", 2)) != fencing_token:
        return "fencing token does not match current lease"
    if status == "offered" and _get(row, "offer_expired", 3):
        return "lease offer expired before claim"
    return f"lease is {status}, not claimable"


def renew_lease(
    conn: Connection,
    *,
    lease_id: str,
    attempt_id: str,
    host_id: str,
    fencing_token: int,
    worker_session_id: str,
) -> Any:
    """Extend an active lease under the exact same authority tuple.

    A lease whose deadline already passed is not renewable — the worker
    must treat authority as lost (§11.5); reconciliation decides what
    happens next. Returns the new expiry.
    """
    row = conn.execute(
        """
        UPDATE placement_leases
           SET last_renewed_at = clock_timestamp(),
               expires_at = clock_timestamp()
                            + make_interval(secs => renewal_ttl_sec),
               last_worker_session_id = %(session)s
         WHERE lease_id = %(lease_id)s
           AND attempt_id = %(attempt_id)s
           AND host_id = %(host_id)s
           AND fencing_token = %(fence)s
           AND status = 'active'
           AND expires_at >= clock_timestamp()
        RETURNING expires_at
        """,
        {
            "session": worker_session_id,
            "lease_id": lease_id,
            "attempt_id": attempt_id,
            "host_id": host_id,
            "fence": fencing_token,
        },
    ).fetchone()
    if row is None:
        raise LeaseRenewRejected(
            "lease not active under this authority (expired, fenced, or rebound)",
            lease_id=lease_id,
            attempt_id=attempt_id,
            fencing_token=fencing_token,
        )
    return _get(row, "expires_at", 0)


def release_lease(
    conn: Connection,
    *,
    lease_id: str,
    attempt_id: str,
    host_id: str,
    fencing_token: int,
) -> bool:
    """Voluntary release by the current authority holder; idempotent."""
    result = conn.execute(
        """
        UPDATE placement_leases
           SET status = 'released',
               released_at = clock_timestamp()
         WHERE lease_id = %s
           AND attempt_id = %s
           AND host_id = %s
           AND fencing_token = %s
           AND status IN ('offered', 'active')
        """,
        (lease_id, attempt_id, host_id, fencing_token),
    )
    return result.rowcount == 1


def require_current_fence(
    conn: Connection,
    *,
    job_id: str,
    attempt_id: str,
    host_id: str,
    fencing_token: int,
) -> None:
    """The §8.1 write gate: worker mutations must carry current authority.

    Verifies the attempt is the job's *active* attempt, on this host,
    with this exact fence, and still in an active status. Raises
    :class:`FencingViolation` otherwise — callers reject the update
    before touching state, billing, routing, or secrets.
    """
    row = conn.execute(
        """
        SELECT 1
          FROM job_attempts a
          JOIN jobs j ON j.job_id = a.job_id
         WHERE a.attempt_id = %s
           AND a.job_id = %s
           AND a.host_id = %s
           AND a.fencing_token = %s
           AND j.active_attempt_id = a.attempt_id
           AND a.status IN ('reserved', 'command_pending', 'lease_offered',
                            'lease_claimed', 'starting', 'running')
        """,
        (attempt_id, job_id, host_id, fencing_token),
    ).fetchone()
    if row is None:
        raise FencingViolation(
            "attempt/host/fence tuple is not the current authority for this job",
            job_id=job_id,
            attempt_id=attempt_id,
            host_id=host_id,
            fencing_token=fencing_token,
        )


def expire_stale_leases(
    conn: Connection,
    *,
    grace_sec: float = DEFAULT_EXPIRY_GRACE_SEC,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Lease-controller sweep (§12.4): fence out dead claims and requeue.

    Two cases, processed with row locks and SKIP LOCKED:

    - ``offered`` past its claim deadline → the worker never claimed;
      attempt fails with ``lease_claim_timeout``.
    - ``active`` past ``expires_at`` + grace → renewals stopped; attempt
      is marked ``lost`` (the container may still be running — §2.1
      distinguishes authority loss from physical stop; restart policy is
      the reconciler's concern, here we only revoke authority).

    In both cases, atomically: lease → expired, attempt → terminal,
    allocations → released (exactly once), job → pending with a durable
    reason and cleared active attempt, outbox event appended. A retry
    then mints a *new* attempt with a higher fence via the normal
    scheduler path.
    """
    stale = conn.execute(
        """
        SELECT lease_id, job_id, attempt_id, host_id, fencing_token, status
          FROM placement_leases
         WHERE (status = 'offered' AND claim_deadline < clock_timestamp())
            OR (status = 'active'
                AND expires_at + make_interval(secs => %(grace)s)
                    < clock_timestamp())
         ORDER BY lease_id
         LIMIT %(limit)s
           FOR UPDATE SKIP LOCKED
        """,
        {"grace": grace_sec, "limit": limit},
    ).fetchall()
    expired: list[dict[str, Any]] = []
    for row in stale:
        lease_id = str(_get(row, "lease_id", 0))
        job_id = str(_get(row, "job_id", 1))
        attempt_id = str(_get(row, "attempt_id", 2))
        was = str(_get(row, "status", 5))
        terminal = "failed" if was == "offered" else "lost"
        failure_code = (
            "lease_claim_timeout" if was == "offered" else "lease_renewal_timeout"
        )

        conn.execute(
            "UPDATE placement_leases SET status = 'expired' WHERE lease_id = %s",
            (lease_id,),
        )
        conn.execute(
            """
            UPDATE job_attempts
               SET status = %s, failure_code = %s, ended_at = clock_timestamp()
             WHERE attempt_id = %s
               AND status NOT IN ('succeeded', 'failed', 'cancelled',
                                  'preempted', 'lost', 'fenced')
            """,
            (terminal, failure_code, attempt_id),
        )
        conn.execute(
            """
            UPDATE gpu_device_allocations
               SET status = 'released',
                   released_at = clock_timestamp(),
                   release_reason = %s
             WHERE attempt_id = %s
               AND status = 'active'
            """,
            (failure_code, attempt_id),
        )
        # Undelivered start command for this attempt is dead: cancel it so
        # a late worker fetch cannot start a fenced-out attempt.
        conn.execute(
            """
            UPDATE agent_commands
               SET status = 'cancelled'
             WHERE attempt_id = %s
               AND status IN ('pending', 'claimed')
            """,
            (attempt_id,),
        )
        conn.execute(
            """
            UPDATE jobs
               SET phase = 'pending',
                   status = 'queued',
                   host_id = NULL,
                   active_attempt_id = NULL,
                   reason_code = %s,
                   next_schedule_at = clock_timestamp(),
                   version = version + 1,
                   updated_at = clock_timestamp()
             WHERE job_id = %s
               AND active_attempt_id = %s
            """,
            (failure_code, job_id, attempt_id),
        )
        conn.execute(
            """
            INSERT INTO outbox_events
                (aggregate_type, aggregate_id, event_type, payload,
                 destination_class, idempotency_key)
            VALUES ('job', %s, 'job.v1.lease_expired',
                    jsonb_build_object(
                        'attempt_id', %s::text,
                        'lease_id', %s::text,
                        'reason', %s::text
                    ),
                    'default', %s)
            ON CONFLICT (destination_class, idempotency_key) DO NOTHING
            """,
            (job_id, attempt_id, lease_id, failure_code, f"lease_expired:{lease_id}"),
        )
        expired.append(
            {
                "lease_id": lease_id,
                "job_id": job_id,
                "attempt_id": attempt_id,
                "was": was,
                "attempt_terminal": terminal,
            }
        )
    return expired
