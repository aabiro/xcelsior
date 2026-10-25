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
- expiry sweeps close the attempt and release its allocations exactly once;
  only work still desired running (or explicitly restarting) is requeued
  with a *higher* fence. Stop and terminal intent survive authority loss.

All functions run inside the caller's transaction
(``control_plane.db.run_transaction``).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, cast

from psycopg import Connection, sql

from control_plane.state import job_phase

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


class LeaseStartRejected(LeaseRenewRejected):
    """The current attempt may settle, but its execution intent no longer runs."""

    code = "lease_start_rejected"


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
    gpu_device_uuids: tuple[str, ...]


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
    # Status reporting and expiry lock job before lease; claim follows the
    # same order while projecting the claim and its outbox intent atomically.
    conn.execute("SELECT job_id FROM jobs WHERE job_id=%s FOR UPDATE", (job_id,)).fetchone()
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
           AND EXISTS (
               SELECT 1 FROM jobs j JOIN job_attempts a ON a.attempt_id=j.active_attempt_id
               WHERE j.job_id=placement_leases.job_id
                 AND j.active_attempt_id=placement_leases.attempt_id
                 AND j.host_id=placement_leases.host_id
                 AND j.spec->>'execution_spec_version'='v1'
                 AND j.spec_hash=a.spec_hash AND j.generation=a.job_generation
                 AND a.fencing_token=placement_leases.fencing_token)
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
    # The successful claim grants concrete devices, never a host-wide count.
    # Check inside this transaction so missing/retired/foreign allocations roll
    # back both the lease claim and its attempt projection.
    requested = conn.execute("SELECT spec FROM jobs WHERE job_id=%s", (job_id,)).fetchone()
    spec = _get(requested, "spec", 0)
    count = spec.get("num_gpus", 1)
    devices = conn.execute(
        """SELECT a.host_id AS allocation_host, d.host_id AS device_host,
                  d.gpu_uuid, d.retired_at, a.allocation_mode
           FROM gpu_device_allocations a JOIN host_gpu_devices d USING(gpu_device_id)
           WHERE a.attempt_id=%s AND a.job_id=%s AND a.released_at IS NULL
           ORDER BY a.gpu_device_id FOR SHARE OF a,d""",
        (attempt_id, job_id),
    ).fetchall()
    uuids = tuple(_get(d, "gpu_uuid", 2) for d in devices)
    if (
        type(count) is not int
        or not 1 <= count <= 64
        or len(devices) != count
        or any(
            _get(d, "allocation_host", 0) != host_id
            or _get(d, "device_host", 1) != host_id
            or _get(d, "retired_at", 3) is not None
            or _get(d, "allocation_mode", 4) != "exclusive"
            for d in devices
        )
        or any(
            not isinstance(u, str) or not re.fullmatch(r"GPU-[A-Za-z0-9-]{1,100}", u) for u in uuids
        )
        or len(set(uuids)) != len(uuids)
    ):
        raise LeaseClaimRejected("gpu_allocation_authority_missing")
    from control_plane.attempts import report_attempt_status

    report_attempt_status(
        conn,
        job_id=job_id,
        attempt_id=attempt_id,
        host_id=host_id,
        fencing_token=fencing_token,
        status="lease_claimed",
    )
    return LeaseGrant(
        lease_id=lease_id,
        job_id=job_id,
        attempt_id=attempt_id,
        host_id=host_id,
        fencing_token=fencing_token,
        expires_at=_get(row, "expires_at", 0),
        renewal_ttl_sec=int(_get(row, "renewal_ttl_sec", 1)),
        gpu_device_uuids=uuids,
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
    require_running_intent: bool = False,
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
           AND (NOT %(start)s OR EXISTS (
               SELECT 1 FROM jobs j JOIN job_attempts a ON a.attempt_id=j.active_attempt_id
               WHERE j.job_id=placement_leases.job_id
                 AND a.attempt_id=placement_leases.attempt_id
                 AND j.host_id=placement_leases.host_id
                 AND a.host_id=placement_leases.host_id
                 AND a.fencing_token=placement_leases.fencing_token
                 AND a.spec_hash=j.spec_hash AND a.job_generation=j.generation
                 AND j.desired_state='running' AND j.lifecycle_intent IS NULL
                 AND a.status IN ('lease_claimed','starting')))
        RETURNING expires_at
        """,
        {
            "session": worker_session_id,
            "lease_id": lease_id,
            "attempt_id": attempt_id,
            "host_id": host_id,
            "fence": fencing_token,
            "start": require_running_intent,
        },
    ).fetchone()
    if row is None:
        if (
            require_running_intent
            and conn.execute(
                """SELECT 1 FROM jobs j JOIN job_attempts a ON a.attempt_id=j.active_attempt_id
               WHERE a.attempt_id=%s AND a.host_id=%s AND a.fencing_token=%s
                 AND (j.desired_state<>'running' OR j.lifecycle_intent IS NOT NULL)""",
                (attempt_id, host_id, fencing_token),
            ).fetchone()
        ):
            raise LeaseStartRejected("execution intent changed before container start")
        raise LeaseRenewRejected(
            "lease not active under this authority (expired, fenced, or rebound)",
            lease_id=lease_id,
            attempt_id=attempt_id,
            fencing_token=fencing_token,
        )
    expires_at = _get(row, "expires_at", 0)
    conn.execute(
        "UPDATE serverless_attempt_credentials SET expires_at=%s "
        "WHERE attempt_id=%s AND revoked_at IS NULL",
        (expires_at, attempt_id),
    )
    return expires_at


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
    current_job_id: str | None = None,
    lock_nowait: bool = False,
) -> list[dict[str, Any]]:
    """Lease-controller sweep (§12.4): fence dead claims, preserve job intent.

    Two cases, processed with row locks and SKIP LOCKED:

    - ``offered`` past its claim deadline → the worker never claimed;
      attempt fails with ``lease_claim_timeout``.
    - ``active`` past ``expires_at`` + grace → renewals stopped; attempt
      is marked ``lost`` (the container may still be running — §2.1
      distinguishes authority loss from physical stop; restart policy is
      the reconciler's concern, here we only revoke authority).

    In both cases, atomically: lease → expired, attempt → terminal,
    allocations → released (exactly once), job authority cleared and desired
    stop/terminal state preserved, outbox event appended. Work still desired
    running is requeued with a durable reason, except serverless workers:
    their scheduler job closes and endpoint reconciliation provisions a fresh
    worker/job identity for queued requests. A normal job retry
    then mints a *new* attempt with a higher fence via the normal
    scheduler path.
    """
    stale = conn.execute(
        """
        SELECT l.lease_id, l.job_id, l.attempt_id, l.host_id, l.fencing_token, l.status,
               j.status AS job_status, j.desired_state,
               j.lifecycle_intent,
               j.spec->>'job_type' AS job_type,
               (j.active_attempt_id=l.attempt_id AND j.host_id=l.host_id) AS current_authority
          FROM placement_leases l JOIN jobs j ON j.job_id = l.job_id
         WHERE ((l.status = 'offered' AND l.claim_deadline < clock_timestamp())
            OR (l.status = 'active'
                AND l.expires_at + make_interval(secs => %(grace)s)
                    < clock_timestamp()))
           AND (%(job)s::text IS NULL OR (
               j.job_id = %(job)s AND j.active_attempt_id = l.attempt_id
               AND j.host_id = l.host_id AND EXISTS (
                   SELECT 1 FROM job_attempts a WHERE a.attempt_id = l.attempt_id
                     AND a.job_id = j.job_id AND a.host_id = l.host_id
                     AND a.fencing_token = l.fencing_token)))
         ORDER BY l.lease_id
         LIMIT %(limit)s
           FOR UPDATE OF j, l
        """ + (" NOWAIT" if lock_nowait else " SKIP LOCKED"),
        {"grace": grace_sec, "limit": limit, "job": current_job_id},
    ).fetchall()
    expired: list[dict[str, Any]] = []
    for row in stale:
        lease_id = str(_get(row, "lease_id", 0))
        job_id = str(_get(row, "job_id", 1))
        attempt_id = str(_get(row, "attempt_id", 2))
        from control_plane.billing_effects import serialize_native_period
        from psycopg.errors import LockNotAvailable

        if not serialize_native_period(conn, job_id, wait=not lock_nowait):
            raise LockNotAvailable("billing period is settling; retry lease expiry")
        if lock_nowait:
            # Reconciliation owns a queue row already. Runtime writers can
            # hold these rows while waiting for that queue, so fail fast and
            # let process_due roll back its savepoint instead of deadlocking.
            for table in (
                "job_attempts",
                "gpu_device_allocations",
                "agent_commands",
                "usage_meters",
                "serverless_workers",
            ):
                conn.execute(
                    sql.SQL("SELECT 1 FROM {} WHERE attempt_id = %s FOR UPDATE NOWAIT").format(
                        sql.Identifier(table)
                    ),
                    (attempt_id,),
                ).fetchall()
            conn.execute(
                "SELECT 1 FROM wallet_holds WHERE job_id=%s FOR UPDATE NOWAIT", (job_id,)
            ).fetchall()
            conn.execute(
                "SELECT 1 FROM serverless_jobs WHERE worker_id IN "
                "(SELECT worker_id FROM serverless_workers WHERE attempt_id=%s) FOR UPDATE NOWAIT",
                (attempt_id,),
            ).fetchall()
        was = str(_get(row, "status", 5))
        terminal = "failed" if was == "offered" else "lost"
        failure_code = "lease_claim_timeout" if was == "offered" else "lease_renewal_timeout"
        job_status = str(_get(row, "job_status", 6))
        desired = str(_get(row, "desired_state", 7))
        intent = _get(row, "lifecycle_intent", 8)
        serverless = _get(row, "job_type", 9) == "serverless_worker"
        current_authority = bool(_get(row, "current_authority", 10))
        if job_status not in {
            "completed",
            "failed",
            "terminated",
            "cancelled",
            "stopped",
            "paused",
        }:
            if desired == "stopped":
                job_status = {
                    "terminate": "terminated",
                    "cancel": "cancelled",
                    "restart": "queued",
                }.get(intent, "stopped")
            else:
                job_status = "queued"
        # A serverless worker represents one attempt and one billing span.
        # Close it permanently; endpoint reconciliation creates fresh worker
        # and scheduler-job IDs instead of lending this identity to a retry.
        if serverless and job_status == "queued":
            job_status = "failed"

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
        if current_authority:
            from control_plane.attempt_billing import sync_attempt_billing
            from control_plane.serverless_workers import sync_serverless_attempt

            sync_attempt_billing(conn, attempt_id)
            sync_serverless_attempt(conn, job_id=job_id, attempt_id=attempt_id)
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
               SET status = %(status)s,
                   phase = %(phase)s,
                   desired_state = CASE
                       WHEN %(status)s = 'queued' THEN 'running'
                       WHEN %(phase)s = 'stopped' THEN 'stopped'
                       ELSE desired_state END,
                   queued_at = CASE WHEN %(status)s = 'queued'
                       THEN statement_timestamp() ELSE queued_at END,
                   submitted_at = CASE WHEN %(status)s = 'queued'
                       THEN EXTRACT(EPOCH FROM statement_timestamp()) ELSE submitted_at END,
                   host_id = NULL,
                   active_attempt_id = NULL,
                   lifecycle_intent = NULL,
                   reason_code = %(reason)s,
                   next_schedule_at = CASE WHEN %(status)s = 'queued'
                       THEN clock_timestamp() ELSE NULL END,
                   schedule_claim_owner = NULL,
                   schedule_claim_token = NULL,
                   schedule_claim_expires_at = NULL,
                   generation = generation + CASE WHEN %(status)s = 'queued' THEN 1 ELSE 0 END,
                   payload = (CASE WHEN %(status)s = 'queued'
                       THEN COALESCE(payload, '{}'::jsonb) - 'lifecycle_intent'
                            - 'stopped_at' - 'stop_reason' - 'stopping_at'
                       ELSE COALESCE(payload, '{}'::jsonb) - 'lifecycle_intent' END)
                       || jsonb_build_object('status', %(status)s::text,
                           'lease_expired_at', EXTRACT(EPOCH FROM clock_timestamp()))
                       || CASE WHEN %(status)s = 'queued'
                           THEN jsonb_build_object('submitted_at',
                               EXTRACT(EPOCH FROM statement_timestamp()))
                           ELSE '{}'::jsonb END,
                   version = version + 1,
                   updated_at = clock_timestamp()
             WHERE job_id = %(job)s
               AND active_attempt_id = %(attempt)s
            """,
            {
                "reason": failure_code,
                "job": job_id,
                "attempt": attempt_id,
                "status": job_status,
                "phase": job_phase(job_status),
            },
        )
        from control_plane.outbox import append_event

        append_event(
            conn,
            aggregate_type="job",
            aggregate_id=job_id,
            event_type="job.v1.lease_expired",
            payload={"attempt_id": attempt_id, "lease_id": lease_id, "reason": failure_code},
            idempotency_key=f"lease_expired:{lease_id}",
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
