"""Durable agent-command claim/ACK protocol (blueprint §9.4, §13.3).

The v1 path hands commands to workers with ``DELETE ... RETURNING`` — a
worker crash after fetch silently loses the command. This module is the
v2 replacement: fetching *claims* (pending → claimed with an expiring
claim), execution outcome is a durable ACK/NACK, and a claim that expires
un-ACKed is redelivered. A duplicate ACK returns the original result
instead of failing (§8.3: commands are at-least-once delivered,
idempotently executed).

All functions run inside the caller's transaction
(``control_plane.db.run_transaction``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from psycopg import Connection
from psycopg.types.json import Jsonb

DEFAULT_CLAIM_TTL_SEC = 60
DEFAULT_RETRY_BASE_BACKOFF_SEC = 5.0
DEFAULT_RETRY_MAX_BACKOFF_SEC = 300.0
# Acknowledged rows are kept for duplicate-ACK replay and audit until the
# retention sweep removes them.
DEFAULT_RESULT_RETENTION_SEC = 24 * 3600


class CommandProtocolError(Exception):
    """Typed rejection of a claim/ACK/NACK request."""

    def __init__(self, message: str, **details: Any):
        super().__init__(message)
        self.details = details


@dataclass(frozen=True)
class ClaimedCommand:
    command_id: str
    command: str
    args: dict[str, Any]
    job_id: str | None
    attempt_id: str | None
    fencing_token: int | None
    spec_hash: str | None
    idempotency_key: str | None
    attempt_count: int
    claim_expires_at: Any


@dataclass(frozen=True)
class AckOutcome:
    command_id: str
    duplicate: bool
    result: dict[str, Any] | None


@dataclass(frozen=True)
class EnqueuedAttemptCommand:
    command_id: str
    job_id: str
    attempt_id: str
    host_id: str
    fencing_token: int


def _get(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return cast("dict[str, Any]", row)[key]
    return row[index]


def enqueue_current_attempt_command(
    conn: Connection,
    *,
    job_id: str,
    command: str,
    args: dict[str, Any] | None = None,
    created_by: str,
    idempotency_key: str | None = None,
    expires_in_sec: int = 900,
) -> EnqueuedAttemptCommand:
    """Enqueue a lifecycle command bound to the job's current authority.

    The job/attempt/lease rows are locked before the command is created, so
    the tuple in the durable command cannot race a concurrent re-placement.
    Retried API requests collapse on ``idempotency_key``.
    """
    authority = conn.execute(
        """
        SELECT j.active_attempt_id, a.host_id, a.fencing_token, a.spec_hash,
               l.lease_id
          FROM jobs j
          JOIN job_attempts a ON a.attempt_id = j.active_attempt_id
          JOIN placement_leases l ON l.attempt_id = a.attempt_id
         WHERE j.job_id = %s
           AND a.status IN ('lease_claimed', 'starting', 'running')
           AND l.status = 'active'
           AND l.expires_at > clock_timestamp()
         ORDER BY l.claimed_at DESC NULLS LAST
         LIMIT 1
           FOR UPDATE OF j, a, l
        """,
        (job_id,),
    ).fetchone()
    if authority is None:
        raise CommandProtocolError(
            f"job {job_id} has no active fenced authority", job_id=job_id
        )
    attempt_id = str(_get(authority, "active_attempt_id", 0))
    host_id = str(_get(authority, "host_id", 1))
    fencing_token = int(_get(authority, "fencing_token", 2))
    spec_hash = _get(authority, "spec_hash", 3)
    lease_id = str(_get(authority, "lease_id", 4))
    command_key = idempotency_key or f"{command}:{attempt_id}"
    payload = dict(args or {})
    payload.update(
        {
            "job_id": job_id,
            "attempt_id": attempt_id,
            "host_id": host_id,
            "fencing_token": fencing_token,
            "lease_id": lease_id,
        }
    )
    row = conn.execute(
        """
        INSERT INTO agent_commands
            (host_id, command, args, status, created_by, expires_at,
             job_id, attempt_id, fencing_token, spec_hash, idempotency_key)
        VALUES (%s, %s, %s, 'pending', %s,
                EXTRACT(EPOCH FROM NOW()) + %s,
                %s, %s, %s, %s, %s)
        ON CONFLICT (host_id, idempotency_key) WHERE idempotency_key IS NOT NULL
        DO UPDATE SET idempotency_key = EXCLUDED.idempotency_key
        RETURNING command_id
        """,
        (
            host_id,
            command,
            Jsonb(payload),
            created_by,
            int(expires_in_sec),
            job_id,
            attempt_id,
            fencing_token,
            spec_hash,
            command_key,
        ),
    ).fetchone()
    if row is None:  # pragma: no cover - INSERT/UPSERT always returns
        raise CommandProtocolError("attempt command insert returned no row")
    return EnqueuedAttemptCommand(
        command_id=str(_get(row, "command_id", 0)),
        job_id=job_id,
        attempt_id=attempt_id,
        host_id=host_id,
        fencing_token=fencing_token,
    )


def claim_commands(
    conn: Connection,
    *,
    host_id: str,
    worker_session_id: str,
    claim_ttl_sec: int = DEFAULT_CLAIM_TTL_SEC,
    limit: int = 10,
    attempt_commands_only: bool = False,
) -> list[ClaimedCommand]:
    """Claim deliverable commands for one host (pending → claimed).

    SKIP LOCKED keeps concurrent API workers serving the same host from
    double-delivering; a claimed command is invisible to other fetches
    until its claim expires.

    ``attempt_commands_only=True`` restricts the claim to commands bound
    to a placement attempt (``attempt_id IS NOT NULL``) — the /agent/v2
    protocol partition. Plain admin commands stay visible to the v1
    drain, which in turn excludes attempt-bound commands, so the two
    delivery paths can never destroy each other's work.
    """
    rows = conn.execute(
        """
        WITH deliverable AS (
            SELECT id
              FROM agent_commands
             WHERE host_id = %(host_id)s
               AND status = 'pending'
               AND (not_before IS NULL OR not_before <= clock_timestamp())
               AND (next_attempt_at IS NULL
                    OR next_attempt_at <= clock_timestamp())
               AND expires_at >= EXTRACT(EPOCH FROM NOW())
               AND (%(attempt_only)s IS FALSE OR attempt_id IS NOT NULL)
             ORDER BY priority DESC, created_at ASC
             LIMIT %(limit)s
               FOR UPDATE SKIP LOCKED
        )
        UPDATE agent_commands c
           SET status = 'claimed',
               claim_owner = %(host_id)s,
               claim_session = %(session)s,
               claim_expires_at = clock_timestamp()
                                  + make_interval(secs => %(ttl)s),
               attempt_count = c.attempt_count + 1
          FROM deliverable d
         WHERE c.id = d.id
        RETURNING c.command_id, c.command, c.args, c.job_id, c.attempt_id,
                  c.fencing_token, c.spec_hash, c.idempotency_key,
                  c.attempt_count, c.claim_expires_at
        """,
        {
            "host_id": host_id,
            "session": worker_session_id,
            "ttl": claim_ttl_sec,
            "limit": limit,
            "attempt_only": attempt_commands_only,
        },
    ).fetchall()
    return [
        ClaimedCommand(
            command_id=str(_get(r, "command_id", 0)),
            command=str(_get(r, "command", 1)),
            args=_get(r, "args", 2) or {},
            job_id=_get(r, "job_id", 3),
            attempt_id=(
                str(_get(r, "attempt_id", 4))
                if _get(r, "attempt_id", 4) is not None
                else None
            ),
            fencing_token=_get(r, "fencing_token", 5),
            spec_hash=_get(r, "spec_hash", 6),
            idempotency_key=_get(r, "idempotency_key", 7),
            attempt_count=int(_get(r, "attempt_count", 8)),
            claim_expires_at=_get(r, "claim_expires_at", 9),
        )
        for r in rows
    ]


def ack_command(
    conn: Connection,
    *,
    command_id: str,
    host_id: str,
    result: dict[str, Any] | None = None,
    result_retention_sec: int = DEFAULT_RESULT_RETENTION_SEC,
) -> AckOutcome:
    """Durable success ACK; duplicate ACKs replay the original result.

    Only the claiming host may ACK; the terminal transition happens once.
    """
    row = conn.execute(
        """
        UPDATE agent_commands
           SET status = 'acknowledged',
               acked_at = clock_timestamp(),
               ack_result = %(result)s,
               retention_expires_at = clock_timestamp()
                                      + make_interval(secs => %(retention)s)
         WHERE command_id = %(command_id)s
           AND claim_owner = %(host_id)s
           AND status = 'claimed'
        RETURNING command_id
        """,
        {
            "result": Jsonb(result or {}),
            "retention": result_retention_sec,
            "command_id": command_id,
            "host_id": host_id,
        },
    ).fetchone()
    if row is not None:
        return AckOutcome(command_id=command_id, duplicate=False, result=result or {})
    # Not transitioned — either a duplicate ACK (fine, replay stored
    # result) or an illegitimate one (error).
    existing = conn.execute(
        "SELECT status, claim_owner, ack_result FROM agent_commands "
        "WHERE command_id = %s",
        (command_id,),
    ).fetchone()
    if existing is None:
        raise CommandProtocolError(
            f"command {command_id} does not exist", command_id=command_id
        )
    status = _get(existing, "status", 0)
    owner = _get(existing, "claim_owner", 1)
    if status == "acknowledged" and owner == host_id:
        return AckOutcome(
            command_id=command_id,
            duplicate=True,
            result=_get(existing, "ack_result", 2),
        )
    raise CommandProtocolError(
        f"command {command_id} is {status} (claimed by {owner}); ACK rejected",
        command_id=command_id,
        status=status,
    )


def nack_command(
    conn: Connection,
    *,
    command_id: str,
    host_id: str,
    error_code: str,
    error_details: dict[str, Any] | None = None,
    retryable: bool = True,
    base_backoff_sec: float = DEFAULT_RETRY_BASE_BACKOFF_SEC,
    max_backoff_sec: float = DEFAULT_RETRY_MAX_BACKOFF_SEC,
) -> str:
    """Typed failure report. Returns the command's resulting status.

    Retryable failures inside the retry budget go back to ``pending``
    with exponential backoff (§9.4 failed → pending); anything else
    dead-letters and pages through the command controller's metrics.
    """
    row = conn.execute(
        """
        UPDATE agent_commands
           SET status = 'failed',
               error_code = %(code)s,
               error_details = %(details)s
         WHERE command_id = %(command_id)s
           AND claim_owner = %(host_id)s
           AND status = 'claimed'
        RETURNING attempt_count, max_attempts
        """,
        {
            "code": error_code,
            "details": Jsonb(error_details or {}),
            "command_id": command_id,
            "host_id": host_id,
        },
    ).fetchone()
    if row is None:
        raise CommandProtocolError(
            f"command {command_id} is not claimed by {host_id}; NACK rejected",
            command_id=command_id,
        )
    attempt_count = int(_get(row, "attempt_count", 0))
    max_attempts = int(_get(row, "max_attempts", 1))
    if retryable and attempt_count < max_attempts:
        backoff = min(max_backoff_sec, base_backoff_sec * (2 ** (attempt_count - 1)))
        conn.execute(
            """
            UPDATE agent_commands
               SET status = 'pending',
                   claim_owner = NULL,
                   claim_session = NULL,
                   claim_expires_at = NULL,
                   next_attempt_at = clock_timestamp()
                                     + make_interval(secs => %(backoff)s)
             WHERE command_id = %(command_id)s
            """,
            {"backoff": backoff, "command_id": command_id},
        )
        return "pending"
    conn.execute(
        "UPDATE agent_commands SET status = 'dead_letter' WHERE command_id = %s",
        (command_id,),
    )
    return "dead_letter"


def redeliver_expired_claims(conn: Connection, *, limit: int = 100) -> int:
    """Command-controller sweep (§12.4): un-ACKed expired claims.

    The claimer crashed or lost connectivity mid-execution. Within the
    retry budget the command goes back to ``pending`` (redelivery; the
    worker's idempotency journal makes re-execution safe); past it, the
    command dead-letters for operator attention.
    """
    result = conn.execute(
        """
        WITH stale AS (
            SELECT id, attempt_count, max_attempts
              FROM agent_commands
             WHERE status = 'claimed'
               AND claim_expires_at < clock_timestamp()
             LIMIT %(limit)s
               FOR UPDATE SKIP LOCKED
        )
        UPDATE agent_commands c
           SET status = CASE
                   WHEN s.attempt_count >= s.max_attempts THEN 'dead_letter'
                   ELSE 'pending'
               END,
               claim_owner = NULL,
               claim_session = NULL,
               claim_expires_at = NULL
          FROM stale s
         WHERE c.id = s.id
        """,
        {"limit": limit},
    )
    return result.rowcount
