"""Stage B — short transactional queue claim (blueprint §10.2).

A claim is *not* an execution lease. It grants one scheduler replica a
short exclusive window to calculate placement for one job, so expensive
scoring never holds a row lock. Any replica may steal the job after the
claim expires (ADR-003: no global leader, work stealing via row state).

All ordering and expiry decisions use PostgreSQL time
(``clock_timestamp()``), never host clocks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from psycopg import Connection
from psycopg.types.json import Jsonb

DEFAULT_CLAIM_TTL_SEC = 15

# Each claim calculates current weighted ranks in its transaction and persists
# the chosen components. Diagnostic refresh timing never supplies authority.
from control_plane.scheduler.fairness import claim_sql

_CLAIM_SQL = claim_sql()


@dataclass(frozen=True)
class ClaimedJob:
    job_id: str
    claim_token: str
    claim_expires_at: Any
    schedule_attempt_count: int
    version: int
    generation: int
    spec: dict | None
    spec_hash: str | None
    payload: dict
    effective_priority: int


def _row_get(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return cast("dict[str, Any]", row)[key]
    return row[index]


def claim_next_job(
    conn: Connection,
    *,
    replica_id: str,
    claim_ttl_sec: int = DEFAULT_CLAIM_TTL_SEC,
    scope_gpu_models: Sequence[str] | None = None,
) -> ClaimedJob | None:
    """Claim the single highest-ranked schedulable job, or None.

    Safe to call from any number of replicas concurrently: the candidate
    row is taken ``FOR UPDATE SKIP LOCKED``, so two replicas can never
    claim the same job and never block each other — a locked candidate is
    simply skipped in favor of the next one.

    ``scope_gpu_models=None`` claims from the whole queue (active mode);
    a sequence — even an empty one — restricts claims to the canary
    partition (matching gpu_model or explicit v2 opt-in).
    """
    if claim_ttl_sec <= 0:
        raise ValueError("claim_ttl_sec must be positive")
    row = conn.execute(
        _CLAIM_SQL,
        {
            "replica_id": replica_id,
            "ttl_sec": claim_ttl_sec,
            "scope_all": scope_gpu_models is None,
            "scope_models": [m.strip().lower() for m in (scope_gpu_models or [])],
        },
    ).fetchone()
    if row is None:
        return None
    spec = _row_get(row, "spec", 6) or {}
    if _row_get(row, "paid", 12):
        # Ordering evidence commits or rolls back with the claim itself.
        from control_plane.queue_bids import record_claim

        record_claim(
            conn,
            job_id=str(_row_get(row, "job_id", 0)),
            owner=str(_row_get(row, "owner_id", 11) or spec.get("owner") or ""),
            claim_number=int(_row_get(row, "schedule_attempt_count", 3)),
            quote=spec["service_tier_quote"],
            effective_tier=int(_row_get(row, "effective_priority", 10)),
            promoted=bool(_row_get(row, "promoted", 13)),
            aging_steps=int(_row_get(row, "aging_steps", 14)),
            aging_credit=int(_row_get(row, "aging_credit", 15)),
            queue_seconds=_row_get(row, "queue_seconds", 16),
            effective_bid=int(_row_get(row, "effective_bid", 17)),
            promoted_at=_row_get(row, "promoted_at", 18),
        )
    return ClaimedJob(
        job_id=str(_row_get(row, "job_id", 0)),
        claim_token=str(_row_get(row, "schedule_claim_token", 1)),
        claim_expires_at=_row_get(row, "schedule_claim_expires_at", 2),
        schedule_attempt_count=int(_row_get(row, "schedule_attempt_count", 3)),
        version=int(_row_get(row, "version", 4)),
        generation=int(_row_get(row, "generation", 5)),
        spec=_row_get(row, "spec", 6),
        spec_hash=_row_get(row, "spec_hash", 7),
        payload=_row_get(row, "payload", 8) or {},
        effective_priority=int(_row_get(row, "effective_priority", 10)),
    )


def release_claim(
    conn: Connection,
    job_id: str,
    claim_token: str,
    *,
    reason_code: str,
    requeue_delay_sec: float = 0.0,
    reason_details: dict[str, Any] | None = None,
) -> bool:
    """Give the claim back without placing (all candidates failed, §10.5).

    Compare-and-swap on the claim token: a claim that has already expired
    and been stolen by another replica is left alone (returns False), so
    a slow scheduler can never clobber its successor's state. A positive
    ``requeue_delay_sec`` records bounded backoff via ``next_schedule_at``
    and a durable queue reason (§8 invariant 10).
    """
    result = conn.execute(
        """
        UPDATE jobs
           SET schedule_claim_owner = NULL,
               schedule_claim_token = NULL,
               schedule_claim_expires_at = NULL,
               next_schedule_at =
                   clock_timestamp() + make_interval(secs => %(delay)s),
               reason_code = %(reason)s,
               reason_details = CASE WHEN %(details)s::jsonb IS NOT NULL THEN %(details)s::jsonb
                   WHEN %(reason)s='preemption_waiting' THEN reason_details ELSE NULL END,
               last_schedule_conflict_at = CASE
                   WHEN %(reason)s = 'placement_conflict' THEN clock_timestamp()
                   ELSE last_schedule_conflict_at
               END,
               version = version + 1,
               updated_at = clock_timestamp()
         WHERE job_id = %(job_id)s
           AND schedule_claim_token = %(token)s
        RETURNING generation, version, reason_details
        """,
        {
            "job_id": job_id,
            "token": claim_token,
            "delay": max(0.0, requeue_delay_sec),
            "reason": reason_code,
            "details": Jsonb(reason_details) if reason_details is not None else None,
        },
    )
    row = result.fetchone()
    if row is None:
        return False
    # Queue state and cross-replica notification share the claim transaction.
    # A failed append rolls back the release; there is no process-local fallback.
    from control_plane.outbox import append_event

    append_event(
        conn,
        aggregate_type="job",
        aggregate_id=job_id,
        aggregate_version=int(_row_get(row, "version", 1)),
        event_type="job.v1.queue_blocked",
        payload={
            "error": reason_code,
            "message": reason_code.replace("_", " "),
            "details": _row_get(row, "reason_details", 2),
        },
        idempotency_key=f"native_queue_blocked:{job_id}:{_row_get(row, 'generation', 0)}:{reason_code}",
    )
    return True


def clear_expired_claims(conn: Connection, *, limit: int = 100) -> int:
    """Maintenance sweep (§10.2): free claims whose window lapsed.

    The claimer may have crashed mid-scoring; clearing the claim makes
    the job immediately stealable. SKIP LOCKED keeps the sweep from ever
    blocking behind an active reservation transaction.
    """
    result = conn.execute(
        """
        WITH expired AS (
            SELECT job_id
              FROM jobs
             WHERE schedule_claim_expires_at IS NOT NULL
               AND schedule_claim_expires_at < clock_timestamp()
             LIMIT %(limit)s
               FOR UPDATE SKIP LOCKED
        )
        UPDATE jobs j
           SET schedule_claim_owner = NULL,
               schedule_claim_token = NULL,
               schedule_claim_expires_at = NULL,
               version = j.version + 1,
               updated_at = clock_timestamp()
          FROM expired e
         WHERE j.job_id = e.job_id
        """,
        {"limit": limit},
    )
    return result.rowcount
