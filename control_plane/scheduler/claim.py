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

DEFAULT_CLAIM_TTL_SEC = 15

# Queue order is part of the replica contract (§10.7): every replica must
# order work identically or two replicas fight over ordering fairness.
#
# The scope predicate implements the Phase 4 canary partition: in canary
# mode only jobs the new scheduler owns (matching gpu_model, or explicit
# payload {"scheduler": "v2"} opt-in) are claimable; the legacy queue
# walker skips exactly the same set (SchedulerConfig.owns_job), so no job
# is ever contested by both schedulers.
_CLAIM_SQL = """
WITH candidate AS (
    SELECT job_id
      FROM jobs
     WHERE phase = 'pending'
       AND desired_state = 'running'
       AND (next_schedule_at IS NULL OR next_schedule_at <= clock_timestamp())
       AND (schedule_claim_expires_at IS NULL
            OR schedule_claim_expires_at < clock_timestamp())
       AND (%(scope_all)s
            OR lower(COALESCE(payload->>'gpu_model', '')) = ANY(%(scope_models)s)
            OR lower(COALESCE(payload->>'scheduler', '')) = 'v2')
     ORDER BY effective_priority DESC,
              fair_share_finish ASC,
              queued_at ASC NULLS LAST
       FOR UPDATE SKIP LOCKED
     LIMIT 1
)
UPDATE jobs j
   SET schedule_claim_owner = %(replica_id)s,
       schedule_claim_token = gen_random_uuid(),
       schedule_claim_expires_at =
           clock_timestamp() + make_interval(secs => %(ttl_sec)s),
       schedule_attempt_count = j.schedule_attempt_count + 1,
       version = j.version + 1,
       updated_at = clock_timestamp()
  FROM candidate c
 WHERE j.job_id = c.job_id
RETURNING j.job_id, j.schedule_claim_token, j.schedule_claim_expires_at,
          j.schedule_attempt_count, j.version, j.generation, j.spec,
          j.spec_hash, j.payload, j.priority, j.effective_priority
"""


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
               last_schedule_conflict_at = CASE
                   WHEN %(reason)s = 'placement_conflict' THEN clock_timestamp()
                   ELSE last_schedule_conflict_at
               END,
               version = version + 1,
               updated_at = clock_timestamp()
         WHERE job_id = %(job_id)s
           AND schedule_claim_token = %(token)s
        """,
        {
            "job_id": job_id,
            "token": claim_token,
            "delay": max(0.0, requeue_delay_sec),
            "reason": reason_code,
        },
    )
    return result.rowcount == 1


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
