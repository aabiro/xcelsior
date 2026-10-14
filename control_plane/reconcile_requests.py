"""Bounded recovery scan for missed job reconciliation requests (§12.3)."""

from psycopg import Connection


def enqueue_missing_jobs(conn: Connection, *, limit: int = 1000) -> int:
    """Fill missing queue entries without delaying existing or claimed work.

    Active jobs retain their periodic entry, so each batch advances through
    the missing set. Include terminal jobs that still hold attempt authority.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    result = conn.execute(
        """
        INSERT INTO reconciliation_queue
            (resource_type, resource_id, reason, requested_by)
        SELECT 'job', j.job_id, 'recovery_scan', 'job_reconciliation_recovery'
          FROM jobs j
         WHERE (j.active_attempt_id IS NOT NULL
            OR (j.desired_state = 'running'
                AND j.phase IN ('pending', 'scheduled', 'starting', 'running')))
           AND NOT EXISTS (
               SELECT 1 FROM reconciliation_queue q
                WHERE q.resource_type = 'job' AND q.resource_id = j.job_id)
         ORDER BY j.job_id
         LIMIT %s
        ON CONFLICT (resource_type, resource_id) DO NOTHING
        """,
        (limit,),
    )
    return result.rowcount


def recover_job_requests_task() -> dict[str, int]:
    from control_plane.db import run_transaction

    return {"enqueued": run_transaction(enqueue_missing_jobs, what="recover_job_requests")}
