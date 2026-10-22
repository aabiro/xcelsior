"""Project serverless capacity from a verified native attempt transaction."""

from __future__ import annotations

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from control_plane.db import run_transaction


def cancel_unstarted_worker(worker_id: str, job_id: str) -> bool:
    """Cancel only an unplaced job or an exact, still-unclaimed lease offer.

    Locking the offer serializes against lease claim. A committed claim sends
    the caller to the worker stop protocol; an unclaimed offer cannot have
    authorized a container, and may settle without fabricating a stop ACK.
    """

    def transaction(conn):
        from control_plane.attempts import report_attempt_status

        with conn.cursor(row_factory=dict_row) as cur:
            job = cur.execute(
                "SELECT job_id,status,host_id,active_attempt_id,payload FROM jobs "
                "WHERE job_id=%s FOR UPDATE",
                (job_id,),
            ).fetchone()
            payload = (job or {}).get("payload") or {}
            endpoint_id = payload.get("serverless_endpoint_id")
            if (
                not job
                or payload.get("job_type") != "serverless_worker"
                or payload.get("serverless_worker_id") != worker_id
                or not endpoint_id
            ):
                return False
            if job["active_attempt_id"] is not None:
                offer = cur.execute(
                    """SELECT l.attempt_id,l.host_id,l.fencing_token
                         FROM placement_leases l
                         JOIN serverless_workers w ON w.attempt_id=l.attempt_id
                        WHERE l.job_id=%s AND l.attempt_id=%s
                          AND l.status='offered' AND w.worker_id=%s
                          AND w.scheduler_job_id=l.job_id AND w.host_id=l.host_id
                          AND w.endpoint_id=%s AND w.released_at IS NULL
                        FOR UPDATE OF l,w""",
                    (job_id, job["active_attempt_id"], worker_id, endpoint_id),
                ).fetchone()
                if offer is None:
                    return False
                cur.execute(
                    "UPDATE jobs SET payload=COALESCE(payload,'{}'::jsonb)||%s " "WHERE job_id=%s",
                    (Jsonb({"lifecycle_intent": "terminate"}), job_id),
                )
                report_attempt_status(
                    conn,
                    job_id=job_id,
                    attempt_id=str(offer["attempt_id"]),
                    host_id=offer["host_id"],
                    fencing_token=offer["fencing_token"],
                    status="stopped",
                )
                return True
            if job["status"] != "queued" or job["host_id"]:
                return False
            if cur.execute(
                "SELECT 1 FROM job_attempts WHERE job_id=%s LIMIT 1", (job_id,)
            ).fetchone():
                return False
            worker = cur.execute(
                "SELECT worker_id FROM serverless_workers WHERE worker_id=%s "
                "AND scheduler_job_id=%s AND endpoint_id=%s AND attempt_id IS NULL "
                "AND released_at IS NULL FOR UPDATE",
                (worker_id, job_id, endpoint_id),
            ).fetchone()
            if worker is None:
                return False
            cur.execute(
                """UPDATE jobs SET status='cancelled',
                          payload=COALESCE(payload,'{}'::jsonb)||jsonb_build_object(
                            'status','cancelled','completed_at',EXTRACT(EPOCH FROM clock_timestamp())),
                          version=version+1,updated_at=clock_timestamp()
                    WHERE job_id=%s""",
                (job_id,),
            )
            cur.execute(
                """UPDATE serverless_workers SET state='terminated',current_concurrency=0,
                          allocated_at=0,released_at=EXTRACT(EPOCH FROM clock_timestamp()),
                          updated_at=EXTRACT(EPOCH FROM clock_timestamp()) WHERE worker_id=%s""",
                (worker_id,),
            )
            from control_plane.outbox import append_event

            append_event(
                conn,
                aggregate_type="job",
                aggregate_id=job_id,
                event_type="job.v1.cancelled",
                payload={"status": "cancelled"},
                idempotency_key=f"serverless_unplaced_cancel:{job_id}",
            )
            return True

    return run_transaction(transaction, what="serverless_cancel_unstarted")


def sync_serverless_attempt(conn, *, job_id: str, attempt_id: str) -> None:
    """Close routing and requeue work atomically with native attempt settlement.

    Callers hold the current job authority lock or have passed the fence gate.
    Lease expiry revokes authority without asserting a physical container exit.
    All persisted bindings must
    still agree; historical or partly bound workers are never adopted here.
    Billing uses the stored release timestamp on its normal periodic retry.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        worker = cur.execute(
            """SELECT w.worker_id, w.endpoint_id, a.status, e.idle_timeout_sec,
                      EXTRACT(EPOCH FROM a.ended_at)::float8 AS ended_at,
                      a.failure_code
                 FROM serverless_workers w
                 JOIN serverless_endpoints e ON e.endpoint_id=w.endpoint_id
                 JOIN job_attempts a ON a.attempt_id=w.attempt_id
                 JOIN jobs j ON j.job_id=a.job_id
                WHERE a.attempt_id=%s AND a.job_id=%s
                  AND w.scheduler_job_id=j.job_id AND w.host_id=a.host_id
                  AND j.payload->>'job_type'='serverless_worker'
                  AND j.payload->>'serverless_worker_id'=w.worker_id
                  AND j.payload->>'serverless_endpoint_id'=w.endpoint_id
                FOR UPDATE OF w""",
            (attempt_id, job_id),
        ).fetchone()
        if worker is None:
            return
        wid = worker["worker_id"]
        if worker["status"] == "running":
            cur.execute(
                """UPDATE serverless_workers
                      SET state=CASE WHEN state='booting' THEN 'ready' ELSE state END,
                          warm_expires_at=CASE WHEN billing_exempt AND state='booting'
                            THEN EXTRACT(EPOCH FROM clock_timestamp())+%s ELSE warm_expires_at END,
                          last_heartbeat_at=EXTRACT(EPOCH FROM clock_timestamp()),
                          updated_at=EXTRACT(EPOCH FROM clock_timestamp())
                    WHERE worker_id=%s AND released_at IS NULL""",
                (int(worker["idle_timeout_sec"] or 60), wid),
            )
            cur.execute(
                """UPDATE serverless_endpoints SET status='active',
                          updated_at=EXTRACT(EPOCH FROM clock_timestamp())
                    WHERE endpoint_id=%s AND deleted_at=0
                      AND status IN ('provisioning','scaled_down')
                      AND EXISTS (SELECT 1 FROM serverless_workers w
                          WHERE w.worker_id=%s AND w.state IN ('ready','idle')
                            AND w.released_at IS NULL)""",
                (worker["endpoint_id"], wid),
            )
        elif worker["status"] in (
            "succeeded",
            "failed",
            "cancelled",
            "lost",
            "fenced",
            "preempted",
        ):
            # No wall-clock fallback: only the native terminal transaction is
            # proof of release. Replays retain exactly the same billing bound.
            if worker["ended_at"] is None:
                raise ValueError("terminal serverless attempt has no release evidence")
            cur.execute(
                """UPDATE serverless_workers
                      SET state=%s, released_at=COALESCE(released_at,%s),
                          current_concurrency=0,
                          error_message=CASE WHEN %s IN ('failed','lost','fenced','preempted')
                            THEN %s ELSE error_message END,
                          updated_at=EXTRACT(EPOCH FROM clock_timestamp())
                    WHERE worker_id=%s""",
                (
                    (
                        "error"
                        if worker["status"] in ("failed", "lost", "fenced", "preempted")
                        else "terminated"
                    ),
                    worker["ended_at"],
                    worker["status"],
                    worker["failure_code"],
                    wid,
                ),
            )
            cur.execute(
                """UPDATE serverless_jobs
                      SET status='IN_QUEUE',worker_id=NULL,started_at=NULL,
                          queued_at=EXTRACT(EPOCH FROM clock_timestamp()),
                          updated_at=EXTRACT(EPOCH FROM clock_timestamp())
                    WHERE worker_id=%s AND status='IN_PROGRESS'""",
                (wid,),
            )
