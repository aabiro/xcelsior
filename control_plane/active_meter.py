"""Billing-only active meter repair; no new lifecycle or database-role authority."""

from __future__ import annotations

from psycopg.rows import dict_row

from billing import get_billing_engine
from control_plane.job_reconcile import DEFAULT_OBSERVATION_MAX_AGE_SEC
from control_plane.observations import latest_observation


def open_active_meter(conn, attempt_id: str) -> dict:
    """Revalidate and open usage within the caller's savepoint/transaction.

    Job/attempt/host locks freeze binding, status, price and observation ingest.
    NOWAIT avoids waiting against lifecycle/queue writers' inverse lock orders.
    An attempt alone, or a worker-supplied timestamp, cannot authorize a meter.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        current = cur.execute(
            """SELECT j.job_id, j.spec, j.spec_hash AS job_spec_hash, j.owner_id, j.desired_state, j.phase,
                      a.host_id, a.fencing_token, a.spec_hash, a.status,
                      EXTRACT(EPOCH FROM a.started_at)::float8 AS started_at,
                      a.ended_at,a.placement_explanation
                 FROM job_attempts a
                 JOIN jobs j ON j.job_id=a.job_id AND j.active_attempt_id=a.attempt_id
                            AND j.host_id=a.host_id
                 JOIN hosts h ON h.host_id=a.host_id
                WHERE a.attempt_id=%s FOR UPDATE OF j, a, h NOWAIT""",
            (attempt_id,),
        ).fetchone()
        if current is None:
            return {"reason": "current_authority_missing"}
        if (
            current["desired_state"] != "running"
            or current["phase"] != "running"
            or current["status"] != "running"
            or current["ended_at"] is not None
        ):
            return {"reason": "attempt_not_running"}
        lease = cur.execute(
            """SELECT lease_id FROM placement_leases
                WHERE attempt_id=%s AND job_id=%s AND host_id=%s AND fencing_token=%s
                  AND status='active' AND expires_at>clock_timestamp()
                  FOR UPDATE NOWAIT""",
            (attempt_id, current["job_id"], current["host_id"], current["fencing_token"]),
        ).fetchone()
        if lease is None:
            return {"reason": "lease_authority_missing"}
        now = cur.execute("SELECT EXTRACT(EPOCH FROM clock_timestamp()) AS now").fetchone()["now"]
        started = current["started_at"]
        if started is None or not 0 < started <= float(now):
            return {"reason": "start_evidence_missing"}
        observation = latest_observation(conn, current["host_id"])
        if observation is None:
            return {"reason": "observation_missing"}
        evidence = {
            "observation_id": observation["observation_id"],
            "lease_id": str(lease["lease_id"]),
            "attempt_id": attempt_id,
            "fencing_token": current["fencing_token"],
        }
        if float(now) - observation["received_at"].timestamp() > DEFAULT_OBSERVATION_MAX_AGE_SEC:
            return {**evidence, "reason": "observation_stale"}
        matches = [
            w
            for w in observation["workloads"]
            if w["job_id"] == current["job_id"]
            and w["attempt_id"] == attempt_id
            and w["fencing_token"] == current["fencing_token"]
            and current["spec_hash"]
            and w["spec_hash"] == current["spec_hash"]
        ]
        if len(matches) != 1 or matches[0]["state"] != "running":
            return {**evidence, "reason": "observation_authority_mismatch"}
        from control_plane.attempt_billing import bound_billing_inputs

        try:
            payload, frozen_host = bound_billing_inputs(
                current["spec"],
                current["job_spec_hash"],
                current["spec_hash"],
                (current["placement_explanation"] or {}).get("billing_snapshot"),
                current["host_id"],
            )
        except (ValueError, TypeError, OverflowError):
            return {**evidence, "reason": "billing_admission_or_snapshot_unqualified"}
        owner = payload["owner"]
        if current["owner_id"] != owner:
            return {**evidence, "reason": "owner_authority_missing"}
        existing = cur.execute(
            "SELECT meter_id FROM usage_meters WHERE attempt_id=%s",
            (attempt_id,),
        ).fetchone()
        if existing:
            # Never reopen completed usage or silently accept a corrupt binding.
            return {**evidence, "reason": "meter_already_exists"}
        job = {
            **payload,
            "job_id": current["job_id"],
            "attempt_id": attempt_id,
            "owner": owner,
            "started_at": started,
        }
        host = frozen_host
        try:
            with conn.transaction():
                meter = get_billing_engine().meter_job(job, host, conn=conn, active=True)
        except (ValueError, TypeError, OverflowError):
            return {**evidence, "reason": "pricing_or_resource_evidence_invalid"}
        if meter.completed_at > 0:
            return {**evidence, "reason": "meter_already_completed"}
        return {**evidence, "meter_id": meter.meter_id, "reason": None}
