"""Place one claimed job on a Slurm cluster (blueprint Phase 2.2).

A host placement binds devices, a lease and an agent command. A Slurm
placement does none of that. It opens one attempt and one ``slurm_submissions``
row, leaves ``jobs.host_id`` empty, and records the target as
``slurm:<cluster_id>`` on the attempt. The submitter turns that row into at
most one Slurm job.

The caller holds no host lock. Lock order is the job (already claimed), the
team policy, then the cluster row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from psycopg import Connection
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from control_plane.job_specs import validate_execution_spec
from control_plane.scheduler.filters import FILTER_POLICY_VERSION
from control_plane.scheduler.reservation import (
    ClaimLost,
    JobNotSchedulable,
    JobSpecUnqualified,
    QueuePolicyChanged,
    ReservationConflict,
    TeamQuotaConflict,
)

# Same token Slurm and slurm_adapter accept as a job id / submission name.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class SlurmPlacementRefused(ReservationConflict):
    """The cluster cannot take this job. The caller releases the claim."""

    def __init__(self, code: str, message: str, **details):
        super().__init__(message, **details)
        self.code = code


@dataclass(frozen=True)
class SlurmPlacement:
    job_id: str
    attempt_id: str
    attempt_number: int
    cluster_id: str
    submission_name: str
    target: str  # slurm:<cluster_id>
    fencing_token: int


def _require(row):
    if row is None:
        raise RuntimeError("statement expected to return a row returned none")
    return row


def _gpu_count(spec: dict) -> int:
    raw = spec.get("num_gpus")
    if raw is None:
        return 0
    if isinstance(raw, bool):
        raise JobSpecUnqualified("job GPU count is unqualified")
    try:
        count = int(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise JobSpecUnqualified("job GPU count is unqualified") from exc
    if count < 0:
        raise JobSpecUnqualified("job GPU count is unqualified")
    return count


def place_on_cluster(
    conn: Connection,
    *,
    job_id: str,
    claim_token: str,
    replica_id: str,
    cluster_id: str,
    expected_spec_hash: str | None,
    trace_id: str | None,
) -> SlurmPlacement:
    """Bind ``job_id`` to ``cluster_id`` inside the caller's transaction.

    Raises a :class:`ReservationConflict` when the claim, the spec or the
    cluster no longer allows it. The transaction rolls back with the exception.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        job = cur.execute(
            """
            SELECT status, phase, desired_state, generation, version, spec, spec_hash,
                   schedule_claim_token, tenant_id, queue_policy_version
              FROM jobs
             WHERE job_id = %s
               FOR UPDATE
            """,
            (job_id,),
        ).fetchone()
    if job is None:
        raise JobNotSchedulable(f"job {job_id} no longer exists", job_id=job_id)
    claim_ok = conn.execute(
        """
        SELECT 1 FROM jobs
         WHERE job_id = %s
           AND schedule_claim_token = %s
           AND schedule_claim_expires_at >= clock_timestamp()
        """,
        (job_id, claim_token),
    ).fetchone()
    if claim_ok is None:
        raise ClaimLost(
            f"schedule claim for {job_id} is no longer held by this replica",
            job_id=job_id,
            replica_id=replica_id,
        )
    if job["phase"] != "pending" or job["desired_state"] != "running":
        raise JobNotSchedulable(
            f"job {job_id} is {job['phase']}/{job['desired_state']}, not pending/running",
            phase=job["phase"],
            desired_state=job["desired_state"],
        )
    spec_hash = job["spec_hash"]
    if expected_spec_hash is not None and spec_hash != expected_spec_hash:
        raise JobSpecUnqualified("job execution spec changed after its scheduling claim")
    try:
        spec = validate_execution_spec(job["spec"], spec_hash, require_admission=True)
    except ValueError as exc:
        raise JobSpecUnqualified("job execution spec is unqualified", reason=str(exc)) from exc
    if str(spec.get("preferred_host_id") or "").strip():
        raise SlurmPlacementRefused(
            "slurm_and_host_pin",
            "a job cannot name both a Slurm cluster and a host",
            job_id=job_id,
        )
    named = str(spec.get("hpc_cluster_id") or "").strip()
    if named != cluster_id:
        raise JobNotSchedulable(
            f"job {job_id} no longer names cluster {cluster_id}",
            cluster_id=named,
        )

    num_gpus = _gpu_count(spec)
    from control_plane.scheduler.fairness import QuotaExceeded, lock_team_policy

    try:
        team_policy = lock_team_policy(conn, job_id=job_id, requested_gpus=num_gpus)
    except QuotaExceeded as exc:
        raise TeamQuotaConflict(str(exc)) from exc
    if int(job["queue_policy_version"] or 0) != int(team_policy.version):
        raise QueuePolicyChanged("team policy changed after the queue claim")

    with conn.cursor(row_factory=dict_row) as cur:
        cluster = cur.execute(
            """
            SELECT cluster_id, enabled, research_only
              FROM slurm_clusters
             WHERE cluster_id = %s
               FOR UPDATE
            """,
            (cluster_id,),
        ).fetchone()
    if cluster is None:
        raise SlurmPlacementRefused(
            "slurm_cluster_not_found",
            f"Slurm cluster {cluster_id} is not registered",
            cluster_id=cluster_id,
        )
    if not cluster["enabled"]:
        raise SlurmPlacementRefused(
            "slurm_cluster_disabled",
            f"Slurm cluster {cluster_id} is disabled",
            cluster_id=cluster_id,
        )
    if cluster["research_only"]:
        raise SlurmPlacementRefused(
            "slurm_cluster_research_only",
            f"Slurm cluster {cluster_id} is research-only and cannot take tenant jobs",
            cluster_id=cluster_id,
        )

    attempt_number = int(
        _require(
            conn.execute(
                "SELECT COALESCE(MAX(attempt_number), 0) + 1 FROM job_attempts WHERE job_id = %s",
                (job_id,),
            ).fetchone()
        )[0]
    )
    submission_name = f"{job_id}-a{attempt_number}"
    if _NAME_RE.match(submission_name) is None:
        raise SlurmPlacementRefused(
            "slurm_submission_name_invalid",
            f"submission name {submission_name!r} is not a legal Slurm job name",
            submission_name=submission_name,
        )
    target = f"slurm:{cluster_id}"
    fencing_token = int(
        _require(conn.execute("SELECT nextval('placement_fencing_token_seq')").fetchone())[0]
    )
    explanation = {"target": target, "cluster_id": cluster_id}
    tenant_id = spec.get("tenant_id") or job["tenant_id"]
    if tenant_id is not None:
        tenant_id = str(tenant_id)

    try:
        attempt_row = conn.execute(
            """
            INSERT INTO job_attempts
                (job_id, attempt_number, status, host_id, fencing_token,
                 job_generation, spec_hash, policy_version, placement_explanation,
                 created_by, trace_id)
            VALUES (%s, %s, 'reserved', NULL, %s, %s, %s, %s, %s, %s, %s)
            RETURNING attempt_id
            """,
            (
                job_id,
                attempt_number,
                fencing_token,
                int(job["generation"]),
                spec_hash,
                FILTER_POLICY_VERSION,
                Jsonb(explanation),
                f"scheduler:{replica_id}",
                trace_id,
            ),
        ).fetchone()
        attempt_id = str(_require(attempt_row)[0])
        conn.execute(
            """
            INSERT INTO slurm_submissions
                (submission_name, cluster_id, xcelsior_job_id, attempt_id, tenant_id, state)
            VALUES (%s, %s, %s, %s, %s, 'pending')
            """,
            (submission_name, cluster_id, job_id, attempt_id, tenant_id),
        )
    except UniqueViolation as exc:
        raise SlurmPlacementRefused(
            "slurm_attempt_conflict",
            f"job {job_id} already has an active attempt",
            job_id=job_id,
        ) from exc

    updated = conn.execute(
        """
        UPDATE jobs
           SET phase = 'scheduled',
               status = 'assigned',
               host_id = NULL,
               active_attempt_id = %s,
               error_message = '',
               exit_code = NULL,
               payload = COALESCE(payload,'{}'::jsonb)-ARRAY[
                   'http_ports','auto_launch_ports','container_name','container_id',
                   'public_ssh_port','ssh_info','started_at','completed_at']::text[],
               reason_code = NULL,
               reason_details = NULL,
               schedule_claim_owner = NULL,
               schedule_claim_token = NULL,
               schedule_claim_expires_at = NULL,
               next_schedule_at = NULL,
               version = version + 1,
               updated_at = clock_timestamp()
         WHERE job_id = %s
           AND schedule_claim_token = %s
        """,
        (attempt_id, job_id, claim_token),
    )
    if updated.rowcount != 1:
        raise ClaimLost(f"claim token for {job_id} vanished inside reservation", job_id=job_id)

    if num_gpus >= 1:
        from control_plane.scheduler.fairness import spend_service_credit

        spend_service_credit(conn, team_policy, gpu_count=num_gpus)

    from control_plane.outbox import append_event

    append_event(
        conn,
        aggregate_type="job",
        aggregate_id=job_id,
        aggregate_version=int(job["generation"]),
        event_type="job.v1.placement_reserved",
        payload={
            "job_id": job_id,
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "host_id": target,
            "target": target,
            "cluster_id": cluster_id,
            "submission_name": submission_name,
            "fencing_token": fencing_token,
            "policy_version": FILTER_POLICY_VERSION,
        },
        destination_class="default",
        idempotency_key=f"placement:{attempt_id}",
        trace_id=trace_id,
    )
    return SlurmPlacement(
        job_id=job_id,
        attempt_id=attempt_id,
        attempt_number=attempt_number,
        cluster_id=cluster_id,
        submission_name=submission_name,
        target=target,
        fencing_token=fencing_token,
    )
