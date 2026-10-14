"""Job controller for blueprint §12.4; remediation defaults to report-only.

Compare desired state with the current attempt, lease, and fresh observation.
Do not infer a successful exit from an ``exited`` container or replay a stale
worker report. Lifecycle settlement still belongs to the fenced attempt and
lease services. New findings ship report-only before remediation is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from psycopg import Connection

from control_plane.observations import latest_observation
from control_plane.reconcile import (
    DEFAULT_MISSING_GRACE_SEC,
    ActionPolicy,
    _get,
    _open_finding,
    _record_action,
    action_policy_for,
)

DEFAULT_RECHECK_SEC = 60
DEFAULT_OBSERVATION_MAX_AGE_SEC = 180
_RUNTIME_STATUSES = frozenset({"lease_claimed", "starting", "running"})
_PRE_RUNTIME_STATUSES = frozenset({"reserved", "command_pending", "lease_offered"})
_FINDING_TYPES = (
    "job_active_attempt_missing",
    "job_attempt_authority_mismatch",
    "job_lease_authority_missing",
    "job_observation_missing",
    "job_observation_stale",
    "job_observation_authority_mismatch",
    "job_stop_command_missing",
    "job_stop_not_observed",
    "job_attempt_status_mismatch",
    "job_terminal_observation",
    "job_active_meter_missing",
)


@dataclass(frozen=True)
class JobReconcileReport:
    job_id: str
    findings_opened: list[str] = field(default_factory=list)
    findings_resolved: int = 0
    actions_taken: list[str] = field(default_factory=list)
    observation_id: str | None = None
    recheck_after_sec: int | None = None


def reconcile_job(
    conn: Connection,
    job_id: str,
    *,
    missing_grace_sec: float = DEFAULT_MISSING_GRACE_SEC,
    observation_max_age_sec: float = DEFAULT_OBSERVATION_MAX_AGE_SEC,
) -> JobReconcileReport:
    """Reconcile one job under the caller's transaction and resource lock.

    Pending unassigned work is already the durable scheduler queue. Completed
    and stopped work must never be re-admitted merely because legacy desired
    state projects ``completed``/``failed`` to ``running``. A live job without
    authority is reported, not restarted with an invented fence.
    """
    row = conn.execute(
        """
        SELECT j.status, j.desired_state, j.phase, j.host_id,
               j.active_attempt_id::text, j.generation, j.tenant_id,
               a.job_id AS attempt_job_id, a.host_id AS attempt_host_id,
               a.fencing_token, a.status AS attempt_status, a.spec_hash,
               EXTRACT(EPOCH FROM (clock_timestamp() -
                   COALESCE(a.lease_claimed_at, a.reserved_at))) AS attempt_age,
               EXTRACT(EPOCH FROM clock_timestamp()) AS checked_at,
               j.payload->>'lifecycle_intent' AS lifecycle_intent
          FROM jobs j
          LEFT JOIN job_attempts a ON a.attempt_id = j.active_attempt_id
         WHERE j.job_id = %s
           FOR UPDATE OF j SKIP LOCKED
        """,
        (job_id,),
    ).fetchone()
    # Writers lock jobs before the trigger touches the queue; this controller
    # already holds the queue row. Never wait on the inverse lock order.
    if (
        row is None
        and conn.execute("SELECT 1 FROM jobs WHERE job_id = %s", (job_id,)).fetchone() is not None
    ):
        raise RuntimeError("job_locked")
    current: dict[str, Any] = {}
    if row is not None:
        keys = (
            "status",
            "desired_state",
            "phase",
            "host_id",
            "active_attempt_id",
            "generation",
            "tenant_id",
            "attempt_job_id",
            "attempt_host_id",
            "fencing_token",
            "attempt_status",
            "spec_hash",
            "attempt_age",
            "checked_at",
            "lifecycle_intent",
        )
        current = {key: _get(row, key, i) for i, key in enumerate(keys)}
    opened: list[str] = []
    actions: list[str] = []
    still_open: list[str] = []
    observation_id = None
    recheck = None

    def finding(kind: str, summary: str, observed: dict[str, Any] | None = None) -> None:
        still_open.append(kind)
        if (
            _open_finding(
                conn,
                resource_type="job",
                resource_id=job_id,
                finding_type=kind,
                severity="warning",
                summary=summary,
                desired={
                    "state": current.get("desired_state"),
                    "attempt_id": current.get("active_attempt_id"),
                    "host_id": current.get("host_id"),
                    "fencing_token": current.get("fencing_token"),
                },
                observed=observed,
                tenant_id=current.get("tenant_id"),
            )
            is not None
        ):
            opened.append(kind)

    if current:
        aid = current["active_attempt_id"]
        active = current["phase"] in ("pending", "scheduled", "starting", "running")
        if aid or (active and current["desired_state"] == "running"):
            recheck = DEFAULT_RECHECK_SEC
        if not aid:
            if current["phase"] in ("scheduled", "starting", "running"):
                finding("job_active_attempt_missing", "live job has no active attempt authority")
        elif (
            current["attempt_job_id"] != job_id
            or current["attempt_host_id"] != current["host_id"]
            or current["attempt_status"] not in _RUNTIME_STATUSES | _PRE_RUNTIME_STATUSES
        ):
            finding(
                "job_attempt_authority_mismatch",
                "active attempt does not match job authority",
                {
                    "job_id": current["attempt_job_id"],
                    "host_id": current["attempt_host_id"],
                    "status": current["attempt_status"],
                },
            )
        else:
            runtime_active = current["attempt_status"] in _RUNTIME_STATUSES
            lease = conn.execute(
                """
                SELECT 1 FROM placement_leases
                 WHERE attempt_id = %s AND job_id = %s AND host_id = %s
                   AND fencing_token = %s
                   AND ((%s AND status = 'active' AND expires_at > clock_timestamp())
                     OR (NOT %s AND status = 'offered' AND claim_deadline > clock_timestamp()))
                """,
                (
                    aid,
                    job_id,
                    current["host_id"],
                    current["fencing_token"],
                    runtime_active,
                    runtime_active,
                ),
            ).fetchone()
            if lease is None:
                finding("job_lease_authority_missing", "active attempt has no valid matching lease")

            if current["desired_state"] == "stopped":
                from control_plane.lifecycle import desired_stop_intent

                intent = desired_stop_intent(current["status"], current["lifecycle_intent"])
                command = conn.execute(
                    """
                    SELECT status,
                           EXTRACT(EPOCH FROM clock_timestamp()) - created_at AS age
                      FROM agent_commands
                     WHERE job_id = %s AND attempt_id = %s AND host_id = %s
                       AND fencing_token = %s AND command = 'stop_attempt'
                       AND spec_hash IS NOT DISTINCT FROM %s
                       AND args->>'intent' = %s
                       AND args->'preserve' = to_jsonb(%s::boolean)
                       AND (status = 'acknowledged'
                         OR (status IN ('pending', 'claimed')
                             AND expires_at > EXTRACT(EPOCH FROM clock_timestamp())))
                     ORDER BY created_at DESC LIMIT 1
                    """,
                    (
                        job_id,
                        aid,
                        current["host_id"],
                        current["fencing_token"],
                        current["spec_hash"],
                        intent,
                        intent == "stop",
                    ),
                ).fetchone()
                if command is None:
                    finding(
                        "job_stop_command_missing",
                        "desired stopped has no valid fenced stop command",
                    )
                    if action_policy_for("job_stop_command_missing") == ActionPolicy.ENFORCE:
                        from control_plane.lifecycle import reconcile_fenced_stop

                        result = reconcile_fenced_stop(
                            conn, job_id=job_id, created_by="reconciler:job_stop_command_missing"
                        )
                        # Include findings opened during a previous report-only
                        # pass: enabling repair must not depend on a new finding.
                        finding_row = conn.execute(
                            "SELECT finding_id FROM reconciliation_findings "
                            "WHERE resource_type = 'job' AND resource_id = %s "
                            "AND finding_type = 'job_stop_command_missing' AND resolved_at IS NULL",
                            (job_id,),
                        ).fetchone()
                        if finding_row is not None:
                            _record_action(
                                conn,
                                str(_get(finding_row, "finding_id", 0)),
                                "enqueue_fenced_stop" if result.ok else "fenced_stop_refused",
                                {
                                    "ok": result.ok,
                                    "reason": result.reason,
                                    "command_id": result.command_id,
                                    "intent": result.intent,
                                },
                            )
                        if result.ok and result.command_created:
                            actions.append("enqueue_fenced_stop")
                elif float(_get(command, "age", 1)) >= missing_grace_sec:
                    finding(
                        "job_stop_not_observed", "stop delivery has not settled execution authority"
                    )

            observation = latest_observation(conn, str(current["host_id"]))
            past_grace = float(current["attempt_age"] or 0) >= missing_grace_sec
            if observation is None:
                if runtime_active and past_grace:
                    finding("job_observation_missing", "active attempt has no host observation")
            elif (
                float(current["checked_at"]) - observation["received_at"].timestamp()
                > observation_max_age_sec
            ):
                if runtime_active:
                    finding("job_observation_stale", "latest host observation is stale")
            else:
                observation_id = observation["observation_id"]
                matches = [
                    w
                    for w in observation["workloads"]
                    if current["spec_hash"]
                    and w["job_id"] == job_id
                    and w["attempt_id"] == aid
                    and w["fencing_token"] == current["fencing_token"]
                    and w["spec_hash"] == current["spec_hash"]
                ]
                if not matches:
                    if runtime_active and past_grace:
                        finding(
                            "job_observation_authority_mismatch",
                            "fresh observation has no workload matching job/attempt/fence/spec",
                            {"observation_id": observation_id},
                        )
                elif len(matches) != 1:
                    finding(
                        "job_observation_authority_mismatch",
                        "fresh observation has multiple workloads under one attempt authority",
                        {"observation_id": observation_id, "matches": len(matches)},
                    )
                else:
                    seen = matches[0]
                    if seen["state"] == "exited":
                        finding(
                            "job_terminal_observation",
                            "container exited; a fenced terminal report must settle the attempt",
                            {"observation_id": observation_id, "workload": seen},
                        )
                    elif seen["state"] == "running":
                        if current["attempt_status"] != "running":
                            finding(
                                "job_attempt_status_mismatch",
                                "observed running but attempt status differs",
                            )
                        elif current["desired_state"] == "running" and lease is not None:
                            # This is an observation projection, never a lifecycle transition.
                            conn.execute(
                                "UPDATE jobs SET observed_generation = generation "
                                "WHERE job_id = %s AND observed_generation <> generation",
                                (job_id,),
                            )
                        meter = conn.execute(
                            "SELECT 1 FROM usage_meters WHERE attempt_id = %s "
                            "AND (completed_at IS NULL OR completed_at = 0)",
                            (aid,),
                        ).fetchone()
                        if meter is None:
                            finding(
                                "job_active_meter_missing",
                                "observed running has no active attempt meter",
                            )
                    elif seen["state"] != "preparing" or current["attempt_status"] == "running":
                        finding(
                            "job_attempt_status_mismatch",
                            "observed workload state differs from active attempt",
                            {"observation_id": observation_id, "workload": seen},
                        )

    # Resolve only findings owned by this controller; billing/host controllers
    # must never lose their findings just because a job pass did not evaluate them.
    resolved = conn.execute(
        """
        UPDATE reconciliation_findings SET resolved_at = clock_timestamp()
         WHERE resource_type = 'job' AND resource_id = %s AND resolved_at IS NULL
           AND finding_type = ANY(%s) AND NOT (finding_type = ANY(%s))
        """,
        (job_id, list(_FINDING_TYPES), still_open),
    ).rowcount
    return JobReconcileReport(
        job_id=job_id,
        findings_opened=opened,
        findings_resolved=resolved,
        actions_taken=actions,
        observation_id=observation_id,
        recheck_after_sec=recheck,
    )
