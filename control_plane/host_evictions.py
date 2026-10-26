"""Human-approved host eviction through existing fenced attempt settlement.

Approval binds the exact native authority, not just a host name. Commands and
the executing plan commit together. Neither a command ACK nor elapsed grace
releases capacity; only terminal attempt/lease/allocation proof completes it.
There is deliberately no force escalation in this operator action.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json

from psycopg.rows import dict_row
from psycopg.errors import LockNotAvailable
from psycopg.types.json import Jsonb

from control_plane.commands import CommandProtocolError, enqueue_current_attempt_command
from control_plane.job_specs import validate_execution_spec
from control_plane.launch import action_plans
from control_plane.launch.service import PlanConflict
from control_plane.outbox import append_event
from control_plane.sweep_isolation import SweepQuarantine

ACTION = "evict_host_workloads"
POLICY = "host-eviction/v1"


def canonical_hash(canonical):
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def snapshot(conn, *, host_id, expected_version):
    """Lock jobs before host, matching placement order; reject orphan authority.

    A drained host cannot accept new reservations. NOWAIT avoids a partial job
    lock graph behind lifecycle writers. The API retries the whole transaction.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        ids = [
            r["job_id"]
            for r in cur.execute(
                """SELECT job_id FROM jobs WHERE host_id=%s
                AND (active_attempt_id IS NOT NULL OR phase IN ('scheduled','starting','running')
                    OR status IN ('assigned','leased','starting','running','stopping','restarting'))
            UNION SELECT job_id FROM placement_leases WHERE host_id=%s
                AND status IN ('offered','active')
            UNION SELECT job_id FROM gpu_device_allocations WHERE host_id=%s
                AND released_at IS NULL ORDER BY job_id LIMIT 257""",
                (host_id, host_id, host_id),
            ).fetchall()
        ]
        if len(ids) > 256:
            raise PlanConflict("eviction_too_large", "evict at most 256 workloads per plan")
        try:
            jobs = cur.execute(
                "SELECT * FROM jobs WHERE job_id=ANY(%s) ORDER BY job_id FOR UPDATE NOWAIT", (ids,)
            ).fetchall()
        except LockNotAvailable as exc:
            conflict = PlanConflict("eviction_busy", "workloads are changing; retry this plan")
            conflict.retryable = True
            raise conflict from exc
        host = cur.execute(
            "SELECT status,version FROM hosts WHERE host_id=%s FOR UPDATE", (host_id,)
        ).fetchone()
        if not host or host["status"] != "draining":
            raise PlanConflict("host_not_draining", "drain the host before planning eviction")
        if host["version"] != expected_version:
            raise PlanConflict("version_conflict", "host changed; create a new eviction preview")
        if cur.execute(
            """SELECT 1 FROM action_plans WHERE action_type=%s AND status='executing'
            AND canonical_args->>'host_id'=%s LIMIT 1""",
            (ACTION, host_id),
        ).fetchone():
            raise PlanConflict(
                "eviction_in_progress", "wait for the current host eviction to settle"
            )
        if len(jobs) != len(ids):
            raise PlanConflict("eviction_authority_unavailable", "host has orphaned authority")
        victims = []
        for job in jobs:
            if (
                job["host_id"] != host_id
                or not job["active_attempt_id"]
                or job["desired_state"] != "running"
                or job["lifecycle_intent"]
                or job["history_disposition"] is not None
            ):
                raise PlanConflict(
                    "eviction_authority_unavailable",
                    "workload is changing or lacks native authority",
                )
            try:
                spec = validate_execution_spec(
                    job["spec"], job["spec_hash"], require_admission=True
                )
            except ValueError as exc:
                raise PlanConflict(
                    "eviction_authority_unavailable", "workload lacks a valid native admission"
                ) from exc
            authority = cur.execute(
                """SELECT a.attempt_id,a.fencing_token,a.spec_hash,a.job_generation,
                    l.lease_id FROM job_attempts a JOIN placement_leases l USING(attempt_id)
                WHERE a.attempt_id=%s AND a.job_id=%s AND a.host_id=%s
                    AND a.status IN ('lease_claimed','starting','running')
                    AND l.job_id=a.job_id AND l.host_id=a.host_id
                    AND l.fencing_token=a.fencing_token AND l.status='active'
                    AND l.expires_at>clock_timestamp() FOR UPDATE OF a,l""",
                (job["active_attempt_id"], job["job_id"], host_id),
            ).fetchall()
            if (
                len(authority) != 1
                or authority[0]["spec_hash"] != job["spec_hash"]
                or authority[0]["job_generation"] != job["generation"]
            ):
                raise PlanConflict(
                    "eviction_authority_unavailable", "wait for a current, claimed native lease"
                )
            a = authority[0]
            if (
                cur.execute(
                    """SELECT 1 FROM placement_leases WHERE host_id=%s AND job_id=%s
                AND status IN ('offered','active') AND lease_id<>%s""",
                    (host_id, job["job_id"], a["lease_id"]),
                ).fetchone()
                or cur.execute(
                    """SELECT 1 FROM gpu_device_allocations WHERE host_id=%s AND job_id=%s
                AND released_at IS NULL AND attempt_id<>%s""",
                    (host_id, job["job_id"], a["attempt_id"]),
                ).fetchone()
            ):
                raise PlanConflict(
                    "eviction_authority_unavailable",
                    "conflicting host authority needs reconciliation",
                )
            devices = cur.execute(
                """SELECT gpu_device_id FROM gpu_device_allocations WHERE attempt_id=%s
                    AND host_id=%s AND released_at IS NULL ORDER BY gpu_device_id""",
                (a["attempt_id"], host_id),
            ).fetchall()
            victims.append(
                dict(
                    job_id=job["job_id"],
                    attempt_id=str(a["attempt_id"]),
                    lease_id=str(a["lease_id"]),
                    fencing_token=a["fencing_token"],
                    generation=job["generation"],
                    spec_hash=job["spec_hash"],
                    gpu_device_ids=[str(d["gpu_device_id"]) for d in devices],
                    settlement=(
                        "terminate" if spec.get("job_type") == "serverless_worker" else "requeue"
                    ),
                )
            )
        return victims


def approved_canonical(plan, host_id):
    canonical = dict(plan["canonical_args"])
    if canonical.get("host_id") != host_id:
        raise PlanConflict("resource_mismatch", "plan belongs to a different host")
    if (
        canonical_hash(canonical) != plan["canonical_args_hash"]
        or canonical.get("policy") != POLICY
        or canonical.get("allow_force") is not False
        or type(canonical.get("grace_seconds")) is not int
        or not 1 <= canonical["grace_seconds"] <= 300
        or not isinstance(canonical.get("victims"), list)
    ):
        raise PlanConflict("argument_hash_mismatch", "create and approve a new eviction preview")
    return canonical


def progress(conn, plan):
    """Read physical authority closure, never infer it from command delivery."""
    if plan["status"] == "succeeded":
        return dict(plan["idempotent_response"])
    canonical = plan["canonical_args"]
    response = dict(plan.get("idempotent_response") or {})
    victims = []
    with conn.cursor(row_factory=dict_row) as cur:
        for victim in canonical["victims"]:
            proof = cur.execute(
                """SELECT a.status,l.status AS lease_status,
                    NOT EXISTS(SELECT 1 FROM gpu_device_allocations g
                        WHERE g.attempt_id=a.attempt_id AND g.released_at IS NULL) AS released
                FROM job_attempts a JOIN placement_leases l ON l.lease_id=%s
                WHERE a.attempt_id=%s AND a.job_id=%s AND a.host_id=%s
                    AND a.fencing_token=%s AND a.spec_hash=%s AND a.job_generation=%s
                    AND l.attempt_id=a.attempt_id AND l.fencing_token=a.fencing_token
                    AND l.job_id=a.job_id AND l.host_id=a.host_id""",
                (
                    victim["lease_id"],
                    victim["attempt_id"],
                    victim["job_id"],
                    canonical["host_id"],
                    victim["fencing_token"],
                    victim["spec_hash"],
                    victim["generation"],
                ),
            ).fetchone()
            closed = bool(
                proof
                and proof["status"]
                in ("succeeded", "failed", "cancelled", "lost", "fenced", "preempted")
                and proof["lease_status"] not in ("active", "offered")
                and proof["released"]
            )
            command = cur.execute(
                "SELECT status FROM agent_commands WHERE host_id=%s AND idempotency_key=%s",
                (canonical["host_id"], f'evict:{plan["plan_id"]}:{victim["attempt_id"]}'),
            ).fetchone()
            victims.append(
                dict(
                    job_id=victim["job_id"],
                    attempt_id=victim["attempt_id"],
                    released=closed,
                    command_status=command["status"] if command else "missing",
                )
            )
    complete = all(v["released"] for v in victims)
    return {
        **response,
        "ok": True,
        "plan_id": str(plan["plan_id"]),
        "host_id": canonical["host_id"],
        "pending": not complete,
        "status": "succeeded" if complete else "executing",
        "victims": victims,
        "allow_force": False,
    }


def reconcile(conn, plan):
    """Caller owns the action plan lock. A restart can resume this proof check."""
    response = progress(conn, plan)
    if not response["pending"]:
        conn.execute(
            """UPDATE action_plans SET status='succeeded',resulting_resource_id=%s,
                idempotent_response=%s,version=version+1 WHERE plan_id=%s AND status='executing'""",
            (response["host_id"], Jsonb(response), plan["plan_id"]),
        )
        append_event(
            conn,
            aggregate_type="host",
            aggregate_id=response["host_id"],
            event_type="host.workloads_evicted",
            tenant_id=str(plan["tenant_id"]),
            payload=dict(plan_id=str(plan["plan_id"]), victims=response["victims"]),
            idempotency_key=f'host-eviction-settled:{plan["plan_id"]}',
        )
    return response


def execute(conn, plan, *, host_id, created_by):
    """An approved request is an atomic command set, never an optimistic requeue."""
    canonical = approved_canonical(plan, host_id)
    if plan["status"] == "succeeded":
        return {**dict(plan["idempotent_response"]), "idempotent": True}
    if plan["status"] == "executing":
        return {**reconcile(conn, plan), "idempotent": True}
    if plan["status"] != "approved":
        raise PlanConflict("approval_required", "approve the eviction plan before execution")
    if action_plans.is_expired(plan):
        raise PlanConflict("plan_expired", "create and approve a fresh eviction preview")
    if (
        snapshot(conn, host_id=host_id, expected_version=canonical["expected_version"])
        != canonical["victims"]
    ):
        raise PlanConflict(
            "eviction_snapshot_changed", "workloads changed; create and approve a new preview"
        )
    # Consume the host snapshot as well as the plan. Concurrent approved plans
    # cannot share one unchanged host version, even if their victims match.
    conn.execute("UPDATE hosts SET version=version+1 WHERE host_id=%s", (host_id,))
    deadline = datetime.now(timezone.utc) + timedelta(seconds=canonical["grace_seconds"])
    commands = []
    for victim in canonical["victims"]:
        try:
            command = enqueue_current_attempt_command(
                conn,
                job_id=victim["job_id"],
                command="stop_attempt",
                created_by=created_by,
                idempotency_key=f'evict:{plan["plan_id"]}:{victim["attempt_id"]}',
                args=dict(
                    preemption=True,
                    preemption_plan_id=str(plan["plan_id"]),
                    grace_deadline=deadline.isoformat(),
                    checkpoint_policy="none",
                    preserve=False,
                    allow_force=False,
                    force=False,
                ),
            )
        except CommandProtocolError as exc:
            raise PlanConflict(
                "eviction_authority_unavailable", "authority changed; create a new preview"
            ) from exc
        commands.append(command.command_id)
        conn.execute(
            """UPDATE jobs SET reason_code='operator_eviction_requested',reason_details=%s,
                version=version+1 WHERE job_id=%s""",
            (
                Jsonb(dict(plan_id=str(plan["plan_id"]), grace_deadline=deadline.isoformat())),
                victim["job_id"],
            ),
        )
    plan = action_plans.mark_executing(conn, str(plan["plan_id"]))
    response = dict(command_ids=commands, grace_deadline=deadline.isoformat())
    conn.execute(
        "UPDATE action_plans SET idempotent_response=%s WHERE plan_id=%s",
        (Jsonb(response), plan["plan_id"]),
    )
    plan["idempotent_response"] = response
    append_event(
        conn,
        aggregate_type="host",
        aggregate_id=host_id,
        event_type="host.eviction_requested",
        tenant_id=str(plan["tenant_id"]),
        payload=dict(plan_id=str(plan["plan_id"]), command_ids=commands, created_by=created_by),
        idempotency_key=f'host-eviction-requested:{plan["plan_id"]}',
    )
    return {**reconcile(conn, plan), "idempotent": False}


_SWEEP_QUARANTINE = SweepQuarantine("host_eviction")


def sweep(conn, *, limit=100):
    """The scheduler's existing sweep finishes plans after worker settlement.

    Each plan is checked in its own savepoint. A plan that no longer validates
    (``PlanConflict``) or fails for any other reason is logged and backed off
    instead of failing first on every tick and taking the sweep down with it.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        plans = cur.execute(
            """SELECT * FROM action_plans WHERE action_type=%s AND status='executing'
            AND plan_id <> ALL(%s::uuid[])
            ORDER BY consumed_at LIMIT %s FOR UPDATE SKIP LOCKED""",
            (ACTION, _SWEEP_QUARANTINE.skipped(), limit),
        ).fetchall()

    def settle(plan):
        approved_canonical(plan, (plan["canonical_args"] or {}).get("host_id"))
        return reconcile(conn, plan)

    settled = 0
    for plan in plans:
        ok, _ = _SWEEP_QUARANTINE.run(conn, str(plan["plan_id"]), lambda p=plan: settle(p))
        settled += ok
    return settled
