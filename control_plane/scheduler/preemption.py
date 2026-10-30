"""Versioned preemption plans; only fenced settlement releases capacity.

No planner or deadline sweep frees an allocation. Device nominations protect
the selected capacity until a new reservation transfers it under a new fence.
The default team policy disables preemption and force escalation.
"""

from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from control_plane.commands import enqueue_current_attempt_command
from control_plane.job_specs import validate_execution_spec
from control_plane.scheduler.fairness import lock_team_policy, require_team_quota, QuotaExceeded
from control_plane.scheduler.filters import (
    SCHEDULABLE_DEVICE_HEALTH,
    evaluate_host,
    gpu_model_matches,
    is_physical_gpu_uuid,
)
from control_plane.scheduler.reservation import CapacityConflict
from control_plane.scheduler.snapshot import summarize_devices
from control_plane.sweep_isolation import SweepQuarantine


def is_preemptible(spec):
    return (
        spec.get("preemptible") is True
        or spec.get("spot") is True
        or spec.get("pricing_mode") == "spot"
    )


POLICY_VERSION = "preemption/v1"


class PreemptionWaiting(CapacityConflict):
    code = "preemption_waiting"


def create_plan(
    conn, *, job_id: str, claim_token: str, host_id: str, replica_id: str, scope_host_ids=None
) -> str | None:
    """Lock jobs before host/devices, then persist plan, nomination and stops.

    NOWAIT on the complete sorted job set avoids taking a partial lock graph
    behind a concurrent lifecycle writer or another planner. The caller retries
    contention using the existing bounded transaction budget.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        ids = [
            r["job_id"]
            for r in cur.execute(
                """SELECT job_id FROM jobs
            WHERE job_id=%s OR (host_id=%s AND active_attempt_id IS NOT NULL)
            ORDER BY job_id LIMIT 257""",
                (job_id, host_id),
            ).fetchall()
        ]
        if len(ids) > 256:
            return None
        locked = cur.execute(
            "SELECT * FROM jobs WHERE job_id=ANY(%s) ORDER BY job_id FOR UPDATE NOWAIT", (ids,)
        ).fetchall()
        jobs = {r["job_id"]: r for r in locked}
        nominee = jobs.get(job_id)
        if not nominee or nominee["phase"] != "pending" or nominee["desired_state"] != "running":
            return None
        if str(nominee["schedule_claim_token"]) != claim_token:
            return None
        if not cur.execute(
            """SELECT 1 FROM jobs WHERE job_id=%s
            AND schedule_claim_expires_at>=statement_timestamp()""",
            (job_id,),
        ).fetchone():
            return None
        spec = validate_execution_spec(
            nominee["spec"], nominee["spec_hash"], require_admission=True
        )
        needed = spec.get("num_gpus") or 1
        if type(needed) is not int or not 1 <= needed <= 1024:
            return None
        if is_preemptible(spec):
            return None
        existing = cur.execute(
            """SELECT plan_id FROM preemption_plans
            WHERE nominated_job_id=%s AND state IN ('stopping','ready')""",
            (job_id,),
        ).fetchone()
        if existing:
            return str(existing["plan_id"])
        try:
            policy = lock_team_policy(conn, job_id=job_id, requested_gpus=needed, check_quota=False)
        except QuotaExceeded:
            return None
        if policy.version != nominee["queue_policy_version"]:
            return None
        if not policy.preemption_enabled:
            return None
        host = cur.execute("SELECT * FROM hosts WHERE host_id=%s FOR UPDATE", (host_id,)).fetchone()
        if (
            not host
            or host["administrative_state"] != "admitted"
            or host["availability_state"] != "ready"
        ):
            return None
        # The scheduler's fresh snapshot already rejected stale observations;
        # recheck the timestamp here as well before issuing disruption.
        if not cur.execute(
            "SELECT 1 FROM hosts WHERE host_id=%s AND last_observed_at>=statement_timestamp()-interval '5 minutes'",
            (host_id,),
        ).fetchone():
            return None
        host_view = {
            **(host["payload"] or {}),
            "host_id": host_id,
            "status": host["status"],
            "administrative_state": host["administrative_state"],
        }
        host_view["availability_state"] = host["availability_state"]
        # The observed capability column, as the snapshot and reservation read
        # it: a purchased tier is only placed (or preempted for) where the
        # worker qualified its isolation.
        host_view["capabilities"] = host["capabilities"] or {}
        host_view.pop("lease_claim_backoff", None)
        devices = cur.execute(
            """SELECT * FROM host_gpu_devices WHERE host_id=%s
            AND retired_at IS NULL ORDER BY gpu_uuid FOR UPDATE""",
            (host_id,),
        ).fetchall()
        # Hypothetical capacity: every schedulable physical device of this host
        # as if it were free. The filters then answer "could this host ever
        # carry the nominee", with the same per-device model/VRAM rule the
        # reservation applies.
        physical = [
            d
            for d in devices
            if is_physical_gpu_uuid(d["gpu_uuid"]) and d["health"] in SCHEDULABLE_DEVICE_HEALTH
        ]
        summarize_devices(
            host_view,
            [
                {
                    "gpu_device_id": str(d["gpu_device_id"]),
                    "model": d["model"],
                    "allocatable_vram_mb": d["allocatable_vram_mb"],
                    "free": True,
                }
                for d in physical
            ],
        )
        if evaluate_host(spec, host_view):
            return None
        from control_plane.scheduler.native_preference import require_preferred_host

        require_preferred_host(
            conn,
            spec=spec,
            host_id=host_id,
            scope_host_ids=scope_host_ids,
            hypothetical_host=host_view,
        )
        held = {
            str(r["gpu_device_id"])
            for r in cur.execute(
                """SELECT gpu_device_id FROM
            preemption_plan_devices d JOIN host_gpu_devices g USING(gpu_device_id) WHERE d.held AND g.host_id=%s""",
                (host_id,),
            ).fetchall()
        }
        allocations = cur.execute(
            """SELECT g.gpu_device_id,g.attempt_id,g.job_id,
                a.fencing_token,a.status,a.spec_hash AS attempt_spec_hash,a.job_generation,l.lease_id,l.status AS lease_status,
                l.expires_at>statement_timestamp() AS lease_fresh
            FROM gpu_device_allocations g JOIN job_attempts a USING(attempt_id)
            LEFT JOIN placement_leases l ON l.attempt_id=a.attempt_id AND l.status='active'
            WHERE g.host_id=%s AND g.released_at IS NULL
            """,
            (host_id,),
        ).fetchall()
        by_device = {str(r["gpu_device_id"]): r for r in allocations}
        requested_vram = int(float(spec.get("vram_needed_gb") or 0) * 1024)
        usable = [
            d
            for d in physical
            if str(d["gpu_device_id"]) not in held
            and d["allocation_mode"] == "exclusive"
            and d["allocatable_vram_mb"] >= requested_vram
            and gpu_model_matches(spec.get("gpu_model"), d["model"])
        ]
        selected = [
            str(d["gpu_device_id"]) for d in usable if str(d["gpu_device_id"]) not in by_device
        ]
        if len(selected) >= needed:
            return None  # Normal reservation can place without disrupting anyone.
        victims: dict[str, dict[str, Any]] = {}
        # The accounting clock lock also serializes operator policy publication.
        # Recheck current class/entitlement; a revision can protect a running team.
        # The originally purchased service tier ranks first: queue bids and aging
        # buy no eviction right, and team policy orders work only within a tier.
        from control_plane.scheduler.fairness import team_key_sql

        key = team_key_sql()
        ranks = {
            r["job_id"]: (r["service_tier"], r["rank"], r["entitlement"])
            for r in cur.execute(
                f"""
            SELECT j.job_id,CASE j.spec->'service_tier_quote'->>'tier'
                    WHEN 'dedicated' THEN 3 WHEN 'premium' THEN 2 ELSE 1 END AS service_tier,
                COALESCE(p.administrative_class,0)*1000
                +GREATEST(0,LEAST(COALESCE(j.priority,0),979)) AS rank,
                COALESCE(p.reservation_entitlement,0) AS entitlement
            FROM jobs j LEFT JOIN LATERAL (SELECT * FROM scheduling_team_policies
                WHERE team_key={key} ORDER BY version DESC LIMIT 1) p ON TRUE
            WHERE j.job_id=ANY(%s)""",
                (ids,),
            ).fetchall()
        }
        nominee_rank = ranks[job_id]
        for d in usable:
            allocation = by_device.get(str(d["gpu_device_id"]))
            if not allocation:
                continue
            victim = jobs.get(allocation["job_id"])
            if not victim or victim["desired_state"] != "running" or victim["phase"] != "running":
                continue
            try:
                victim_spec = validate_execution_spec(
                    victim["spec"], victim["spec_hash"], require_admission=True
                )
            except ValueError:
                continue
            if not is_preemptible(victim_spec):
                continue
            if ranks[victim["job_id"]] > nominee_rank:
                continue
            if (
                str(victim["active_attempt_id"]) != str(allocation["attempt_id"])
                or victim["spec_hash"] != allocation["attempt_spec_hash"]
                or victim["generation"] != allocation["job_generation"]
                or victim["lifecycle_intent"]
                or allocation["status"] != "running"
                or allocation["lease_status"] != "active"
                or not allocation["lease_fresh"]
            ):
                continue
            if cur.execute(
                "SELECT 1 FROM preemption_plan_victims WHERE attempt_id=%s AND NOT released",
                (allocation["attempt_id"],),
            ).fetchone():
                continue
            victims[allocation["job_id"]] = dict(allocation)
        chosen = []
        for victim_id in sorted(
            victims,
            key=lambda jid: (
                *ranks[jid],
                jid,
            ),
        ):
            victim = victims[victim_id]
            chosen.append(victim)
            selected.extend(
                str(d["gpu_device_id"])
                for d in usable
                if (by_device.get(str(d["gpu_device_id"])) or {}).get("attempt_id")
                == victim["attempt_id"]
            )
            if len(selected) >= needed:
                break
        if len(selected) < needed or not chosen:
            return None
        try:
            require_team_quota(
                conn,
                policy,
                requested_gpus=needed,
                replacing_attempts=[str(v["attempt_id"]) for v in chosen],
            )
        except QuotaExceeded:
            return None
        plan = cur.execute(
            """INSERT INTO preemption_plans
            (nominated_job_id,nominated_generation,nominated_spec_hash,host_id,policy_version,
             team_policy_version,grace_deadline,nomination_expires_at,force_after_deadline,reason_code)
            VALUES(%s,%s,%s,%s,%s,%s,statement_timestamp()+make_interval(secs=>%s),
                statement_timestamp()+make_interval(secs=>%s),%s,'higher_priority_admission')
            RETURNING plan_id,grace_deadline""",
            (
                job_id,
                nominee["generation"],
                nominee["spec_hash"],
                host_id,
                POLICY_VERSION,
                policy.version,
                policy.grace_seconds,
                policy.grace_seconds + 900,
                policy.force_after_deadline,
            ),
        ).fetchone()
        plan_id = str(plan["plan_id"])
        for device_id in selected[:needed]:
            cur.execute(
                "INSERT INTO preemption_plan_devices(plan_id,gpu_device_id) VALUES(%s,%s)",
                (plan_id, device_id),
            )
        for victim in chosen:
            command = enqueue_current_attempt_command(
                conn,
                job_id=victim["job_id"],
                command="stop_attempt",
                created_by=f"scheduler:{replica_id}",
                idempotency_key=f'preempt:{plan_id}:{victim["attempt_id"]}',
                expires_in_sec=policy.grace_seconds + 900,
                args=dict(
                    preemption=True,
                    preemption_plan_id=plan_id,
                    grace_deadline=plan["grace_deadline"].isoformat(),
                    allow_force=policy.force_after_deadline,
                    force=False,
                    checkpoint_policy="none",
                    preserve=False,
                ),
            )
            cur.execute(
                """INSERT INTO preemption_plan_victims(plan_id,attempt_id,job_id,
                lease_id,fencing_token,stop_command_id) VALUES(%s,%s,%s,%s,%s,%s)""",
                (
                    plan_id,
                    victim["attempt_id"],
                    victim["job_id"],
                    victim["lease_id"],
                    victim["fencing_token"],
                    command.command_id,
                ),
            )
            cur.execute(
                "UPDATE jobs SET reason_code='preemption_requested',reason_details=%s,version=version+1 WHERE job_id=%s",
                (
                    Jsonb(
                        dict(
                            plan_id=plan_id,
                            grace_deadline=plan["grace_deadline"].isoformat(),
                            checkpoint_policy="none",
                        )
                    ),
                    victim["job_id"],
                ),
            )
        cur.execute(
            "UPDATE jobs SET reason_code='preemption_waiting',reason_details=%s,version=version+1 WHERE job_id=%s",
            (Jsonb(dict(plan_id=plan_id, policy=POLICY_VERSION, host_id=host_id)), job_id),
        )
        from control_plane.outbox import append_event

        append_event(
            conn,
            aggregate_type="job",
            aggregate_id=job_id,
            event_type="job.v1.queue_blocked",
            payload=dict(job_id=job_id, reason="preemption_waiting", plan_id=plan_id),
            idempotency_key=f"preemption:{plan_id}",
        )
        return plan_id


def reconcile_plan(conn, plan_id: str) -> str:
    """Resume a plan after any scheduler restart without inferring release."""
    with conn.cursor(row_factory=dict_row) as cur:
        plan = cur.execute("SELECT * FROM preemption_plans WHERE plan_id=%s", (plan_id,)).fetchone()
        if not plan or plan["state"] not in ("stopping", "ready"):
            return plan["state"] if plan else "missing"
        victims = cur.execute(
            "SELECT * FROM preemption_plan_victims WHERE plan_id=%s ORDER BY job_id", (plan_id,)
        ).fetchall()
        ids = sorted({plan["nominated_job_id"], *(v["job_id"] for v in victims)})
        jobs = {
            r["job_id"]: r
            for r in cur.execute(
                "SELECT * FROM jobs WHERE job_id=ANY(%s) ORDER BY job_id FOR UPDATE NOWAIT", (ids,)
            ).fetchall()
        }
        plan = cur.execute(
            """SELECT *,nomination_expires_at<=statement_timestamp() AS expired,
            grace_deadline<=statement_timestamp() AS grace_elapsed FROM preemption_plans
            WHERE plan_id=%s FOR UPDATE NOWAIT""",
            (plan_id,),
        ).fetchone()
        # The initial read is only a lock-order hint. Another scheduler may
        # have placed/cancelled the plan before these job and plan locks landed.
        if not plan or plan["state"] not in ("stopping", "ready"):
            return plan["state"] if plan else "missing"
        nominee = jobs.get(plan["nominated_job_id"])
        cancelled = (
            not nominee
            or nominee["desired_state"] != "running"
            or nominee["generation"] != plan["nominated_generation"]
            or nominee["spec_hash"] != plan["nominated_spec_hash"]
            or plan["expired"]
        )
        if nominee and nominee["active_attempt_id"]:
            # Only reserve_and_bind may mark a nomination placed. An unrelated
            # replacement cancels this plan instead of inheriting its devices.
            cancelled = True
        released_all = True
        for victim in victims:
            proof = cur.execute(
                """SELECT a.status,l.status AS lease_status,
                NOT EXISTS(SELECT 1 FROM gpu_device_allocations g WHERE
                    g.attempt_id=a.attempt_id AND g.released_at IS NULL) AS allocation_released
                FROM job_attempts a JOIN placement_leases l ON l.lease_id=%s
                WHERE a.attempt_id=%s AND a.fencing_token=%s
                  AND l.attempt_id=a.attempt_id AND l.fencing_token=a.fencing_token""",
                (victim["lease_id"], victim["attempt_id"], victim["fencing_token"]),
            ).fetchone()
            released = bool(
                proof
                and proof["status"]
                in ("succeeded", "failed", "cancelled", "lost", "fenced", "preempted")
                and proof["lease_status"] not in ("active", "offered")
                and proof["allocation_released"]
            )
            if released:
                cur.execute(
                    "UPDATE preemption_plan_victims SET released=TRUE WHERE plan_id=%s AND attempt_id=%s",
                    (plan_id, victim["attempt_id"]),
                )
                continue
            released_all = False
            current = jobs.get(victim["job_id"])
            if (
                not cancelled
                and plan["force_after_deadline"]
                and plan["grace_elapsed"]
                and not victim["force_command_id"]
                and current
                and str(current["active_attempt_id"]) == str(victim["attempt_id"])
                and current["desired_state"] == "running"
            ):
                from control_plane.commands import CommandProtocolError

                try:
                    command = enqueue_current_attempt_command(
                        conn,
                        job_id=victim["job_id"],
                        command="stop_attempt",
                        created_by="scheduler:preemption-deadline",
                        idempotency_key=f'preempt-force:{plan_id}:{victim["attempt_id"]}',
                        args=dict(
                            preemption=True,
                            preemption_plan_id=plan_id,
                            grace_deadline=plan["grace_deadline"].isoformat(),
                            allow_force=True,
                            force=True,
                            checkpoint_policy="none",
                            preserve=False,
                        ),
                    )
                except CommandProtocolError:
                    continue
                cur.execute(
                    "UPDATE preemption_plan_victims SET force_command_id=%s WHERE plan_id=%s AND attempt_id=%s",
                    (command.command_id, plan_id, victim["attempt_id"]),
                )
        state = "cancelled" if cancelled else "ready" if released_all else "stopping"
        cur.execute(
            "UPDATE preemption_plans SET state=%s,updated_at=statement_timestamp() WHERE plan_id=%s",
            (state, plan_id),
        )
        if state == "cancelled":
            cur.execute(
                "UPDATE preemption_plan_devices SET held=FALSE WHERE plan_id=%s", (plan_id,)
            )
            # A cancelled plan must not keep stopping its victims. Undelivered
            # stops are cancelled; one already claimed by the worker is in
            # flight and finishes (its stop still requeues the victim), but it
            # can no longer be redelivered after a lost claim.
            cur.execute(
                """UPDATE agent_commands c
                   SET status = CASE WHEN c.status = 'pending' THEN 'cancelled' ELSE c.status END,
                       expires_at = LEAST(c.expires_at, EXTRACT(EPOCH FROM NOW()))
                  FROM preemption_plan_victims v
                 WHERE v.plan_id = %s
                   AND c.command_id IN (v.stop_command_id, v.force_command_id)
                   AND c.status IN ('pending', 'claimed')""",
                (plan_id,),
            )
            # A victim whose stop never reached the worker keeps running; it
            # must stay eligible for a later plan.
            cur.execute(
                """UPDATE preemption_plan_victims v SET released = TRUE
                 WHERE v.plan_id = %s AND NOT v.released
                   AND NOT EXISTS (SELECT 1 FROM agent_commands c
                                    WHERE c.command_id IN (v.stop_command_id, v.force_command_id)
                                      AND c.attempt_count > 0)""",
                (plan_id,),
            )
        elif state == "ready":
            cur.execute(
                "UPDATE jobs SET next_schedule_at=NULL WHERE job_id=%s", (plan["nominated_job_id"],)
            )
        return state


_PLAN_QUARANTINE = SweepQuarantine("preemption_plan")


def sweep_plans(conn, *, limit=100) -> int:
    """Reconcile live plans, each in its own savepoint.

    A plan whose jobs are locked (NOWAIT) or whose reconciliation raises for
    any other reason rolls back alone and backs off; it can no longer keep the
    rest of the batch, or the scheduler tick behind it, from running.
    """
    rows = conn.execute(
        """SELECT plan_id FROM preemption_plans WHERE state IN ('stopping','ready')
        AND plan_id <> ALL(%s::uuid[]) ORDER BY updated_at LIMIT %s""",
        (_PLAN_QUARANTINE.skipped(), limit),
    ).fetchall()
    reconciled = 0
    for row in rows:
        plan_id = str(row["plan_id"] if isinstance(row, dict) else row[0])
        ok, _ = _PLAN_QUARANTINE.run(conn, plan_id, lambda p=plan_id: reconcile_plan(conn, p))
        reconciled += ok
    return reconciled


def nomination_for_reservation(conn, *, job_id, host_id, generation, spec_hash):
    """Caller holds the job; transfer only after fenced settlement."""
    with conn.cursor(row_factory=dict_row) as cur:
        plan = cur.execute(
            """SELECT *,nomination_expires_at<=statement_timestamp() AS expired
            FROM preemption_plans WHERE nominated_job_id=%s AND state IN ('stopping','ready')
            FOR UPDATE""",
            (job_id,),
        ).fetchone()
        if plan is None:
            return None
        if (
            plan["expired"]
            or plan["nominated_generation"] != generation
            or plan["nominated_spec_hash"] != spec_hash
        ):
            raise CapacityConflict("preemption nomination expired or changed")
        if plan["state"] != "ready" or plan["host_id"] != host_id:
            raise PreemptionWaiting("preemption settlement or nominated host required")
        return str(plan["plan_id"])
