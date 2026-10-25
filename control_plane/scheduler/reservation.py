"""Stage E — the atomic reservation transaction (blueprint §10.5).

One PostgreSQL transaction takes a scored candidate from "claimed idea"
to "bound placement": it locks the job and the host's concrete GPU
devices, revalidates every fact the scoring decision depended on,
allocates a monotonic fencing token, and inserts the attempt, device
allocations, lease offer, durable start command, and outbox intents —
or raises and leaves nothing behind.

Designed to run inside ``control_plane.db.run_transaction``: domain
conflicts raise :class:`ReservationConflict` subclasses (never retried —
the caller moves to the next candidate or releases the claim with a
reason), while transient serialization/deadlock errors bubble to the
retry wrapper.

Lock order is fixed and canonical (§2.4): job row → host row → that
host's device rows in ``(host_id, gpu_uuid)`` order. Every reservation
takes locks in this exact order, which makes lock cycles impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from psycopg import Connection
from psycopg.types.json import Jsonb

from control_plane.scheduler.filters import FILTER_POLICY_VERSION
from control_plane.job_specs import validate_execution_spec

# v1 agents GC pending commands past expires_at (routes/agent.py). Bound
# start commands must outlive shadow/canary windows until the v2 claim/ACK
# protocol owns delivery, so they get a long explicit expiry instead of
# the legacy 15-minute default.
_START_COMMAND_EXPIRY_SEC = 7 * 24 * 3600

_DEVICE_HEALTH_SCHEDULABLE = ("healthy", "unknown")


class ReservationConflict(Exception):
    """Base: this candidate cannot be reserved; roll back and move on."""

    code = "reservation_conflict"

    def __init__(self, message: str, **details: Any):
        super().__init__(message)
        self.details = details


class ClaimLost(ReservationConflict):
    """Another replica stole the schedule claim (or it was released)."""

    code = "claim_lost"


class JobNotSchedulable(ReservationConflict):
    """Job left the pending/running-desired state since it was claimed."""

    code = "job_not_schedulable"


class JobSpecUnqualified(JobNotSchedulable):
    """Admission evidence cannot safely authorize execution."""

    code = "execution_spec_unqualified"


class HostNotEligible(ReservationConflict):
    """Host administrative/availability state changed since scoring."""

    code = "host_not_eligible"


class InventoryChanged(ReservationConflict):
    """Host GPU inventory generation moved; the score is stale."""

    code = "inventory_changed"


class CapacityConflict(ReservationConflict):
    """Not enough unallocated devices under lock (a rival won the race)."""

    code = "capacity_conflict"


class QueuePolicyChanged(ReservationConflict):
    """A policy revision invalidated the persisted queue decision."""

    code = "queue_policy_changed"


class TeamQuotaConflict(ReservationConflict):
    """The serialized team capacity check refuses a new allocation."""

    code = "team_quota_exceeded"


@dataclass(frozen=True)
class Reservation:
    job_id: str
    attempt_id: str
    attempt_number: int
    fencing_token: int
    lease_id: str
    command_id: str
    host_id: str
    gpu_device_ids: list[str] = field(default_factory=list)
    spec_hash: str = ""


def _one(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return cast("dict[str, Any]", row)[key]
    return row[index]


def _require(row: Any) -> Any:
    """RETURNING / sequence fetches always yield a row; enforce for typing."""
    if row is None:  # pragma: no cover - PostgreSQL contract
        raise RuntimeError("statement expected to return a row returned none")
    return row


def reserve_and_bind(
    conn: Connection,
    *,
    job_id: str,
    claim_token: str,
    replica_id: str,
    host_id: str,
    num_gpus: int = 1,
    requested_vram_mb: int = 0,
    expected_inventory_generation: int | None = None,
    placement_score: int | None = None,
    placement_explanation: dict[str, Any] | None = None,
    lease_claim_ttl_sec: int = 60,
    lease_renewal_ttl_sec: int = 300,
    trace_id: str | None = None,
    expected_spec_hash: str | None = None,
    scope_host_ids: frozenset[str] | None = None,
    host_freshness_timeout_sec: int = 300,
) -> Reservation:
    """Execute §10.5 steps 2–14 inside the caller's open transaction.

    Raises a :class:`ReservationConflict` subclass when any revalidated
    fact no longer holds; the caller must roll back (run_transaction does
    this automatically when the exception propagates).
    """
    if num_gpus < 1:
        raise ValueError("num_gpus must be >= 1")

    # ── Step 2-3: lock job, verify claim + schedulability ────────────
    job = conn.execute(
        """
        SELECT status, phase, desired_state, generation, version, payload,
               spec, spec_hash, schedule_claim_token, schedule_claim_expires_at, owner_id, queue_policy_version
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
    phase = _one(job, "phase", 1)
    desired = _one(job, "desired_state", 2)
    if phase != "pending" or desired != "running":
        raise JobNotSchedulable(
            f"job {job_id} is {phase}/{desired}, not pending/running",
            phase=phase,
            desired_state=desired,
        )

    # Recheck the pin under the job lock; a stale candidate snapshot never
    # authorizes starting a paid marketplace launch on a different host.
    # Only a native admission can authorize placement. A recovered historical
    # execution record cannot substitute for a customer's approved launch.
    spec = _one(job, "spec", 6)
    spec_hash = _one(job, "spec_hash", 7)
    if expected_spec_hash is not None and spec_hash != expected_spec_hash:
        raise JobSpecUnqualified("job execution spec changed after its scheduling claim")
    try:
        spec = validate_execution_spec(spec, spec_hash, require_admission=True)
    except ValueError as exc:
        raise JobSpecUnqualified("job execution spec is unqualified", reason=str(exc)) from exc
    if "execution_spec_version" in spec:
        try:
            num_gpus = max(1, int(spec.get("num_gpus") or 1))
            requested_vram_mb = int(float(spec.get("vram_needed_gb") or 0) * 1024)
        except (TypeError, ValueError, OverflowError) as exc:
            raise JobSpecUnqualified("job resource requirements are unqualified") from exc
    preferred_host_id = spec.get("preferred_host_id")
    if preferred_host_id and str(preferred_host_id) != host_id:
        raise HostNotEligible("job is pinned to another host", host_id=host_id)

    from control_plane.scheduler.preemption import nomination_for_reservation

    nomination = nomination_for_reservation(
        conn,
        job_id=job_id,
        host_id=host_id,
        generation=_one(job, "generation", 3),
        spec_hash=spec_hash,
    )

    from control_plane.scheduler.fairness import lock_team_policy, QuotaExceeded

    try:
        team_policy = lock_team_policy(conn, job_id=job_id, requested_gpus=num_gpus)
    except QuotaExceeded as exc:
        raise TeamQuotaConflict(str(exc)) from exc
    if _one(job, "queue_policy_version", 11) != team_policy.version:
        raise QueuePolicyChanged("team policy changed after the queue claim")

    # ── Step 4-5: lock host, verify state + inventory generation ────
    host = conn.execute(
        """
        SELECT status, administrative_state, inventory_generation, payload,availability_state
          FROM hosts
         WHERE host_id = %s
           FOR UPDATE
        """,
        (host_id,),
    ).fetchone()
    if host is None:
        raise HostNotEligible(f"host {host_id} no longer exists", host_id=host_id)
    admin_state = _one(host, "administrative_state", 1)
    if admin_state != "admitted":
        raise HostNotEligible(f"host {host_id} is {admin_state}", administrative_state=admin_state)
    if _one(host, "availability_state", 4) != "ready":
        raise HostNotEligible("host availability changed before reservation", host_id=host_id)
    inventory_generation = int(_one(host, "inventory_generation", 2))
    if (
        expected_inventory_generation is not None
        and inventory_generation != expected_inventory_generation
    ):
        raise InventoryChanged(
            f"host {host_id} inventory moved "
            f"{expected_inventory_generation} -> {inventory_generation}",
            expected=expected_inventory_generation,
            actual=inventory_generation,
        )

    # ── Step 4b/6: lock ALL host devices in canonical order, then pick.
    # Locking the full (small) device set of one host in gpu_uuid order
    # keeps the lock graph acyclic and lets us recalculate capacity
    # exactly, at the cost of briefly serializing reservations per host —
    # which is precisely the §2.1 correctness guarantee we want.
    devices = conn.execute(
        """
        SELECT d.gpu_device_id, d.gpu_uuid, d.allocation_mode, d.health,
               d.allocatable_vram_mb
          FROM host_gpu_devices d
         WHERE d.host_id = %s
           AND d.retired_at IS NULL
         ORDER BY d.gpu_uuid
           FOR UPDATE
        """,
        (host_id,),
    ).fetchall()
    if not devices:
        raise CapacityConflict(f"host {host_id} has no registered GPU devices", host_id=host_id)
    # Spot capacity is revalidated while holding the host/device locks. Native
    # allocations, including unclaimed reservations, consume the pool; a raw
    # compatibility status cannot release that capacity.
    if spec.get("pricing_mode") == "spot":
        from control_plane.scheduler.filters import FilterContext, spot_capacity
        import json

        policy = _one(host, "payload", 3)
        policy = policy if isinstance(policy, dict) else json.loads(policy or "{}")
        spot_used = conn.execute(
            """SELECT count(DISTINCT a.gpu_device_id) FROM gpu_device_allocations a
            JOIN job_attempts t ON t.attempt_id=a.attempt_id JOIN jobs j ON j.job_id=t.job_id
            WHERE a.host_id=%s AND a.released_at IS NULL AND j.spec->>'pricing_mode'='spot'""",
            (host_id,),
        ).fetchone()[0]
        reason = spot_capacity(
            spec,
            {**policy, "gpu_count": len(devices), "active_spot_gpu_count": spot_used},
            FilterContext(),
        )
        if reason is not None:
            raise HostNotEligible(reason.message, spot_policy_reason=reason.code)

    allocated = {
        str(_one(row, "gpu_device_id", 0))
        for row in conn.execute(
            """
            SELECT a.gpu_device_id
              FROM gpu_device_allocations a
              JOIN host_gpu_devices d ON d.gpu_device_id = a.gpu_device_id
             WHERE d.host_id = %s
               AND a.released_at IS NULL
            """,
            (host_id,),
        ).fetchall()
    }
    held = {
        str(_one(r, "gpu_device_id", 0))
        for r in conn.execute(
            """SELECT n.gpu_device_id FROM preemption_plan_devices n
            JOIN host_gpu_devices d USING(gpu_device_id)
            WHERE d.host_id=%s AND n.held AND (%s::uuid IS NULL OR n.plan_id<>%s::uuid)""",
            (host_id, nomination, nomination),
        ).fetchall()
    }
    nominated = (
        {
            str(_one(r, "gpu_device_id", 0))
            for r in conn.execute(
                "SELECT gpu_device_id FROM preemption_plan_devices WHERE plan_id=%s AND held",
                (nomination,),
            ).fetchall()
        }
        if nomination
        else set()
    )
    free: list[tuple[str, str]] = []
    for row in devices:
        device_id = str(_one(row, "gpu_device_id", 0))
        health = _one(row, "health", 3)
        if health not in _DEVICE_HEALTH_SCHEDULABLE:
            continue
        if (
            device_id in allocated
            or device_id in held
            or (nomination and device_id not in nominated)
        ):
            continue
        if requested_vram_mb and int(_one(row, "allocatable_vram_mb", 4)) < requested_vram_mb:
            continue
        free.append((device_id, str(_one(row, "gpu_uuid", 1))))
    if len(free) < num_gpus:
        # Multi-GPU is all-or-nothing (§8.2): nothing is inserted.
        raise CapacityConflict(
            f"host {host_id} has {len(free)} free schedulable GPUs, " f"need {num_gpus}",
            free=len(free),
            requested=num_gpus,
        )
    selected = free[:num_gpus]
    if nomination:
        conn.execute(
            "UPDATE preemption_plans SET state='placed',updated_at=statement_timestamp() WHERE plan_id=%s",
            (nomination,),
        )
        conn.execute(
            "UPDATE preemption_plan_devices SET held=FALSE WHERE plan_id=%s", (nomination,)
        )

    from control_plane.scheduler.native_preference import require_preferred_host

    preference_evidence = require_preferred_host(
        conn,
        spec=spec,
        host_id=host_id,
        scope_host_ids=scope_host_ids,
        freshness_sec=host_freshness_timeout_sec,
    )

    # ── Step 8: fencing token ────────────────────────────────────────
    fencing_token = int(
        _require(conn.execute("SELECT nextval('placement_fencing_token_seq')").fetchone())[0]
    )

    # ── Step 7/9: attempt row (unique active-attempt index backstops) ─
    from control_plane.attempt_billing import capture_billing_snapshot

    host_payload = {**(_one(host, "payload", 3) or {}), "host_id": host_id}
    try:
        billing_snapshot = capture_billing_snapshot(
            dict(spec), _one(job, "owner_id", 10), host_payload
        )
    except (ValueError, TypeError, OverflowError) as exc:
        raise HostNotEligible("billable placement evidence is invalid", reason=str(exc)) from exc
    explanation = {**(placement_explanation or {})}
    # This field is generated only from the locked server rows, never from
    # a client explanation or from a later host-price observation.
    explanation["billing_snapshot"] = billing_snapshot
    if preference_evidence:
        explanation["placement_preference"] = preference_evidence
    attempt_number = int(
        _require(
            conn.execute(
                "SELECT COALESCE(MAX(attempt_number), 0) + 1 FROM job_attempts "
                "WHERE job_id = %s",
                (job_id,),
            ).fetchone()
        )[0]
    )
    job_generation = int(_one(job, "generation", 3))
    attempt_id_row = conn.execute(
        """
            INSERT INTO job_attempts
                (job_id, attempt_number, status, host_id, fencing_token,
                 job_generation, spec_hash, policy_version, placement_score,
                 placement_explanation, created_by, trace_id)
            VALUES (%s, %s, 'reserved', %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING attempt_id
            """,
        (
            job_id,
            attempt_number,
            host_id,
            fencing_token,
            job_generation,
            spec_hash,
            FILTER_POLICY_VERSION,
            placement_score,
            Jsonb(explanation),
            f"scheduler:{replica_id}",
            trace_id,
        ),
    ).fetchone()
    attempt_id = str(_require(attempt_id_row)[0])

    # ── Step 10: device allocations (partial unique index backstops) ─
    for device_id, _uuid in selected:
        conn.execute(
            """
            INSERT INTO gpu_device_allocations
                (attempt_id, job_id, host_id, gpu_device_id, allocation_mode,
                 requested_vram_mb, requested_shares)
            VALUES (%s, %s, %s, %s, 'exclusive', %s, 1)
            """,
            (attempt_id, job_id, host_id, device_id, requested_vram_mb),
        )

    # ── Step 10b: bind a serverless worker to its fenced attempt (B3.1) ─
    # A serverless_worker job carries its worker id in the payload. Stamping it
    # here — in the same transaction that created the attempt and its
    # allocations — makes serverless capacity attempt-scoped: one worker, one
    # fenced attempt, one allocation set, closed once when the attempt is fenced.
    serverless_worker_id = spec.get("serverless_worker_id")
    if serverless_worker_id:
        # serverless_workers keeps epoch-float timestamps (not timestamptz), so
        # updated_at takes EXTRACT(EPOCH …), matching that table's convention.
        bound_worker = conn.execute(
            """
            UPDATE serverless_workers
               SET attempt_id = %s,
                   host_id = %s,
                   allocated_at = EXTRACT(EPOCH FROM clock_timestamp()),
                   updated_at = EXTRACT(EPOCH FROM clock_timestamp())
             WHERE worker_id = %s AND scheduler_job_id=%s AND endpoint_id=%s
               AND released_at IS NULL AND attempt_id IS NULL AND state IN ('booting','ready','idle')
            RETURNING worker_id
            """,
            (
                attempt_id,
                host_id,
                str(serverless_worker_id),
                job_id,
                spec.get("serverless_endpoint_id"),
            ),
        ).fetchone()
        if bound_worker is None:
            raise CapacityConflict("serverless worker binding is missing or closed")

    # ── Step 11: lease offer bound to attempt/host/fence ─────────────
    lease_id_row = conn.execute(
        """
            INSERT INTO placement_leases
                (job_id, attempt_id, host_id, fencing_token, status,
                 claim_deadline, claim_ttl_sec, renewal_ttl_sec)
            VALUES (%s, %s, %s, %s, 'offered',
                    clock_timestamp() + make_interval(secs => %s), %s, %s)
            RETURNING lease_id
            """,
        (
            job_id,
            attempt_id,
            host_id,
            fencing_token,
            lease_claim_ttl_sec,
            lease_claim_ttl_sec,
            lease_renewal_ttl_sec,
        ),
    ).fetchone()
    lease_id = str(_require(lease_id_row)[0])

    # ── Step 12: durable start command (§11.1 payload shape) ─────────
    command_args = {
        "command_type": "start_attempt",
        "job_id": job_id,
        "attempt_id": attempt_id,
        "lease_id": lease_id,
        "fencing_token": fencing_token,
        "spec_hash": spec_hash,
        "spec": dict(spec),
    }
    command_id_row = conn.execute(
        """
            INSERT INTO agent_commands
                (host_id, command, args, status, created_by, expires_at,
                 job_id, attempt_id, fencing_token, spec_hash,
                 idempotency_key, trace_id)
            VALUES (%s, 'start_attempt', %s, 'pending', %s,
                    EXTRACT(EPOCH FROM NOW()) + %s,
                    %s, %s, %s, %s, %s, %s)
            RETURNING command_id
            """,
        (
            host_id,
            Jsonb(command_args),
            f"scheduler:{replica_id}",
            _START_COMMAND_EXPIRY_SEC,
            job_id,
            attempt_id,
            fencing_token,
            spec_hash,
            f"start:{attempt_id}",
            trace_id,
        ),
    ).fetchone()
    command_id = str(_require(command_id_row)[0])
    conn.execute(
        "UPDATE job_attempts SET command_created_at=to_timestamp(c.created_at) "
        "FROM agent_commands c WHERE job_attempts.attempt_id=%s AND c.command_id=%s",
        (attempt_id, command_id),
    )

    # ── Step 13: job projection update + claim consumption ───────────
    updated = conn.execute(
        """
        UPDATE jobs
           SET phase = 'scheduled',
               status = 'assigned',
               host_id = %s,
               active_attempt_id = %s,
               error_message = '',
               exit_code = NULL,
               payload = COALESCE(payload,'{}'::jsonb)-ARRAY[
                   'http_ports','auto_launch_ports','container_name','container_id',
                   'public_ssh_port','ssh_info','started_at','completed_at']::text[],
               spec = %s,
               spec_hash = %s,
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
        (host_id, attempt_id, Jsonb(spec), spec_hash, job_id, claim_token),
    )
    if updated.rowcount != 1:  # pragma: no cover - job row is locked by us
        raise ClaimLost(f"claim token for {job_id} vanished inside reservation", job_id=job_id)

    from control_plane.scheduler.fairness import spend_service_credit

    spend_service_credit(conn, team_policy, gpu_count=len(selected))

    # ── Step 14: outbox intents (same transaction, §16.1) ────────────
    event_payload = {
        "job_id": job_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "host_id": host_id,
        "fencing_token": fencing_token,
        "gpu_device_ids": [d for d, _ in selected],
        "policy_version": FILTER_POLICY_VERSION,
    }
    for destination in ("default", "agent_wake"):
        from control_plane.outbox import append_event

        append_event(
            conn,
            aggregate_type="job",
            aggregate_id=job_id,
            aggregate_version=job_generation,
            event_type="job.v1.placement_reserved",
            payload=event_payload,
            destination_class=destination,
            idempotency_key=f"placement:{attempt_id}",
            trace_id=trace_id,
        )

    return Reservation(
        job_id=job_id,
        attempt_id=attempt_id,
        attempt_number=attempt_number,
        fencing_token=fencing_token,
        lease_id=lease_id,
        command_id=command_id,
        host_id=host_id,
        gpu_device_ids=[d for d, _ in selected],
        spec_hash=spec_hash,
    )
