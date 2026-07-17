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

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, cast

from psycopg import Connection
from psycopg.types.json import Jsonb

from control_plane.scheduler.filters import FILTER_POLICY_VERSION

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


class HostNotEligible(ReservationConflict):
    """Host administrative/availability state changed since scoring."""

    code = "host_not_eligible"


class InventoryChanged(ReservationConflict):
    """Host GPU inventory generation moved; the score is stale."""

    code = "inventory_changed"


class CapacityConflict(ReservationConflict):
    """Not enough unallocated devices under lock (a rival won the race)."""

    code = "capacity_conflict"


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


def canonical_spec_hash(spec: dict[str, Any]) -> str:
    """sha256 over canonical JSON — stable across replicas and reruns."""
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


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
               spec, spec_hash, schedule_claim_token, schedule_claim_expires_at
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

    # ── Step 4-5: lock host, verify state + inventory generation ────
    host = conn.execute(
        """
        SELECT status, administrative_state, inventory_generation
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
        raise HostNotEligible(
            f"host {host_id} is {admin_state}", administrative_state=admin_state
        )
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
        raise CapacityConflict(
            f"host {host_id} has no registered GPU devices", host_id=host_id
        )
    allocated = {
        str(_one(row, "gpu_device_id", 0))
        for row in conn.execute(
            """
            SELECT a.gpu_device_id
              FROM gpu_device_allocations a
              JOIN host_gpu_devices d ON d.gpu_device_id = a.gpu_device_id
             WHERE d.host_id = %s
               AND a.status = 'active'
            """,
            (host_id,),
        ).fetchall()
    }
    free: list[tuple[str, str]] = []
    for row in devices:
        device_id = str(_one(row, "gpu_device_id", 0))
        health = _one(row, "health", 3)
        if health not in _DEVICE_HEALTH_SCHEDULABLE:
            continue
        if device_id in allocated:
            continue
        if requested_vram_mb and int(_one(row, "allocatable_vram_mb", 4)) < requested_vram_mb:
            continue
        free.append((device_id, str(_one(row, "gpu_uuid", 1))))
    if len(free) < num_gpus:
        # Multi-GPU is all-or-nothing (§8.2): nothing is inserted.
        raise CapacityConflict(
            f"host {host_id} has {len(free)} free schedulable GPUs, "
            f"need {num_gpus}",
            free=len(free),
            requested=num_gpus,
        )
    selected = free[:num_gpus]

    # ── Step 8: fencing token ────────────────────────────────────────
    fencing_token = int(
        _require(conn.execute("SELECT nextval('placement_fencing_token_seq')").fetchone())[0]
    )

    # ── Step 7/9: attempt row (unique active-attempt index backstops) ─
    payload = _one(job, "payload", 5) or {}
    spec = _one(job, "spec", 6) or payload
    spec_hash = _one(job, "spec_hash", 7) or canonical_spec_hash(dict(spec))
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
    attempt_id_row = (
        conn.execute(
            """
            INSERT INTO job_attempts
                (job_id, attempt_number, status, host_id, fencing_token,
                 job_generation, spec_hash, policy_version, placement_score,
                 placement_explanation, created_by, trace_id)
            VALUES (%s, %s, 'reserved', %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING attempt_id
            """,
            (
                job_id, attempt_number, host_id, fencing_token, job_generation,
                spec_hash, FILTER_POLICY_VERSION, placement_score,
                Jsonb(placement_explanation) if placement_explanation else None,
                f"scheduler:{replica_id}", trace_id,
            ),
        ).fetchone()
    )
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

    # ── Step 11: lease offer bound to attempt/host/fence ─────────────
    lease_id_row = (
        conn.execute(
            """
            INSERT INTO placement_leases
                (job_id, attempt_id, host_id, fencing_token, status,
                 claim_deadline, claim_ttl_sec, renewal_ttl_sec)
            VALUES (%s, %s, %s, %s, 'offered',
                    clock_timestamp() + make_interval(secs => %s), %s, %s)
            RETURNING lease_id
            """,
            (
                job_id, attempt_id, host_id, fencing_token,
                lease_claim_ttl_sec, lease_claim_ttl_sec, lease_renewal_ttl_sec,
            ),
        ).fetchone()
    )
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
    command_id_row = (
        conn.execute(
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
                host_id, Jsonb(command_args), f"scheduler:{replica_id}",
                _START_COMMAND_EXPIRY_SEC, job_id, attempt_id, fencing_token,
                spec_hash, f"start:{attempt_id}", trace_id,
            ),
        ).fetchone()
    )
    command_id = str(_require(command_id_row)[0])

    # ── Step 13: job projection update + claim consumption ───────────
    updated = conn.execute(
        """
        UPDATE jobs
           SET phase = 'scheduled',
               status = 'assigned',
               host_id = %s,
               active_attempt_id = %s,
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
        (host_id, attempt_id, spec_hash, job_id, claim_token),
    )
    if updated.rowcount != 1:  # pragma: no cover - job row is locked by us
        raise ClaimLost(
            f"claim token for {job_id} vanished inside reservation", job_id=job_id
        )

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
        conn.execute(
            """
            INSERT INTO outbox_events
                (aggregate_type, aggregate_id, aggregate_version, event_type,
                 payload, destination_class, idempotency_key)
            VALUES ('job', %s, %s, 'job.v1.placement_reserved', %s, %s, %s)
            """,
            (
                job_id, job_generation, Jsonb(event_payload), destination,
                f"placement:{attempt_id}",
            ),
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
