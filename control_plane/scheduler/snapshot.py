"""Consistent native queue and fleet snapshot for scheduling diagnostics.

Immutable admission specs supply workload constraints. Native phase, desired
state and persisted queue keys supply eligibility/order; only unreleased GPU
allocations consume capacity. Host telemetry is observation data, while the
normalized administrative state supplies admission authority.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from psycopg import Connection


def _decode(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, (str, bytes)):
        try:
            decoded = json.loads(payload)
        except (ValueError, TypeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _get(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return row[key]
    return row[index]


@dataclass(frozen=True)
class SchedulerSnapshot:
    """Immutable read of the world at one DB instant."""

    taken_at: Any  # timestamptz from the snapshot transaction's clock
    jobs: list[dict[str, Any]] = field(default_factory=list)  # claim order
    hosts: list[dict[str, Any]] = field(default_factory=list)
    stale_host_ids: frozenset[str] = frozenset()


def take_snapshot(
    conn: Connection,
    *,
    host_freshness_timeout_sec: int = 300,
) -> SchedulerSnapshot:
    """Read jobs + hosts as one consistent picture on the caller's txn.

    Must be the first work done in the transaction: REPEATABLE READ can
    only be set before the transaction's first query.
    """
    conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
    now_row = conn.execute("SELECT clock_timestamp() AS now_ts").fetchone()
    if now_row is None:  # pragma: no cover - SELECT always returns a row
        raise RuntimeError("clock_timestamp returned no row")
    taken_at = _get(now_row, "now_ts", 0)

    # ── Admitted jobs, in the same persisted order as the claimer ─────
    job_rows = conn.execute("""
        SELECT job_id, status, host_id, priority, submitted_at, spec AS payload
          FROM jobs
         WHERE phase='pending' AND desired_state='running'
           AND spec->>'execution_spec_version'='v1'
         ORDER BY effective_priority DESC, fair_share_finish ASC,
                  queued_at ASC NULLS LAST, job_id ASC
        """).fetchall()
    jobs: list[dict[str, Any]] = []
    for row in job_rows:
        job = _decode(_get(row, "payload", 5))
        job["job_id"] = str(_get(row, "job_id", 0))
        job["status"] = _get(row, "status", 1) or job.get("status")
        job["host_id"] = _get(row, "host_id", 2)
        job.setdefault("priority", _get(row, "priority", 3))
        jobs.append(job)

    fleet = read_fleet(conn, host_freshness_timeout_sec=host_freshness_timeout_sec)
    return SchedulerSnapshot(
        taken_at=taken_at, jobs=jobs, hosts=fleet.hosts, stale_host_ids=fleet.stale_host_ids
    )


# A host whose offered leases keep timing out unclaimed cannot start work
# (wrong agent build, broken claim path, a GPU it cannot bind). After this many
# consecutive claim timeouts within the window it receives no new offers until
# the oldest timeout ages out, which then admits one probe placement.
LEASE_CLAIM_BACKOFF_THRESHOLD = 3
LEASE_CLAIM_BACKOFF_WINDOW_SEC = 900


def summarize_devices(host: dict[str, Any], devices: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive the scheduling capacity view of a host from its device rows.

    ``devices`` are the host's schedulable physical GPUs, each with a ``free``
    flag (no active allocation, no preemption hold). Count, model and VRAM come
    from these rows, never from the heartbeat payload, which describes GPU0
    and a ``gpu_count`` nothing in production writes.
    """
    free = [d for d in devices if d.get("free")]
    models = sorted({str(d.get("model") or "") for d in devices})
    host["gpu_devices"] = devices
    host["gpu_count"] = len(devices)
    host["free_gpu_count"] = len(free)
    if models:
        host["gpu_model"] = models[0] if len(models) == 1 else ", ".join(models)
    host["free_vram_gb"] = max(
        (int(d.get("allocatable_vram_mb") or 0) / 1024 for d in free), default=0.0
    )
    return host


def read_fleet(
    conn: Connection,
    *,
    host_freshness_timeout_sec: int = 300,
    claim_backoff_threshold: int = LEASE_CLAIM_BACKOFF_THRESHOLD,
    claim_backoff_window_sec: int = LEASE_CLAIM_BACKOFF_WINDOW_SEC,
) -> SchedulerSnapshot:
    """Read current native capacity within an existing transaction snapshot."""
    from control_plane.scheduler.filters import (
        PHYSICAL_GPU_UUID_PATTERN,
        SCHEDULABLE_DEVICE_HEALTH,
    )

    now = conn.execute("SELECT statement_timestamp() AS now_ts").fetchone()
    if now is None:
        raise RuntimeError("database time unavailable")
    taken_at = _get(now, "now_ts", 0)
    # ── Active device allocations, independent of compatibility status ──
    usage_rows = conn.execute(
        """SELECT a.host_id,
        count(DISTINCT a.gpu_device_id) FILTER (WHERE j.spec->>'pricing_mode'='spot') AS spot_gpus
        FROM gpu_device_allocations a JOIN job_attempts t ON t.attempt_id=a.attempt_id
        JOIN jobs j ON j.job_id=t.job_id WHERE a.released_at IS NULL GROUP BY a.host_id"""
    ).fetchall()
    spot_gpus_used = {
        str(_get(row, "host_id", 0)): int(_get(row, "spot_gpus", 1)) for row in usage_rows
    }

    # ── Physical devices: the only capacity a reservation can bind ──
    device_rows = conn.execute(
        """SELECT d.host_id, d.gpu_device_id, d.model, d.allocatable_vram_mb,
                  EXISTS (SELECT 1 FROM gpu_device_allocations a
                           WHERE a.gpu_device_id=d.gpu_device_id
                             AND a.released_at IS NULL) AS allocated,
                  (SELECT n.plan_id FROM preemption_plan_devices n
                    WHERE n.gpu_device_id=d.gpu_device_id AND n.held
                    ORDER BY n.plan_id LIMIT 1) AS held_by_plan
             FROM host_gpu_devices d
            WHERE d.retired_at IS NULL
              AND d.health = ANY(%s)
              AND d.gpu_uuid ~ %s
            ORDER BY d.host_id, d.gpu_uuid""",
        (list(SCHEDULABLE_DEVICE_HEALTH), PHYSICAL_GPU_UUID_PATTERN),
    ).fetchall()
    devices_by_host: dict[str, list[dict[str, Any]]] = {}
    for row in device_rows:
        held_by = _get(row, "held_by_plan", 5)
        allocated = bool(_get(row, "allocated", 4))
        devices_by_host.setdefault(str(_get(row, "host_id", 0)), []).append(
            {
                "gpu_device_id": str(_get(row, "gpu_device_id", 1)),
                "model": str(_get(row, "model", 2) or ""),
                "allocatable_vram_mb": int(_get(row, "allocatable_vram_mb", 3) or 0),
                "allocated": allocated,
                "held_by_plan": str(held_by) if held_by is not None else None,
                "free": not allocated and held_by is None,
            }
        )

    # ── Hosts whose offered leases keep expiring unclaimed ──
    backoff_rows = conn.execute(
        """SELECT a.host_id, count(*) AS timeouts, max(a.ended_at) AS last_timeout_at
             FROM job_attempts a
            WHERE a.failure_code='lease_claim_timeout' AND a.host_id IS NOT NULL
              AND a.ended_at > statement_timestamp() - make_interval(secs => %s)
              AND NOT EXISTS (SELECT 1 FROM job_attempts b
                               WHERE b.host_id=a.host_id AND b.lease_claimed_at > a.ended_at)
            GROUP BY a.host_id HAVING count(*) >= %s""",
        (claim_backoff_window_sec, claim_backoff_threshold),
    ).fetchall()
    backoff = {
        str(_get(row, "host_id", 0)): {
            "consecutive_claim_timeouts": int(_get(row, "timeouts", 1)),
            "last_timeout_at": str(_get(row, "last_timeout_at", 2)),
        }
        for row in backoff_rows
    }

    # ── Host fleet: payload merged with control-plane columns ─────────
    # Freshness is the DB-clock receipt time of the host's last accepted
    # observation, never the payload's last_seen (an ICMP ping refreshes that).
    host_rows = conn.execute(
        """
        SELECT host_id, status, payload,
               administrative_state, availability_state, inventory_generation,
               region, country, province, capabilities,
               (last_observed_at IS NULL
                OR last_observed_at < statement_timestamp()
                                      - make_interval(secs => %s)) AS observation_stale
          FROM hosts
        """,
        (host_freshness_timeout_sec,),
    ).fetchall()
    hosts: list[dict[str, Any]] = []
    stale: set[str] = set()
    for row in host_rows:
        host = _decode(_get(row, "payload", 2))
        host_id = str(_get(row, "host_id", 0))
        host["host_id"] = host_id
        host["status"] = _get(row, "status", 1) or host.get("status")

        admin_state = _get(row, "administrative_state", 3)
        if not admin_state:
            admin_state = "pending"
        host["administrative_state"] = admin_state
        host["availability_state"] = _get(row, "availability_state", 4) or "unknown"
        host["inventory_generation"] = int(_get(row, "inventory_generation", 5) or 0)
        # Placement must see the same normalized facts as admission and API
        # reads. A mutable payload alias cannot move the host or grant egress.
        for field_name, index in (("region", 6), ("country", 7), ("province", 8)):
            host[field_name] = _get(row, field_name, index)
        host["capabilities"] = _decode(_get(row, "capabilities", 9))
        host.pop("lease_claim_backoff", None)
        if host_id in backoff:
            host["lease_claim_backoff"] = backoff[host_id]

        summarize_devices(host, devices_by_host.get(host_id, []))
        host["active_spot_gpu_count"] = spot_gpus_used.get(host_id, 0)

        if bool(_get(row, "observation_stale", 10)):
            stale.add(host_id)
        hosts.append(host)

    return SchedulerSnapshot(
        taken_at=taken_at,
        jobs=[],
        hosts=hosts,
        stale_host_ids=frozenset(stale),
    )


def with_nominated_devices(host: dict[str, Any], plan_id: str) -> dict[str, Any]:
    """The nominee's view of its preemption host: only its held devices are free.

    Reservation binds a nominee exclusively to the devices its plan holds;
    every other device is either another job's or another plan's.
    """
    view = dict(host)
    devices = [
        {
            **d,
            "free": not d.get("allocated") and d.get("held_by_plan") == str(plan_id),
        }
        for d in host.get("gpu_devices") or []
    ]
    return summarize_devices(view, devices)
