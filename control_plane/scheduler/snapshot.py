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
    now_row = conn.execute(
        "SELECT clock_timestamp() AS now_ts, " "EXTRACT(EPOCH FROM clock_timestamp()) AS now_epoch"
    ).fetchone()
    if now_row is None:  # pragma: no cover - SELECT always returns a row
        raise RuntimeError("clock_timestamp returned no row")
    taken_at = _get(now_row, "now_ts", 0)
    now_epoch = float(_get(now_row, "now_epoch", 1))

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

    # ── Active device allocations, independent of compatibility status ──
    usage_rows = conn.execute("""SELECT host_id,count(DISTINCT gpu_device_id) AS allocated_gpus
        FROM gpu_device_allocations WHERE released_at IS NULL GROUP BY host_id""").fetchall()
    gpus_used = {
        str(_get(row, "host_id", 0)): int(_get(row, "allocated_gpus", 1)) for row in usage_rows
    }

    # ── Host fleet: payload merged with control-plane columns ─────────
    host_rows = conn.execute("""
        SELECT host_id, status, payload,
               administrative_state, availability_state, inventory_generation
          FROM hosts
        """).fetchall()
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
        availability = _get(row, "availability_state", 4)
        if availability:
            host["availability_state"] = availability
        host["inventory_generation"] = int(_get(row, "inventory_generation", 5) or 0)

        gpu_count = max(1, int(host.get("gpu_count", 1) or 1))
        host["free_gpu_count"] = max(0, gpu_count - gpus_used.get(host_id, 0))

        last_seen = host.get("last_seen")
        if isinstance(last_seen, (int, float)) and last_seen > 0:
            if now_epoch - float(last_seen) > host_freshness_timeout_sec:
                stale.add(host_id)
        hosts.append(host)

    return SchedulerSnapshot(
        taken_at=taken_at,
        jobs=jobs,
        hosts=hosts,
        stale_host_ids=frozenset(stale),
    )
