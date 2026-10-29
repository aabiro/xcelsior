"""Consistent queue/fleet snapshot for the shadow scheduler (Phase 3).

One REPEATABLE READ transaction reads queued jobs, the host fleet, and
the active-job capacity picture as of a single MVCC snapshot, then maps
them into the dict shapes the pure Stage C/D modules
(:mod:`control_plane.scheduler.filters` / ``scoring``) consume.

Runtime writers do not yet maintain the 054 projection columns
(``phase`` / ``queued_at`` / ``administrative_state`` were backfilled
once by the migration; ``DatabaseOps.upsert_job`` still writes only the
legacy columns + payload). Until A2.5 routes writers through the new
core, the *legacy* columns are the operational truth — so this snapshot
reads legacy truth and computes the new pipeline's projections itself:

- schedulable job   = ``status = 'queued'``
- claim order       = ``priority DESC, submitted_at ASC`` — the legacy
  scheduler's exact queue order. ``effective_priority`` / ``queued_at``
  cannot be used yet: 054 declared them ``NOT NULL DEFAULT 0`` / NULL and
  only the one-time backfill populated them, so for every job created
  since, ``effective_priority`` is 0 regardless of ``priority``. The
  §10.2 claim order takes over at the A2.5 cutover, when writers
  maintain the projection.
- administrative_state = column when set, else projected from the
  payload's legacy ``admitted`` flag
- free GPU capacity = host ``gpu_count`` minus GPUs of active jobs
  (mirrors ``scheduler._gpus_active_on_host``)
- staleness         = payload ``last_seen`` older than the configured
  freshness timeout (measured against DB time, not the Python clock)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from psycopg import Connection

# Mirrors scheduler._ACTIVE_GPU_STATUSES — job statuses that consume GPUs.
ACTIVE_GPU_STATUSES = ("running", "starting", "assigned", "leased")


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
        "SELECT clock_timestamp() AS now_ts, "
        "EXTRACT(EPOCH FROM clock_timestamp()) AS now_epoch"
    ).fetchone()
    if now_row is None:  # pragma: no cover - SELECT always returns a row
        raise RuntimeError("clock_timestamp returned no row")
    taken_at = _get(now_row, "now_ts", 0)
    now_epoch = float(_get(now_row, "now_epoch", 1))

    # ── Queued jobs, in legacy queue order (see module docstring) ─────
    job_rows = conn.execute(
        """
        SELECT job_id, status, host_id, priority, submitted_at, payload
          FROM jobs
         WHERE status = 'queued'
         ORDER BY COALESCE(priority, 0) DESC,
                  submitted_at ASC NULLS LAST
        """
    ).fetchall()
    jobs: list[dict[str, Any]] = []
    for row in job_rows:
        job = _decode(_get(row, "payload", 5))
        job["job_id"] = str(_get(row, "job_id", 0))
        job["status"] = _get(row, "status", 1) or job.get("status")
        job["host_id"] = _get(row, "host_id", 2)
        job.setdefault("priority", _get(row, "priority", 3))
        jobs.append(job)

    # ── Active-job GPU usage per host (capacity truth) ────────────────
    usage_rows = conn.execute(
        """
        SELECT host_id, payload
          FROM jobs
         WHERE status = ANY(%s)
           AND host_id IS NOT NULL
        """,
        (list(ACTIVE_GPU_STATUSES),),
    ).fetchall()
    gpus_used: dict[str, int] = {}
    for row in usage_rows:
        host_id = str(_get(row, "host_id", 0))
        payload = _decode(_get(row, "payload", 1))
        try:
            needed = max(1, int(payload.get("num_gpus", 1) or 1))
        except (TypeError, ValueError):
            needed = 1
        gpus_used[host_id] = gpus_used.get(host_id, 0) + needed

    # ── Host fleet: payload merged with control-plane columns ─────────
    host_rows = conn.execute(
        """
        SELECT host_id, status, payload,
               administrative_state, availability_state, inventory_generation
          FROM hosts
        """
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
            # Projection of the legacy admission flag (054 backfill rule).
            admin_state = "admitted" if host.get("admitted") else "pending"
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
