"""Worker observation ingest (blueprint §12.2, Phase 6).

The agent periodically reports *everything it actually sees running* —
each ``xcl-*`` container with its attempt/fence labels — as one immutable
snapshot per (host, session, observation generation). API receipt time
(``received_at``, DB clock) is the authoritative freshness signal; the
worker's own clock is diagnostic only.

Ingest is write-only and fast: comparison against desired state happens
in :mod:`control_plane.reconcile`, driven by the PK-coalesced
reconciliation queue this module feeds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

from psycopg import Connection
from psycopg.types.json import Jsonb

log = logging.getLogger("xcelsior.control_plane.observations")

OBSERVED_STATES = (
    "preparing", "running", "paused", "exited", "removing", "unmanaged", "unknown",
)


@dataclass(frozen=True)
class IngestResult:
    observation_id: str | None
    duplicate: bool
    workloads: int


def _get(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return cast("dict[str, Any]", row)[key]
    return row[index]


def _norm_state(state: Any) -> str:
    s = str(state or "").strip().lower()
    return s if s in OBSERVED_STATES else "unknown"


def ingest_observation(
    conn: Connection,
    *,
    host_id: str,
    session_id: str,
    observation_generation: int,
    workloads: list[dict[str, Any]],
    agent_version: str | None = None,
    worker_reported_at: float | None = None,
    capabilities: dict[str, Any] | None = None,
    conditions: dict[str, Any] | None = None,
    gpu_inventory: list[dict[str, Any]] | None = None,
) -> IngestResult:
    """Store one full-state observation inside the caller's transaction.

    Snapshots are immutable: a repeated (host, session, generation)
    triple is a duplicate delivery and ingests nothing (at-least-once
    transport safe). Also enqueues the host for reconciliation and
    freshens ``hosts.last_observed_at``.
    """
    row = conn.execute(
        """
        INSERT INTO host_observations
            (host_id, session_id, inventory_generation, agent_version,
             capabilities, conditions, gpu_inventory,
             observed_workload_count, worker_reported_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                CASE WHEN %s::float8 IS NULL THEN NULL
                     ELSE to_timestamp(%s) END)
        ON CONFLICT (host_id, session_id, inventory_generation) DO NOTHING
        RETURNING observation_id
        """,
        (
            host_id, session_id, observation_generation, agent_version,
            Jsonb(capabilities or {}), Jsonb(conditions or {}),
            Jsonb(gpu_inventory or []), len(workloads),
            worker_reported_at, worker_reported_at or 0.0,
        ),
    ).fetchone()
    if row is None:
        return IngestResult(observation_id=None, duplicate=True, workloads=0)
    observation_id = str(_get(row, "observation_id", 0))

    for w in workloads:
        conn.execute(
            """
            INSERT INTO observed_workloads
                (observation_id, host_id, session_id, job_id, attempt_id,
                 fencing_token, container_id, container_name, spec_hash,
                 state, details)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                observation_id, host_id, session_id,
                w.get("job_id") or None,
                w.get("attempt_id") or None,
                int(w["fencing_token"]) if w.get("fencing_token") else None,
                w.get("container_id") or None,
                w.get("container_name") or None,
                w.get("spec_hash") or None,
                _norm_state(w.get("state")),
                Jsonb(w.get("details") or {}),
            ),
        )

    # received_at (DB clock) is the freshness signal §12.2.
    conn.execute(
        "UPDATE hosts SET last_observed_at = clock_timestamp() WHERE host_id = %s",
        (host_id,),
    )
    # PK-coalesced: one pending reconcile per host, soonest due_at wins.
    conn.execute(
        """
        INSERT INTO reconciliation_queue (resource_type, resource_id, reason,
                                          requested_by)
        VALUES ('host', %s, 'observation_received', %s)
        ON CONFLICT (resource_type, resource_id) DO UPDATE
           SET due_at = LEAST(reconciliation_queue.due_at, clock_timestamp()),
               reason = EXCLUDED.reason,
               updated_at = clock_timestamp()
        """,
        (host_id, f"agent:{session_id}"),
    )
    return IngestResult(
        observation_id=observation_id, duplicate=False, workloads=len(workloads)
    )


def latest_observation(conn: Connection, host_id: str) -> dict[str, Any] | None:
    """The newest snapshot header for a host, with its workloads."""
    row = conn.execute(
        """
        SELECT observation_id, session_id, inventory_generation, received_at
          FROM host_observations
         WHERE host_id = %s
         ORDER BY received_at DESC
         LIMIT 1
        """,
        (host_id,),
    ).fetchone()
    if row is None:
        return None
    observation_id = str(_get(row, "observation_id", 0))
    workloads = [
        {
            "job_id": _get(w, "job_id", 0),
            "attempt_id": (
                str(_get(w, "attempt_id", 1))
                if _get(w, "attempt_id", 1) is not None else None
            ),
            "fencing_token": _get(w, "fencing_token", 2),
            "container_name": _get(w, "container_name", 3),
            "state": str(_get(w, "state", 4)),
        }
        for w in conn.execute(
            """
            SELECT job_id, attempt_id, fencing_token, container_name, state
              FROM observed_workloads
             WHERE observation_id = %s
            """,
            (observation_id,),
        ).fetchall()
    ]
    return {
        "observation_id": observation_id,
        "session_id": str(_get(row, "session_id", 1)),
        "observation_generation": int(_get(row, "inventory_generation", 2)),
        "received_at": _get(row, "received_at", 3),
        "workloads": workloads,
    }


def prune_observations(
    conn: Connection, *, retention_days: int = 3, limit: int = 2000
) -> int:
    """Observations are high-volume diagnostics; keep a short window."""
    result = conn.execute(
        """
        DELETE FROM host_observations
         WHERE observation_id IN (
             SELECT observation_id FROM host_observations
              WHERE received_at < clock_timestamp()
                                  - make_interval(days => %(days)s)
              LIMIT %(limit)s
         )
        """,
        {"days": retention_days, "limit": limit},
    )
    return result.rowcount
