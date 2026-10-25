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
    "preparing",
    "running",
    "paused",
    "exited",
    "removing",
    "unmanaged",
    "unknown",
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
    if gpu_inventory is not None:
        from control_plane.physical_inventory import validate_inventory

        validate_inventory(gpu_inventory)
    host = conn.execute(
        "SELECT observation_session_id FROM hosts WHERE host_id=%s FOR UPDATE", (host_id,)
    ).fetchone()
    if host is None:
        raise ValueError("observation host is not registered")
    current_session = _get(host, "observation_session_id", 0)
    previous = conn.execute(
        "SELECT highest_generation,retired_at FROM host_observation_sessions "
        "WHERE host_id=%s AND session_id=%s",
        (host_id, session_id),
    ).fetchone()
    # Snapshot retention cannot erase a session's retirement or accepted
    # generation. A known non-current session is diagnostic evidence only.
    stale = previous is not None and (
        current_session != session_id
        or _get(previous, "retired_at", 1) is not None
        or observation_generation <= _get(previous, "highest_generation", 0)
    )
    if previous is None:
        # A writer that predates the registry (an older runtime still serving
        # during a handoff) can record a current session without a registry
        # row. Register it from its retained snapshots instead of refusing
        # every later report, so its generation fence still holds.
        conn.execute(
            "UPDATE host_observation_sessions SET retired_at=clock_timestamp() "
            "WHERE host_id=%s AND retired_at IS NULL AND session_id IS DISTINCT FROM %s",
            (host_id, session_id if current_session == session_id else None),
        )
        if current_session is not None and current_session != session_id:
            # Record the session the previous writer left behind even when its
            # snapshots were already pruned. A later replay stays retired.
            conn.execute(
                """INSERT INTO host_observation_sessions
                   (host_id,session_id,highest_generation,retired_at)
                   VALUES (%s,%s,COALESCE((
                       SELECT max(inventory_generation) FROM host_observations
                        WHERE host_id=%s AND session_id=%s),-1),clock_timestamp())
                   ON CONFLICT(host_id,session_id) DO NOTHING""",
                (host_id, current_session, host_id, current_session),
            )
        previous = conn.execute(
            """INSERT INTO host_observation_sessions(host_id,session_id,highest_generation)
               VALUES (%s,%s,COALESCE((
                   SELECT max(inventory_generation) FROM host_observations
                    WHERE host_id=%s AND session_id=%s),-1))
               ON CONFLICT(host_id,session_id) DO UPDATE
                  SET highest_generation=host_observation_sessions.highest_generation
               RETURNING highest_generation,retired_at""",
            (host_id, session_id, host_id, session_id),
        ).fetchone()
        stale = _get(previous, "retired_at", 1) is not None or (
            observation_generation <= _get(previous, "highest_generation", 0)
        )
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
            host_id,
            session_id,
            observation_generation,
            agent_version,
            Jsonb(capabilities or {}),
            Jsonb(conditions or {}),
            Jsonb(gpu_inventory or []),
            len(workloads),
            worker_reported_at,
            worker_reported_at or 0.0,
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
                observation_id,
                host_id,
                session_id,
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

    if stale:
        return IngestResult(
            observation_id=observation_id, duplicate=False, workloads=len(workloads)
        )
    conn.execute(
        "UPDATE host_observation_sessions SET highest_generation=%s,latest_observation_id=%s "
        "WHERE host_id=%s AND session_id=%s AND retired_at IS NULL",
        (observation_generation, observation_id, host_id, session_id),
    )
    # received_at (DB clock) is the freshness signal §12.2. Capability and
    # condition observations confer no operator admission or execution intent.
    conn.execute(
        "UPDATE hosts SET last_observed_at=clock_timestamp(),observation_session_id=%s, "
        "capabilities=COALESCE(%s::jsonb,capabilities),conditions=COALESCE(%s::jsonb,conditions), "
        "availability_state=CASE WHEN %s THEN 'not_ready' "
        "WHEN %s AND status='active' THEN 'ready' ELSE availability_state END "
        "WHERE host_id=%s",
        (
            session_id,
            Jsonb(capabilities) if capabilities is not None else None,
            Jsonb(conditions) if conditions is not None else None,
            gpu_inventory == [] or (conditions or {}).get("gpu_inventory") == "unavailable",
            bool(gpu_inventory),
            host_id,
        ),
    )
    if gpu_inventory is not None:
        from control_plane.physical_inventory import sync_physical_inventory

        sync_physical_inventory(conn, host_id=host_id, inventory=gpu_inventory)
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
    # Use server-owned placement identity, never workload-supplied job IDs.
    # An empty snapshot still wakes jobs whose containers have disappeared.
    conn.execute(
        """
        INSERT INTO reconciliation_queue
            (resource_type, resource_id, reason, requested_by)
        SELECT 'job', j.job_id, 'observation_received', %s
          FROM jobs j
          JOIN job_attempts a ON a.attempt_id = j.active_attempt_id
         WHERE a.host_id = %s
         ORDER BY j.job_id
        ON CONFLICT (resource_type, resource_id) DO UPDATE
           SET due_at = LEAST(reconciliation_queue.due_at, clock_timestamp()),
               reason = EXCLUDED.reason, updated_at = clock_timestamp()
        """,
        (f"agent:{session_id}", host_id),
    )
    return IngestResult(observation_id=observation_id, duplicate=False, workloads=len(workloads))


def latest_observation(conn: Connection, host_id: str) -> dict[str, Any] | None:
    """The newest snapshot header for a host, with its workloads."""
    row = conn.execute(
        """
        SELECT o.observation_id, o.session_id, o.inventory_generation, o.received_at
          FROM host_observations o
          JOIN host_observation_sessions s ON s.host_id=o.host_id AND s.session_id=o.session_id
              AND s.latest_observation_id=o.observation_id
              AND s.highest_generation=o.inventory_generation AND s.retired_at IS NULL
          JOIN hosts h ON h.host_id=o.host_id AND h.observation_session_id=o.session_id
         WHERE o.host_id = %s
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
                str(_get(w, "attempt_id", 1)) if _get(w, "attempt_id", 1) is not None else None
            ),
            "fencing_token": _get(w, "fencing_token", 2),
            "container_name": _get(w, "container_name", 3),
            "state": str(_get(w, "state", 4)),
            "spec_hash": _get(w, "spec_hash", 5),
            "details": _get(w, "details", 6),
        }
        for w in conn.execute(
            """
            SELECT job_id, attempt_id, fencing_token, container_name, state,
                   spec_hash, details
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


def prune_observations(conn: Connection, *, retention_days: int = 3, limit: int = 2000) -> int:
    """Prune diagnostic snapshots; durable session fences deliberately survive."""
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
