"""Telemetry snapshots retention task and table pruning (Track B B7.6, DA§9.4).

DA§9.4 establishes:
"Do not turn it into an unbounded raw telemetry table in the control-plane
database. Choose one of two explicit paths:
- repurpose it as a bounded, downsampled business/SLA history (for example
  one-minute or five-minute summaries tied to a host/job), partitioned and
  retained for a limited interval; or
- deprecate it after metrics and BigQuery projections are implemented."

This module implements the bounded retention policy for `telemetry_snapshots`,
pruning rows older than the retention window (default 14 days) in bounded
batches to prevent table bloat without starving application transactions.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from db import pg_connection

log = logging.getLogger("xcelsior.control_plane.telemetry_retention")

DEFAULT_RETENTION_DAYS = 14
DEFAULT_PRUNE_BATCH_SIZE = 5000


def prune_telemetry_snapshots(
    conn: Any,
    *,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    batch_limit: int = DEFAULT_PRUNE_BATCH_SIZE,
) -> int:
    """Delete rows from telemetry_snapshots older than `retention_days`.

    Executes in bounded batches to avoid holding long locks or exhausting WAL.
    Returns the total count of deleted rows.
    """
    total_deleted = 0
    days = max(1, int(retention_days))
    limit = max(1, int(batch_limit))

    while True:
        cur = conn.execute(
            """
            DELETE FROM telemetry_snapshots
            WHERE id IN (
                SELECT id FROM telemetry_snapshots
                WHERE recorded_at < clock_timestamp() - make_interval(days => %s)
                ORDER BY recorded_at ASC
                LIMIT %s
            )
            RETURNING id
            """,
            (days, limit),
        )
        deleted = len(cur.fetchall())
        total_deleted += deleted
        if deleted < limit:
            break

    if total_deleted > 0:
        log.info(
            "telemetry_snapshots retention pruned %d row(s) older than %d day(s)",
            total_deleted,
            days,
        )
    return total_deleted


def telemetry_retention_task() -> dict[str, Any]:
    """Scheduled task entrypoint for periodic telemetry retention enforcement."""
    retention_days = int(
        os.environ.get("XCELSIOR_TELEMETRY_SNAPSHOTS_RETENTION_DAYS", DEFAULT_RETENTION_DAYS)
    )
    batch_limit = int(
        os.environ.get("XCELSIOR_TELEMETRY_RETENTION_BATCH_LIMIT", DEFAULT_PRUNE_BATCH_SIZE)
    )

    with pg_connection() as conn:
        deleted = prune_telemetry_snapshots(
            conn, retention_days=retention_days, batch_limit=batch_limit
        )
        conn.commit()

    return {
        "ok": True,
        "pruned_rows": deleted,
        "retention_days": retention_days,
    }
