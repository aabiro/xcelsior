"""Monthly partition maintenance for `audit_events_v2` (Track B B4.1).

Partitions are created **ahead of time** by a scheduled task so a write never
has to create one inline in a request handler (companion §4.5). The DEFAULT
partition is only a safety net; keeping the window full means it stays empty and
the partition-lag alert stays quiet.

Mirrors Track A's telemetry partition pattern; idempotent (`CREATE TABLE IF NOT
EXISTS … PARTITION OF`), so a task that runs every day is a no-op once the window
is full.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any

log = logging.getLogger("xcelsior")


def _month_bounds(start: _dt.date, offset: int) -> tuple[str, str, str]:
    """(suffix, from_iso, to_iso) for the month `offset` months after start."""
    year = start.year + (start.month - 1 + offset) // 12
    month = (start.month - 1 + offset) % 12 + 1
    frm = _dt.date(year, month, 1)
    to = _dt.date(year + 1, 1, 1) if month == 12 else _dt.date(year, month + 1, 1)
    return f"{year:04d}{month:02d}", frm.isoformat(), to.isoformat()


#: Every range-partitioned, monthly table whose window this task keeps full.
#:
#: A list rather than a second copy of the loop below. `placement_decisions`
#: arrived needing exactly this, and a partition maintainer duplicated per table
#: is how one of them silently stops advancing while the other looks fine — the
#: DEFAULT partition absorbs the writes and nothing complains until someone
#: tries to prune.
PARTITIONED_TABLES = ("audit_events_v2", "placement_decisions")


def ensure_monthly_partitions(
    conn: Any,
    table: str,
    *,
    months_ahead: int = 3,
    today: _dt.date | None = None,
) -> list[str]:
    """Create this month + the next `months_ahead` monthly partitions if missing.

    Returns the partition suffixes that now exist for the window. Idempotent.
    Takes an open connection so the caller owns the transaction boundary.
    """
    if table not in PARTITIONED_TABLES:
        # The name is interpolated into DDL, so it is never taken from a caller
        # unchecked.
        raise ValueError(f"{table!r} is not a known partitioned table")
    base = (today or _dt.date.today()).replace(day=1)
    ensured: list[str] = []
    for offset in range(months_ahead + 1):
        suffix, frm, to = _month_bounds(base, offset)
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table}_{suffix} "
            f"PARTITION OF {table} FOR VALUES FROM ('{frm}') TO ('{to}')"
        )
        ensured.append(suffix)
    return ensured


def ensure_audit_partitions(conn: Any, *, months_ahead: int = 3, today: _dt.date | None = None) -> list[str]:
    """Back-compatible wrapper for the original single-table entry point."""
    return ensure_monthly_partitions(
        conn, "audit_events_v2", months_ahead=months_ahead, today=today
    )


def audit_partition_maintenance_task() -> None:
    """Durable `scheduled_tasks` entry point — keep every window full."""
    from control_plane.db import control_plane_transaction

    for table in PARTITIONED_TABLES:
        with control_plane_transaction() as conn:
            # One transaction per table: a table whose partition creation fails
            # must not take the others' windows down with it.
            try:
                ensured = ensure_monthly_partitions(conn, table)
            except Exception:
                log.exception("partition maintenance failed for %s", table)
                continue
        log.debug("%s partitions ensured: %s", table, ensured)
