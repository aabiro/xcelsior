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


def ensure_audit_partitions(conn: Any, *, months_ahead: int = 3, today: _dt.date | None = None) -> list[str]:
    """Create this month + the next `months_ahead` monthly partitions if missing.

    Returns the partition suffixes that now exist for the window. Idempotent.
    Takes an open connection so the caller owns the transaction boundary.
    """
    base = (today or _dt.date.today()).replace(day=1)
    ensured: list[str] = []
    for offset in range(months_ahead + 1):
        suffix, frm, to = _month_bounds(base, offset)
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS audit_events_v2_{suffix} "
            f"PARTITION OF audit_events_v2 FOR VALUES FROM ('{frm}') TO ('{to}')"
        )
        ensured.append(suffix)
    return ensured


def audit_partition_maintenance_task() -> None:
    """Durable `scheduled_tasks` entry point — keep the partition window full."""
    from control_plane.db import control_plane_transaction

    with control_plane_transaction() as conn:
        ensured = ensure_audit_partitions(conn)
    log.debug("audit_events_v2 partitions ensured: %s", ensured)
