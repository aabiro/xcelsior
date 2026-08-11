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

#: How long append-only audit data is kept, in months.
#:
#: **This number is the retention policy, and it is enforced here or nowhere.**
#: The tables in `PARTITIONED_TABLES` carry an append-only trigger: rows cannot
#: be UPDATEd or DELETEd, so dropping a whole partition is the only mechanism
#: that can ever remove one. Until this existed, partitions were created ahead
#: of time and never dropped — a stated retention period with nothing enforcing
#: it, which is a policy claim rather than a policy.
#:
#: The ruling behind the number, decided 2026-08-11 by Aaryn Biro: this data is
#: **retained under a documented legal basis** rather than pseudonymised at
#: erasure time. Audit records of placement and access decisions are a standard
#: legitimate-interest / legal-obligation retention under GDPR Art. 17(3) and
#: the equivalent carve-outs in other privacy regimes, so a subject's erasure
#: request does not reach them.
#: What that basis requires in exchange is a stated period, disclosure, and
#: enforcement — hence this constant, the privacy-policy line, and the task
#: below. See `docs/audit-retention.md`.
#:
#: Changing it changes what is deleted. It is not a tuning knob.
WORM_RETENTION_MONTHS = 24


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


def ensure_audit_partitions(
    conn: Any, *, months_ahead: int = 3, today: _dt.date | None = None
) -> list[str]:
    """Back-compatible wrapper for the original single-table entry point."""
    return ensure_monthly_partitions(
        conn, "audit_events_v2", months_ahead=months_ahead, today=today
    )


def _partition_is_expired(suffix: str, cutoff: _dt.date) -> bool:
    """True when a `YYYYMM` partition's month ends at or before `cutoff`.

    The comparison is on the month's **end**, not its start: a partition holding
    January still holds rows written on 31 January, so it may only be dropped
    once the whole month is outside the window.
    """
    # Exactly six digits, checked before parsing. Slicing alone is not enough:
    # `"20249"` slices to year 2024 and month 9 and would be dropped as if it
    # were September, which is a partition nobody named being deleted on a
    # guess. A name this function cannot read is a name it does not act on.
    if len(suffix) != 6 or not suffix.isdigit():
        # The DEFAULT partition, or something a human made. Never dropped by an
        # automatic sweep.
        return False
    year, month = int(suffix[:4]), int(suffix[4:6])
    if not 1 <= month <= 12:
        return False
    end = _dt.date(year + 1, 1, 1) if month == 12 else _dt.date(year, month + 1, 1)
    return end <= cutoff


def expired_partitions(
    conn: Any,
    table: str,
    *,
    retention_months: int = WORM_RETENTION_MONTHS,
    today: _dt.date | None = None,
) -> list[str]:
    """The partitions of `table` that fall entirely outside the retention window.

    Read-only, and separate from the drop on purpose: "what would this remove"
    is answerable without removing it, and the drop below is thin enough to
    read in one go because the selection lives here.
    """
    if table not in PARTITIONED_TABLES:
        raise ValueError(f"{table!r} is not a known partitioned table")
    base = (today or _dt.date.today()).replace(day=1)
    # `retention_months` back from the first of this month.
    total = base.year * 12 + (base.month - 1) - int(retention_months)
    cutoff = _dt.date(total // 12, total % 12 + 1, 1)

    rows = conn.execute(
        """
        SELECT c.relname
          FROM pg_inherits i
          JOIN pg_class c      ON c.oid = i.inhrelid
          JOIN pg_class parent ON parent.oid = i.inhparent
         WHERE parent.relname = %s
        """,
        (table,),
    ).fetchall()
    names = [r[0] if not isinstance(r, dict) else r["relname"] for r in rows]

    prefix = f"{table}_"
    return sorted(
        name
        for name in names
        if name.startswith(prefix) and _partition_is_expired(name[len(prefix) :], cutoff)
    )


def drop_expired_partitions(
    conn: Any,
    table: str,
    *,
    retention_months: int = WORM_RETENTION_MONTHS,
    today: _dt.date | None = None,
) -> list[str]:
    """Drop whole partitions older than the retention window. Returns the names.

    This is the **only** way a row leaves one of these tables. The append-only
    trigger rejects DELETE unconditionally, which is what makes the table
    trustworthy and also what makes a retention period unenforceable by any
    other means.

    Dropping a partition is not a per-subject erasure and is not offered as one:
    it removes a month for every tenant at once. That is a consequence of the
    ruling, not a gap in it — the data is retained under a documented basis
    until the period lapses, and then it goes for everybody.
    """
    dropped = expired_partitions(conn, table, retention_months=retention_months, today=today)
    if not dropped:
        return []

    # `DROP TABLE` on a partition takes ACCESS EXCLUSIVE on the parent, and a
    # lock request queues *ahead* of every later request on that table. Waiting
    # behind one long read would stall every audit write for as long as the wait
    # lasts — a retention sweep taking the audit trail offline is a far worse
    # outcome than a month of data surviving until tomorrow's run.
    #
    # Same reasoning as `migrations/lock_safe.py` rule 5, and the same trade:
    # give up rather than queue. This task runs daily and the drop is
    # idempotent, so a timed-out sweep simply retries tomorrow.
    conn.execute("SET LOCAL lock_timeout = '5s'")

    for name in dropped:
        # The names came from `pg_inherits` for this table and matched a
        # `YYYYMM` suffix, so they are not caller input; they are still checked
        # rather than trusted, because this is DDL.
        if not name.startswith(f"{table}_") or not name[len(table) + 1 :].isdigit():
            raise ValueError(f"refusing to drop {name!r}")
        conn.execute(f"DROP TABLE IF EXISTS {name}")
        log.info("RETENTION dropped %s (outside %s months)", name, retention_months)
    return dropped


def audit_partition_maintenance_task() -> None:
    """Durable `scheduled_tasks` entry point — keep every window full, prune the tail."""
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

        with control_plane_transaction() as conn:
            # A separate transaction from the creation above, and deliberately
            # after it: if the drop fails, the window is still extended, and the
            # table keeps accepting writes. The reverse order would let a failing
            # drop stop partition creation and take writes down with it.
            try:
                dropped = drop_expired_partitions(conn, table)
            except Exception:
                log.exception("retention drop failed for %s", table)
                continue
        if dropped:
            log.info("%s retention dropped: %s", table, dropped)
