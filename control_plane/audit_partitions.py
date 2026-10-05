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

#: Partitioned the same way, but not audit data, so it is kept out of
#: `PARTITIONED_TABLES` — the audit task would otherwise apply the 24-month
#: legal retention below to GPU temperature samples. It has its own task and its
#: own policy; see `telemetry_partition_maintenance_task`.
TELEMETRY_TABLE = "telemetry_samples"

#: Every table whose name these functions will interpolate into DDL.
_MAINTAINED_TABLES = PARTITIONED_TABLES + (TELEMETRY_TABLE,)

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
    if table not in _MAINTAINED_TABLES:
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
    if table not in _MAINTAINED_TABLES:
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

    failures: list[str] = []
    for table in PARTITIONED_TABLES:
        with control_plane_transaction() as conn:
            # One transaction per table: a table whose partition creation fails
            # must not take the others' windows down with it.
            try:
                ensured = ensure_monthly_partitions(conn, table)
            except Exception as exc:
                log.exception("partition maintenance failed for %s", table)
                failures.append(f"{table} create: {exc}")
                continue
        log.debug("%s partitions ensured: %s", table, ensured)

        with control_plane_transaction() as conn:
            # A separate transaction from the creation above, and deliberately
            # after it: if the drop fails, the window is still extended, and the
            # table keeps accepting writes. The reverse order would let a failing
            # drop stop partition creation and take writes down with it.
            try:
                dropped = drop_expired_partitions(conn, table)
            except Exception as exc:
                log.exception("retention drop failed for %s", table)
                failures.append(f"{table} retention: {exc}")
                continue
        if dropped:
            log.info("%s retention dropped: %s", table, dropped)

    # Isolating the tables from each other is right; reporting the run as a
    # success when one of them failed is not. The exceptions above were caught
    # and logged, so the scheduler recorded `succeeded` and the only trace of a
    # stalled window was a log line. Raised after every table has had its turn.
    if failures:
        raise RuntimeError("; ".join(failures))


#: Defaults for the policy row migration 057 seeds. The row is authoritative;
#: these apply only if its payload is missing a key.
TELEMETRY_MONTHS_AHEAD = 2
TELEMETRY_RETENTION_MONTHS = 6
TELEMETRY_TASK_NAME = "telemetry_partition_maintenance"


def telemetry_policy(conn: Any) -> tuple[int, int]:
    """(months_ahead, retention_months) from the task's own `scheduled_tasks` row.

    Migration 057 seeded `{"months_ahead": 2, "retention_months": 6}` as the
    policy. The dispatcher calls tasks with no arguments, so the task reads it
    here rather than restating the numbers in code where they could drift.
    """
    row = conn.execute(
        "SELECT payload FROM scheduled_tasks WHERE task_name = %s",
        (TELEMETRY_TASK_NAME,),
    ).fetchone()
    if row is None:
        payload: Any = {}
    elif isinstance(row, dict):
        payload = row.get("payload") or {}
    else:
        payload = row[0] or {}
    if isinstance(payload, str):
        import json

        payload = json.loads(payload)
    ahead = int(payload.get("months_ahead", TELEMETRY_MONTHS_AHEAD))
    keep = int(payload.get("retention_months", TELEMETRY_RETENTION_MONTHS))
    # A retention of zero would drop last month, and a negative one would drop
    # the month being written to. Refuse a nonsense policy rather than act on it.
    if ahead < 1 or keep < 1:
        raise ValueError(
            f"{TELEMETRY_TASK_NAME} policy is invalid: months_ahead={ahead}, "
            f"retention_months={keep} (both must be at least 1)"
        )
    return ahead, keep


def telemetry_partition_maintenance_task() -> None:
    """The handler migration 057 promised and nothing ever registered.

    057 created `telemetry_samples` with partitions for its own month and the
    next two, and said that "after that, the seeded
    'telemetry_partition_maintenance' scheduled task owns partition lifecycle".
    The row was seeded; no function was registered under its name. The
    dispatcher claimed it daily, logged "not found in registry", and recorded
    `failed` — in production it had never once succeeded. Its only test asserted
    that the row *existed*.

    So the window stopped at September 2026. From 1 October every sample would
    have landed in `telemetry_samples_default`, which is worse than unpruned:
    once DEFAULT holds rows inside a month's range, `CREATE TABLE … PARTITION
    OF` for that month is refused outright, and the gap can no longer be closed
    by this task at all. It was caught with DEFAULT still empty.

    Unlike the audit task this raises instead of logging: there is one table,
    so there is nothing to isolate it from, and a swallowed exception is exactly
    how a dead maintainer stays invisible.

    If it ever does fail with "updated partition constraint for default
    partition … would be violated", rows for that month are already in DEFAULT.
    The repair is deliberate, not automatic: create the month as a standalone
    table, move its rows out of DEFAULT, and ATTACH it, in one transaction.
    """
    from control_plane.db import control_plane_transaction

    with control_plane_transaction() as conn:
        months_ahead, retention_months = telemetry_policy(conn)
        ensured = ensure_monthly_partitions(conn, TELEMETRY_TABLE, months_ahead=months_ahead)
    log.debug("%s partitions ensured: %s", TELEMETRY_TABLE, ensured)

    # After creation and in its own transaction, for the reason given in the
    # audit task: a failing drop must never stop the window from advancing.
    with control_plane_transaction() as conn:
        dropped = drop_expired_partitions(
            conn, TELEMETRY_TABLE, retention_months=retention_months
        )
    if dropped:
        log.info("%s retention dropped: %s", TELEMETRY_TABLE, dropped)
