"""The stated retention period must be enforced, not asserted.

The WORM/erasure ruling (2026-08-11, Aaryn Biro) is that placement and audit
records are **retained under a documented legal basis** rather than
pseudonymised at erasure time — GDPR Art. 17(3) and the equivalent carve-outs
in other privacy regimes. That basis is not free. It requires a stated period, disclosure, and a mechanism that
actually applies the period. `docs/audit-retention.md` carries the first two.
This file is about the third.

## Why it could not have been enforced before

These tables carry an append-only trigger: `DELETE` is rejected unconditionally.
That is exactly what makes them worth keeping — and it means no ordinary sweep
can implement a retention period on them. Dropping a whole partition is the only
mechanism that can remove a row, and until `drop_expired_partitions` existed,
partitions were **created ahead of time and never dropped**. A retention period
with nothing enforcing it is a claim, not a policy, and the claim was about to
be published in a privacy policy.

The first test below is the load-bearing one: it proves `DELETE` is refused *and*
that the drop removes the same rows. Either half alone proves nothing — a
passing `DELETE` would mean the table is not WORM, and a drop that removed
nothing would mean the retention period is decorative.
"""

from __future__ import annotations

import datetime as dt
import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

try:
    from control_plane.db import control_plane_transaction as pg_transaction

    with pg_transaction() as _c:
        _has = _c.execute("SELECT to_regclass('placement_decisions')").fetchone()[0] is not None
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no control-plane db: {_e}")
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("test database is behind migration 106")

from control_plane.audit_partitions import (  # noqa: E402
    PARTITIONED_TABLES,
    WORM_RETENTION_MONTHS,
    drop_expired_partitions,
    expired_partitions,
)

#: A month far enough in the past that no real partition shares the name, so
#: these tests never drop a partition another test is using.
ANCIENT = "195003"
ANCIENT_START = dt.date(1950, 3, 1)


@pytest.fixture
def ancient_partition():
    """A real partition of `placement_decisions` holding one real row."""
    name = f"placement_decisions_{ANCIENT}"
    tenant = str(uuid.uuid4())
    with pg_transaction() as conn:
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {name} PARTITION OF placement_decisions "
            f"FOR VALUES FROM ('1950-03-01') TO ('1950-04-01')"
        )
        conn.execute(
            """
            INSERT INTO placement_decisions
                   (tenant_id, outcome, host_id, decided_at)
            VALUES (%s, 'placed', 'h-retention-fixture', '1950-03-15')
            """,
            (tenant,),
        )
    yield name, tenant
    with pg_transaction() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {name}")


def _count(conn, tenant: str) -> int:
    return conn.execute(
        "SELECT count(*) FROM placement_decisions WHERE tenant_id = %s", (tenant,)
    ).fetchone()[0]


# ── The load-bearing pair ─────────────────────────────────────────────


def test_a_row_delete_is_refused_and_the_partition_drop_removes_the_same_rows(
    ancient_partition,
):
    """Both halves, in one test, because either alone proves nothing.

    A passing DELETE would mean the table is not append-only and the whole
    retention argument is moot. A drop that removed nothing would mean the
    stated period is decorative. The pair is the claim.
    """
    _, tenant = ancient_partition

    with pg_transaction() as conn:
        assert _count(conn, tenant) == 1, "the fixture row is not there to begin with"

    # Half one: the ordinary mechanism cannot remove it.
    with pytest.raises(Exception) as refused:
        with pg_transaction() as conn:
            conn.execute("DELETE FROM placement_decisions WHERE tenant_id = %s", (tenant,))
    assert "append" in str(refused.value).lower() or "worm" in str(refused.value).lower(), (
        f"DELETE failed, but not because the table is append-only: {refused.value}"
    )

    with pg_transaction() as conn:
        assert _count(conn, tenant) == 1, "the refused DELETE removed rows anyway"

    # Half two: the retention mechanism can.
    with pg_transaction() as conn:
        dropped = drop_expired_partitions(conn, "placement_decisions", today=dt.date(2026, 8, 1))
    assert f"placement_decisions_{ANCIENT}" in dropped, dropped

    with pg_transaction() as conn:
        assert _count(conn, tenant) == 0, (
            "the partition was dropped but the rows are still readable"
        )


# ── The window ────────────────────────────────────────────────────────


def test_a_partition_inside_the_window_is_not_dropped(ancient_partition):
    """The period is a period, not a synonym for "everything old"."""
    name, tenant = ancient_partition
    # Pretend "today" is inside the fixture's own retention window.
    today = dt.date(1950, 6, 1)
    with pg_transaction() as conn:
        assert name not in expired_partitions(conn, "placement_decisions", today=today)
        assert drop_expired_partitions(conn, "placement_decisions", today=today) == []
        assert _count(conn, tenant) == 1


def test_the_boundary_month_expires_and_the_next_one_does_not():
    """A partition is droppable once its **last day** is outside the window.

    Comparing the month's start would drop a partition that still holds rows
    written on its final day — inside the stated period, deleted anyway.
    """
    from control_plane.audit_partitions import _partition_is_expired

    # 24 months back from 2026-08-01 is 2024-08-01.
    cutoff = dt.date(2024, 8, 1)
    assert _partition_is_expired("202407", cutoff), "July 2024 ends 2024-08-01; it is out"
    assert not _partition_is_expired("202408", cutoff), "August 2024 is the first month in"
    assert not _partition_is_expired("202409", cutoff)


def test_the_retention_period_is_the_documented_one():
    """The number is the policy. A silent change is a silent change to deletion."""
    assert WORM_RETENTION_MONTHS == 24, (
        "the retention period changed. It is disclosed in the privacy policy and "
        "in docs/audit-retention.md; change those in the same commit or not at all."
    )


# ── What must never be dropped ────────────────────────────────────────


def test_the_default_partition_is_never_dropped():
    """It has no month, so it cannot be outside a window, so it stays.

    It is also the safety net that catches a write when the maintainer has
    stopped advancing — dropping it converts a quiet degradation into lost
    writes.
    """
    from control_plane.audit_partitions import _partition_is_expired

    cutoff = dt.date(2099, 1, 1)
    for name in ("default", "", "20249", "abcdef", "999913"):
        assert not _partition_is_expired(name, cutoff), name


def test_an_unknown_table_is_refused_before_any_ddl():
    """The name reaches a DROP TABLE, so it is never taken on trust."""
    with pg_transaction() as conn:
        with pytest.raises(ValueError, match="not a known partitioned table"):
            expired_partitions(conn, "users; DROP TABLE wallets")
        with pytest.raises(ValueError, match="not a known partitioned table"):
            drop_expired_partitions(conn, "users")


def test_every_partitioned_table_is_swept_by_the_same_task():
    """One retention mechanism, not one per table.

    A table added to `PARTITIONED_TABLES` gets the retention drop for free. The
    failure this prevents is a new WORM table whose partitions accumulate
    forever while the policy claims 24 months.
    """
    import inspect

    from control_plane.audit_partitions import audit_partition_maintenance_task

    source = inspect.getsource(audit_partition_maintenance_task)
    assert "PARTITIONED_TABLES" in source, (
        "the maintenance task no longer iterates the table list; a new WORM "
        "table would silently keep everything"
    )
    assert "drop_expired_partitions" in source, (
        "the maintenance task no longer prunes; retention would be asserted only"
    )
    assert len(PARTITIONED_TABLES) >= 2


def test_the_drop_gives_up_rather_than_queueing_behind_a_reader(ancient_partition):
    """`DROP TABLE` on a partition takes ACCESS EXCLUSIVE on the parent.

    A lock request queues ahead of every later request on that table, so a
    sweep waiting behind one long read stalls every audit write for the
    duration. A retention sweep that takes the audit trail offline is far worse
    than a month of data surviving until tomorrow's run — the task is daily and
    the drop is idempotent.

    Read back from the session rather than grepped out of the source: what
    matters is that the setting is in force when the DDL runs.
    """
    with pg_transaction() as conn:
        before = conn.execute("SHOW lock_timeout").fetchone()[0]
        dropped = drop_expired_partitions(conn, "placement_decisions", today=dt.date(2026, 8, 1))
        assert dropped, "nothing was dropped, so this asserts nothing"
        assert conn.execute("SHOW lock_timeout").fetchone()[0] == "5s", (
            f"the drop ran without a lock timeout (session default {before!r}); "
            "it would queue ahead of every audit write instead of giving up"
        )
