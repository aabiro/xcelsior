"""Partition maintenance recovers when rows have landed in a DEFAULT partition.

Each monthly table keeps a DEFAULT partition as a safety net: a write past the
pre-created window lands there instead of failing. But once DEFAULT holds a row
for some month, PostgreSQL refuses `CREATE TABLE … PARTITION OF` for that month
("updated partition constraint for default partition … would be violated"), so
a maintainer that lapsed long enough failed on that month every day, for good.

Moving the rows out is not an option on the audit tables: their append-only
trigger refuses DELETE. So maintenance swaps the whole DEFAULT partition out
instead, in one transaction: detach it, give the table a fresh empty one, create
the months the old one covered, copy its rows back through the parent, and drop
the detached table. That is the same move migration 106 used to carry
`placement_decisions` across: INSERT … SELECT and DROP TABLE, never an UPDATE
or a DELETE on a row.

These tests use months far from the live window (2001 and 2033) and drop the
partitions they cause, which is how retention removes rows from these tables.
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
    ensure_monthly_partitions,
    expired_partitions,
)

EXPIRED = "200101"  # far outside the 24-month retention window
FUTURE = "203304"  # far beyond any window the maintainer has created
WINDOW = dt.date(2033, 4, 1)  # the window under test: April 2033 only


def _insert(conn, table: str, marker: str, when: str) -> None:
    """One row in `table`, keyed by `marker`, at timestamp `when`."""
    if table == "placement_decisions":
        conn.execute(
            "INSERT INTO placement_decisions (tenant_id, outcome, refusal_code, decided_at) "
            "VALUES (%s, 'refused', 'no_capacity', %s)",
            (marker, when),
        )
    elif table == "audit_events_v2":
        conn.execute(
            "INSERT INTO audit_events_v2 (stream_type, stream_id, stream_sequence, "
            "event_type, prev_hash, event_hash, created_at) "
            "VALUES ('test', %s, 1, 'partition.recovery', 'prev', md5(%s), %s)",
            (marker, marker + when, when),
        )
    else:
        conn.execute(
            "INSERT INTO telemetry_samples (host_id, sample, received_at) "
            "VALUES (%s, '{\"gpu_temp_c\": 61}'::jsonb, %s)",
            (marker, when),
        )


_MARKER_COLUMN = {
    "placement_decisions": "tenant_id",
    "audit_events_v2": "stream_id",
    "telemetry_samples": "host_id",
}


def _rows(conn, table: str, marker: str) -> list[tuple[str, str]]:
    """(partition, whole row as text) for each row carrying `marker`."""
    return [
        (r[0], r[1])
        for r in conn.execute(
            f"SELECT t.tableoid::regclass::text, t::text FROM {table} t "
            f"WHERE {_MARKER_COLUMN[table]} = %s ORDER BY 2",
            (marker,),
        ).fetchall()
    ]


def _partitions(conn, table: str, marker: str) -> set[str]:
    return {partition for partition, _ in _rows(conn, table, marker)}


def _default_partition(conn, table: str) -> str | None:
    row = conn.execute(
        "SELECT NULLIF(partdefid, 0)::regclass::text FROM pg_partitioned_table "
        "WHERE partrelid = %s::regclass",
        (table,),
    ).fetchone()
    return row[0]


def _drop(table: str, *suffixes: str) -> None:
    """Remove the partitions a test caused, the way retention removes rows."""
    with pg_transaction() as conn:
        for suffix in suffixes:
            conn.execute(f"DROP TABLE IF EXISTS {table}_{suffix}")


@pytest.mark.parametrize("table", ["placement_decisions", "audit_events_v2", "telemetry_samples"])
def test_rows_in_default_no_longer_block_the_window(table):
    marker = f"recovery-{uuid.uuid4().hex[:12]}"
    try:
        with pg_transaction() as conn:
            _insert(conn, table, marker, "2001-01-15 12:00:00+00")
            _insert(conn, table, marker, "2033-04-10 12:00:00+00")
            before = _rows(conn, table, marker)
            assert {p for p, _ in before} == {f"{table}_default"}, "setup: rows belong in DEFAULT"

            ensured = ensure_monthly_partitions(conn, table, months_ahead=0, today=WINDOW)

        assert ensured == [FUTURE]
        with pg_transaction() as conn:
            after = _rows(conn, table, marker)
            # Each row now sits in its own month's partition, byte for byte the
            # row it was: the hash chain, identity and timestamps all survive.
            assert {p for p, _ in after} == {f"{table}_{EXPIRED}", f"{table}_{FUTURE}"}
            assert [t for _, t in after] == [t for _, t in before]
            # The table still has a safety net, and it is empty.
            assert _default_partition(conn, table) == f"{table}_default"
            assert conn.execute(f"SELECT count(*) FROM {table}_default").fetchone()[0] == 0
            assert (
                conn.execute("SELECT to_regclass(%s)", (f"{table}_default_repair",)).fetchone()[0]
                is None
            )
            # Retention can reach the old row again. Stuck in DEFAULT, it could
            # never have been dropped: DEFAULT is not a month.
            expired = expired_partitions(conn, table, today=dt.date(2026, 11, 1))
            assert f"{table}_{EXPIRED}" in expired
            assert f"{table}_{FUTURE}" not in expired
    finally:
        _drop(table, EXPIRED, FUTURE)


@pytest.mark.parametrize("table", PARTITIONED_TABLES)
def test_the_recovered_partitions_are_still_append_only(table):
    marker = f"recovery-{uuid.uuid4().hex[:12]}"
    column = _MARKER_COLUMN[table]
    try:
        with pg_transaction() as conn:
            _insert(conn, table, marker, "2033-04-10 12:00:00+00")
            ensure_monthly_partitions(conn, table, months_ahead=0, today=WINDOW)

        for statement in (
            f"UPDATE {table} SET {column} = {column} || '-edited' WHERE {column} = %s",
            f"DELETE FROM {table} WHERE {column} = %s",
        ):
            with pytest.raises(Exception, match="append-only"):
                with pg_transaction() as conn:
                    conn.execute(statement, (marker,))

        with pg_transaction() as conn:
            # The fresh DEFAULT carries the trigger too, so the next row to land
            # there is as immutable as the last.
            triggers = conn.execute(
                "SELECT count(*) FROM pg_trigger WHERE tgrelid = %s::regclass AND tgname = %s",
                (f"{table}_default", f"trg_{table}_immutable"),
            ).fetchone()[0]
            assert triggers == 1
            assert _partitions(conn, table, marker) == {f"{table}_{FUTURE}"}
    finally:
        _drop(table, FUTURE)


def test_a_month_that_cannot_be_created_leaves_default_untouched():
    """Recovery is all or nothing. If one month cannot be created, the rows stay
    exactly where they were, DEFAULT stays attached, and the run fails loudly."""
    table = "placement_decisions"
    marker = f"recovery-{uuid.uuid4().hex[:12]}"
    try:
        with pg_transaction() as conn:
            _insert(conn, table, marker, "2033-04-10 12:00:00+00")
            # A table of the month's name that is not a partition: something a
            # human made, which recovery must not adopt or replace.
            conn.execute(f"CREATE TABLE {table}_{FUTURE} (LIKE {table})")

        with pytest.raises(Exception, match="already exists"):
            with pg_transaction() as conn:
                ensure_monthly_partitions(conn, table, months_ahead=0, today=WINDOW)

        with pg_transaction() as conn:
            assert _partitions(conn, table, marker) == {f"{table}_default"}
            assert _default_partition(conn, table) == f"{table}_default"
            conn.execute(f"DROP TABLE {table}_{FUTURE}")

        # With the obstruction gone, the next run recovers.
        with pg_transaction() as conn:
            ensure_monthly_partitions(conn, table, months_ahead=0, today=WINDOW)
            assert _partitions(conn, table, marker) == {f"{table}_{FUTURE}"}
    finally:
        _drop(table, FUTURE)


def test_an_empty_default_is_left_alone():
    """The common case costs one cheap read: no detach, no lock on the parent."""
    table = "placement_decisions"
    try:
        with pg_transaction() as conn:
            # Whatever an earlier run left in DEFAULT is recovered first.
            ensure_monthly_partitions(conn, table, months_ahead=0, today=WINDOW)
        with pg_transaction() as conn:
            oid = conn.execute("SELECT %s::regclass::oid", (f"{table}_default",)).fetchone()[0]
            ensure_monthly_partitions(conn, table, months_ahead=0, today=WINDOW)
            after = conn.execute("SELECT %s::regclass::oid", (f"{table}_default",)).fetchone()[0]
            assert after == oid
    finally:
        _drop(table, EXPIRED, FUTURE)
