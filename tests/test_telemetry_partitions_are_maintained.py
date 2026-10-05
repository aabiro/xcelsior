"""`telemetry_samples` keeps its monthly window full, under its own policy.

See `telemetry_partition_maintenance_task` for the history: the task 057 seeded
was never registered, the window stopped at September 2026, and the routing
test began failing on 1 October — not because routing broke, but because the
month it was routing into had never been created. That test now runs the
maintainer first, because "the current month lands in a monthly partition" is
only true *given* maintenance, and asserting it without maintenance made it a
calendar-dependent test that passed until the day it didn't.
"""

from __future__ import annotations

import datetime as dt
import os

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from control_plane.audit_partitions import (  # noqa: E402
    PARTITIONED_TABLES,
    TELEMETRY_TABLE,
    WORM_RETENTION_MONTHS,
    expired_partitions,
    telemetry_policy,
)


class _Conn:
    """Answers the two queries the policy and the expiry selection make."""

    def __init__(self, *, payload=None, partitions=()):  # noqa: ANN001
        self._payload = payload
        self._partitions = list(partitions)
        self._last = ""

    def execute(self, sql, params=None):  # noqa: ANN001
        self._last = sql
        return self

    def fetchone(self):
        return None if self._payload is None else (self._payload,)

    def fetchall(self):
        return [(name,) for name in self._partitions]


def test_telemetry_is_not_swept_under_the_audit_retention() -> None:
    """24 months is a legal ruling about audit records, not about GPU samples."""
    assert TELEMETRY_TABLE not in PARTITIONED_TABLES


def test_the_seeded_policy_is_the_one_applied() -> None:
    assert telemetry_policy(_Conn(payload={"months_ahead": 2, "retention_months": 6})) == (2, 6)


def test_a_missing_policy_row_falls_back_to_the_seeded_values() -> None:
    assert telemetry_policy(_Conn(payload=None)) == (2, 6)
    assert telemetry_policy(_Conn(payload='{"retention_months": 9}')) == (2, 9)


@pytest.mark.parametrize("payload", [{"retention_months": 0}, {"months_ahead": 0}, {"retention_months": -1}])
def test_a_nonsense_policy_is_refused_rather_than_obeyed(payload) -> None:  # noqa: ANN001
    """Retention 0 drops last month; a negative one drops the month being written."""
    with pytest.raises(ValueError, match="policy is invalid"):
        telemetry_policy(_Conn(payload=payload))


def test_retention_drops_only_whole_months_outside_six() -> None:
    conn = _Conn(
        partitions=[
            "telemetry_samples_202603",  # ends 2026-04-01: outside a 6-month window from October
            "telemetry_samples_202604",  # ends 2026-05-01: inside
            "telemetry_samples_202609",
            "telemetry_samples_default",
        ]
    )
    expired = expired_partitions(
        conn, TELEMETRY_TABLE, retention_months=6, today=dt.date(2026, 10, 5)
    )
    assert expired == ["telemetry_samples_202603"], expired
    assert WORM_RETENTION_MONTHS != 6  # the two policies really are different


def _partitions(conn) -> set[str]:  # noqa: ANN001
    rows = conn.execute(
        """
        SELECT c.relname FROM pg_inherits i
          JOIN pg_class c ON c.oid = i.inhrelid
          JOIN pg_class p ON p.oid = i.inhparent
         WHERE p.relname = 'telemetry_samples'
        """
    ).fetchall()
    return {r[0] if not isinstance(r, dict) else r["relname"] for r in rows}


def test_the_registered_task_extends_the_window_in_the_real_database() -> None:
    """Run the actual entry point against the actual schema."""
    try:
        from control_plane.db import control_plane_transaction
        from control_plane.audit_partitions import telemetry_partition_maintenance_task

        telemetry_partition_maintenance_task()
        with control_plane_transaction() as conn:
            present = _partitions(conn)
    except Exception as exc:  # pragma: no cover
        if "telemetry_samples" in str(exc) and "does not exist" in str(exc):
            pytest.skip(f"test database predates migration 057: {exc}")
        raise
    today = dt.date.today()
    for offset in range(3):  # this month and the two the policy keeps ahead
        total = today.year * 12 + today.month - 1 + offset
        suffix = f"{total // 12:04d}{total % 12 + 1:02d}"
        assert f"telemetry_samples_{suffix}" in present, (
            f"telemetry_samples_{suffix} missing after maintenance; have {sorted(present)}"
        )


def test_the_audit_task_reports_a_failed_table_instead_of_succeeding(monkeypatch) -> None:  # noqa: ANN001
    """It caught every exception and the scheduler recorded `succeeded`.

    Isolating tables from each other is right — one bad table must not stop the
    others — but a run where a window failed to advance is not a success, and
    `scheduled_tasks.last_status` is the only place anyone looks.
    """
    import contextlib

    import control_plane.audit_partitions as ap
    import control_plane.db as cpdb

    @contextlib.contextmanager
    def _txn():
        yield object()

    attempted: list[str] = []

    def _ensure(conn, table, **kw):  # noqa: ANN001, ANN003
        attempted.append(table)
        if table == "audit_events_v2":
            raise RuntimeError("relation is locked")
        return []

    monkeypatch.setattr(cpdb, "control_plane_transaction", _txn)
    monkeypatch.setattr(ap, "ensure_monthly_partitions", _ensure)
    monkeypatch.setattr(ap, "drop_expired_partitions", lambda *a, **k: [])

    with pytest.raises(RuntimeError, match="audit_events_v2 create: relation is locked"):
        ap.audit_partition_maintenance_task()
    assert attempted == list(PARTITIONED_TABLES), (
        "a failure in one table stopped the others from being maintained"
    )
