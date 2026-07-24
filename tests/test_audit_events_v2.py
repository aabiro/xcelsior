"""Track B B4.1 — partitioned, append-only audit_events_v2 (§13.6/§4.5).

Proves the durable audit stream's structural guarantees:
  * WORM — a written row can never be UPDATEd or DELETEd;
  * partition maintenance creates monthly partitions **ahead** of the write, so
    a write never has to create one inline;
  * a write beyond the pre-created window lands in the DEFAULT partition rather
    than failing;
  * no request handler creates a partition ad hoc.

Real PostgreSQL; cleans up with TRUNCATE (statement-level, so the row-level WORM
trigger does not block teardown).
"""

from __future__ import annotations

import ast
import pathlib
import uuid

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _has = _c.execute("SELECT to_regclass('audit_events_v2')").fetchone()[0] is not None
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("audit_events_v2 missing — upgrade to >= 072")

from control_plane.audit_partitions import ensure_audit_partitions
from psycopg.errors import RestrictViolation, UniqueViolation

REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _truncate_audit():
    yield
    if _pool is None:
        return
    with _pool.connection() as conn:
        # TRUNCATE is statement-level; the FOR EACH ROW WORM trigger does not
        # fire, so this is the sanctioned way to reset the append-only table.
        conn.execute("TRUNCATE audit_events_v2")
        conn.commit()


def _insert(conn, *, stream_id: str, seq: int, created_at: str | None = None) -> str:
    event_id = str(uuid.uuid4())
    cols = "event_id, stream_type, stream_id, stream_sequence, event_type, event_hash"
    vals = "%s, 'job', %s, %s, 'job.v1.created', %s"
    params: list = [event_id, stream_id, seq, f"h{seq}"]
    if created_at is not None:
        cols += ", created_at"
        vals += ", %s"
        params.append(created_at)
    conn.execute(f"INSERT INTO audit_events_v2 ({cols}) VALUES ({vals})", params)
    return event_id


def test_worm_rejects_update_and_delete():
    with _pool.connection() as conn:
        sid = f"s-{uuid.uuid4().hex[:8]}"
        eid = _insert(conn, stream_id=sid, seq=1)
        conn.commit()
    with _pool.connection() as conn:
        with pytest.raises(RestrictViolation):
            conn.execute("UPDATE audit_events_v2 SET event_type='x' WHERE event_id=%s", (eid,))
        conn.rollback()
    with _pool.connection() as conn:
        with pytest.raises(RestrictViolation):
            conn.execute("DELETE FROM audit_events_v2 WHERE event_id=%s", (eid,))
        conn.rollback()
    # The row survives both attempts.
    with _pool.connection() as conn:
        assert conn.execute("SELECT count(*) FROM audit_events_v2 WHERE event_id=%s", (eid,)).fetchone()[0] == 1


def test_stream_sequence_unique_backstop_within_partition():
    # The DB backstop is UNIQUE (stream_id, stream_sequence, created_at) — the
    # partition key must be included on a partitioned table, so a duplicate must
    # match created_at too. Global per-stream monotonicity is app-enforced by the
    # append path's per-stream advisory lock (like the existing events chain).
    ts = "2026-07-15 10:00:00+00"
    with _pool.connection() as conn:
        sid = f"s-{uuid.uuid4().hex[:8]}"
        _insert(conn, stream_id=sid, seq=1, created_at=ts)
        conn.commit()
    with _pool.connection() as conn:
        with pytest.raises(UniqueViolation):
            _insert(conn, stream_id=sid, seq=1, created_at=ts)
        conn.rollback()


def test_maintenance_creates_month_partitions_ahead():
    import datetime as _dt

    far = _dt.date(2031, 3, 1)
    try:
        with _pool.connection() as conn:
            ensured = ensure_audit_partitions(conn, months_ahead=1, today=far)
            conn.commit()
        assert "203103" in ensured and "203104" in ensured
        # A write into the pre-created month lands in its own partition, not default.
        with _pool.connection() as conn:
            eid = _insert(conn, stream_id="s-far", seq=1, created_at="2031-03-15")
            conn.commit()
            part = conn.execute(
                "SELECT tableoid::regclass::text FROM audit_events_v2 WHERE event_id=%s", (eid,)
            ).fetchone()[0]
        assert part == "audit_events_v2_203103"
    finally:
        with _pool.connection() as conn:
            conn.execute("DROP TABLE IF EXISTS audit_events_v2_203103")
            conn.execute("DROP TABLE IF EXISTS audit_events_v2_203104")
            conn.commit()


def test_write_beyond_window_lands_in_default_not_a_failure():
    # A month with no pre-created partition — the write must succeed (DEFAULT
    # safety net), never fail and never create a partition inline.
    with _pool.connection() as conn:
        eid = _insert(conn, stream_id="s-ovf", seq=1, created_at="2040-11-20")
        conn.commit()
        part = conn.execute(
            "SELECT tableoid::regclass::text FROM audit_events_v2 WHERE event_id=%s", (eid,)
        ).fetchone()[0]
    assert part == "audit_events_v2_default"


def test_no_request_handler_creates_audit_partitions():
    """Static guard: no route handler does ad-hoc partition DDL for the table."""
    offenders = []
    for path in (REPO / "routes").glob("*.py"):
        src = path.read_text()
        if "PARTITION OF audit_events_v2" in src or "CREATE TABLE" in src and "audit_events_v2" in src:
            offenders.append(path.name)
    assert not offenders, f"request handler creates audit partitions inline (forbidden): {offenders}"
