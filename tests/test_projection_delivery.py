"""Track B B4.4 — two-stage per-sink projection delivery (§12.1).

Proves the crash-safe fan-out invariants:
  * prepare materializes exactly one obligation per (event, sink), idempotently;
  * a crash between prepare and deliver still yields exactly one logical delivery
    per sink after restart;
  * delivery is recorded by the sink's stable external id, so a replay is a
    no-op — exactly-once logical delivery on at-least-once I/O;
  * a failing delivery backs off and eventually dead-letters;
  * a sink added later receives nothing until an explicit backfill range.
"""

from __future__ import annotations

import datetime as _dt
import uuid

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _has = _c.execute("SELECT to_regclass('projection_deliveries')").fetchone()[0] is not None
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("projection_deliveries missing — upgrade >= 074")

from control_plane.projection_delivery import (
    backfill_sink,
    deliver_pending,
    ensure_sink,
    prepare_fanout,
    record_failure,
)


def _clear_projection_state() -> None:
    with _pool.connection() as conn:
        conn.execute("TRUNCATE projection_deliveries")
        conn.execute("TRUNCATE projection_checkpoints")
        conn.commit()


@pytest.fixture
def scratch():
    # Clean at SETUP too: the test database is shared, and earlier modules
    # (e.g. the audit projection runtime registering its 'audit_log' sink)
    # leave active checkpoints behind. prepare_fanout() fans out to every
    # active sink, so a foreign checkpoint changes this module's assertions.
    if _pool is not None:
        _clear_projection_state()
    made = {"idems": []}
    yield made
    if _pool is None:
        return
    _clear_projection_state()
    with _pool.connection() as conn:
        for idem in made["idems"]:
            conn.execute("DELETE FROM outbox_events WHERE idempotency_key=%s", (idem,))
        conn.commit()


def _append(scratch, *, created_at: _dt.datetime | None = None) -> str:
    idem = f"idem-{uuid.uuid4().hex[:12]}"
    scratch["idems"].append(idem)
    with _pool.connection() as conn:
        cols = "aggregate_type, aggregate_id, event_type, idempotency_key"
        vals = "'job', 'j1', 'job.v1.created', %s"
        params: list = [idem]
        if created_at is not None:
            cols += ", created_at"
            vals += ", %s"
            params.append(created_at)
        eid = conn.execute(
            f"INSERT INTO outbox_events ({cols}) VALUES ({vals}) RETURNING event_id", params
        ).fetchone()[0]
        conn.commit()
    return str(eid)


def _deliveries(event_id: str, sink: str | None = None) -> list[tuple]:
    with _pool.connection() as conn:
        if sink:
            rows = conn.execute(
                "SELECT sink, status, external_id, attempt_count FROM projection_deliveries "
                "WHERE event_id=%s AND sink=%s", (event_id, sink)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT sink, status, external_id, attempt_count FROM projection_deliveries "
                "WHERE event_id=%s ORDER BY sink", (event_id,)
            ).fetchall()
    return rows


def test_prepare_materializes_one_per_sink_idempotently(scratch):
    with _pool.connection() as conn:
        ensure_sink(conn, "warehouse")
        ensure_sink(conn, "sse")
        conn.commit()
    eid = _append(scratch)
    with _pool.connection() as conn:
        n = prepare_fanout(conn, only_event_ids=[eid])
        conn.commit()
    assert n == 1
    assert {r[0] for r in _deliveries(eid)} == {"warehouse", "sse"}
    # Re-prepare: the event is already prepared → no new work, no duplicates.
    with _pool.connection() as conn:
        assert prepare_fanout(conn, only_event_ids=[eid]) == 0
        conn.commit()
    assert len(_deliveries(eid)) == 2
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT fanout_prepared_at IS NOT NULL FROM outbox_events WHERE event_id=%s", (eid,)
        ).fetchone()[0] is True


def test_deliver_exactly_once_and_replay_is_noop(scratch):
    with _pool.connection() as conn:
        ensure_sink(conn, "warehouse")
        conn.commit()
    eid = _append(scratch)
    with _pool.connection() as conn:
        prepare_fanout(conn, only_event_ids=[eid])
        conn.commit()

    calls = {"n": 0}

    def deliver(ev):
        calls["n"] += 1
        return f"ext-{ev.event_id}"  # stable external id

    # Crash-recovery shape: the obligation persisted at prepare; deliver now.
    assert deliver_pending(_pool, "warehouse", deliver, limit=10) == 1
    rows = _deliveries(eid, "warehouse")
    assert rows[0][1] == "delivered" and rows[0][2] == f"ext-{eid}"
    # Replay: nothing pending → no second logical delivery.
    assert deliver_pending(_pool, "warehouse", deliver, limit=10) == 0
    assert calls["n"] == 1


def test_failed_delivery_backs_off_then_dead_letters(scratch):
    with _pool.connection() as conn:
        ensure_sink(conn, "webhook")
        conn.commit()
    eid = _append(scratch)
    with _pool.connection() as conn:
        prepare_fanout(conn, only_event_ids=[eid])
        # Tighten max_attempts so the dead-letter is reachable quickly, and clear
        # backoff so successive record_failure calls re-qualify immediately.
        conn.execute("UPDATE projection_deliveries SET max_attempts=2 WHERE event_id=%s", (eid,))
        conn.commit()

    def boom(ev):
        raise RuntimeError("sink unavailable")

    # First failure: still pending (backed off), attempt_count=1.
    assert deliver_pending(_pool, "webhook", boom, limit=10) == 0
    assert _deliveries(eid, "webhook")[0][3] == 1
    # Drive to the dead-letter threshold via the failure primitive (no backoff wait).
    with _pool.connection() as conn:
        record_failure(conn, eid, "webhook", "still down", backoff_sec=0)
        conn.commit()
    assert _deliveries(eid, "webhook")[0][1] == "dead_lettered"


def test_late_sink_gets_nothing_until_explicit_backfill(scratch):
    old = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(minutes=5)
    with _pool.connection() as conn:
        # Warehouse explicitly owns this historical range; the later SSE sink
        # does not inherit it merely by being registered.
        ensure_sink(conn, "warehouse", backfill_from=old - _dt.timedelta(minutes=1))
        conn.commit()
    eid = _append(scratch, created_at=old)
    with _pool.connection() as conn:
        prepare_fanout(conn, only_event_ids=[eid])  # warehouse gets E1; E1 is now prepared
        conn.commit()
    assert _deliveries(eid, "warehouse")  # warehouse has it

    # A sink added later sees nothing from before it existed …
    with _pool.connection() as conn:
        ensure_sink(conn, "sse")
        assert prepare_fanout(conn, only_event_ids=[eid]) == 0  # E1 already prepared — not re-fanned
        conn.commit()
    assert _deliveries(eid, "sse") == []

    # … until an explicit backfill range materializes the obligation. Use a
    # narrow window around E1 so concurrent tests' recent events don't fall in.
    with _pool.connection() as conn:
        made = backfill_sink(
            conn, "sse", frm=old - _dt.timedelta(minutes=1), to=old + _dt.timedelta(minutes=1)
        )
        conn.commit()
    # The shared test database can contain unrelated outbox rows inside the
    # explicit time window; the target obligation must be among those created.
    assert made >= 1
    assert _deliveries(eid, "sse")


def test_new_sink_defaults_to_registration_time_not_unbounded_backlog(scratch):
    """A NULL boundary used to make a newly enabled sink consume old backlog."""
    old = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=1)
    eid = _append(scratch, created_at=old)
    with _pool.connection() as conn:
        ensure_sink(conn, "webhook")
        checkpoint = conn.execute(
            "SELECT backfilled_from FROM projection_checkpoints WHERE sink='webhook'"
        ).fetchone()[0]
        assert checkpoint is not None
        assert checkpoint > old
        # The event is safely stamped as considered, but no obligation is
        # invented for a sink that did not exist when it was created.
        assert prepare_fanout(conn, only_event_ids=[eid]) == 1
        conn.commit()
    assert _deliveries(eid, "webhook") == []


def test_backfill_rejects_invalid_range_and_unregistered_sink(scratch):
    now = _dt.datetime.now(_dt.timezone.utc)
    with _pool.connection() as conn:
        with pytest.raises(ValueError, match="not registered"):
            backfill_sink(
                conn,
                "typo-sink",
                frm=now - _dt.timedelta(minutes=1),
                to=now,
            )
        ensure_sink(conn, "warehouse")
        with pytest.raises(ValueError, match="frm < to"):
            backfill_sink(conn, "warehouse", frm=now, to=now)
        conn.rollback()
