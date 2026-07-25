"""Track B B4.6 — durable, resumable SSE cursor (§16.3).

The crux of "kill a replica mid-stream; the client reconnects and receives every
transition exactly once, in order": a durable, totally-ordered replay from the
outbox projection. This drives that mechanism directly — a client that saw up to
a cursor, after a gap, replays exactly the missed events and nothing it already
saw.
"""

from __future__ import annotations

import datetime as _dt
import json
import uuid

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT to_regclass('outbox_events')").fetchone()
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None

from control_plane.event_stream import decode_cursor, encode_cursor, resume_after


@pytest.fixture
def scratch():
    made = {"idems": []}
    yield made
    if _pool is None:
        return
    with _pool.connection() as conn:
        for idem in made["idems"]:
            conn.execute("DELETE FROM outbox_events WHERE idempotency_key=%s", (idem,))
        conn.commit()


def _append(scratch, *, seq: int, created_at: str) -> str:
    idem = f"strm-{uuid.uuid4().hex[:10]}-{seq}"
    scratch["idems"].append(idem)
    with _pool.connection() as conn:
        eid = conn.execute(
            """INSERT INTO outbox_events (aggregate_type, aggregate_id, event_type, payload,
                                          idempotency_key, created_at)
               VALUES ('job', 'j-strm', %s, %s, %s, %s) RETURNING event_id""",
            (f"job.v1.e{seq}", json.dumps({"seq": seq}), idem, created_at),
        ).fetchone()[0]
        conn.commit()
    return str(eid)


def test_cursor_encode_decode_roundtrip():
    now = _dt.datetime(2026, 7, 20, 12, 0, 0, tzinfo=_dt.timezone.utc)
    eid = str(uuid.uuid4())
    ts, back_id = decode_cursor(encode_cursor(now, eid))
    assert back_id == eid
    assert abs((ts - now).total_seconds()) < 0.001
    assert decode_cursor(None) is None
    assert decode_cursor("garbage") is None


def _resume(cursor, my: set[str]) -> list[str]:
    """resume_after over a properly-scoped connection, filtered to this test's
    events (the shared outbox has unrelated rows)."""
    with _pool.connection() as conn:
        return [e for e in resume_after(conn, cursor, limit=2000) if e.event_id in my]


def test_resume_replays_gap_exactly_once_in_order(scratch):
    base = "2026-07-20 08:00:0"
    ids = [_append(scratch, seq=i, created_at=f"{base}{i}+00") for i in range(4)]  # e0..e3
    my = set(ids)

    all_ev = _resume(None, my)
    assert [e.event_id for e in all_ev] == ids  # earliest-first, in order

    # The client saw through e1 (index 1). A gap happens; it reconnects with that
    # cursor and must get exactly e2, e3 — never e0/e1 again.
    gap = _resume(all_ev[1].cursor, my)
    assert [e.event_id for e in gap] == ids[2:]

    # Caught up: resuming after the last delivered cursor yields nothing of ours.
    assert _resume(all_ev[-1].cursor, my) == []


def test_total_order_is_stable_across_equal_timestamps(scratch):
    # Two events at the SAME created_at — the UUID tiebreak gives a stable order,
    # so a cursor at the first returns the second (never both, never neither).
    ts = "2026-07-20 09:00:00+00"
    a = _append(scratch, seq=10, created_at=ts)
    b = _append(scratch, seq=11, created_at=ts)
    my = {a, b}
    ordered = _resume(None, my)
    assert len(ordered) == 2
    after_first = [e.event_id for e in _resume(ordered[0].cursor, my)]
    assert after_first == [ordered[1].event_id]
