"""Track B B4.2 — the event archive/retention path works and reads real columns.

The archive was dead code: it queried `created_at` / `chain_hash` on the live
`events` table, which has `timestamp` / `event_hash`. This proves the corrected
retention path actually moves rows to cold storage (preserving the chain hash),
and a static guard fails if the archive query ever references a column the live
`events` table does not have again (DA§2.2).
"""

from __future__ import annotations

import inspect
import re
import time
import uuid

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT to_regclass('events_archive')").fetchone()
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None

from events import EventSnapshotManager

_LIVE_EVENTS_COLUMNS = {
    "event_id", "event_type", "entity_type", "entity_id", "timestamp",
    "actor", "data", "metadata", "prev_hash", "event_hash",
}


def _insert_event(conn, *, event_id: str, ts: float, event_hash: str) -> None:
    conn.execute(
        """INSERT INTO events (event_id, event_type, entity_type, entity_id,
                               timestamp, actor, data, metadata, prev_hash, event_hash)
           VALUES (%s, 'job.v1.created', 'job', %s, %s, 'tester',
                   '{}'::jsonb, '{}'::jsonb, '', %s)""",
        (event_id, event_id, ts, event_hash),
    )


@pytest.fixture
def scratch():
    made = {"events": []}
    yield made
    if _pool is None:
        return
    with _pool.connection() as conn:
        for eid in made["events"]:
            conn.execute("DELETE FROM events_archive WHERE event_id=%s", (eid,))
            conn.execute("DELETE FROM events WHERE event_id=%s", (eid,))
        conn.commit()


def test_archive_moves_old_events_and_preserves_chain_hash(scratch):
    now = time.time()
    old_id = f"ev-old-{uuid.uuid4().hex[:8]}"
    fresh_id = f"ev-new-{uuid.uuid4().hex[:8]}"
    scratch["events"] += [old_id, fresh_id]
    with _pool.connection() as conn:
        _insert_event(conn, event_id=old_id, ts=now - 200 * 86400, event_hash="chainhash_old")
        _insert_event(conn, event_id=fresh_id, ts=now, event_hash="chainhash_fresh")
        conn.commit()

    moved = EventSnapshotManager().archive_old_events(max_age_days=90)
    assert moved >= 1

    with _pool.connection() as conn:
        # Old event left the hot table …
        assert conn.execute("SELECT count(*) FROM events WHERE event_id=%s", (old_id,)).fetchone()[0] == 0
        # … and arrived in cold storage with its chain hash preserved.
        arch = conn.execute(
            "SELECT chain_hash FROM events_archive WHERE event_id=%s", (old_id,)
        ).fetchone()
        assert arch is not None and arch[0] == "chainhash_old"
        # The fresh event is untouched.
        assert conn.execute("SELECT count(*) FROM events WHERE event_id=%s", (fresh_id,)).fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM events_archive WHERE event_id=%s", (fresh_id,)).fetchone()[0] == 0


def test_archive_query_references_only_live_events_columns():
    """Static guard: the archive path must not read a column `events` lacks.

    Extracts the column tokens the archive selects/filters against `events` and
    asserts each is a real `events` column — so the `created_at`/`chain_hash`
    regression cannot silently return.
    """
    src = inspect.getsource(EventSnapshotManager.archive_old_events)
    # The SELECT … FROM events and WHERE clause against the hot table.
    forbidden = {"created_at", "chain_hash"}  # live-events columns these are NOT
    # `chain_hash` and `created_at` are legitimate on events_archive (the INSERT
    # target), so only flag them where the query reads FROM/WHERE `events`.
    select_from_events = re.search(r"FROM\s+events\b(.*?)RETURNING", src, re.S | re.I)
    where_events = re.findall(r"(?:FROM|DELETE FROM)\s+events\b[^%]*?WHERE\s+(\w+)", src, re.I)
    read_region = (select_from_events.group(1) if select_from_events else "")
    for col in forbidden:
        assert col not in read_region, f"archive reads `{col}` from events (not a live column)"
    for col in where_events:
        assert col in _LIVE_EVENTS_COLUMNS, f"archive filters events on `{col}` (not a live column)"
