"""Per-stream event hash chains without global table exclusive lock.

Drives the shipped ``EventStore.append`` / ``verify_chain`` path against
real Postgres:

- concurrent appends on two distinct entity streams succeed
- each stream's prev_hash chain verifies independently
- a broken link fails closed on that stream
- production append source no longer issues ``LOCK TABLE events IN EXCLUSIVE MODE``
"""

from __future__ import annotations

import concurrent.futures
import inspect
import time
import uuid
from pathlib import Path

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _has_events = (
            _c.execute("SELECT to_regclass('public.events')").fetchone()[0]
            is not None
        )
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None
else:
    if not _has_events:  # pragma: no cover
        pytestmark = pytest.mark.skip("events table missing")

from events import Event, EventStore, EventType, event_stream_id


@pytest.fixture
def clean_events():
    """Isolate tests to streams we create (delete by entity_id prefix)."""
    prefix = f"stream-test-{uuid.uuid4().hex[:8]}"
    yield prefix
    if _pool is None:
        return
    with _pool.connection() as conn:
        conn.execute(
            "DELETE FROM events WHERE entity_id LIKE %s",
            (f"{prefix}%",),
        )
        conn.commit()


def _store() -> EventStore:
    return EventStore()


def test_event_stream_id_is_entity_scoped():
    assert event_stream_id("job", "j1") == event_stream_id("job", "j1")
    assert event_stream_id("job", "j1") != event_stream_id("job", "j2")
    assert event_stream_id("job", "j1") != event_stream_id("host", "j1")


def test_append_path_has_no_table_exclusive_lock():
    """Structural: production append must not serialize the whole table."""
    src = Path(__file__).resolve().parents[1] / "events.py"
    text = src.read_text()
    # Locate the EventStore.append body.
    start = text.find("def append(self, event: Event)")
    assert start > 0
    end = text.find("\n    def verify_chain", start)
    body = text[start:end]
    assert "LOCK TABLE events" not in body
    assert "EXCLUSIVE MODE" not in body
    assert "pg_advisory_xact_lock" in body
    assert "entity_type = %s AND entity_id = %s" in body


def test_same_stream_chains_and_cross_stream_do_not(clean_events):
    prefix = clean_events
    es = _store()
    a1 = es.append(
        Event(
            event_type=EventType.JOB_SUBMITTED,
            entity_type="job",
            entity_id=f"{prefix}-job-a",
            actor="test",
        )
    )
    a2 = es.append(
        Event(
            event_type=EventType.JOB_ASSIGNED,
            entity_type="job",
            entity_id=f"{prefix}-job-a",
            actor="test",
        )
    )
    b1 = es.append(
        Event(
            event_type=EventType.HOST_REGISTERED,
            entity_type="host",
            entity_id=f"{prefix}-host-b",
            actor="test",
        )
    )
    assert a1.prev_hash == ""
    assert a2.prev_hash == a1.event_hash
    # New stream starts its own chain — does not link to job-a's head.
    assert b1.prev_hash == ""
    assert b1.event_hash != a2.event_hash

    ok_a = es.verify_chain(entity_type="job", entity_id=f"{prefix}-job-a")
    ok_b = es.verify_chain(entity_type="host", entity_id=f"{prefix}-host-b")
    assert ok_a["valid"] is True and ok_a["events_checked"] == 2
    assert ok_b["valid"] is True and ok_b["events_checked"] == 1


def test_verify_chain_detects_broken_stream_link(clean_events):
    prefix = clean_events
    es = _store()
    eid = f"{prefix}-job-break"
    e1 = es.append(
        Event(
            event_type=EventType.JOB_SUBMITTED,
            entity_type="job",
            entity_id=eid,
            actor="test",
        )
    )
    e2 = es.append(
        Event(
            event_type=EventType.JOB_ASSIGNED,
            entity_type="job",
            entity_id=eid,
            actor="test",
        )
    )
    assert e2.prev_hash == e1.event_hash

    # Tamper with the stored link (simulate disk/DB corruption).
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE events SET prev_hash = %s WHERE event_id = %s",
            ("0" * 64, e2.event_id),
        )
        conn.commit()

    result = es.verify_chain(entity_type="job", entity_id=eid)
    assert result["valid"] is False
    assert result["broken_at"] == e2.event_id
    assert "prev_hash mismatch" in (result.get("reason") or "")


def test_verify_chain_detects_tampered_event_hash(clean_events):
    prefix = clean_events
    es = _store()
    eid = f"{prefix}-job-hash"
    evt = es.append(
        Event(
            event_type=EventType.JOB_SUBMITTED,
            entity_type="job",
            entity_id=eid,
            actor="test",
            data={"k": "v"},
        )
    )
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE events SET event_hash = %s WHERE event_id = %s",
            ("f" * 64, evt.event_id),
        )
        conn.commit()

    result = es.verify_chain(entity_type="job", entity_id=eid)
    assert result["valid"] is False
    assert result["broken_at"] == evt.event_id
    assert "tampered" in (result.get("reason") or "")


def _append_n(entity_type: str, entity_id: str, n: int, actor: str) -> list[str]:
    es = EventStore()
    ids = []
    for i in range(n):
        evt = es.append(
            Event(
                event_type=EventType.JOB_SUBMITTED
                if entity_type == "job"
                else EventType.HOST_REGISTERED,
                entity_type=entity_type,
                entity_id=entity_id,
                actor=actor,
                data={"i": i, "ts": time.time()},
            )
        )
        ids.append(evt.event_id)
    return ids


def test_concurrent_distinct_streams_append_and_verify(clean_events):
    """Two connections append to different streams without table exclusive lock."""
    prefix = clean_events
    stream_a = f"{prefix}-job-conc-a"
    stream_b = f"{prefix}-host-conc-b"
    n = 12

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fa = pool.submit(_append_n, "job", stream_a, n, "worker-a")
        fb = pool.submit(_append_n, "host", stream_b, n, "worker-b")
        ids_a = fa.result(timeout=30)
        ids_b = fb.result(timeout=30)

    assert len(ids_a) == n
    assert len(ids_b) == n
    assert set(ids_a).isdisjoint(set(ids_b))

    es = _store()
    va = es.verify_chain(entity_type="job", entity_id=stream_a)
    vb = es.verify_chain(entity_type="host", entity_id=stream_b)
    assert va["valid"] is True, va
    assert va["events_checked"] == n
    assert vb["valid"] is True, vb
    assert vb["events_checked"] == n

    # Full multi-stream verify over our prefixes only via get + verify each.
    # Global verify includes other test pollution; check our streams only.
    events_a = es.get_events(entity_type="job", entity_id=stream_a, limit=100)
    assert len(events_a) == n
    assert events_a[0].prev_hash == ""
    for i in range(1, n):
        assert events_a[i].prev_hash == events_a[i - 1].event_hash


def test_concurrent_same_stream_serializes_chain(clean_events):
    """Same-stream concurrent appends still form one correct chain."""
    prefix = clean_events
    stream = f"{prefix}-job-same"
    n_each = 8

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_append_n, "job", stream, n_each, "w1")
        f2 = pool.submit(_append_n, "job", stream, n_each, "w2")
        f1.result(timeout=30)
        f2.result(timeout=30)

    es = _store()
    result = es.verify_chain(entity_type="job", entity_id=stream)
    assert result["valid"] is True, result
    assert result["events_checked"] == n_each * 2

    events = es.get_events(entity_type="job", entity_id=stream, limit=100)
    assert len(events) == n_each * 2
    assert events[0].prev_hash == ""
    for i in range(1, len(events)):
        assert events[i].prev_hash == events[i - 1].event_hash


def test_prelock_inverted_timestamps_do_not_fork_stream(clean_events):
    """Pre-lock wall-clock timestamps must not invert chain head selection.

    Skeptic case: Event.timestamp set before the stream lock (caller or
    concurrent construction) can put a later causal append *earlier* in
    wall-clock order. Head SELECT / verify_chain order by timestamp would
    then diverge from prev_hash link order and fork the stream.
    """
    prefix = clean_events
    stream = f"{prefix}-job-ts-invert"
    es = _store()
    now = time.time()
    # Deliberately inverted pre-lock timestamps (future first, past second).
    e1 = es.append(
        Event(
            event_type=EventType.JOB_SUBMITTED,
            entity_type="job",
            entity_id=stream,
            actor="test",
            timestamp=now + 10_000.0,
            data={"phase": "first"},
        )
    )
    e2 = es.append(
        Event(
            event_type=EventType.JOB_ASSIGNED,
            entity_type="job",
            entity_id=stream,
            actor="test",
            timestamp=now - 10_000.0,
            data={"phase": "second"},
        )
    )
    # Append owns causal timestamp under the lock (pre-lock +10k/-10k discarded).
    assert e2.prev_hash == e1.event_hash
    assert e2.timestamp >= e1.timestamp
    assert abs(e1.timestamp - now) < 60.0  # not the pre-lock future +10k
    assert abs(e2.timestamp - now) < 60.0  # not the pre-lock past -10k

    result = es.verify_chain(entity_type="job", entity_id=stream)
    assert result["valid"] is True, result
    assert result["events_checked"] == 2

    # Third append must link to e2 (true head), not re-link to e1.
    e3 = es.append(
        Event(
            event_type=EventType.JOB_RUNNING,
            entity_type="job",
            entity_id=stream,
            actor="test",
            timestamp=now - 20_000.0,
            data={"phase": "third"},
        )
    )
    assert e3.prev_hash == e2.event_hash
    assert e3.timestamp >= e2.timestamp
    assert es.verify_chain(entity_type="job", entity_id=stream)["valid"] is True


def _append_n_skewed(entity_type: str, entity_id: str, n: int, actor: str, skew: float):
    """Append n events with deliberately skewed pre-lock timestamps."""
    es = EventStore()
    ids = []
    base = time.time() + skew
    for i in range(n):
        # Decreasing pre-lock timestamps — worst case vs lock order.
        evt = es.append(
            Event(
                event_type=EventType.JOB_SUBMITTED
                if entity_type == "job"
                else EventType.HOST_REGISTERED,
                entity_type=entity_type,
                entity_id=entity_id,
                actor=actor,
                timestamp=base - float(i),
                data={"i": i, "skew": skew},
            )
        )
        ids.append(evt.event_id)
    return ids


def test_concurrent_same_stream_with_skewed_prelock_timestamps(clean_events):
    """Concurrent same-stream appends with inverted pre-lock timestamps.

    Stresses lock-queue inversion: writers build Event objects with
    future/past timestamps *before* contending on the stream lock.
    Chain must still verify after both finish.
    """
    prefix = clean_events
    stream = f"{prefix}-job-skew"
    n_each = 10

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(
            _append_n_skewed, "job", stream, n_each, "skew-future", 50_000.0
        )
        f2 = pool.submit(
            _append_n_skewed, "job", stream, n_each, "skew-past", -50_000.0
        )
        f1.result(timeout=30)
        f2.result(timeout=30)

    es = _store()
    result = es.verify_chain(entity_type="job", entity_id=stream)
    assert result["valid"] is True, result
    assert result["events_checked"] == n_each * 2

    events = es.get_events(entity_type="job", entity_id=stream, limit=100)
    assert len(events) == n_each * 2
    # Monotonic causal timestamps under lock; prev_hash matches order.
    for i in range(1, len(events)):
        assert events[i].timestamp >= events[i - 1].timestamp
        assert events[i].prev_hash == events[i - 1].event_hash


def test_production_callers_still_use_event_store_append():
    """Static inventory: audit / volumes go through EventStore.append only."""
    root = Path(__file__).resolve().parents[1]
    deps = (root / "routes" / "_deps.py").read_text()
    assert "get_event_store().append" in deps
    assert "append_user_audit_event" in deps

    volumes = (root / "volumes.py").read_text()
    assert "EventStore" in volumes
    # volumes constructs EventStore and appends lifecycle events
    assert ".append(" in volumes

    events_src = (root / "events.py").read_text()
    assert "def append(self, event: Event)" in events_src
    # Single authoritative append method still owns chain policy.
    assert events_src.count("def append(self, event: Event)") == 1


def test_append_source_uses_advisory_not_table_lock_inspect():
    es = EventStore()
    src = inspect.getsource(es.append)
    assert "LOCK TABLE" not in src
    assert "pg_advisory_xact_lock" in src
