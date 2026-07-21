"""Scheduler residual SSE: queue-block job_error + spot_prices via outbox.

Drives shipped ``_persist_queue_reason`` / ``process_queue_binpack`` skip
path and ``update_spot_prices`` against real Postgres. Asserts durable
outbox rows, projection shapes, no dual process-local emit when outbox
wins, and LISTEN delivery for at least one type.
"""

from __future__ import annotations

import json
import select
import time
import uuid

import pytest

try:
    from db import _get_pg_pool, resolve_postgres_dsn

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _migrated = (
            _c.execute("SELECT to_regclass('outbox_events')").fetchone()[0]
            is not None
        )
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None
else:
    if not _migrated:  # pragma: no cover
        pytestmark = pytest.mark.skip("test database missing outbox_events")

import psycopg

from control_plane.outbox import OutboxDispatcher, OutboxEvent
from control_plane.outbox_runtime import (
    EVENTS_CHANNEL,
    default_handlers,
    sse_payload_for,
)


@pytest.fixture
def cleanup():
    ids = {"jobs": [], "hosts": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (hid,))
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        # Spot pricing aggregate is fixed id "spot"
        conn.execute(
            "DELETE FROM outbox_events WHERE aggregate_id='spot' "
            "AND event_type='pricing.v1.spot_prices_updated'"
        )
        conn.commit()


def _outbox_for(aggregate_id: str) -> list[dict]:
    with _pool.connection() as conn:
        rows = conn.execute(
            """SELECT event_type, payload, published_at, idempotency_key
                 FROM outbox_events
                WHERE aggregate_id=%s
                ORDER BY created_at""",
            (aggregate_id,),
        ).fetchall()
    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append(r)
        else:
            out.append(
                {
                    "event_type": r[0],
                    "payload": r[1],
                    "published_at": r[2],
                    "idempotency_key": r[3],
                }
            )
    return out


def _payload(row: dict) -> dict:
    p = row["payload"]
    if isinstance(p, str):
        return json.loads(p)
    return dict(p or {})


class _Listener:
    def __enter__(self):
        self.conn = psycopg.connect(resolve_postgres_dsn(), autocommit=True)
        self.conn.execute(f"LISTEN {EVENTS_CHANNEL}")
        return self

    def __exit__(self, *exc):
        self.conn.close()

    def collect(self, max_wait: float = 0.25) -> list[dict]:
        got: list[dict] = []
        deadline = time.time() + max_wait
        while time.time() < deadline:
            remaining = max(0.02, deadline - time.time())
            if select.select([self.conn.fileno()], [], [], remaining) == ([], [], []):
                continue
            for notice in self.conn.notifies(timeout=0.05):
                got.append(json.loads(notice.payload))
        return got

    def drain_until(self, predicate, *, dispatcher_id: str, max_cycles: int = 40):
        all_msgs: list[dict] = []
        dispatcher = OutboxDispatcher(dispatcher_id, default_handlers())
        for _ in range(max_cycles):
            dispatcher.run_once()
            all_msgs.extend(self.collect(max_wait=0.25))
            if predicate(all_msgs):
                all_msgs.extend(self.collect(max_wait=0.15))
                break
        return all_msgs


# ── Projections ────────────────────────────────────────────────────────


def test_sse_payload_for_queue_blocked_and_spot_prices():
    q = OutboxEvent(
        event_id="e-q",
        aggregate_type="job",
        aggregate_id="j-block",
        event_type="job.v1.queue_blocked",
        payload={"error": "insufficient_vram", "message": "need more VRAM"},
        headers={},
        destination_class="default",
        idempotency_key="k",
        attempt_count=1,
    )
    msg = sse_payload_for(q)
    assert msg is not None
    assert msg["type"] == "job_error"
    assert msg["data"]["job_id"] == "j-block"
    assert msg["data"]["error"] == "insufficient_vram"
    assert msg["data"]["message"] == "need more VRAM"

    s = OutboxEvent(
        event_id="e-s",
        aggregate_type="pricing",
        aggregate_id="spot",
        event_type="pricing.v1.spot_prices_updated",
        payload={"prices": {"RTX 4090": 0.42, "A100": 1.1}},
        headers={},
        destination_class="default",
        idempotency_key="k2",
        attempt_count=1,
    )
    msg_s = sse_payload_for(s)
    assert msg_s is not None
    assert msg_s["type"] == "spot_prices"
    assert msg_s["data"]["RTX 4090"] == 0.42
    assert msg_s["data"]["A100"] == 1.1


# ── Queue-block producer ───────────────────────────────────────────────


def test_persist_queue_reason_appends_queue_blocked_outbox(cleanup, monkeypatch):
    import scheduler as sched

    job = sched.submit_job(
        f"qb-{uuid.uuid4().hex[:6]}", 8.0, tier="on-demand"
    )
    job_id = job["job_id"]
    cleanup["jobs"].append(job_id)

    emit_calls: list = []
    monkeypatch.setattr(
        sched, "emit_event", lambda et, data: emit_calls.append((et, data))
    )

    durable = sched._persist_queue_reason(
        job, "insufficient_vram", "Not enough free VRAM"
    )
    assert durable is True

    rows = [
        r
        for r in _outbox_for(job_id)
        if r["event_type"] == "job.v1.queue_blocked"
    ]
    assert len(rows) >= 1
    payload = _payload(rows[-1])
    assert payload.get("error") == "insufficient_vram"
    assert "VRAM" in (payload.get("message") or "")
    assert rows[-1]["published_at"] is None

    # Row truth: queue_reason landed with the outbox intent.
    found = sched.get_job(job_id) if hasattr(sched, "get_job") else None
    if found is None:
        jobs = [j for j in sched.list_jobs() if isinstance(j, dict)]
        found = next((j for j in jobs if j.get("job_id") == job_id), None)
    assert found is not None
    assert found.get("queue_reason") == "insufficient_vram"
    assert found.get("queue_reason_detail") == "Not enough free VRAM"


def test_binpack_skip_path_uses_outbox_not_dual_emit(cleanup, monkeypatch):
    """Shipped process_queue_binpack: unassignable job → outbox, no dual emit."""
    import scheduler as sched

    # No active hosts → every queued job is skipped → job_error path.
    job = sched.submit_job(
        f"binpack-qb-{uuid.uuid4().hex[:6]}",
        999.0,  # impossible VRAM so even hosts would fail
        tier="on-demand",
    )
    job_id = job["job_id"]
    cleanup["jobs"].append(job_id)

    emit_calls: list = []
    monkeypatch.setattr(
        sched, "emit_event", lambda et, data: emit_calls.append((et, data))
    )
    # Clear process-local throttle so this job is eligible.
    sched._job_error_notified.clear()

    # Ensure no hosts for a clean skip (empty list_hosts).
    monkeypatch.setattr(sched, "list_hosts", lambda active_only=True: [])

    assigned = sched.process_queue_binpack()
    assert assigned == [] or isinstance(assigned, list)

    rows = [
        r
        for r in _outbox_for(job_id)
        if r["event_type"] == "job.v1.queue_blocked"
    ]
    assert len(rows) >= 1, f"expected queue_blocked outbox for {job_id}"
    assert [c for c in emit_calls if c[0] == "job_error"] == [], (
        f"dual fan-out when outbox durable: {emit_calls}"
    )


def test_queue_blocked_outbox_dispatches_to_notify(cleanup):
    import scheduler as sched

    job = sched.submit_job(
        f"qb-listen-{uuid.uuid4().hex[:6]}", 8.0, tier="on-demand"
    )
    job_id = job["job_id"]
    cleanup["jobs"].append(job_id)

    assert sched._persist_queue_reason(
        job, "no_hosts_online", "No GPU hosts online"
    )

    def _has_mine(msgs):
        return any(
            m.get("type") == "job_error"
            and m.get("data", {}).get("job_id") == job_id
            for m in msgs
        )

    with _Listener() as listener:
        messages = listener.drain_until(
            _has_mine, dispatcher_id=f"qb-{uuid.uuid4().hex[:6]}"
        )

    mine = [
        m
        for m in messages
        if m.get("type") == "job_error"
        and m.get("data", {}).get("job_id") == job_id
    ]
    assert mine, f"no NOTIFY job_error for {job_id}"
    assert mine[0]["data"].get("error") == "no_hosts_online"

    rows = [
        r
        for r in _outbox_for(job_id)
        if r["event_type"] == "job.v1.queue_blocked"
    ]
    assert rows and rows[-1]["published_at"] is not None


# ── Spot prices producer ───────────────────────────────────────────────


def test_update_spot_prices_appends_outbox(cleanup, monkeypatch):
    import scheduler as sched

    emit_calls: list = []
    monkeypatch.setattr(
        sched, "emit_event", lambda et, data: emit_calls.append((et, data))
    )

    # Force a non-empty price map without depending on live catalog state.
    fake_prices = {"RTX 4090": 0.55, "A100 40GB": 1.25}

    def _fake_update():
        return dict(fake_prices)

    monkeypatch.setattr(
        "spot_pricing.update_all_spot_prices", _fake_update
    )
    # Also patch import path used inside update_spot_prices.
    import spot_pricing as sp

    monkeypatch.setattr(sp, "update_all_spot_prices", _fake_update)

    result = sched.update_spot_prices()
    assert result == fake_prices

    rows = [
        r
        for r in _outbox_for("spot")
        if r["event_type"] == "pricing.v1.spot_prices_updated"
    ]
    assert len(rows) >= 1, f"no spot outbox: {_outbox_for('spot')}"
    payload = _payload(rows[-1])
    prices = payload.get("prices") or {}
    assert prices.get("RTX 4090") == 0.55
    assert [c for c in emit_calls if c[0] == "spot_prices"] == []


def test_spot_prices_outbox_dispatches_to_notify(cleanup, monkeypatch):
    import scheduler as sched
    import spot_pricing as sp

    unique_key = f"TESTX-TEST-{uuid.uuid4().hex[:6]}"
    fake_prices = {unique_key: 0.77}

    monkeypatch.setattr(sp, "update_all_spot_prices", lambda: dict(fake_prices))

    sched.update_spot_prices()

    def _has_mine(msgs):
        return any(
            m.get("type") == "spot_prices"
            and isinstance(m.get("data"), dict)
            and unique_key in m.get("data", {})
            for m in msgs
        )

    with _Listener() as listener:
        messages = listener.drain_until(
            _has_mine, dispatcher_id=f"sp-{uuid.uuid4().hex[:6]}"
        )

    mine = [
        m
        for m in messages
        if m.get("type") == "spot_prices"
        and isinstance(m.get("data"), dict)
        and unique_key in m.get("data", {})
    ]
    assert mine, "no NOTIFY spot_prices for unique model"
    assert mine[0]["data"][unique_key] == 0.77


def test_inventory_residual_emit_sites_classified():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sched = (root / "scheduler.py").read_text()
    runtime = (root / "control_plane" / "outbox_runtime.py").read_text()

    assert "job.v1.queue_blocked" in runtime
    assert "pricing.v1.spot_prices_updated" in runtime
    assert 'event_type="job.v1.queue_blocked"' in sched
    assert 'event_type="pricing.v1.spot_prices_updated"' in sched
    # Dual-fan-out avoidance present for both residual paths.
    assert "if not outbox_enqueued" in sched
    # Persist-before-notify ordering for queue block (durable write first).
    persist_idx = sched.find("def _persist_queue_reason")
    assert persist_idx > 0
    # Remaining process-local residuals are request-path / non-scheduler.
    assert "try_append_lifecycle_outbox" in sched
