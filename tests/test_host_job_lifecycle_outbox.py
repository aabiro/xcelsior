"""Host/job lifecycle SSE via durable outbox (multi-replica).

Drives shipped scheduler producers that previously only process-local
``emit_event``'d: host register/drain/remove, job submit, job preempt.
Asserts outbox rows + projection shape + real LISTEN delivery for at
least one type. Process-local dual fan-out is not required when outbox
succeeds.
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
    try_append_lifecycle_outbox,
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
        conn.commit()


def _outbox_for(aggregate_id: str) -> list[dict]:
    with _pool.connection() as conn:
        rows = conn.execute(
            """SELECT event_type, payload, published_at, idempotency_key,
                      destination_class
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
                    "destination_class": r[4],
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

    def collect(self, max_wait: float = 3.0, *, stop_on_first: bool = False) -> list[dict]:
        got: list[dict] = []
        deadline = time.time() + max_wait
        while time.time() < deadline:
            remaining = max(0.02, deadline - time.time())
            if select.select([self.conn.fileno()], [], [], remaining) == ([], [], []):
                if got and stop_on_first:
                    break
                continue
            for notice in self.conn.notifies(timeout=0.05):
                got.append(json.loads(notice.payload))
            if got and stop_on_first:
                break
        return got

    def drain_until(
        self,
        predicate,
        *,
        dispatcher_id: str,
        max_cycles: int = 40,
    ) -> list[dict]:
        """Run dispatcher cycles until predicate(messages) is true or budget ends."""
        all_msgs: list[dict] = []
        dispatcher = OutboxDispatcher(dispatcher_id, default_handlers())
        for _ in range(max_cycles):
            dispatcher.run_once()
            all_msgs.extend(self.collect(max_wait=0.25))
            if predicate(all_msgs):
                # One more short collect for trailing notifies.
                all_msgs.extend(self.collect(max_wait=0.15))
                break
        return all_msgs


# ── Pure projection ────────────────────────────────────────────────────


def test_sse_payload_projections_for_host_and_job_lifecycle():
    host_evt = OutboxEvent(
        event_id="e-h",
        aggregate_type="host",
        aggregate_id="h1",
        event_type="host.v1.status_changed",
        payload={"status": "draining"},
        headers={},
        destination_class="default",
        idempotency_key="k",
        attempt_count=1,
    )
    msg = sse_payload_for(host_evt)
    assert msg is not None
    assert msg["type"] == "host_update"
    assert msg["data"]["host_id"] == "h1"
    assert msg["data"]["status"] == "draining"

    rem = OutboxEvent(
        event_id="e-r",
        aggregate_type="host",
        aggregate_id="h2",
        event_type="host.v1.removed",
        payload={},
        headers={},
        destination_class="default",
        idempotency_key="k2",
        attempt_count=1,
    )
    assert sse_payload_for(rem) == {
        "type": "host_removed",
        "data": {"host_id": "h2"},
    }

    sub = OutboxEvent(
        event_id="e-s",
        aggregate_type="job",
        aggregate_id="j1",
        event_type="job.v1.submitted",
        payload={"name": "n", "tier": "on-demand"},
        headers={},
        destination_class="default",
        idempotency_key="k3",
        attempt_count=1,
    )
    msg_s = sse_payload_for(sub)
    assert msg_s is not None
    assert msg_s["type"] == "job_submitted"
    assert msg_s["data"]["job_id"] == "j1"
    assert msg_s["data"]["name"] == "n"

    pre = OutboxEvent(
        event_id="e-p",
        aggregate_type="job",
        aggregate_id="j2",
        event_type="job.v1.preempted",
        payload={"name": "spotty", "preemption_count": 2},
        headers={},
        destination_class="default",
        idempotency_key="k4",
        attempt_count=1,
    )
    msg_p = sse_payload_for(pre)
    assert msg_p is not None
    assert msg_p["type"] == "job_preempted"
    assert msg_p["data"]["job_id"] == "j2"
    assert msg_p["data"]["preemption_count"] == 2


def test_try_append_lifecycle_outbox_savepoint_isolates_failure(cleanup, monkeypatch):
    """Bad append must not leave the open txn aborted."""
    import control_plane.outbox as outbox_mod

    host_id = f"h-sp-{uuid.uuid4().hex[:8]}"
    cleanup["hosts"].append(host_id)

    def _boom(conn, **kwargs):
        conn.execute("SELECT 1 FROM definitely_no_such_lifecycle_outbox_table")

    monkeypatch.setattr(outbox_mod, "append_event", _boom)

    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload)
               VALUES (%s, 'active', %s, %s)""",
            (host_id, time.time(), json.dumps({"host_id": host_id})),
        )
        ok = try_append_lifecycle_outbox(
            conn,
            aggregate_type="host",
            aggregate_id=host_id,
            event_type="host.v1.status_changed",
            payload={"status": "active"},
            idempotency_key=f"test-sp:{host_id}",
        )
        assert ok is False
        # Outer txn still usable — host insert must commit.
        conn.commit()

    with _pool.connection() as conn:
        row = conn.execute(
            "SELECT host_id FROM hosts WHERE host_id=%s", (host_id,)
        ).fetchone()
    assert row is not None


# ── Shipped producer entry points ──────────────────────────────────────


def test_register_host_appends_host_status_outbox(cleanup, monkeypatch):
    import scheduler as sched

    host_id = f"h-reg-{uuid.uuid4().hex[:8]}"
    cleanup["hosts"].append(host_id)
    emit_calls: list = []
    monkeypatch.setattr(
        sched, "emit_event", lambda et, data: emit_calls.append((et, data))
    )

    host = sched.register_host(
        host_id, "10.0.0.9", "RTX 4090", 24.0, 24.0, cost_per_hour=0.5
    )
    assert host is not None
    assert host["host_id"] == host_id

    rows = [
        r
        for r in _outbox_for(host_id)
        if r["event_type"] == "host.v1.status_changed"
    ]
    assert len(rows) >= 1, f"no host status outbox: {_outbox_for(host_id)}"
    assert _payload(rows[-1]).get("status") in ("active", "draining", "pending")
    assert rows[-1]["published_at"] is None
    # Outbox won — no process-local dual fan-out required.
    host_emits = [c for c in emit_calls if c[0] == "host_update"]
    assert host_emits == [], f"unexpected dual emit: {emit_calls}"


def test_set_host_draining_appends_outbox(cleanup, monkeypatch):
    import scheduler as sched

    host_id = f"h-drain-{uuid.uuid4().hex[:8]}"
    cleanup["hosts"].append(host_id)
    sched.register_host(host_id, "10.0.0.10", "A100", 40.0, 40.0)

    emit_calls: list = []
    monkeypatch.setattr(
        sched, "emit_event", lambda et, data: emit_calls.append((et, data))
    )

    host = sched.set_host_draining(host_id, True)
    assert host is not None
    assert host["status"] == "draining"

    rows = [
        r
        for r in _outbox_for(host_id)
        if r["event_type"] == "host.v1.status_changed"
        and _payload(r).get("status") == "draining"
    ]
    assert rows, f"no draining outbox row: {_outbox_for(host_id)}"
    assert emit_calls == []


def test_remove_host_appends_removed_outbox(cleanup, monkeypatch):
    import scheduler as sched

    host_id = f"h-rm-{uuid.uuid4().hex[:8]}"
    cleanup["hosts"].append(host_id)
    sched.register_host(host_id, "10.0.0.11", "A100", 40.0, 40.0)

    emit_calls: list = []
    monkeypatch.setattr(
        sched, "emit_event", lambda et, data: emit_calls.append((et, data))
    )

    sched.remove_host(host_id)
    rows = [
        r for r in _outbox_for(host_id) if r["event_type"] == "host.v1.removed"
    ]
    assert len(rows) >= 1
    assert emit_calls == []


def test_submit_job_appends_submitted_outbox(cleanup, monkeypatch):
    import scheduler as sched

    emit_calls: list = []
    monkeypatch.setattr(
        sched, "emit_event", lambda et, data: emit_calls.append((et, data))
    )

    job = sched.submit_job("lifecycle-outbox-job", 8.0, tier="on-demand")
    job_id = job["job_id"]
    cleanup["jobs"].append(job_id)

    rows = [
        r for r in _outbox_for(job_id) if r["event_type"] == "job.v1.submitted"
    ]
    assert len(rows) == 1
    payload = _payload(rows[0])
    assert payload.get("name") == "lifecycle-outbox-job"
    assert payload.get("tier") == "on-demand"
    assert rows[0]["published_at"] is None
    assert [c for c in emit_calls if c[0] == "job_submitted"] == []


def test_preempt_job_appends_preempted_outbox(cleanup, monkeypatch):
    import scheduler as sched

    host_id = f"h-pre-{uuid.uuid4().hex[:8]}"
    cleanup["hosts"].append(host_id)
    sched.register_host(host_id, "10.0.0.12", "A100", 40.0, 40.0)

    job = sched.submit_job(
        "spot-preempt-outbox",
        8.0,
        pricing_mode="spot",
        owner="spot-owner@test",
    )
    job_id = job["job_id"]
    cleanup["jobs"].append(job_id)

    # Force running on host (spot preemption only acts on running jobs).
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET status='running', host_id=%s WHERE job_id=%s",
            (host_id, job_id),
        )
        conn.execute(
            """UPDATE jobs SET payload = payload
                   || jsonb_build_object(
                        'status', 'running',
                        'host_id', %s::text,
                        'spot', true,
                        'pricing_mode', 'spot',
                        'preemptible', true
                   )
               WHERE job_id=%s""",
            (host_id, job_id),
        )
        conn.commit()

    emit_calls: list = []
    monkeypatch.setattr(
        sched, "emit_event", lambda et, data: emit_calls.append((et, data))
    )

    result = sched.preempt_job(job_id)
    assert result is not None

    rows = [
        r for r in _outbox_for(job_id) if r["event_type"] == "job.v1.preempted"
    ]
    assert len(rows) >= 1, f"no preempt outbox: {_outbox_for(job_id)}"
    payload = _payload(rows[-1])
    assert payload.get("previous_host_id") == host_id
    assert [c for c in emit_calls if c[0] == "job_preempted"] == []


def test_submit_job_outbox_dispatches_to_notify(cleanup):
    """End-to-end: submit → outbox row → dispatcher → LISTEN."""
    import scheduler as sched

    job = sched.submit_job("dispatch-listen-job", 4.0, tier="on-demand")
    job_id = job["job_id"]
    cleanup["jobs"].append(job_id)

    def _has_mine(msgs: list[dict]) -> bool:
        return any(
            m.get("type") == "job_submitted"
            and m.get("data", {}).get("job_id") == job_id
            for m in msgs
        )

    with _Listener() as listener:
        messages = listener.drain_until(
            _has_mine,
            dispatcher_id=f"life-{uuid.uuid4().hex[:6]}",
        )

    mine = [
        m
        for m in messages
        if m.get("type") == "job_submitted"
        and m.get("data", {}).get("job_id") == job_id
    ]
    assert mine, f"no NOTIFY/SSE for submitted job {job_id} after drain"
    assert mine[0]["data"].get("name") == "dispatch-listen-job"

    rows = [
        r for r in _outbox_for(job_id) if r["event_type"] == "job.v1.submitted"
    ]
    assert rows and rows[-1]["published_at"] is not None


def test_host_drain_outbox_dispatches_to_notify(cleanup):
    import scheduler as sched

    host_id = f"h-disp-{uuid.uuid4().hex[:8]}"
    cleanup["hosts"].append(host_id)
    sched.register_host(host_id, "10.0.0.13", "A100", 40.0, 40.0)
    # Drain so we can key the LISTEN message on status=draining.
    sched.set_host_draining(host_id, True)

    def _has_mine(msgs: list[dict]) -> bool:
        return any(
            m.get("type") == "host_update"
            and m.get("data", {}).get("host_id") == host_id
            and m.get("data", {}).get("status") == "draining"
            for m in msgs
        )

    with _Listener() as listener:
        messages = listener.drain_until(
            _has_mine,
            dispatcher_id=f"life-h-{uuid.uuid4().hex[:6]}",
        )

    mine = [
        m
        for m in messages
        if m.get("type") == "host_update"
        and m.get("data", {}).get("host_id") == host_id
        and m.get("data", {}).get("status") == "draining"
    ]
    assert mine, f"no NOTIFY/SSE for host drain {host_id} after drain"


def test_inventory_process_local_emit_sites_classified():
    """Static inventory: lifecycle sites use outbox helper; residuals noted."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sched_src = (root / "scheduler.py").read_text()
    runtime_src = (root / "control_plane" / "outbox_runtime.py").read_text()

    assert "try_append_lifecycle_outbox" in runtime_src
    assert "host.v1.status_changed" in runtime_src
    assert "host.v1.removed" in runtime_src
    assert "job.v1.submitted" in runtime_src
    assert "job.v1.preempted" in runtime_src

    for marker in (
        'event_type="host.v1.status_changed"',
        'event_type="host.v1.removed"',
        'event_type="job.v1.submitted"',
        'event_type="job.v1.preempted"',
    ):
        assert marker in sched_src, f"missing wired producer: {marker}"

    # Still process-local (not this cutover): queue-block job_error, spot_prices.
    assert 'emit_event(\n            "job_error"' in sched_src or 'emit_event(\n        "job_error"' in sched_src or '"job_error"' in sched_src
    assert "spot_prices" in sched_src
