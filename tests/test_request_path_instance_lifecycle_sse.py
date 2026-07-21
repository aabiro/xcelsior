"""Request-path instance lifecycle SSE via durable outbox (multi-replica).

Drives shipped ``routes.instances._broadcast_instance_lifecycle_sse`` and
projections for stop/start/restart/terminate/cancel. Asserts outbox row
presence, dual-fan-out avoidance when durable, and real LISTEN delivery
for at least one type.
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
    enqueue_lifecycle_sse_outbox,
    sse_payload_for,
)


@pytest.fixture
def cleanup():
    ids = {"jobs": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
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
        all_msgs: list[dict] = []
        dispatcher = OutboxDispatcher(dispatcher_id, default_handlers())
        for _ in range(max_cycles):
            dispatcher.run_once()
            all_msgs.extend(self.collect(max_wait=0.25))
            if predicate(all_msgs):
                all_msgs.extend(self.collect(max_wait=0.15))
                break
        return all_msgs


# ── Pure projections ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "event_type,sse_type",
    [
        ("job.v1.instance_stopped", "instance_stopped"),
        ("job.v1.instance_started", "instance_started"),
        ("job.v1.instance_restarted", "instance_restarted"),
        ("job.v1.instance_terminated", "instance_terminated"),
        ("job.v1.cancelled", "job_cancelled"),
    ],
)
def test_sse_payload_projections_for_instance_lifecycle(event_type, sse_type):
    evt = OutboxEvent(
        event_id="e-il",
        aggregate_type="job",
        aggregate_id="j-il-1",
        event_type=event_type,
        payload={"status": "stopping"} if event_type == "job.v1.cancelled" else {},
        headers={},
        destination_class="default",
        idempotency_key="k",
        attempt_count=1,
    )
    msg = sse_payload_for(evt)
    assert msg is not None
    assert msg["type"] == sse_type
    assert msg["data"]["job_id"] == "j-il-1"
    if event_type == "job.v1.cancelled":
        assert msg["data"]["status"] == "stopping"


# ── enqueue helper ─────────────────────────────────────────────────────


def test_enqueue_lifecycle_sse_outbox_appends_row(cleanup):
    job_id = f"j-rpsse-{uuid.uuid4().hex[:10]}"
    cleanup["jobs"].append(job_id)
    ok = enqueue_lifecycle_sse_outbox(
        aggregate_type="job",
        aggregate_id=job_id,
        event_type="job.v1.instance_stopped",
        payload={"job_id": job_id},
        idempotency_key=f"instance_stopped:{job_id}:test",
    )
    assert ok is True
    rows = [
        r
        for r in _outbox_for(job_id)
        if r["event_type"] == "job.v1.instance_stopped"
    ]
    assert len(rows) == 1
    assert rows[0]["published_at"] is None
    assert rows[0]["destination_class"] == "default"
    assert _payload(rows[0]).get("job_id") == job_id

    # Idempotent re-append with same key does not create a second row.
    ok2 = enqueue_lifecycle_sse_outbox(
        aggregate_type="job",
        aggregate_id=job_id,
        event_type="job.v1.instance_stopped",
        payload={"job_id": job_id},
        idempotency_key=f"instance_stopped:{job_id}:test",
    )
    assert ok2 is True
    rows2 = [
        r
        for r in _outbox_for(job_id)
        if r["event_type"] == "job.v1.instance_stopped"
    ]
    assert len(rows2) == 1


# ── Shipped request-path helper ────────────────────────────────────────


def test_broadcast_instance_lifecycle_sse_uses_outbox_not_dual_emit(
    cleanup, monkeypatch
):
    import routes.instances as inst

    job_id = f"j-rpsse-h-{uuid.uuid4().hex[:10]}"
    cleanup["jobs"].append(job_id)
    local_calls: list = []
    monkeypatch.setattr(
        inst, "broadcast_sse", lambda et, data: local_calls.append((et, data))
    )

    inst._broadcast_instance_lifecycle_sse(
        "instance_started",
        {"job_id": job_id},
        event_type="job.v1.instance_started",
        job_id=job_id,
        idempotency_key=f"instance_started:{job_id}:helper",
    )

    rows = [
        r
        for r in _outbox_for(job_id)
        if r["event_type"] == "job.v1.instance_started"
    ]
    assert len(rows) == 1, f"no outbox: {_outbox_for(job_id)}"
    assert local_calls == [], f"unexpected dual process-local emit: {local_calls}"


def test_broadcast_instance_lifecycle_sse_falls_back_when_outbox_fails(
    cleanup, monkeypatch
):
    import routes.instances as inst

    job_id = f"j-rpsse-fb-{uuid.uuid4().hex[:10]}"
    cleanup["jobs"].append(job_id)
    local_calls: list = []
    monkeypatch.setattr(
        inst, "broadcast_sse", lambda et, data: local_calls.append((et, data))
    )
    monkeypatch.setattr(
        "control_plane.outbox_runtime.enqueue_lifecycle_sse_outbox",
        lambda **kwargs: False,
    )

    inst._broadcast_instance_lifecycle_sse(
        "instance_terminated",
        {"job_id": job_id},
        event_type="job.v1.instance_terminated",
        job_id=job_id,
        idempotency_key=f"instance_terminated:{job_id}:fb",
    )

    assert local_calls == [
        ("instance_terminated", {"job_id": job_id})
    ], local_calls


@pytest.mark.parametrize(
    "sse_event,event_type",
    [
        ("instance_stopped", "job.v1.instance_stopped"),
        ("instance_restarted", "job.v1.instance_restarted"),
        ("job_cancelled", "job.v1.cancelled"),
    ],
)
def test_broadcast_covers_remaining_lifecycle_types(
    cleanup, monkeypatch, sse_event, event_type
):
    import routes.instances as inst

    job_id = f"j-rpsse-p-{uuid.uuid4().hex[:8]}"
    cleanup["jobs"].append(job_id)
    monkeypatch.setattr(inst, "broadcast_sse", lambda *a, **k: None)

    payload = {"job_id": job_id}
    if sse_event == "job_cancelled":
        payload["status"] = "stopping"

    inst._broadcast_instance_lifecycle_sse(
        sse_event,
        payload,
        event_type=event_type,
        job_id=job_id,
        idempotency_key=f"{sse_event}:{job_id}:param",
    )
    rows = [r for r in _outbox_for(job_id) if r["event_type"] == event_type]
    assert len(rows) == 1


# ── LISTEN e2e ─────────────────────────────────────────────────────────


def test_dispatcher_delivers_instance_stopped_via_listen(cleanup):
    job_id = f"j-rpsse-listen-{uuid.uuid4().hex[:10]}"
    cleanup["jobs"].append(job_id)

    ok = enqueue_lifecycle_sse_outbox(
        aggregate_type="job",
        aggregate_id=job_id,
        event_type="job.v1.instance_stopped",
        payload={"job_id": job_id},
        idempotency_key=f"instance_stopped:{job_id}:listen",
    )
    assert ok is True

    with _Listener() as listener:
        msgs = listener.drain_until(
            lambda m: any(
                x.get("type") == "instance_stopped"
                and (x.get("data") or {}).get("job_id") == job_id
                for x in m
            ),
            dispatcher_id=f"test-rpsse-{uuid.uuid4().hex[:8]}",
        )

    hits = [
        m
        for m in msgs
        if m.get("type") == "instance_stopped"
        and (m.get("data") or {}).get("job_id") == job_id
    ]
    assert hits, f"no instance_stopped NOTIFY; got {msgs!r}"


def test_dispatcher_delivers_job_cancelled_via_listen(cleanup):
    job_id = f"j-rpsse-cancel-{uuid.uuid4().hex[:10]}"
    cleanup["jobs"].append(job_id)

    ok = enqueue_lifecycle_sse_outbox(
        aggregate_type="job",
        aggregate_id=job_id,
        event_type="job.v1.cancelled",
        payload={"job_id": job_id, "status": "stopping"},
        idempotency_key=f"job_cancelled:{job_id}:listen",
    )
    assert ok is True

    with _Listener() as listener:
        msgs = listener.drain_until(
            lambda m: any(
                x.get("type") == "job_cancelled"
                and (x.get("data") or {}).get("job_id") == job_id
                for x in m
            ),
            dispatcher_id=f"test-rpsse-c-{uuid.uuid4().hex[:8]}",
        )

    hits = [
        m
        for m in msgs
        if m.get("type") == "job_cancelled"
        and (m.get("data") or {}).get("job_id") == job_id
    ]
    assert hits, f"no job_cancelled NOTIFY; got {msgs!r}"
    assert hits[0]["data"].get("status") == "stopping"


# ── Structural inventory ───────────────────────────────────────────────


def test_instance_lifecycle_routes_use_durable_helper():
    import inspect

    import routes.instances as inst

    src = inspect.getsource(inst)
    # Wired request-path sites use the helper, not raw broadcast_sse.
    for needle in (
        "instance_stopped",
        "instance_started",
        "instance_restarted",
        "instance_terminated",
        "job_cancelled",
    ):
        # At least one helper call mentioning the SSE vocabulary remains.
        assert (
            f'"{needle}"' in src or f"'{needle}'" in src
        ), f"missing {needle} reference in routes.instances"
    assert "def _broadcast_instance_lifecycle_sse" in src
    assert "enqueue_lifecycle_sse_outbox" in src
    # Stop/start/restart/terminate handlers must not call bare broadcast_sse
    # for the lifecycle vocabulary after the mutation success path.
    stop_src = inspect.getsource(inst.api_stop_instance)
    start_src = inspect.getsource(inst.api_start_instance)
    restart_src = inspect.getsource(inst.api_restart_instance)
    term_src = inspect.getsource(inst.api_terminate_instance)
    for name, body in (
        ("stop", stop_src),
        ("start", start_src),
        ("restart", restart_src),
        ("terminate", term_src),
    ):
        assert "_broadcast_instance_lifecycle_sse" in body, name
        assert "broadcast_sse(" not in body, f"{name} still uses bare broadcast_sse"

    # Cancel route: both fenced and legacy paths use the helper.
    # Find cancel function by scanning for the job_cancelled event_type wiring.
    assert 'event_type="job.v1.cancelled"' in src
    assert src.count('event_type="job.v1.cancelled"') >= 2
