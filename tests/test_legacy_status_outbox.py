"""Legacy update_job_status side effects via durable outbox.

Drives the *shipped* ``scheduler.update_job_status`` path: status mutation
and outbox append share one transaction; the existing outbox dispatcher
projects ``job.v1.legacy_status_changed`` onto ``xcelsior_events`` NOTIFY
(the same channel API replicas LISTEN for SSE).
"""

from __future__ import annotations

import os
os.environ["XCELSIOR_DB_BACKEND"] = "postgres"

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
def cleanup_job():
    ids: list[str] = []
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        conn.commit()


def _insert_queued_job(job_id: str) -> None:
    payload = {
        "job_id": job_id,
        "name": job_id,
        "status": "queued",
        "priority": 0,
        "vram_needed_gb": 8.0,
        "submitted_at": time.time(),
    }
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload)
               VALUES (%s, 'queued', 0, %s, %s)""",
            (job_id, time.time(), json.dumps(payload)),
        )
        conn.commit()


def _outbox_rows(job_id: str) -> list[dict]:
    with _pool.connection() as conn:
        rows = conn.execute(
            """SELECT event_type, payload, published_at, idempotency_key
                 FROM outbox_events
                WHERE aggregate_id=%s
                ORDER BY created_at""",
            (job_id,),
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


class _Listener:
    def __enter__(self):
        self.conn = psycopg.connect(resolve_postgres_dsn(), autocommit=True)
        self.conn.execute(f"LISTEN {EVENTS_CHANNEL}")
        return self

    def __exit__(self, *exc):
        self.conn.close()

    def collect(self, max_wait: float = 3.0) -> list[dict]:
        got: list[dict] = []
        deadline = time.time() + max_wait
        while time.time() < deadline:
            remaining = max(0.05, deadline - time.time())
            if select.select([self.conn.fileno()], [], [], remaining) == ([], [], []):
                break
            for notice in self.conn.notifies(timeout=0.2):
                got.append(json.loads(notice.payload))
            if got:
                break
        return got


class TestLegacyStatusOutboxProjection:
    def test_sse_payload_for_legacy_status(self):
        evt = OutboxEvent(
            event_id="e1",
            aggregate_type="job",
            aggregate_id="j-leg",
            event_type="job.v1.legacy_status_changed",
            payload={
                "status": "running",
                "previous_status": "assigned",
                "host_id": "h1",
            },
            headers={},
            destination_class="default",
            idempotency_key="k",
            attempt_count=1,
        )
        msg = sse_payload_for(evt)
        assert msg is not None
        assert msg["type"] == "job_status"
        assert msg["data"]["job_id"] == "j-leg"
        assert msg["data"]["status"] == "running"
        assert msg["data"]["host_id"] == "h1"
        assert msg["data"]["previous_status"] == "assigned"


class TestUpdateJobStatusEnqueuesOutbox:
    def test_update_job_status_appends_outbox_row(self, cleanup_job):
        """Shipped update_job_status must append job.v1.legacy_status_changed."""
        import scheduler as sched

        job_id = f"j-stout-{uuid.uuid4().hex[:8]}"
        cleanup_job.append(job_id)
        _insert_queued_job(job_id)

        updated = sched.update_job_status(job_id, "assigned", host_id="h-stout-1")
        assert updated is not None
        assert updated.get("status") == "assigned"

        rows = _outbox_rows(job_id)
        legacy = [
            r for r in rows if r["event_type"] == "job.v1.legacy_status_changed"
        ]
        assert len(legacy) >= 1, f"no outbox row for {job_id}: {rows}"
        payload = legacy[-1]["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        assert payload.get("status") == "assigned"
        assert payload.get("host_id") == "h-stout-1"
        assert payload.get("previous_status") == "queued"
        # Not yet published — dispatcher owns delivery.
        assert legacy[-1]["published_at"] is None

    def test_update_job_status_dispatches_to_notify(self, cleanup_job):
        """Outbox intent from real status path → dispatcher → LISTEN payload."""
        import scheduler as sched

        with _pool.connection() as conn:
            conn.execute("DELETE FROM outbox_events")
            conn.commit()

        job_id = f"j-stout-d-{uuid.uuid4().hex[:8]}"
        cleanup_job.append(job_id)
        _insert_queued_job(job_id)

        sched.update_job_status(job_id, "assigned", host_id="h-stout-d")


        with _Listener() as listener:
            stats = OutboxDispatcher(
                f"stout-{uuid.uuid4().hex[:6]}", default_handlers()
            ).run_once()
            messages = listener.collect()

        assert stats["published"] >= 1
        mine = [
            m
            for m in messages
            if m.get("type") == "job_status"
            and m.get("data", {}).get("job_id") == job_id
        ]
        assert mine, f"no NOTIFY/SSE for {job_id}: {messages}"
        assert mine[0]["data"]["status"] == "assigned"
        assert mine[0]["data"]["host_id"] == "h-stout-d"

        rows = _outbox_rows(job_id)
        legacy = [
            r for r in rows if r["event_type"] == "job.v1.legacy_status_changed"
        ]
        assert legacy and legacy[-1]["published_at"] is not None

    def test_idempotent_repeat_running_does_not_double_append(self, cleanup_job):
        """running→running early-return must not invent a second outbox row."""
        import scheduler as sched

        job_id = f"j-stout-i-{uuid.uuid4().hex[:8]}"
        cleanup_job.append(job_id)
        _insert_queued_job(job_id)
        sched.update_job_status(job_id, "assigned", host_id="h-i")
        # Force payload path to running (skip invalid transition noise).
        with _pool.connection() as conn:
            conn.execute(
                "UPDATE jobs SET status='running' WHERE job_id=%s", (job_id,)
            )
            conn.execute(
                """UPDATE jobs SET payload =
                       jsonb_set(payload, '{status}', '"running"')
                    WHERE job_id=%s""",
                (job_id,),
            )
            conn.commit()
        before = len(_outbox_rows(job_id))
        again = sched.update_job_status(job_id, "running", host_id="h-i")
        assert again is not None
        after = len(_outbox_rows(job_id))
        # Idempotent path returns without append.
        assert after == before

    def test_append_sql_failure_savepoint_status_still_commits(self, cleanup_job, monkeypatch):
        """Skeptic regression: bad SQL in append must not split-brain.

        Forces append_event to run invalid SQL on the *same* connection used
        by the open mutation. Without SAVEPOINT the job upsert would roll
        back while update_job_status returned the in-memory new status.
        With SAVEPOINT: status commits, outbox not durable, emit_event
        fallback path is taken (outbox_enqueued=False).
        """
        import scheduler as sched
        import control_plane.outbox as outbox_mod

        job_id = f"j-stout-sp-{uuid.uuid4().hex[:8]}"
        cleanup_job.append(job_id)
        _insert_queued_job(job_id)

        def _boom_append(conn, **kwargs):
            # Abort the open transaction with real SQL unless SAVEPOINT
            # isolates it — this is the failure mode under test.
            conn.execute("SELECT 1 FROM definitely_no_such_table_outbox_sp")

        monkeypatch.setattr(outbox_mod, "append_event", _boom_append)

        emit_calls: list = []
        monkeypatch.setattr(
            sched,
            "emit_event",
            lambda et, data: emit_calls.append((et, data)),
        )

        updated = sched.update_job_status(job_id, "assigned", host_id="h-sp")
        assert updated is not None
        assert updated.get("status") == "assigned"

        # Durable truth: job row must match the returned status (no split-brain).
        with _pool.connection() as conn:
            row = conn.execute(
                "SELECT status, host_id FROM jobs WHERE job_id=%s", (job_id,)
            ).fetchone()
        status = row[0] if not isinstance(row, dict) else row["status"]
        host = row[1] if not isinstance(row, dict) else row["host_id"]
        assert status == "assigned", f"DB status {status!r} != returned assigned"
        assert host == "h-sp"

        # Append failed → no durable outbox row; emit_event fallback used.
        legacy = [
            r
            for r in _outbox_rows(job_id)
            if r["event_type"] == "job.v1.legacy_status_changed"
        ]
        assert legacy == []
        assert any(
            et == "job_status" and data.get("job_id") == job_id
            for et, data in emit_calls
        )
