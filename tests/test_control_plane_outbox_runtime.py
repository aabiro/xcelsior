"""Phase 7 — outbox dispatcher runtime: SSE projection, NOTIFY delivery,
settlement, retention. Runs against the real test PostgreSQL with a real
LISTEN connection asserting what API replicas would receive.
"""

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
            _c.execute("SELECT to_regclass('outbox_events')").fetchone()[0] is not None
        )
except Exception as _e:  # pragma: no cover - skip path
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None
else:
    if not _migrated:  # pragma: no cover - skip path
        pytestmark = pytest.mark.skip("test database not migrated to >= 056")

import psycopg

from control_plane.db import run_transaction
from control_plane.outbox import OutboxDispatcher, OutboxEvent, append_event
from control_plane.outbox_runtime import (
    EVENTS_CHANNEL,
    default_handlers,
    prune_settled_events,
    sse_payload_for,
)


@pytest.fixture
def cleanup():
    ids = {"aggregates": [], "jobs": [], "hosts": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for agg in ids["aggregates"]:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (agg,))
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (jid,))
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


class _Listener:
    """Real LISTEN connection — what every API replica runs."""

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


def _mk_event(cleanup, event_type, payload, *, destination="default"):
    agg = f"job-obx-{uuid.uuid4().hex[:8]}"
    cleanup["aggregates"].append(agg)
    run_transaction(
        lambda c: append_event(
            c,
            aggregate_type="job",
            aggregate_id=agg,
            event_type=event_type,
            payload=payload,
            destination_class=destination,
            idempotency_key=f"test:{agg}:{event_type}",
        ),
        what="test_append",
    )
    return agg


def _event_row(agg):
    with _pool.connection() as conn:
        row = conn.execute(
            "SELECT published_at, dead_lettered_at, attempt_count "
            "FROM outbox_events WHERE aggregate_id=%s",
            (agg,),
        ).fetchone()
    if row is None or isinstance(row, dict):
        return row
    return {"published_at": row[0], "dead_lettered_at": row[1], "attempt_count": row[2]}


class TestSseProjection:
    def _evt(self, event_type, payload):
        return OutboxEvent(
            event_id="e1", aggregate_type="job", aggregate_id="job-x",
            event_type=event_type, payload=payload, headers={},
            destination_class="default", idempotency_key="k", attempt_count=1,
        )

    def test_placement_reserved(self):
        msg = sse_payload_for(
            self._evt("job.v1.placement_reserved", {"host_id": "h1", "attempt_id": "a1"})
        )
        assert msg == {
            "type": "job_status",
            "data": {"job_id": "job-x", "status": "assigned", "host_id": "h1",
                     "attempt_id": "a1"},
        }

    def test_attempt_status_vocabulary(self):
        for reported, legacy in (
            ("lease_claimed", "leased"), ("starting", "starting"),
            ("running", "running"), ("succeeded", "completed"),
            ("failed", "failed"),
        ):
            msg = sse_payload_for(
                self._evt("job.v1.attempt_status_changed",
                          {"status": reported, "host_id": "h1"})
            )
            assert msg is not None and msg["data"]["status"] == legacy

    def test_lease_expired_requeues(self):
        msg = sse_payload_for(
            self._evt("job.v1.lease_expired", {"reason": "lease_claim_timeout"})
        )
        assert msg is not None
        assert msg["data"]["status"] == "queued"
        assert msg["data"]["reason"] == "lease_claim_timeout"

    def test_unknown_event_has_no_projection(self):
        assert sse_payload_for(self._evt("job.v1.some_future_thing", {})) is None


class TestDispatcherDelivery:
    def test_delivers_notify_and_settles(self, cleanup):
        agg = _mk_event(
            cleanup, "job.v1.placement_reserved",
            {"host_id": "h-obx", "attempt_id": "a-obx"},
        )
        with _Listener() as listener:
            stats = OutboxDispatcher(
                f"test-{uuid.uuid4().hex[:6]}", default_handlers()
            ).run_once()
            messages = listener.collect()
        assert stats["published"] >= 1 and stats["unroutable"] == 0
        mine = [m for m in messages if m["data"].get("job_id") == agg]
        assert len(mine) == 1
        assert mine[0]["type"] == "job_status"
        assert mine[0]["data"]["status"] == "assigned"
        assert _event_row(agg)["published_at"] is not None

    def test_unknown_type_publishes_without_notify(self, cleanup):
        agg = _mk_event(cleanup, "job.v1.mystery", {"x": 1})
        with _Listener() as listener:
            OutboxDispatcher(
                f"test-{uuid.uuid4().hex[:6]}", default_handlers()
            ).run_once()
            messages = listener.collect(max_wait=1.0)
        assert all(m["data"].get("job_id") != agg for m in messages)
        assert _event_row(agg)["published_at"] is not None  # settled, not stuck

    def test_agent_wake_settles(self, cleanup):
        agg = _mk_event(
            cleanup, "job.v1.placement_reserved", {"host_id": "h"},
            destination="agent_wake",
        )
        OutboxDispatcher(f"test-{uuid.uuid4().hex[:6]}", default_handlers()).run_once()
        assert _event_row(agg)["published_at"] is not None

    def test_placement_to_sse_end_to_end(self, cleanup):
        """Reservation transaction → outbox → dispatcher → NOTIFY: the full
        Phase 7 side-effect path for a real placement."""
        from control_plane.scheduler.config import SchedulerConfig, SchedulerMode
        from control_plane.scheduler.service import SchedulerService

        marker = uuid.uuid4().hex[:8]
        model = f"RTX-{marker}"
        host_id = f"h-{marker}-1"
        job_id = f"j-{marker}-run"
        with _pool.connection() as conn:
            conn.execute(
                "INSERT INTO hosts (host_id, status, registered_at, payload) "
                "VALUES (%s, 'active', %s, %s)",
                (host_id, time.time(), json.dumps({
                    "host_id": host_id, "gpu_model": model, "gpu_count": 2,
                    "free_vram_gb": 48.0, "total_vram_gb": 48.0,
                    "cost_per_hour": 1.0, "admitted": True,
                    "last_seen": time.time(),
                })),
            )
            conn.execute(
                "INSERT INTO jobs (job_id, status, priority, submitted_at, payload) "
                "VALUES (%s, 'queued', 0, %s, %s)",
                (job_id, time.time(), json.dumps({
                    "job_id": job_id, "name": job_id, "gpu_model": model,
                    "num_gpus": 1, "vram_needed_gb": 8.0, "status": "queued",
                })),
            )
            conn.commit()
        cleanup["hosts"].append(host_id)
        cleanup["jobs"].append(job_id)

        cfg = SchedulerConfig(
            mode=SchedulerMode.CANARY, replica_id=f"obx-{marker}",
            canary_gpu_models=frozenset({model.lower()}),
            canary_host_ids=frozenset({host_id}),
        )
        report = SchedulerService(cfg).tick()
        assert len(report.placements) == 1

        with _Listener() as listener:
            OutboxDispatcher(f"test-{uuid.uuid4().hex[:6]}", default_handlers()).run_once()
            messages = listener.collect()
        mine = [m for m in messages if m["data"].get("job_id") == job_id]
        assert mine, f"no SSE projection delivered for {job_id}: {messages}"
        assert mine[0]["data"]["status"] == "assigned"
        assert mine[0]["data"]["host_id"] == host_id


class TestRetention:
    def test_prune_respects_settlement_and_age(self, cleanup):
        old_pub = _mk_event(cleanup, "job.v1.placement_reserved", {"n": 1})
        fresh_pub = _mk_event(cleanup, "job.v1.placement_reserved", {"n": 2})
        never_pub = _mk_event(cleanup, "job.v1.placement_reserved", {"n": 3})
        with _pool.connection() as conn:
            conn.execute(
                "UPDATE outbox_events SET published_at = clock_timestamp() - interval '30 days' "
                "WHERE aggregate_id=%s", (old_pub,),
            )
            conn.execute(
                "UPDATE outbox_events SET published_at = clock_timestamp() "
                "WHERE aggregate_id=%s", (fresh_pub,),
            )
            conn.execute(
                "UPDATE outbox_events SET created_at = clock_timestamp() - interval '30 days' "
                "WHERE aggregate_id=%s", (never_pub,),
            )
            conn.commit()
        deleted = run_transaction(
            lambda c: prune_settled_events(c, retention_days=7), what="test_prune"
        )
        assert deleted >= 1
        assert _event_row(old_pub) is None  # old + published: gone
        assert _event_row(fresh_pub) is not None  # recent: kept
        assert _event_row(never_pub) is not None  # unpublished: NEVER pruned
