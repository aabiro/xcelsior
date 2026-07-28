"""Real runtime gates for Track B B4.4 projection delivery."""

from __future__ import annotations

import inspect
import uuid

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _conn:
        _ready = all(value is not None for value in _conn.execute("""
                SELECT to_regclass('event_contracts'),
                       to_regclass('projection_deliveries'),
                       to_regclass('projection_checkpoints'),
                       to_regclass('audit_events_v2')
                """).fetchone())
except Exception as _exc:  # pragma: no cover - environment gate
    pytestmark = pytest.mark.skip(f"no pg pool: {_exc}")
    _pool = None
else:
    if not _ready:  # pragma: no cover
        pytestmark = pytest.mark.skip("projection runtime schema missing — upgrade >= 075")

import bg_worker
from control_plane.outbox import append_event
from control_plane.outbox_runtime import prune_settled_events
from control_plane.projection_delivery import (
    deliver_pending,
    health_snapshot,
    prepare_fanout,
    record_delivery,
)
from control_plane.projection_runtime import (
    AUDIT_SINK,
    bootstrap_projection_runtime,
    deliver_audit_event,
)


@pytest.fixture
def clean_projection_runtime():
    made: list[tuple[str, str]] = []
    if _pool is not None:
        with _pool.connection() as conn:
            conn.execute("TRUNCATE projection_deliveries")
            conn.execute("TRUNCATE projection_checkpoints")
            conn.commit()
    yield made
    if _pool is not None:
        with _pool.connection() as conn:
            conn.execute(
                "DELETE FROM projection_deliveries WHERE event_id = ANY(%s)",
                ([event_id for event_id, _ in made],),
            )
            for event_id, idempotency_key in made:
                conn.execute(
                    "DELETE FROM outbox_events WHERE event_id=%s OR idempotency_key=%s",
                    (event_id, idempotency_key),
                )
            conn.execute("DELETE FROM projection_checkpoints WHERE sink=%s", (AUDIT_SINK,))
            conn.commit()


def _append(cleanup, *, event_type: str = "job.v1.submitted") -> tuple[str, str]:
    idempotency_key = f"projection-runtime-{uuid.uuid4().hex}"
    secret = f"must-not-survive-{uuid.uuid4().hex}"
    with _pool.connection() as conn:
        event_id = append_event(
            conn,
            aggregate_type="job",
            aggregate_id=f"job-{uuid.uuid4().hex[:12]}",
            event_type=event_type,
            payload={
                "job_id": "projection-runtime-test",
                "status": "queued",
                "registry_password": secret,
                "env": {"SAFE": "visible", "API_TOKEN": secret},
                "variables": [{"name": "ACCESS_TOKEN", "value": secret}],
            },
            headers={
                "tenant_id": "tenant-projection-test",
                "trace_id": uuid.uuid4().hex,
                "principal_id": "principal-projection-test",
            },
            destination_class="default",
            idempotency_key=idempotency_key,
        )
        assert event_id is not None
        conn.commit()
    cleanup.append((event_id, idempotency_key))
    return event_id, secret


def test_outbox_to_audit_projection_is_durable_redacted_and_idempotent(
    clean_projection_runtime,
):
    event_id, secret = _append(clean_projection_runtime)
    with _pool.connection() as conn:
        assert bootstrap_projection_runtime(conn)
        assert prepare_fanout(conn, only_event_ids=[event_id]) == 1
        conn.commit()

    assert (
        deliver_pending(
            _pool,
            AUDIT_SINK,
            deliver_audit_event,
            owner="projection-runtime-test",
            limit=10,
        )
        == 1
    )
    with _pool.connection() as conn:
        audit = conn.execute(
            """
            SELECT event_type, tenant_id, classification, payload, event_hash,
                   stream_sequence
              FROM audit_events_v2
             WHERE event_id=%s
            """,
            (event_id,),
        ).fetchone()
        delivery = conn.execute(
            """
            SELECT status, external_id
              FROM projection_deliveries
             WHERE event_id=%s AND sink=%s
            """,
            (event_id, AUDIT_SINK),
        ).fetchone()
    assert audit is not None
    assert audit[0:3] == ("job.v1.submitted", "tenant-projection-test", "internal")
    assert secret not in str(audit[3])
    assert audit[3]["registry_password"] == "[REDACTED]"
    assert audit[3]["env"] == "[REDACTED]"
    assert audit[3]["variables"][0]["value"] == "[REDACTED]"
    assert len(audit[4]) == 64
    assert int(audit[5]) >= 1
    assert delivery == ("delivered", f"audit:{event_id}")

    # Retry after the sink insert committed returns the same stable identity and
    # never appends another immutable audit row.
    with _pool.connection() as conn:
        source = conn.execute(
            """
            SELECT event_id, event_type, payload, aggregate_type, aggregate_id,
                   aggregate_version, headers, event_version, tenant_id,
                   occurred_at, classification, payload_sha256, correlation_id,
                   causation_id, trace_id, created_at
              FROM outbox_events
             WHERE event_id=%s
            """,
            (event_id,),
        ).fetchone()
    from control_plane.projection_delivery import DeliverableEvent

    replay = DeliverableEvent(
        event_id=str(source[0]),
        event_type=str(source[1]),
        payload=source[2],
        sink=AUDIT_SINK,
        aggregate_type=str(source[3]),
        aggregate_id=str(source[4]),
        aggregate_version=int(source[5]),
        headers=source[6],
        event_version=int(source[7]),
        tenant_id=source[8],
        occurred_at=source[9],
        classification=source[10],
        payload_sha256=source[11],
        correlation_id=source[12],
        causation_id=source[13],
        trace_id=source[14],
        created_at=source[15],
    )
    assert deliver_audit_event(replay) == f"audit:{event_id}"
    with _pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM audit_events_v2 WHERE event_id=%s",
                (event_id,),
            ).fetchone()[0]
            == 1
        )


def test_unknown_contract_dead_letters_without_retry_loop(clean_projection_runtime):
    event_id, _ = _append(
        clean_projection_runtime,
        event_type="unknown.v1.projection_runtime_test",
    )
    with _pool.connection() as conn:
        assert bootstrap_projection_runtime(conn)
        assert prepare_fanout(conn, only_event_ids=[event_id]) == 1
        conn.commit()
    assert (
        deliver_pending(
            _pool,
            AUDIT_SINK,
            deliver_audit_event,
            owner="projection-runtime-contract-test",
            limit=10,
        )
        == 0
    )
    with _pool.connection() as conn:
        row = conn.execute(
            """
            SELECT status, attempt_count, last_error
              FROM projection_deliveries
             WHERE event_id=%s AND sink=%s
            """,
            (event_id, AUDIT_SINK),
        ).fetchone()
    assert row[0] == "dead_lettered"
    assert row[1] == 1
    assert "no active contract" in row[2]


def test_outbox_retention_waits_for_projection_settlement(clean_projection_runtime):
    event_id, _ = _append(clean_projection_runtime)
    with _pool.connection() as conn:
        assert bootstrap_projection_runtime(conn)
        assert prepare_fanout(conn, only_event_ids=[event_id]) == 1
        conn.execute(
            """
            UPDATE outbox_events
               SET published_at = clock_timestamp() - interval '40 days'
             WHERE event_id=%s
            """,
            (event_id,),
        )
        assert prune_settled_events(conn, retention_days=1, limit=100) == 0
        assert conn.execute("SELECT 1 FROM outbox_events WHERE event_id=%s", (event_id,)).fetchone()
        assert record_delivery(
            conn,
            event_id,
            AUDIT_SINK,
            f"audit:{event_id}",
        )
        assert prune_settled_events(conn, retention_days=1, limit=100) == 1
        # Delivered receipts intentionally outlive their pruned source as
        # operator evidence; that normal state must not degrade health.
        assert health_snapshot(conn)["orphaned"] == 0
        conn.commit()
    clean_projection_runtime[:] = [pair for pair in clean_projection_runtime if pair[0] != event_id]


def test_health_snapshot_and_bg_worker_registration(clean_projection_runtime):
    with _pool.connection() as conn:
        assert bootstrap_projection_runtime(conn)
        snapshot = health_snapshot(conn)
        conn.rollback()
    assert snapshot["orphaned"] == 0
    assert any(sink["sink"] == AUDIT_SINK for sink in snapshot["sinks"])

    source = inspect.getsource(bg_worker.main)
    assert 'register_task("projection_fanout_prepare"' in source
    assert 'register_task("projection_deliver_audit_log"' in source
    assert 'register_task("projection_delivery_retention"' in source
