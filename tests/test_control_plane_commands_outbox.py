"""Command claim/ACK protocol (§9.4) and transactional outbox (§16.1)."""

import time
import uuid

import pytest

try:
    from db import _get_pg_pool

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

from control_plane.commands import (
    CommandProtocolError,
    ack_command,
    claim_commands,
    nack_command,
    redeliver_expired_claims,
)
from control_plane.db import run_transaction
from control_plane.outbox import (
    OutboxDispatcher,
    append_event,
    claim_batch,
    mark_failed,
    mark_published,
)


# ── Command protocol ─────────────────────────────────────────────────


@pytest.fixture
def host_id():
    hid = f"cmd-host-{uuid.uuid4().hex[:10]}"
    yield hid
    if _pool is None:
        return
    with _pool.connection() as conn:
        conn.execute("DELETE FROM agent_commands WHERE host_id=%s", (hid,))
        conn.commit()


def _enqueue(host_id, *, command="start_attempt", max_attempts=3, not_before=None):
    with _pool.connection() as conn:
        row = conn.execute(
            """INSERT INTO agent_commands
                   (host_id, command, args, status, max_attempts, not_before,
                    idempotency_key, expires_at)
               VALUES (%s, %s, '{"k": 1}'::jsonb, 'pending', %s,
                       CASE WHEN %s::float8 IS NULL THEN NULL
                            ELSE clock_timestamp() + make_interval(secs => %s) END,
                       %s, EXTRACT(EPOCH FROM NOW()) + 86400)
               RETURNING command_id""",
            (host_id, command, max_attempts, not_before, not_before or 0.0,
             f"{command}:{uuid.uuid4().hex[:8]}"),
        ).fetchone()
        conn.commit()
    return str(row[0])


def _claim(host_id, session="wkr-1", ttl=60, limit=10):
    return run_transaction(
        lambda conn: claim_commands(
            conn, host_id=host_id, worker_session_id=session,
            claim_ttl_sec=ttl, limit=limit,
        )
    )


class TestCommandClaim:
    def test_claim_is_exclusive_until_expiry(self, host_id):
        _enqueue(host_id)
        first = _claim(host_id, session="wkr-1")
        assert len(first) == 1 and first[0].attempt_count == 1
        assert _claim(host_id, session="wkr-2") == []

    def test_not_before_defers_delivery(self, host_id):
        _enqueue(host_id, not_before=3600.0)
        assert _claim(host_id) == []

    def test_claim_carries_authority_fields(self, host_id):
        cid = _enqueue(host_id)
        got = _claim(host_id)[0]
        assert got.command_id == cid
        assert got.command == "start_attempt"
        assert got.args == {"k": 1}


class TestAckNack:
    def test_ack_then_duplicate_ack_replays_result(self, host_id):
        _enqueue(host_id)
        cmd = _claim(host_id)[0]
        first = run_transaction(
            lambda conn: ack_command(
                conn, command_id=cmd.command_id, host_id=host_id,
                result={"container_id": "abc123"},
            )
        )
        assert first.duplicate is False
        dup = run_transaction(
            lambda conn: ack_command(
                conn, command_id=cmd.command_id, host_id=host_id,
                result={"container_id": "SHOULD-BE-IGNORED"},
            )
        )
        assert dup.duplicate is True
        assert dup.result == {"container_id": "abc123"}

    def test_ack_from_wrong_host_rejected(self, host_id):
        _enqueue(host_id)
        cmd = _claim(host_id)[0]
        with pytest.raises(CommandProtocolError):
            run_transaction(
                lambda conn: ack_command(
                    conn, command_id=cmd.command_id, host_id="host-imposter",
                )
            )

    def test_retryable_nack_requeues_with_backoff(self, host_id):
        _enqueue(host_id, max_attempts=3)
        cmd = _claim(host_id)[0]
        status = run_transaction(
            lambda conn: nack_command(
                conn, command_id=cmd.command_id, host_id=host_id,
                error_code="image_pull_backoff",
            )
        )
        assert status == "pending"
        # Backoff: not immediately deliverable again.
        assert _claim(host_id) == []
        with _pool.connection() as conn:
            row = conn.execute(
                "SELECT error_code, next_attempt_at > clock_timestamp() "
                "FROM agent_commands WHERE command_id=%s",
                (cmd.command_id,),
            ).fetchone()
        assert row[0] == "image_pull_backoff" and row[1] is True

    def test_nack_past_budget_dead_letters(self, host_id):
        _enqueue(host_id, max_attempts=1)
        cmd = _claim(host_id)[0]
        status = run_transaction(
            lambda conn: nack_command(
                conn, command_id=cmd.command_id, host_id=host_id,
                error_code="runtime_unavailable",
            )
        )
        assert status == "dead_letter"

    def test_non_retryable_nack_dead_letters_immediately(self, host_id):
        _enqueue(host_id, max_attempts=5)
        cmd = _claim(host_id)[0]
        status = run_transaction(
            lambda conn: nack_command(
                conn, command_id=cmd.command_id, host_id=host_id,
                error_code="spec_hash_mismatch", retryable=False,
            )
        )
        assert status == "dead_letter"


class TestRedelivery:
    def test_expired_claim_redelivers(self, host_id):
        _enqueue(host_id)
        assert len(_claim(host_id, session="wkr-crash", ttl=1)) == 1
        time.sleep(1.2)
        moved = run_transaction(lambda conn: redeliver_expired_claims(conn))
        assert moved >= 1
        redelivered = _claim(host_id, session="wkr-2")
        assert len(redelivered) == 1 and redelivered[0].attempt_count == 2

    def test_expired_claim_past_budget_dead_letters(self, host_id):
        cid = _enqueue(host_id, max_attempts=1)
        assert len(_claim(host_id, ttl=1)) == 1
        time.sleep(1.2)
        run_transaction(lambda conn: redeliver_expired_claims(conn))
        with _pool.connection() as conn:
            row = conn.execute(
                "SELECT status FROM agent_commands WHERE command_id=%s", (cid,)
            ).fetchone()
        assert row[0] == "dead_letter"


# ── Outbox ───────────────────────────────────────────────────────────


@pytest.fixture
def aggregate_id():
    agg = f"outbox-{uuid.uuid4().hex[:10]}"
    yield agg
    if _pool is None:
        return
    with _pool.connection() as conn:
        conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (agg,))
        conn.commit()


def _append(aggregate_id, *, event_type="job.v1.test", key=None, dest="default"):
    return run_transaction(
        lambda conn: append_event(
            conn,
            aggregate_type="job",
            aggregate_id=aggregate_id,
            event_type=event_type,
            payload={"x": 1},
            destination_class=dest,
            idempotency_key=key,
        )
    )


class TestOutboxAppend:
    def test_append_and_idempotent_duplicate(self, aggregate_id):
        first = _append(aggregate_id, key="k1")
        assert first is not None
        assert _append(aggregate_id, key="k1") is None  # duplicate intent

    def test_atomic_with_surrounding_transaction(self, aggregate_id):
        with pytest.raises(RuntimeError):
            def fn(conn):
                append_event(
                    conn, aggregate_type="job", aggregate_id=aggregate_id,
                    event_type="job.v1.rolled_back",
                )
                raise RuntimeError("state mutation failed")
            run_transaction(fn)
        with _pool.connection() as conn:
            n = conn.execute(
                "SELECT count(*) FROM outbox_events WHERE aggregate_id=%s",
                (aggregate_id,),
            ).fetchone()[0]
        assert n == 0  # no event without the state mutation


class TestOutboxDispatch:
    def test_claim_publish_lifecycle(self, aggregate_id):
        eid = _append(aggregate_id)
        events = run_transaction(
            lambda conn: claim_batch(conn, dispatcher_id="d1", limit=100)
        )
        mine = [e for e in events if e.event_id == eid]
        assert len(mine) == 1
        # A rival dispatcher gets nothing for the claimed event.
        rival = run_transaction(
            lambda conn: claim_batch(conn, dispatcher_id="d2", limit=100)
        )
        assert eid not in [e.event_id for e in rival]
        run_transaction(lambda conn: mark_published(conn, [eid]))
        again = run_transaction(
            lambda conn: claim_batch(conn, dispatcher_id="d2", limit=100)
        )
        assert eid not in [e.event_id for e in again]

    def test_failure_backoff_then_dead_letter(self, aggregate_id):
        eid = _append(aggregate_id)
        with _pool.connection() as conn:
            conn.execute(
                "UPDATE outbox_events SET max_attempts=2 WHERE event_id=%s", (eid,)
            )
            conn.commit()
        for expected in ("retry_scheduled", "dead_letter"):
            events = run_transaction(
                lambda conn: claim_batch(conn, dispatcher_id="d1", limit=100)
            )
            if expected == "retry_scheduled":
                assert eid in [e.event_id for e in events]
            else:
                # Force availability for the second claim round.
                with _pool.connection() as conn:
                    conn.execute(
                        "UPDATE outbox_events SET available_at=clock_timestamp() "
                        "WHERE event_id=%s AND dead_lettered_at IS NULL",
                        (eid,),
                    )
                    conn.commit()
                events = run_transaction(
                    lambda conn: claim_batch(conn, dispatcher_id="d1", limit=100)
                )
                assert eid in [e.event_id for e in events]
            outcome = run_transaction(
                lambda conn: mark_failed(conn, eid, "sink unavailable")
            )
            assert outcome == expected
        # Dead-lettered events are never claimed again.
        with _pool.connection() as conn:
            conn.execute(
                "UPDATE outbox_events SET available_at=clock_timestamp() "
                "WHERE event_id=%s",
                (eid,),
            )
            conn.commit()
        events = run_transaction(
            lambda conn: claim_batch(conn, dispatcher_id="d1", limit=100)
        )
        assert eid not in [e.event_id for e in events]

    def test_dispatcher_run_once_delivers_and_settles(self, aggregate_id):
        eid_ok = _append(aggregate_id, key="ok")
        eid_bad = _append(aggregate_id, key="bad", event_type="job.v1.explodes")
        seen: list[str] = []

        def handler(event):
            if event.event_type == "job.v1.explodes":
                raise ConnectionError("sink down")
            seen.append(event.event_id)

        dispatcher = OutboxDispatcher("d-test", {"default": handler}, batch_size=500)
        stats = dispatcher.run_once()
        assert stats["claimed"] >= 2
        assert eid_ok in seen
        with _pool.connection() as conn:
            ok_row = conn.execute(
                "SELECT published_at IS NOT NULL FROM outbox_events "
                "WHERE event_id=%s",
                (eid_ok,),
            ).fetchone()
            bad_row = conn.execute(
                "SELECT published_at, last_error FROM outbox_events "
                "WHERE event_id=%s",
                (eid_bad,),
            ).fetchone()
        assert ok_row[0] is True
        assert bad_row[0] is None and "sink down" in bad_row[1]
