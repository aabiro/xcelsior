"""Job requests must run a controller; one broken entry must not poison the queue.

These gates use PostgreSQL transactions, including genuine SQL errors and
concurrent claims. No controller test may silently pass without a database.
"""

from __future__ import annotations

from control_plane.state import initial_desired_state, job_phase

import json
import time
import uuid

import pytest

from control_plane import reconcile
from control_plane.db import try_advisory_xact_lock
from db import _get_pg_pool

pytestmark = pytest.mark.needs_db


@pytest.fixture
def resources():
    pool = _get_pg_pool()
    ids = []
    yield pool, ids
    with pool.connection() as conn:
        for rid in ids:
            conn.execute("DELETE FROM reconciliation_queue WHERE resource_id = %s", (rid,))
            conn.execute("DELETE FROM reconciliation_findings WHERE resource_id = %s", (rid,))
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (rid,))


def _enqueue(resources, resource_type="job", *, status=None, priority=0):
    pool, ids = resources
    rid = f"reconcile-gate-{uuid.uuid4().hex}"
    ids.append(rid)
    with pool.connection() as conn:
        if status is not None:
            fixture_submitted_at = time.time()
            conn.execute(
                """INSERT INTO jobs (job_id, status, priority, submitted_at, payload, phase, desired_state, effective_priority, queued_at) VALUES (%s, %s, 0, %s, %s, %s, %s, 0, CASE WHEN %s IN ('queued','preempted') THEN to_timestamp(%s) ELSE NULL END)""",
                (
                    rid,
                    status,
                    fixture_submitted_at,
                    json.dumps({"job_id": rid, "status": status}),
                    job_phase(status),
                    initial_desired_state(status),
                    status,
                    fixture_submitted_at,
                ),
            )
        conn.execute(
            "INSERT INTO reconciliation_queue (resource_type, resource_id, reason, priority) "
            "VALUES (%s, %s, 'test_request', %s) "
            "ON CONFLICT (resource_type, resource_id) DO UPDATE "
            "SET reason = EXCLUDED.reason, priority = EXCLUDED.priority",
            (resource_type, rid, 10000 + priority),
        )
    return rid


def _process(resources, **kwargs):
    pool, ids = resources
    # The durable trigger also queues jobs owned by other suites. Process this
    # fixture's higher-priority entries, without deleting unrelated requests.
    kwargs.setdefault("limit", len(ids))
    with pool.connection() as conn:
        return reconcile.process_due(conn, worker_id="job-reconcile-gate", **kwargs)


def test_running_job_without_authority_is_reported_and_periodically_rechecked(resources):
    pool, _ = resources
    rid = _enqueue(resources, status="running")
    stats = _process(resources)
    assert stats["reconciled"] == 1 and stats["failed"] == 0
    assert stats["findings_opened"] == 1
    with pool.connection() as conn:
        assert conn.execute(
            "SELECT finding_type, action_taken FROM reconciliation_findings "
            "WHERE resource_type = 'job' AND resource_id = %s AND resolved_at IS NULL",
            (rid,),
        ).fetchone() == ("job_active_attempt_missing", "report_only")
        assert conn.execute("SELECT status FROM jobs WHERE job_id = %s", (rid,)).fetchone() == (
            "running",
        )
        assert conn.execute(
            "SELECT claim_owner IS NULL, due_at > clock_timestamp(), last_error "
            "FROM reconciliation_queue WHERE resource_type = 'job' AND resource_id = %s",
            (rid,),
        ).fetchone() == (True, True, None)


@pytest.mark.parametrize("status", ["queued", "completed", "failed", "stopped", "terminated"])
def test_normal_unassigned_jobs_are_not_resurrected(resources, status):
    pool, _ = resources
    rid = _enqueue(resources, status=status)
    stats = _process(resources)
    assert stats["reconciled"] == 1 and stats["failed"] == 0
    assert stats["findings_opened"] == 0
    with pool.connection() as conn:
        assert conn.execute("SELECT status FROM jobs WHERE job_id = %s", (rid,)).fetchone() == (
            status,
        )


def test_deleted_job_request_settles_idempotently(resources):
    pool, _ = resources
    rid = _enqueue(resources)
    stats = _process(resources)
    assert stats["reconciled"] == 1 and stats["failed"] == 0
    with pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM reconciliation_queue WHERE resource_id = %s", (rid,)
            ).fetchone()
            is None
        )


def test_unknown_controller_retains_request_with_error_and_backoff(resources):
    pool, _ = resources
    rid = _enqueue(resources, "unsupported_gate_type")
    stats = _process(resources)
    assert stats["reconciled"] == 0 and stats["failed"] == 1
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT last_error, attempt_count, claim_owner, due_at > clock_timestamp() "
            "FROM reconciliation_queue WHERE resource_id = %s",
            (rid,),
        ).fetchone()
    assert row is not None
    assert "unsupported_resource_type" in row[0]
    assert row[1:] == (1, None, True)


def test_sql_failure_rolls_back_partial_work_and_continues_next_entry(resources, monkeypatch):
    pool, _ = resources
    broken = _enqueue(resources, "host", priority=1)
    healthy = _enqueue(resources, "host")
    original = reconcile.reconcile_host

    def fail_one(conn, host_id, **kwargs):
        if host_id == broken:
            reconcile._open_finding(
                conn,
                resource_type="host",
                resource_id=broken,
                finding_type="partial_work",
                severity="warning",
                summary="must roll back",
                desired=None,
                observed=None,
            )
            conn.execute("SELECT 1 / 0")
        return original(conn, host_id, **kwargs)

    monkeypatch.setattr(reconcile, "reconcile_host", fail_one)
    stats = _process(resources)
    assert stats["claimed"] == 2 and stats["failed"] == 1 and stats["reconciled"] == 1
    assert stats["findings_opened"] == 0
    with pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM reconciliation_findings WHERE resource_id = %s", (broken,)
            ).fetchone()
            is None
        )
        assert (
            conn.execute(
                "SELECT claim_owner, last_error FROM reconciliation_queue WHERE resource_id = %s",
                (broken,),
            ).fetchone()[0]
            is None
        )
        assert (
            conn.execute(
                "SELECT 1 FROM reconciliation_queue WHERE resource_id = %s", (healthy,)
            ).fetchone()
            is None
        )


def test_resource_lock_contention_backs_off_without_dropping_request(resources):
    pool, _ = resources
    rid = _enqueue(resources, "host")
    with pool.connection() as holder:
        assert try_advisory_xact_lock(holder, "reconcile", f"host:{rid}")
        stats = _process(resources)
        assert stats["failed"] == 1 and stats["reconciled"] == 0
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT last_error FROM reconciliation_queue WHERE resource_id = %s", (rid,)
        ).fetchone()
        assert "resource_locked" in row[0]
        conn.execute(
            "UPDATE reconciliation_queue SET due_at = clock_timestamp() WHERE resource_id = %s",
            (rid,),
        )
    assert _process(resources)["reconciled"] == 1


def test_two_replicas_claim_distinct_entries(resources):
    pool, _ = resources
    _enqueue(resources, "host", priority=1)
    _enqueue(resources, "host")
    with pool.connection() as first:
        one = reconcile.process_due(first, worker_id="replica-1", limit=1)
        # The first transaction is still open: its claimed row must be skipped.
        with pool.connection() as second:
            two = reconcile.process_due(second, worker_id="replica-2", limit=1)
        assert one["reconciled"] == two["reconciled"] == 1
    with pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM reconciliation_queue WHERE resource_id = ANY(%s)",
                (resources[1],),
            ).fetchone()[0]
            == 0
        )
