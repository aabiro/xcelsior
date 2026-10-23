"""Durable enqueue, trusted observation routing, and recovery against PostgreSQL."""

from control_plane.state import initial_desired_state, job_phase

import time
import uuid

import pytest
from psycopg.types.json import Jsonb

from control_plane.job_reconcile import reconcile_job
from control_plane.reconcile_requests import enqueue_missing_jobs
from tests.test_job_reconcile_queue import resources  # noqa: F401
from tests.test_job_reconcile_observations import placed, _observe  # noqa: F401

pytestmark = pytest.mark.needs_db


def _insert(conn, resources, status="queued"):
    jid = f"job-request-{uuid.uuid4().hex}"
    resources[1].append(jid)
    fixture_submitted_at = time.time()
    conn.execute(
        """INSERT INTO jobs (job_id, status, priority, submitted_at, payload, phase, desired_state, effective_priority, queued_at) VALUES (%s, %s, 0, %s, %s, %s, %s, 0, CASE WHEN %s IN ('queued','preempted') THEN to_timestamp(%s) ELSE NULL END)""",
        (
            jid,
            status,
            fixture_submitted_at,
            Jsonb({"job_id": jid, "status": status}),
            job_phase(status),
            initial_desired_state(status),
            status,
            fixture_submitted_at,
        ),
    )
    return jid


def _queue(conn, jid):
    return conn.execute(
        "SELECT reason, claim_owner FROM reconciliation_queue "
        "WHERE resource_type = 'job' AND resource_id = %s",
        (jid,),
    ).fetchone()


def test_job_mutation_and_enqueue_commit_or_rollback_together(resources):
    pool, _ = resources
    with pool.connection() as conn:
        with pytest.raises(RuntimeError, match="rollback"):
            with conn.transaction():
                jid = _insert(conn, resources)
                assert _queue(conn, jid) == ("job_changed", None)
                raise RuntimeError("rollback")
        assert _queue(conn, jid) is None
        assert conn.execute("SELECT 1 FROM jobs WHERE job_id = %s", (jid,)).fetchone() is None
        jid = _insert(conn, resources)
    with pool.connection() as conn:
        assert _queue(conn, jid) == ("job_changed", None)
        conn.execute("DELETE FROM jobs WHERE job_id = %s", (jid,))
        assert _queue(conn, jid) is None


def test_coalescing_preserves_claim_and_ignores_observed_generation(resources):
    pool, _ = resources
    with pool.connection() as conn:
        jid = _insert(conn, resources)
        conn.execute(
            "UPDATE reconciliation_queue SET claim_owner = 'replica', "
            "due_at = clock_timestamp() + interval '1 hour' WHERE resource_id = %s",
            (jid,),
        )
        conn.execute(
            """UPDATE jobs SET status = 'stopping', phase='running', desired_state='stopped' WHERE job_id = %s""",
            (jid,),
        )
        assert _queue(conn, jid) == ("job_changed", "replica")
        assert conn.execute(
            "SELECT due_at < clock_timestamp() FROM reconciliation_queue WHERE resource_id = %s",
            (jid,),
        ).fetchone() == (True,)
        conn.execute("DELETE FROM reconciliation_queue WHERE resource_id = %s", (jid,))
        conn.execute("UPDATE jobs SET observed_generation = generation WHERE job_id = %s", (jid,))
        assert _queue(conn, jid) is None


def test_locked_job_is_retried_not_mistaken_for_deleted(resources):
    pool, _ = resources
    with pool.connection() as conn:
        jid = _insert(conn, resources)
    with pool.connection() as writer:
        writer.execute("SELECT 1 FROM jobs WHERE job_id = %s FOR UPDATE", (jid,))
        with pool.connection() as reconciler:
            reconciler.execute("SET LOCAL statement_timeout = '1s'")
            with pytest.raises(RuntimeError, match="job_locked"):
                reconcile_job(reconciler, jid)
    with pool.connection() as conn:
        assert reconcile_job(conn, jid).recheck_after_sec == 60


def test_recovery_scan_is_bounded_and_advances_without_rescheduling_existing(resources):
    pool, _ = resources
    with pool.connection() as conn:
        ids = [_insert(conn, resources) for _ in range(3)]
        stopped = _insert(conn, resources, "stopped")
        conn.execute(
            "DELETE FROM reconciliation_queue WHERE resource_id = ANY(%s)", (ids + [stopped],)
        )
        # Other fixtures may leave active jobs. First fill them without touching
        # their queue entries, then remove only this test's entries again.
        while enqueue_missing_jobs(conn, limit=1000):
            pass
        conn.execute("DELETE FROM reconciliation_queue WHERE resource_id = ANY(%s)", (ids,))
        assert enqueue_missing_jobs(conn, limit=2) == 2
        assert enqueue_missing_jobs(conn, limit=2) == 1
        assert enqueue_missing_jobs(conn, limit=2) == 0
        assert all(_queue(conn, jid) == ("recovery_scan", None) for jid in ids)
        assert _queue(conn, stopped) is None
        with pytest.raises(ValueError, match="positive"):
            enqueue_missing_jobs(conn, limit=0)
        conn.rollback()  # Do not retain recovery entries for unrelated fixtures.


def test_empty_observation_wakes_host_owned_jobs_not_worker_supplied_ids(placed, resources):
    with placed["pool"].connection() as conn:
        foreign_job = _insert(conn, resources, "running")
        conn.execute(
            "DELETE FROM reconciliation_queue WHERE resource_id = ANY(%s)",
            ([placed["job_id"], foreign_job],),
        )
    _observe(placed, workloads=[])
    with placed["pool"].connection() as conn:
        assert _queue(conn, placed["job_id"])[0] == "observation_received"
        assert _queue(conn, foreign_job) is None
    _observe(placed, workloads=[{"job_id": foreign_job, "state": "running"}])
    with placed["pool"].connection() as conn:
        assert _queue(conn, foreign_job) is None
