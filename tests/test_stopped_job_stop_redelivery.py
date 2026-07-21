"""Stopped-job unfenced stop redelivery — legacy vs fenced-history classes.

Drives the shipped ``bg_worker.reconcile_paused_stopped_jobs`` against
real Postgres:

- pure legacy stopped jobs still receive durable throttled stop redelivery
- attempt-owned / fenced-history stopped jobs get zero unfenced
  ``stop_container`` / ``pause_container`` enqueue
"""

from __future__ import annotations

import json
import time
import uuid

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _has_attempt = (
            _c.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name='jobs' AND column_name='active_attempt_id'"
            ).fetchone()
            is not None
        )
        _has_job_attempts = (
            _c.execute("SELECT to_regclass('public.job_attempts')").fetchone()[0]
            is not None
        )
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None
else:
    if not (_has_attempt and _has_job_attempts):  # pragma: no cover
        pytestmark = pytest.mark.skip("control-plane attempt tables missing")


@pytest.fixture
def cleanup():
    ids = {"jobs": [], "hosts": [], "commands": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute(
                "DELETE FROM agent_commands WHERE args->>'job_id' = %s OR job_id = %s",
                (jid, jid),
            )
            conn.execute("DELETE FROM job_attempts WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


def _mk_host(cleanup, host_id: str) -> str:
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload)
               VALUES (%s, 'active', %s, %s)
               ON CONFLICT (host_id) DO NOTHING""",
            (host_id, time.time(), json.dumps({"host_id": host_id})),
        )
        conn.commit()
    cleanup["hosts"].append(host_id)
    return host_id


def _mk_stopped_legacy(cleanup, *, host_id: str, age_sec: float = 300.0) -> str:
    job_id = f"j-stop-leg-{uuid.uuid4().hex[:8]}"
    stale = time.time() - age_sec
    payload = {
        "job_id": job_id,
        "name": job_id,
        "status": "stopped",
        "container_name": f"xcl-{job_id}",
        "completed_at": stale,
        "submitted_at": stale,
    }
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs
                   (job_id, status, priority, submitted_at, host_id, payload)
               VALUES (%s, 'stopped', 0, %s, %s, %s)""",
            (job_id, stale, host_id, json.dumps(payload)),
        )
        conn.commit()
    cleanup["jobs"].append(job_id)
    return job_id


def _mk_stopped_fenced_history(cleanup, *, host_id: str, age_sec: float = 300.0) -> str:
    """Stopped job with terminal attempt history (active_attempt_id cleared)."""
    job_id = f"j-stop-fenced-{uuid.uuid4().hex[:8]}"
    stale = time.time() - age_sec
    attempt_id = str(uuid.uuid4())
    payload = {
        "job_id": job_id,
        "name": job_id,
        "status": "stopped",
        "container_name": f"xcl-{job_id}",
        "completed_at": stale,
        "submitted_at": stale,
    }
    with _pool.connection() as conn:
        fence = conn.execute(
            "SELECT nextval('placement_fencing_token_seq')"
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO jobs
                   (job_id, status, priority, submitted_at, host_id,
                    payload, active_attempt_id)
               VALUES (%s, 'stopped', 0, %s, %s, %s, NULL)""",
            (job_id, stale, host_id, json.dumps(payload)),
        )
        conn.execute(
            """INSERT INTO job_attempts
                   (attempt_id, job_id, attempt_number, status, host_id,
                    fencing_token, job_generation)
               VALUES (%s, %s, 1, 'succeeded', %s, %s, 1)""",
            (attempt_id, job_id, host_id, fence),
        )
        conn.commit()
    cleanup["jobs"].append(job_id)
    return job_id


def _mk_stopped_attempt_owned(cleanup, *, host_id: str, age_sec: float = 300.0) -> str:
    """Stopped projection still pointing at an active attempt (edge dual-write class)."""
    job_id = f"j-stop-owned-{uuid.uuid4().hex[:8]}"
    stale = time.time() - age_sec
    attempt_id = str(uuid.uuid4())
    payload = {
        "job_id": job_id,
        "name": job_id,
        "status": "stopped",
        "container_name": f"xcl-{job_id}",
        "completed_at": stale,
        "submitted_at": stale,
    }
    with _pool.connection() as conn:
        fence = conn.execute(
            "SELECT nextval('placement_fencing_token_seq')"
        ).fetchone()[0]
        # Job row first (FK parent), then attempt, then active pointer.
        conn.execute(
            """INSERT INTO jobs
                   (job_id, status, priority, submitted_at, host_id,
                    payload, active_attempt_id)
               VALUES (%s, 'stopped', 0, %s, %s, %s, NULL)""",
            (job_id, stale, host_id, json.dumps(payload)),
        )
        conn.execute(
            """INSERT INTO job_attempts
                   (attempt_id, job_id, attempt_number, status, host_id,
                    fencing_token, job_generation)
               VALUES (%s, %s, 1, 'running', %s, %s, 1)""",
            (attempt_id, job_id, host_id, fence),
        )
        conn.execute(
            "UPDATE jobs SET active_attempt_id=%s WHERE job_id=%s",
            (attempt_id, job_id),
        )
        conn.commit()
    cleanup["jobs"].append(job_id)
    return job_id


def _stop_cmds_for(job_id: str) -> list[dict]:
    with _pool.connection() as conn:
        from psycopg.rows import dict_row

        conn.row_factory = dict_row
        return list(
            conn.execute(
                """
                SELECT id, host_id, command, status, created_by, args
                  FROM agent_commands
                 WHERE (args->>'job_id' = %s OR job_id = %s)
                   AND command IN ('stop_container', 'pause_container')
                 ORDER BY id
                """,
                (job_id, job_id),
            ).fetchall()
        )


def test_legacy_stopped_job_gets_durable_stop_redelivery(cleanup):
    import bg_worker

    host_id = _mk_host(cleanup, f"h-stop-leg-{uuid.uuid4().hex[:6]}")
    job_id = _mk_stopped_legacy(cleanup, host_id=host_id, age_sec=400.0)

    before = _stop_cmds_for(job_id)
    assert before == []

    n = bg_worker.reconcile_paused_stopped_jobs()
    assert n >= 1

    cmds = _stop_cmds_for(job_id)
    assert len(cmds) == 1
    assert cmds[0]["command"] == "stop_container"
    assert cmds[0]["status"] == "pending"
    assert cmds[0]["created_by"] == "reconcile_sweep"
    assert cmds[0]["host_id"] == host_id
    assert cmds[0]["args"]["job_id"] == job_id

    # Pending-command dedupe: second pass must not dual-enqueue.
    n2 = bg_worker.reconcile_paused_stopped_jobs()
    assert n2 >= 0
    assert len(_stop_cmds_for(job_id)) == 1


def test_fenced_history_stopped_job_gets_no_unfenced_stop(cleanup):
    import bg_worker

    host_id = _mk_host(cleanup, f"h-stop-fenced-{uuid.uuid4().hex[:6]}")
    job_id = _mk_stopped_fenced_history(cleanup, host_id=host_id, age_sec=400.0)

    n = bg_worker.reconcile_paused_stopped_jobs()
    # May enqueue other legacy fixtures in shared DB; never this job.
    cmds = _stop_cmds_for(job_id)
    assert cmds == [], (
        f"fenced-history job {job_id} must not receive unfenced stop "
        f"(got {cmds}); reconcile returned n={n}"
    )


def test_attempt_owned_stopped_job_gets_no_unfenced_stop(cleanup):
    import bg_worker

    host_id = _mk_host(cleanup, f"h-stop-owned-{uuid.uuid4().hex[:6]}")
    job_id = _mk_stopped_attempt_owned(cleanup, host_id=host_id, age_sec=400.0)

    bg_worker.reconcile_paused_stopped_jobs()
    cmds = _stop_cmds_for(job_id)
    assert cmds == [], (
        f"attempt-owned job {job_id} must not receive unfenced stop (got {cmds})"
    )


def test_legacy_and_fenced_classes_side_by_side(cleanup):
    """One reconcile pass: legacy redelivered, fenced untouched."""
    import bg_worker

    host_id = _mk_host(cleanup, f"h-stop-mix-{uuid.uuid4().hex[:6]}")
    legacy_id = _mk_stopped_legacy(cleanup, host_id=host_id, age_sec=500.0)
    fenced_id = _mk_stopped_fenced_history(cleanup, host_id=host_id, age_sec=500.0)

    bg_worker.reconcile_paused_stopped_jobs()

    legacy_cmds = _stop_cmds_for(legacy_id)
    fenced_cmds = _stop_cmds_for(fenced_id)
    assert len(legacy_cmds) == 1
    assert legacy_cmds[0]["command"] == "stop_container"
    assert fenced_cmds == []
