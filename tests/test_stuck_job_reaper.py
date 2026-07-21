"""Stuck-job reaper — domain transition path (legacy fail + fenced skip).

Drives the shipped ``reaper.reaper_tick`` and
``control_plane.stuck_jobs.fail_stuck_legacy_job`` against real Postgres.
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
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None
else:
    if not _has_attempt:  # pragma: no cover
        pytestmark = pytest.mark.skip("jobs.active_attempt_id missing")


@pytest.fixture
def cleanup():
    ids = {"jobs": [], "hosts": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (jid,))
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM job_attempts WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


def _mk_legacy_stuck(cleanup, *, status: str, age_sec: float) -> str:
    job_id = f"j-stuck-{uuid.uuid4().hex[:8]}"
    stale = time.time() - age_sec
    payload = {
        "job_id": job_id,
        "name": job_id,
        "status": status,
        "priority": 0,
        "submitted_at": stale,
        "updated_at": stale,
        "vram_needed_gb": 8.0,
    }
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs
                   (job_id, status, priority, submitted_at, host_id, payload)
               VALUES (%s, %s, 0, %s, %s, %s)""",
            (
                job_id,
                status,
                stale,
                "h-legacy" if status != "queued" else None,
                json.dumps(payload),
            ),
        )
        conn.commit()
    cleanup["jobs"].append(job_id)
    return job_id


def _mk_attempt_owned_stuck(cleanup, *, age_sec: float) -> str:
    """Assigned job with active_attempt_id set — reaper must not touch it."""
    job_id = f"j-fenced-{uuid.uuid4().hex[:8]}"
    host_id = f"h-fenced-{uuid.uuid4().hex[:6]}"
    attempt_id = str(uuid.uuid4())
    stale = time.time() - age_sec
    payload = {
        "job_id": job_id,
        "name": job_id,
        "status": "assigned",
        "host_id": host_id,
        "updated_at": stale,
        "submitted_at": stale,
    }
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload)
               VALUES (%s, 'active', %s, %s)""",
            (host_id, time.time(), json.dumps({"host_id": host_id})),
        )
        # Job first (attempt FK → jobs), then attempt, then link active_attempt_id.
        conn.execute(
            """INSERT INTO jobs
                   (job_id, status, priority, submitted_at, host_id, payload)
               VALUES (%s, 'assigned', 0, %s, %s, %s)""",
            (job_id, stale, host_id, json.dumps(payload)),
        )
        # fencing_token must be unique — use sequence when available.
        fence = conn.execute(
            "SELECT nextval('placement_fencing_token_seq')"
        ).fetchone()
        fence_val = fence[0] if not isinstance(fence, dict) else fence["nextval"]
        conn.execute(
            """INSERT INTO job_attempts
                   (attempt_id, job_id, attempt_number, status, host_id,
                    fencing_token, job_generation)
               VALUES (%s, %s, 1, 'reserved', %s, %s, 1)""",
            (attempt_id, job_id, host_id, fence_val),
        )
        conn.execute(
            "UPDATE jobs SET active_attempt_id=%s WHERE job_id=%s",
            (attempt_id, job_id),
        )
        conn.commit()
    cleanup["jobs"].append(job_id)
    cleanup["hosts"].append(host_id)
    return job_id


def _job_status(job_id: str) -> dict:
    with _pool.connection() as conn:
        row = conn.execute(
            "SELECT status, active_attempt_id, payload FROM jobs WHERE job_id=%s",
            (job_id,),
        ).fetchone()
    if row is None:
        return {}
    if isinstance(row, dict):
        payload = row["payload"]
        return {
            "status": row["status"],
            "active_attempt_id": row["active_attempt_id"],
            "payload": payload if isinstance(payload, dict) else json.loads(payload or "{}"),
        }
    payload = row[2]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return {
        "status": row[0],
        "active_attempt_id": row[1],
        "payload": payload or {},
    }


class TestStuckJobDomainFail:
    def test_legacy_assigned_past_deadline_fails_via_update_job_status(self, cleanup):
        from reaper import reaper_tick

        # assigned default timeout is 180s — age well past that.
        job_id = _mk_legacy_stuck(cleanup, status="assigned", age_sec=10_000)
        killed = reaper_tick()
        assert killed >= 1
        row = _job_status(job_id)
        assert row["status"] == "failed"
        payload = row["payload"]
        assert payload.get("status") == "failed"
        reason = str(payload.get("failure_reason") or "")
        assert "stuck" in reason.lower() or payload.get("failure_code") == "stuck-no-progress"
        # Durable outbox intent for multi-replica SSE.
        with _pool.connection() as conn:
            n = conn.execute(
                "SELECT count(*) FROM outbox_events "
                "WHERE aggregate_id=%s AND event_type='job.v1.legacy_status_changed'",
                (job_id,),
            ).fetchone()
        count = n[0] if not isinstance(n, dict) else n["count"]
        assert int(count) >= 1

    def test_attempt_owned_stuck_job_untouched(self, cleanup):
        from reaper import reaper_tick

        job_id = _mk_attempt_owned_stuck(cleanup, age_sec=10_000)
        before = _job_status(job_id)
        assert before["status"] == "assigned"
        assert before["active_attempt_id"] is not None

        reaper_tick()
        after = _job_status(job_id)
        assert after["status"] == "assigned"
        assert str(after["active_attempt_id"]) == str(before["active_attempt_id"])

    def test_domain_fail_refuses_when_status_raced(self, cleanup):
        """CAS: expected_status mismatch returns None, no false fail."""
        from control_plane.stuck_jobs import fail_stuck_legacy_job

        job_id = _mk_legacy_stuck(cleanup, status="assigned", age_sec=10)
        # Advance to running so reaper's expected_status='assigned' loses.
        with _pool.connection() as conn:
            conn.execute(
                "UPDATE jobs SET status='running', "
                "payload = jsonb_set(payload, '{status}', '\"running\"') "
                "WHERE job_id=%s",
                (job_id,),
            )
            conn.commit()
        result = fail_stuck_legacy_job(
            job_id, stuck_status="assigned", timeout_sec=180
        )
        assert result is None
        assert _job_status(job_id)["status"] == "running"

    def test_reaper_uses_domain_fail_not_raw_status_sql(self):
        import inspect
        from reaper import reaper_tick

        src = inspect.getsource(reaper_tick)
        assert "fail_stuck_legacy_job" in src
        assert "UPDATE jobs SET status = 'failed'" not in src

    def test_cas_uses_status_under_row_lock_not_stale_snapshot(self, cleanup):
        """Concurrent assigned→running under FOR UPDATE must not be force-failed.

        Session A locks the job and advances status to running, then commits.
        Session B's ``update_job_status(..., expected_status='assigned')`` blocks
        on the same row lock, then must re-read status under the lock and return
        None — leaving running intact (old reaper ``WHERE status=`` semantics).
        """
        import threading

        import scheduler as sched

        job_id = _mk_legacy_stuck(cleanup, status="assigned", age_sec=10_000)
        barrier = threading.Barrier(2)
        results: dict = {}

        def holder():
            # Hold FOR UPDATE, then flip to running so the waiter sees the new status.
            with _pool.connection() as conn:
                conn.execute(
                    "SELECT status FROM jobs WHERE job_id=%s FOR UPDATE",
                    (job_id,),
                ).fetchone()
                barrier.wait(timeout=10)
                # Give waiter time to block on the same row lock.
                time.sleep(0.3)
                conn.execute(
                    "UPDATE jobs SET status='running', "
                    "payload = jsonb_set("
                    "  jsonb_set(payload, '{status}', '\"running\"'::jsonb),"
                    "  '{updated_at}', to_jsonb(EXTRACT(EPOCH FROM NOW()))) "
                    "WHERE job_id=%s",
                    (job_id,),
                )
                conn.commit()

        def waiter():
            barrier.wait(timeout=10)
            try:
                results["updated"] = sched.update_job_status(
                    job_id,
                    "failed",
                    expected_status="assigned",
                    failure_reason="should-not-apply",
                )
            except Exception as e:
                results["error"] = e

        t_hold = threading.Thread(target=holder, name="lock-holder")
        t_wait = threading.Thread(target=waiter, name="cas-waiter")
        t_hold.start()
        t_wait.start()
        t_hold.join(timeout=15)
        t_wait.join(timeout=15)
        assert not t_hold.is_alive() and not t_wait.is_alive()
        assert "error" not in results, results.get("error")
        assert results.get("updated") is None
        assert _job_status(job_id)["status"] == "running"
