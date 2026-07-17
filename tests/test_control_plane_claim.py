"""Stage-B queue claim (§10.2) — ordering, exclusivity, expiry, CAS release.

Runs against the real test PostgreSQL; the two-connection tests exercise
actual FOR UPDATE SKIP LOCKED semantics, not mocks.
"""

import json
import time
import uuid

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _migrated = (
            _c.execute("SELECT to_regclass('job_attempts')").fetchone()[0] is not None
        )
except Exception as _e:  # pragma: no cover - skip path
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None
else:
    if not _migrated:  # pragma: no cover - skip path
        pytestmark = pytest.mark.skip("test database not migrated to >= 055")

from control_plane.scheduler.claim import (
    claim_next_job,
    clear_expired_claims,
    release_claim,
)


@pytest.fixture
def cleanup_jobs():
    ids = []
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids:
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        conn.commit()


def _mkjob(
    cleanup,
    *,
    phase="pending",
    desired="running",
    priority=0,
    fair_share=0,
    queued_offset_sec=0.0,
    next_schedule_delay_sec=None,
):
    job_id = f"job-claim-{uuid.uuid4().hex[:10]}"
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload,
                                 phase, desired_state, effective_priority,
                                 fair_share_finish, queued_at, next_schedule_at)
               VALUES (%s, 'queued', %s, %s, %s, %s, %s, %s, %s,
                       clock_timestamp() + make_interval(secs => %s),
                       CASE WHEN %s::float8 IS NULL THEN NULL
                            ELSE clock_timestamp() + make_interval(secs => %s) END)
            """,
            (
                job_id, priority, time.time(), json.dumps({"name": job_id}),
                phase, desired, priority, fair_share, queued_offset_sec,
                next_schedule_delay_sec, next_schedule_delay_sec or 0.0,
            ),
        )
        conn.commit()
    cleanup.append(job_id)
    return job_id


def _claim(conn=None, replica="replica-test", ttl=15):
    if conn is not None:
        return claim_next_job(conn, replica_id=replica, claim_ttl_sec=ttl)
    with _pool.connection() as c:
        got = claim_next_job(c, replica_id=replica, claim_ttl_sec=ttl)
        c.commit()
        return got


class TestClaimOrdering:
    def test_higher_priority_claimed_first(self, cleanup_jobs):
        low = _mkjob(cleanup_jobs, priority=1)
        high = _mkjob(cleanup_jobs, priority=10)
        got = _claim()
        assert got is not None and got.job_id == high
        got2 = _claim()
        assert got2 is not None and got2.job_id == low

    def test_fair_share_breaks_priority_ties(self, cleanup_jobs):
        behind = _mkjob(cleanup_jobs, priority=5, fair_share=100)
        ahead = _mkjob(cleanup_jobs, priority=5, fair_share=1)
        got = _claim()
        assert got is not None and got.job_id == ahead
        assert _claim().job_id == behind

    def test_fifo_within_same_key(self, cleanup_jobs):
        first = _mkjob(cleanup_jobs, queued_offset_sec=-20)
        second = _mkjob(cleanup_jobs, queued_offset_sec=-10)
        assert _claim().job_id == first
        assert _claim().job_id == second


class TestClaimEligibility:
    def test_ignores_non_pending_and_stopped_desired(self, cleanup_jobs):
        _mkjob(cleanup_jobs, phase="running")
        _mkjob(cleanup_jobs, phase="pending", desired="stopped")
        assert _claim() is None

    def test_ignores_backoff_jobs_until_due(self, cleanup_jobs):
        _mkjob(cleanup_jobs, next_schedule_delay_sec=3600.0)
        assert _claim() is None

    def test_due_backoff_job_is_claimable(self, cleanup_jobs):
        job = _mkjob(cleanup_jobs, next_schedule_delay_sec=-5.0)
        got = _claim()
        assert got is not None and got.job_id == job


class TestClaimExclusivity:
    def test_claimed_job_not_reclaimable_within_ttl(self, cleanup_jobs):
        _mkjob(cleanup_jobs)
        first = _claim(replica="replica-a")
        assert first is not None
        assert _claim(replica="replica-b") is None

    def test_skip_locked_lets_second_replica_take_other_job(self, cleanup_jobs):
        _mkjob(cleanup_jobs, priority=10)
        _mkjob(cleanup_jobs, priority=1)
        with _pool.connection() as c1:
            with _pool.connection() as c2:
                a = claim_next_job(c1, replica_id="replica-a")  # uncommitted lock
                b = claim_next_job(c2, replica_id="replica-b")
                assert a is not None and b is not None
                assert a.job_id != b.job_id  # SKIP LOCKED, no blocking, no dup
                c1.commit()
                c2.commit()

    def test_expired_claim_is_stealable(self, cleanup_jobs):
        job = _mkjob(cleanup_jobs)
        first = _claim(replica="replica-a", ttl=1)
        assert first is not None and first.job_id == job
        time.sleep(1.2)
        second = _claim(replica="replica-b")
        assert second is not None and second.job_id == job
        assert second.schedule_attempt_count == first.schedule_attempt_count + 1


class TestReleaseClaim:
    def test_release_with_backoff_and_reason(self, cleanup_jobs):
        job = _mkjob(cleanup_jobs)
        claimed = _claim()
        with _pool.connection() as conn:
            ok = release_claim(
                conn, claimed.job_id, claimed.claim_token,
                reason_code="no_capacity", requeue_delay_sec=3600.0,
            )
            conn.commit()
        assert ok is True
        assert _claim() is None  # backing off
        with _pool.connection() as conn:
            row = conn.execute(
                "SELECT reason_code, schedule_claim_token FROM jobs WHERE job_id=%s",
                (job,),
            ).fetchone()
        assert row[0] == "no_capacity" and row[1] is None

    def test_release_with_stale_token_is_noop(self, cleanup_jobs):
        _mkjob(cleanup_jobs)
        claimed = _claim()
        with _pool.connection() as conn:
            ok = release_claim(
                conn, claimed.job_id, str(uuid.uuid4()),
                reason_code="no_capacity",
            )
            conn.commit()
        assert ok is False
        # Original claim still holds: nobody else can take the job.
        assert _claim(replica="replica-b") is None


class TestExpiredClaimSweep:
    def test_sweep_frees_expired_claims_only(self, cleanup_jobs):
        expired_job = _mkjob(cleanup_jobs)
        live_job = _mkjob(cleanup_jobs, priority=-5)
        assert _claim(replica="replica-a", ttl=1).job_id == expired_job
        assert _claim(replica="replica-a", ttl=60).job_id == live_job
        time.sleep(1.2)
        with _pool.connection() as conn:
            freed = clear_expired_claims(conn)
            conn.commit()
        assert freed == 1
        with _pool.connection() as conn:
            row = conn.execute(
                "SELECT schedule_claim_owner FROM jobs WHERE job_id=%s",
                (expired_job,),
            ).fetchone()
            live = conn.execute(
                "SELECT schedule_claim_owner FROM jobs WHERE job_id=%s",
                (live_job,),
            ).fetchone()
        assert row[0] is None and live[0] == "replica-a"
