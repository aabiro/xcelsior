"""Stage-B queue claim (§10.2) — ordering, exclusivity, expiry, CAS release.

Runs against the real test PostgreSQL; the two-connection tests exercise
actual FOR UPDATE SKIP LOCKED semantics, not mocks.

Since migration 059 the projection trigger derives ``phase`` /
``desired_state`` / ``effective_priority`` / ``queued_at`` from the
legacy columns on every write, so fixtures control eligibility via
``status`` and ordering via ``submitted_at`` — exactly like production
writers. Every test claims within its own scope (``gpu_model`` carries a
per-test marker) so shared-DB residue can never leak into assertions.
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


@pytest.fixture
def scope():
    """Per-test claim scope marker (doubles as the jobs' gpu_model)."""
    return f"claimscope-{uuid.uuid4().hex[:10]}"


def _mkjob(
    cleanup,
    scope,
    *,
    status="queued",
    priority=0,
    fair_share=0,
    queued_offset_sec=0.0,
    next_schedule_delay_sec=None,
):
    job_id = f"job-claim-{uuid.uuid4().hex[:10]}"
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload,
                                 fair_share_finish, next_schedule_at)
               VALUES (%s, %s, %s, %s, %s, %s,
                       CASE WHEN %s::float8 IS NULL THEN NULL
                            ELSE clock_timestamp() + make_interval(secs => %s) END)
            """,
            (
                job_id, status, priority,
                time.time() + queued_offset_sec,
                json.dumps({"name": job_id, "gpu_model": scope}),
                fair_share,
                next_schedule_delay_sec, next_schedule_delay_sec or 0.0,
            ),
        )
        conn.commit()
    cleanup.append(job_id)
    return job_id


def _claim(scope, conn=None, replica="replica-test", ttl=15):
    if conn is not None:
        return claim_next_job(
            conn, replica_id=replica, claim_ttl_sec=ttl, scope_gpu_models=[scope]
        )
    with _pool.connection() as c:
        got = claim_next_job(
            c, replica_id=replica, claim_ttl_sec=ttl, scope_gpu_models=[scope]
        )
        c.commit()
        return got


class TestClaimOrdering:
    def test_higher_priority_claimed_first(self, cleanup_jobs, scope):
        low = _mkjob(cleanup_jobs, scope, priority=1)
        high = _mkjob(cleanup_jobs, scope, priority=10)
        got = _claim(scope)
        assert got is not None and got.job_id == high
        got2 = _claim(scope)
        assert got2 is not None and got2.job_id == low

    def test_fair_share_breaks_priority_ties(self, cleanup_jobs, scope):
        behind = _mkjob(cleanup_jobs, scope, priority=5, fair_share=100)
        ahead = _mkjob(cleanup_jobs, scope, priority=5, fair_share=1)
        got = _claim(scope)
        assert got is not None and got.job_id == ahead
        assert _claim(scope).job_id == behind

    def test_fifo_within_same_key(self, cleanup_jobs, scope):
        first = _mkjob(cleanup_jobs, scope, queued_offset_sec=-20)
        second = _mkjob(cleanup_jobs, scope, queued_offset_sec=-10)
        assert _claim(scope).job_id == first
        assert _claim(scope).job_id == second


class TestClaimEligibility:
    def test_ignores_non_pending_phases(self, cleanup_jobs, scope):
        # The 059 trigger projects phase/desired_state from status:
        # running → phase 'running'; paused → phase/desired 'stopped'.
        _mkjob(cleanup_jobs, scope, status="running")
        _mkjob(cleanup_jobs, scope, status="paused")
        assert _claim(scope) is None

    def test_ignores_backoff_jobs_until_due(self, cleanup_jobs, scope):
        _mkjob(cleanup_jobs, scope, next_schedule_delay_sec=3600.0)
        assert _claim(scope) is None

    def test_due_backoff_job_is_claimable(self, cleanup_jobs, scope):
        job = _mkjob(cleanup_jobs, scope, next_schedule_delay_sec=-5.0)
        got = _claim(scope)
        assert got is not None and got.job_id == job

    def test_scope_excludes_other_partitions(self, cleanup_jobs, scope):
        """The canary partition predicate: out-of-scope jobs unclaimable."""
        _mkjob(cleanup_jobs, scope)
        assert _claim(f"other-{scope}") is None
        got = _claim(scope)
        assert got is not None

    def test_v2_optin_claimable_in_any_scope(self, cleanup_jobs, scope):
        job_id = f"job-claim-{uuid.uuid4().hex[:10]}"
        with _pool.connection() as conn:
            conn.execute(
                """INSERT INTO jobs (job_id, status, priority, submitted_at, payload)
                   VALUES (%s, 'queued', 0, %s, %s)""",
                (
                    job_id, time.time(),
                    json.dumps({"name": job_id, "scheduler": "v2",
                                "gpu_model": f"unrelated-{scope}"}),
                ),
            )
            conn.commit()
        cleanup_jobs.append(job_id)
        got = _claim(scope)  # scope doesn't match, but v2 opt-in does
        assert got is not None and got.job_id == job_id


class TestClaimExclusivity:
    def test_claimed_job_not_reclaimable_within_ttl(self, cleanup_jobs, scope):
        _mkjob(cleanup_jobs, scope)
        first = _claim(scope, replica="replica-a")
        assert first is not None
        assert _claim(scope, replica="replica-b") is None

    def test_skip_locked_lets_second_replica_take_other_job(self, cleanup_jobs, scope):
        _mkjob(cleanup_jobs, scope, priority=10)
        _mkjob(cleanup_jobs, scope, priority=1)
        with _pool.connection() as c1:
            with _pool.connection() as c2:
                a = _claim(scope, conn=c1, replica="replica-a")  # uncommitted lock
                b = _claim(scope, conn=c2, replica="replica-b")
                assert a is not None and b is not None
                assert a.job_id != b.job_id  # SKIP LOCKED, no blocking, no dup
                c1.commit()
                c2.commit()

    def test_expired_claim_is_stealable(self, cleanup_jobs, scope):
        job = _mkjob(cleanup_jobs, scope)
        first = _claim(scope, replica="replica-a", ttl=1)
        assert first is not None and first.job_id == job
        time.sleep(1.2)
        second = _claim(scope, replica="replica-b")
        assert second is not None and second.job_id == job
        assert second.schedule_attempt_count == first.schedule_attempt_count + 1


class TestReleaseClaim:
    def test_release_with_backoff_and_reason(self, cleanup_jobs, scope):
        job = _mkjob(cleanup_jobs, scope)
        claimed = _claim(scope)
        with _pool.connection() as conn:
            ok = release_claim(
                conn, claimed.job_id, claimed.claim_token,
                reason_code="no_capacity", requeue_delay_sec=3600.0,
            )
            conn.commit()
        assert ok is True
        assert _claim(scope) is None  # backing off
        with _pool.connection() as conn:
            row = conn.execute(
                "SELECT reason_code, schedule_claim_token FROM jobs WHERE job_id=%s",
                (job,),
            ).fetchone()
        assert row[0] == "no_capacity" and row[1] is None

    def test_release_with_stale_token_is_noop(self, cleanup_jobs, scope):
        _mkjob(cleanup_jobs, scope)
        claimed = _claim(scope)
        with _pool.connection() as conn:
            ok = release_claim(
                conn, claimed.job_id, str(uuid.uuid4()),
                reason_code="no_capacity",
            )
            conn.commit()
        assert ok is False
        # Original claim still holds: nobody else can take the job.
        assert _claim(scope, replica="replica-b") is None


class TestExpiredClaimSweep:
    def test_sweep_frees_expired_claims_only(self, cleanup_jobs, scope):
        expired_job = _mkjob(cleanup_jobs, scope)
        live_job = _mkjob(cleanup_jobs, scope, priority=-5)
        assert _claim(scope, replica="replica-a", ttl=1).job_id == expired_job
        assert _claim(scope, replica="replica-a", ttl=60).job_id == live_job
        time.sleep(1.2)
        with _pool.connection() as conn:
            freed = clear_expired_claims(conn)
            conn.commit()
        assert freed >= 1
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
