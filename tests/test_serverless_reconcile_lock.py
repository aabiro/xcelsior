"""SLVRRepo.reconcile_lock — same-connection session advisory lock.

Regression tests for the blueprint §2.5 defect: the old
try_advisory_lock/release_advisory_lock pair used two independent pool
checkouts, so the session lock could be acquired on one pooled
connection and "released" on a different one — leaking the lock until
that pooled session happened to close.
"""

import uuid

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
except Exception as _e:  # pragma: no cover - skip path
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None

from serverless.repo import ServerlessRepo

# Unique key per test run so parallel CI databases never collide.
_TEST_KEY = 0x7E570000 | (uuid.uuid4().int & 0xFFFF)


def _try_lock_raw(conn, key=_TEST_KEY) -> bool:
    row = conn.execute("SELECT pg_try_advisory_lock(%s)", (key,)).fetchone()
    return bool(row[0] if not isinstance(row, dict) else list(row.values())[0])


def _unlock_raw(conn, key=_TEST_KEY) -> None:
    conn.execute("SELECT pg_advisory_unlock(%s)", (key,))
    conn.commit()


class TestReconcileLock:
    def test_lock_excludes_other_sessions_and_releases_on_exit(self):
        repo = ServerlessRepo()
        with _pool.connection() as other:
            with repo.reconcile_lock(lock_key=_TEST_KEY) as acquired:
                assert acquired is True
                # Another physical session must not get the lock while held.
                assert _try_lock_raw(other) is False
            # After exit the lock is gone — proving release happened on the
            # same session that acquired it, not on a random pool checkout.
            assert _try_lock_raw(other) is True
            _unlock_raw(other)

    def test_release_happens_even_when_reconcile_raises(self):
        repo = ServerlessRepo()
        with pytest.raises(RuntimeError):
            with repo.reconcile_lock(lock_key=_TEST_KEY) as acquired:
                assert acquired is True
                raise RuntimeError("reconcile blew up")
        with _pool.connection() as other:
            assert _try_lock_raw(other) is True
            _unlock_raw(other)

    def test_second_holder_sees_not_acquired(self):
        repo_a = ServerlessRepo()
        repo_b = ServerlessRepo()
        with repo_a.reconcile_lock(lock_key=_TEST_KEY) as a:
            assert a is True
            with repo_b.reconcile_lock(lock_key=_TEST_KEY) as b:
                assert b is False
        # A's release frees it for B afterwards.
        with repo_b.reconcile_lock(lock_key=_TEST_KEY) as b2:
            assert b2 is True

    def test_service_reconcile_all_skips_when_lock_held(self, monkeypatch):
        """End-to-end: reconcile_all reports lock_held instead of racing.

        Redirected to a test-unique key: PostgreSQL advisory locks are
        cluster-wide, so the production key can be legitimately held by
        another environment's service sharing this PG instance (the
        pre-fix leak from a long-lived pool session was observed doing
        exactly that during verification).
        """
        from serverless.service import get_serverless_service

        svc = get_serverless_service()
        real_lock = ServerlessRepo.reconcile_lock

        def redirected(self, lock_key=0x534C5652):
            return real_lock(self, lock_key=_TEST_KEY)

        monkeypatch.setattr(ServerlessRepo, "reconcile_lock", redirected)
        with _pool.connection() as holder:
            assert _try_lock_raw(holder, _TEST_KEY) is True
            try:
                result = svc.reconcile_all()
            finally:
                _unlock_raw(holder, _TEST_KEY)
        assert result == {"skipped": True, "reason": "lock_held"}
