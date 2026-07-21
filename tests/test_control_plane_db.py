"""control_plane.db — transaction envelope, typed retry, advisory locks.

Runs against the real test PostgreSQL. Retry-classification paths that
cannot be provoked safely on a healthy database (ambiguous commit) are
exercised with a stub connection instead — the classification logic is
identical either way.
"""

import hashlib
import uuid

import psycopg
import pytest
from psycopg.errors import CheckViolation, SerializationFailure

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _migrated = (
            _c.execute("SELECT to_regclass('scheduled_tasks')").fetchone()[0]
            is not None
        )
except Exception as _e:  # pragma: no cover - skip path
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None
else:
    if not _migrated:  # pragma: no cover - skip path
        pytestmark = pytest.mark.skip("test database not migrated to >= 056")

import control_plane.db as cpdb
from control_plane.db import (
    AmbiguousCommitError,
    RetryBudgetExceeded,
    control_plane_transaction,
    is_transient_error,
    run_transaction,
    stable_advisory_key,
    try_advisory_xact_lock,
)


@pytest.fixture
def task_name():
    """A unique scheduled_tasks row name, deleted after the test."""
    name = f"cp-db-test-{uuid.uuid4().hex[:10]}"
    yield name
    if _pool is None:
        return
    with _pool.connection() as conn:
        conn.execute("DELETE FROM scheduled_tasks WHERE task_name=%s", (name,))
        conn.commit()


def _insert_task(conn, name):
    conn.execute(
        "INSERT INTO scheduled_tasks (task_name, interval_seconds) VALUES (%s, 60)",
        (name,),
    )


def _task_exists(name) -> bool:
    with _pool.connection() as conn:
        return (
            conn.execute(
                "SELECT 1 FROM scheduled_tasks WHERE task_name=%s", (name,)
            ).fetchone()
            is not None
        )


# ── Transaction envelope ─────────────────────────────────────────────


class TestControlPlaneTransaction:
    def test_commit_on_clean_exit(self, task_name):
        with control_plane_transaction() as conn:
            _insert_task(conn, task_name)
        assert _task_exists(task_name)

    def test_rollback_on_error(self, task_name):
        with pytest.raises(RuntimeError):
            with control_plane_transaction() as conn:
                _insert_task(conn, task_name)
                raise RuntimeError("boom")
        assert not _task_exists(task_name)

    def test_local_timeouts_applied(self):
        with control_plane_transaction(
            statement_timeout_ms=1234, lock_timeout_ms=567
        ) as conn:
            assert conn.execute("SHOW statement_timeout").fetchone()[0] == "1234ms"
            assert conn.execute("SHOW lock_timeout").fetchone()[0] == "567ms"

    def test_timeouts_do_not_leak_to_pool(self):
        # SET LOCAL dies with the transaction; the next checkout must see
        # the server/pool default again, not our per-txn values.
        with control_plane_transaction(statement_timeout_ms=1234) as conn:
            pass
        with _pool.connection() as conn:
            assert conn.execute("SHOW statement_timeout").fetchone()[0] != "1234ms"


# ── Typed retry ──────────────────────────────────────────────────────


class TestRunTransaction:
    def test_returns_value_and_commits(self, task_name):
        def fn(conn):
            _insert_task(conn, task_name)
            return 42

        assert run_transaction(fn) == 42
        assert _task_exists(task_name)

    def test_retries_serialization_failure_with_fresh_transaction(self, task_name):
        calls = {"n": 0}

        def fn(conn):
            calls["n"] += 1
            _insert_task(conn, task_name)
            if calls["n"] == 1:
                raise SerializationFailure("simulated 40001")
            return "ok"

        # The first attempt's insert must roll back, or the second
        # attempt's identical insert would hit the primary key.
        assert run_transaction(fn, base_backoff_ms=1, max_backoff_ms=2) == "ok"
        assert calls["n"] == 2
        assert _task_exists(task_name)

    def test_non_transient_error_not_retried(self, task_name):
        calls = {"n": 0}

        def fn(conn):
            calls["n"] += 1
            conn.execute(
                "INSERT INTO scheduled_tasks (task_name, interval_seconds) "
                "VALUES (%s, 0)",  # violates interval_seconds > 0
                (task_name,),
            )

        with pytest.raises(CheckViolation):
            run_transaction(fn, base_backoff_ms=1, max_backoff_ms=2)
        assert calls["n"] == 1

    def test_retry_budget_exceeded(self):
        calls = {"n": 0}

        def fn(conn):
            calls["n"] += 1
            raise SerializationFailure("simulated 40001")

        with pytest.raises(RetryBudgetExceeded) as excinfo:
            run_transaction(fn, max_attempts=3, base_backoff_ms=1, max_backoff_ms=2)
        assert calls["n"] == 3
        assert isinstance(excinfo.value.__cause__, SerializationFailure)

    def test_ambiguous_commit_never_retried(self, monkeypatch):
        class _StubConn:
            def execute(self, *_a, **_k):
                return self

            def fetchone(self):
                return (1,)

            def rollback(self):
                pass

            def commit(self):
                raise psycopg.OperationalError("connection lost during COMMIT")

        class _StubCtx:
            def __enter__(self):
                return _StubConn()

            def __exit__(self, *exc):
                return False

        class _StubPool:
            def connection(self):
                return _StubCtx()

        monkeypatch.setattr(cpdb, "_pool", lambda: _StubPool())
        calls = {"n": 0}

        def fn(conn):
            calls["n"] += 1
            return "value"

        with pytest.raises(AmbiguousCommitError):
            run_transaction(fn, base_backoff_ms=1, max_backoff_ms=2)
        assert calls["n"] == 1  # ambiguous outcome is never blindly retried


class TestTransientClassification:
    def test_sqlstates(self):
        assert is_transient_error(SerializationFailure("x"))
        assert is_transient_error(psycopg.errors.DeadlockDetected("x"))
        assert is_transient_error(psycopg.OperationalError("connection refused"))
        assert not is_transient_error(CheckViolation("x"))
        assert not is_transient_error(psycopg.errors.UniqueViolation("x"))
        assert not is_transient_error(ValueError("x"))


# ── Advisory locks (§2.5) ────────────────────────────────────────────


class TestAdvisoryXactLock:
    def test_key_is_stable_documented_mapping(self):
        digest = hashlib.sha256(b"serverless_endpoint:ep-123").digest()
        expected = int.from_bytes(digest[:8], "big", signed=True)
        assert stable_advisory_key("serverless_endpoint", "ep-123") == expected
        # Same inputs, same key; different inputs, different keys.
        assert stable_advisory_key("serverless_endpoint", "ep-123") == expected
        assert stable_advisory_key("serverless_endpoint", "ep-124") != expected
        assert stable_advisory_key("job", "ep-123") != expected

    def test_lock_excludes_concurrent_transaction_and_releases_on_commit(self):
        resource = f"ep-{uuid.uuid4().hex[:8]}"
        with _pool.connection() as c1:
            with _pool.connection() as c2:
                assert try_advisory_xact_lock(c1, "test_lock", resource) is True
                # Second transaction on a different connection must lose.
                assert try_advisory_xact_lock(c2, "test_lock", resource) is False
                c1.commit()  # transaction ends → lock releases automatically
                assert try_advisory_xact_lock(c2, "test_lock", resource) is True
                c2.commit()

    def test_reentrant_within_same_transaction(self):
        resource = f"ep-{uuid.uuid4().hex[:8]}"
        with _pool.connection() as conn:
            assert try_advisory_xact_lock(conn, "test_lock", resource) is True
            assert try_advisory_xact_lock(conn, "test_lock", resource) is True
            conn.commit()


# ── Schema compatibility contract (§13.8 / ADR-009) ──────────────────


class TestSchemaCompat:
    def test_current_database_is_compatible(self):
        from control_plane.schema_compat import assert_schema_compatible

        with _pool.connection() as conn:
            compat = assert_schema_compatible(conn)
        assert compat.compatible
        assert compat.current is not None and int(compat.current) >= 57

    def test_min_revision_enforced(self, monkeypatch):
        from control_plane.schema_compat import (
            SchemaIncompatibleError,
            assert_schema_compatible,
            check_schema_compatible,
        )

        monkeypatch.setenv("XCELSIOR_DB_SCHEMA_MIN_REVISION", "999")
        with _pool.connection() as conn:
            compat = check_schema_compatible(conn)
            assert not compat.compatible and "requires >= 999" in compat.reason
            with pytest.raises(SchemaIncompatibleError):
                assert_schema_compatible(conn)

    def test_max_revision_enforced(self, monkeypatch):
        from control_plane.schema_compat import check_schema_compatible

        monkeypatch.setenv("XCELSIOR_DB_SCHEMA_MIN_REVISION", "001")
        monkeypatch.setenv("XCELSIOR_DB_SCHEMA_MAX_REVISION", "010")
        with _pool.connection() as conn:
            compat = check_schema_compatible(conn)
        assert not compat.compatible and "exceeds supported maximum" in compat.reason

    def test_non_numeric_revision_is_incompatible(self, monkeypatch):
        """A min revision that is not an ancestor of current is incompatible.

        Hash / unknown branch ids are resolved via the Alembic graph when
        available; a fabricated id is never treated as a silent pass.
        """
        from control_plane.schema_compat import check_schema_compatible

        monkeypatch.setenv("XCELSIOR_DB_SCHEMA_MIN_REVISION", "abcdef")
        with _pool.connection() as conn:
            compat = check_schema_compatible(conn)
        assert not compat.compatible
        assert (
            "non-numeric" in compat.reason
            or "requires >=" in compat.reason
            or "unknown" in compat.reason
        )
