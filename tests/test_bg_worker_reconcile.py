"""bg_worker reconcile + user_images sweeper regression.

Structural tests: stub pool + enqueue, drive the shipped
``reconcile_paused_stopped_jobs`` path (and task registration via main).
Real Postgres class coverage lives in
``tests/test_stopped_job_stop_redelivery.py``.
"""

import os
import time
from unittest.mock import patch

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")


class _FakeCursor:
    def __init__(self, rows=None, rowcount=0):
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self, *, rows_for_select=None, rowcount=0, pending=False):
        self._rows = rows_for_select or []
        self._rowcount = rowcount
        self._pending = pending
        self.executed: list[tuple[str, tuple]] = []
        self.row_factory = None

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        s = " ".join(sql.split()).lower()
        if "from agent_commands" in s:
            return _FakeCursor(rows=[{"ok": 1}] if self._pending else [])
        if (
            s.startswith("select") or "select" in s.split("update", 1)[0]
            if "update" in s
            else s.startswith("select")
        ):
            return _FakeCursor(rows=self._rows)
        return _FakeCursor(rowcount=self._rowcount)

    def commit(self):
        pass


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def connection(self):
        pool_self = self

        class _Ctx:
            def __enter__(self_inner):
                return pool_self._conn

            def __exit__(self_inner, *a):
                return False

        return _Ctx()


def _extract_registrations():
    """Run bg_worker.main with durable register_task captured (no DB/threads)."""
    import bg_worker

    collected: dict[str, tuple] = {}

    def capture(name, func, interval, enabled=True):
        collected[name] = (func, interval, enabled)

    # Hijack _stop so main() exits its wait immediately after wiring.
    bg_worker._stop.set()

    with (
        patch("bg_worker.register_task", side_effect=capture),
        patch("bg_worker.threading.Thread") as T,
    ):
        T.return_value.start = lambda: None
        T.return_value.join = lambda *a, **kw: None
        try:
            bg_worker.main()
        except SystemExit:
            pass
    return collected


def test_reconcile_task_registered_with_60s_interval():
    closures = _extract_registrations()
    assert "reconcile_paused_stopped" in closures, list(closures)
    _, interval, _ = closures["reconcile_paused_stopped"]
    assert interval == 60


def test_serverless_reconcile_task_registered_with_45s_interval():
    closures = _extract_registrations()
    assert "serverless_reconcile" in closures, list(closures)
    _, interval, _ = closures["serverless_reconcile"]
    assert interval == 45


def test_user_images_sweeper_registered_with_300s_interval():
    closures = _extract_registrations()
    assert "user_images_pending_sweeper" in closures, list(closures)
    _, interval, _ = closures["user_images_pending_sweeper"]
    assert interval == 300


def test_is_fenced_history_job_classifier():
    import bg_worker

    assert bg_worker.is_fenced_history_job(active_attempt_id=None, has_fenced_history=False) is False
    assert bg_worker.is_fenced_history_job(active_attempt_id="aid", has_fenced_history=False) is True
    assert bg_worker.is_fenced_history_job(active_attempt_id=None, has_fenced_history=True) is True
    assert bg_worker.is_fenced_history_job(active_attempt_id="aid", has_fenced_history=True) is True


def test_reconcile_reenqueues_stale_stopped_jobs():
    import bg_worker

    stale_ts = time.time() - 300.0  # > 120s old
    rows = [
        {
            "job_id": "job-A",
            "status": "stopped",
            "host_id": "host-1",
            "container_name": "xcl-job-A",
            "state_age_ts": stale_ts,
            "active_attempt_id": None,
            "has_fenced_history": False,
        },
        {
            "job_id": "job-B",
            "status": "stopped",
            "host_id": "host-2",
            "container_name": "xcl-job-B",
            "state_age_ts": stale_ts,
            "active_attempt_id": None,
            "has_fenced_history": False,
        },
    ]
    conn = _FakeConn(rows_for_select=rows, pending=False)
    pool = _FakePool(conn)

    enqueued: list[tuple[str, str, dict]] = []

    def fake_enqueue(host_id, cmd, args, **kw):
        enqueued.append((host_id, cmd, args))
        return "cmd-x"

    with (
        patch("db._get_pg_pool", return_value=pool),
        patch("routes.agent.enqueue_agent_command", side_effect=fake_enqueue),
    ):
        n = bg_worker.reconcile_paused_stopped_jobs()

    names = {(h, c) for h, c, _ in enqueued}
    assert ("host-1", "stop_container") in names
    assert ("host-2", "stop_container") in names
    assert n == 2


def test_reconcile_skips_fenced_history_even_if_row_returned():
    """Python gate: fenced-history rows must never get unfenced stop."""
    import bg_worker

    stale_ts = time.time() - 300.0
    rows = [
        {
            "job_id": "job-fenced",
            "status": "stopped",
            "host_id": "host-1",
            "container_name": "xcl-job-fenced",
            "state_age_ts": stale_ts,
            "active_attempt_id": None,
            "has_fenced_history": True,
        },
        {
            "job_id": "job-attempt-owned",
            "status": "stopped",
            "host_id": "host-2",
            "container_name": "xcl-job-owned",
            "state_age_ts": stale_ts,
            "active_attempt_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "has_fenced_history": False,
        },
    ]
    conn = _FakeConn(rows_for_select=rows, pending=False)
    pool = _FakePool(conn)
    enqueued: list = []

    with (
        patch("db._get_pg_pool", return_value=pool),
        patch(
            "routes.agent.enqueue_agent_command",
            side_effect=lambda *a, **kw: enqueued.append(a),
        ),
    ):
        n = bg_worker.reconcile_paused_stopped_jobs()

    assert enqueued == [], "fenced-history / attempt-owned must not dual-command stop"
    assert n == 0


def test_reconcile_skips_fresh_jobs():
    import bg_worker

    fresh_ts = time.time() - 30.0  # < 120s
    rows = [
        {
            "job_id": "job-fresh",
            "status": "stopped",
            "host_id": "host-1",
            "container_name": "xcl-job-fresh",
            "state_age_ts": fresh_ts,
            "active_attempt_id": None,
            "has_fenced_history": False,
        }
    ]
    conn = _FakeConn(rows_for_select=rows)
    pool = _FakePool(conn)

    enqueued = []
    with (
        patch("db._get_pg_pool", return_value=pool),
        patch(
            "routes.agent.enqueue_agent_command", side_effect=lambda *a, **kw: enqueued.append(a)
        ),
    ):
        bg_worker.reconcile_paused_stopped_jobs()
    assert enqueued == [], "fresh jobs must not be reconciled"


def test_reconcile_skips_when_pending_command_exists():
    import bg_worker

    stale_ts = time.time() - 300.0
    rows = [
        {
            "job_id": "job-pending",
            "status": "stopped",
            "host_id": "host-1",
            "container_name": "xcl-job-pending",
            "state_age_ts": stale_ts,
            "active_attempt_id": None,
            "has_fenced_history": False,
        }
    ]
    conn = _FakeConn(rows_for_select=rows, pending=True)  # pending cmd exists
    pool = _FakePool(conn)

    enqueued = []
    with (
        patch("db._get_pg_pool", return_value=pool),
        patch(
            "routes.agent.enqueue_agent_command", side_effect=lambda *a, **kw: enqueued.append(a)
        ),
    ):
        bg_worker.reconcile_paused_stopped_jobs()
    assert enqueued == [], "must not duplicate a pending reconcile command"


def test_reconcile_throttles_recent_stop_attempts():
    import bg_worker

    stale_ts = time.time() - 300.0
    rows = [
        {
            "job_id": "job-recent",
            "status": "stopped",
            "host_id": "host-1",
            "container_name": "xcl-job-recent",
            "state_age_ts": stale_ts,
            "last_reconcile_stop_at": time.time() - 60.0,
            "reconcile_stop_count": 1,
            "active_attempt_id": None,
            "has_fenced_history": False,
        }
    ]
    conn = _FakeConn(rows_for_select=rows, pending=False)
    pool = _FakePool(conn)

    enqueued = []
    with (
        patch("db._get_pg_pool", return_value=pool),
        patch(
            "routes.agent.enqueue_agent_command", side_effect=lambda *a, **kw: enqueued.append(a)
        ),
    ):
        bg_worker.reconcile_paused_stopped_jobs()
    assert enqueued == [], "recent reconcile attempts must be throttled"


def test_reconcile_sql_excludes_fenced_history():
    """Static gate: production SELECT must exclude attempt/fenced rows."""
    import inspect

    import bg_worker

    src = inspect.getsource(bg_worker.reconcile_paused_stopped_jobs)
    assert "active_attempt_id IS NULL" in src
    assert "NOT EXISTS" in src
    assert "job_attempts" in src
    assert "stop_container" in src


def test_user_images_sweeper_marks_stale_pending_failed():
    closures = _extract_registrations()
    func, _, _ = closures["user_images_pending_sweeper"]

    conn = _FakeConn(rowcount=3)
    pool = _FakePool(conn)

    with patch("db._get_pg_pool", return_value=pool):
        func()

    # Verify an UPDATE ... SET status='failed' WHERE status='pending' ran.
    sqls = [sql for sql, _ in conn.executed]
    joined = " ".join(" ".join(s.split()).lower() for s in sqls)
    assert "update user_images" in joined
    assert "set status = 'failed'" in joined
    assert "status = 'pending'" in joined
    assert "deleted_at = 0" in joined
