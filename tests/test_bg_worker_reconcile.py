"""Maintenance registrations exclude retired unfenced redelivery."""

import os
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


def test_unfenced_stop_redelivery_is_not_registered():
    closures = _extract_registrations()
    assert "reconcile_paused_stopped" not in closures


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


def test_unfenced_stop_redelivery_code_is_removed():
    import bg_worker
    from control_plane.task_identities import TASK_SERVICES

    assert not hasattr(bg_worker, "reconcile_paused_stopped_jobs")
    assert "reconcile_paused_stopped" not in TASK_SERVICES


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
