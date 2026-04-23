"""P3/A4 + A5 — bg_worker reconcile + user_images sweeper regression.

These are structural tests that extract the two task closures from
bg_worker.main(), substitute a stub pool, and assert the SQL + enqueue
behavior. We deliberately avoid importing a real pg pool so the tests
run fast in any environment.
"""

import os
import time
from unittest.mock import MagicMock, patch

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
        if s.startswith("select") or "select" in s.split("update", 1)[0] if "update" in s else s.startswith("select"):
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


def _extract_closures():
    """Pull the two closure functions out of bg_worker.main without
    actually launching threads. We monkeypatch threading.Thread so
    main() returns immediately after wiring tasks."""
    import bg_worker

    collected: dict[str, object] = {}

    orig_append = list.append

    def fake_thread_start(self, *a, **kw):
        return None

    # Hijack _stop so main() exits its _stop.wait() immediately.
    bg_worker._stop.set()

    with patch("bg_worker.threading.Thread") as T:
        T.return_value.start = lambda: None
        T.return_value.join = lambda *a, **kw: None
        # Capture tasks list via patching list.append is overkill; simpler:
        # replay the tasks after main builds them. main() assigns tasks
        # locally, so we capture via signal.signal wrapper — but easiest
        # is to re-import and call main with a small wrapper that
        # replaces _bg_loop so tasks are introspectable. Simpler still:
        # monkeypatch main itself. Instead, import bg_worker fresh and
        # scrape the closures via inspect after running main to the
        # thread launch point.
        # Use a lighter approach: call main() but interrupt via _stop.
        try:
            bg_worker.main()
        except SystemExit:
            pass

    # main() dropped out after wiring. Pull closures out of its frame:
    # tasks is a local — we need to introspect via the thread-Mock's
    # call_args_list which received (target=_bg_loop, args=(name, func, interval)).
    calls = T.call_args_list
    for c in calls:
        kwargs = c.kwargs
        args = kwargs.get("args") or (c.args[1] if len(c.args) > 1 else None)
        if not args:
            continue
        name, func, interval = args
        collected[name] = (func, interval)
    return collected


def test_reconcile_task_registered_with_60s_interval():
    closures = _extract_closures()
    assert "reconcile_paused_stopped" in closures, list(closures)
    _, interval = closures["reconcile_paused_stopped"]
    assert interval == 60


def test_user_images_sweeper_registered_with_300s_interval():
    closures = _extract_closures()
    assert "user_images_pending_sweeper" in closures, list(closures)
    _, interval = closures["user_images_pending_sweeper"]
    assert interval == 300


def test_reconcile_reenqueues_stale_stopped_jobs():
    closures = _extract_closures()
    func, _ = closures["reconcile_paused_stopped"]

    stale_ts = time.time() - 300.0  # > 120s old
    rows = [
        {
            "job_id": "job-A",
            "status": "stopped",
            "host_id": "host-1",
            "container_name": "xcl-job-A",
            "state_age_ts": stale_ts,
        },
        {
            "job_id": "job-B",
            "status": "stopped",
            "host_id": "host-2",
            "container_name": "xcl-job-B",
            "state_age_ts": stale_ts,
        },
    ]
    conn = _FakeConn(rows_for_select=rows, pending=False)
    pool = _FakePool(conn)

    enqueued: list[tuple[str, str, dict]] = []

    def fake_enqueue(host_id, cmd, args, **kw):
        enqueued.append((host_id, cmd, args))
        return "cmd-x"

    with patch("db._get_pg_pool", return_value=pool), \
         patch("routes.agent.enqueue_agent_command", side_effect=fake_enqueue):
        func()

    names = {(h, c) for h, c, _ in enqueued}
    assert ("host-1", "stop_container") in names
    assert ("host-2", "stop_container") in names


def test_reconcile_skips_fresh_jobs():
    closures = _extract_closures()
    func, _ = closures["reconcile_paused_stopped"]

    fresh_ts = time.time() - 30.0  # < 120s
    rows = [{
        "job_id": "job-fresh",
        "status": "stopped",
        "host_id": "host-1",
        "container_name": "xcl-job-fresh",
        "state_age_ts": fresh_ts,
    }]
    conn = _FakeConn(rows_for_select=rows)
    pool = _FakePool(conn)

    enqueued = []
    with patch("db._get_pg_pool", return_value=pool), \
         patch("routes.agent.enqueue_agent_command",
               side_effect=lambda *a, **kw: enqueued.append(a)):
        func()
    assert enqueued == [], "fresh jobs must not be reconciled"


def test_reconcile_skips_when_pending_command_exists():
    closures = _extract_closures()
    func, _ = closures["reconcile_paused_stopped"]

    stale_ts = time.time() - 300.0
    rows = [{
        "job_id": "job-pending",
        "status": "stopped",
        "host_id": "host-1",
        "container_name": "xcl-job-pending",
        "state_age_ts": stale_ts,
    }]
    conn = _FakeConn(rows_for_select=rows, pending=True)  # pending cmd exists
    pool = _FakePool(conn)

    enqueued = []
    with patch("db._get_pg_pool", return_value=pool), \
         patch("routes.agent.enqueue_agent_command",
               side_effect=lambda *a, **kw: enqueued.append(a)):
        func()
    assert enqueued == [], "must not duplicate a pending reconcile command"


def test_user_images_sweeper_marks_stale_pending_failed():
    closures = _extract_closures()
    func, _ = closures["user_images_pending_sweeper"]

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
