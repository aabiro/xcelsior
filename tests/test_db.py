"""Tests for Xcelsior database abstraction — SQLite backend, CRUD, state, dual-write engine."""

import json
import os
import tempfile
import time

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_DB_BACKEND", "sqlite")

from db import (
    DatabaseOps,
    sqlite_connection,
    sqlite_transaction,
)


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Redirect SQLite DB to a temp directory for test isolation."""
    db_file = str(tmp_path / "test_xcelsior.db")
    monkeypatch.setenv("XCELSIOR_DB_PATH", db_file)
    # Also patch the module-level default
    import db as db_mod
    monkeypatch.setattr(db_mod, "DEFAULT_DB_FILE", db_file)
    yield db_file


# ── SQLite Connection ─────────────────────────────────────────────────


class TestSQLiteConnection:
    """Verify SQLite connection management and table creation."""

    def test_connection_creates_tables(self):
        with sqlite_connection() as conn:
            # Check that tables exist
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            names = [t["name"] for t in tables]
            assert "state" in names
            assert "jobs" in names
            assert "hosts" in names

    def test_wal_mode_enabled(self):
        with sqlite_connection() as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()
            assert mode[0] == "wal"


class TestSQLiteTransaction:
    def test_commit_on_success(self):
        with sqlite_transaction() as conn:
            conn.execute(
                "INSERT INTO state(namespace, payload) VALUES (?, ?)",
                ("test-ns", '{"ok": true}'),
            )
        # Verify outside transaction
        with sqlite_connection() as conn:
            row = conn.execute(
                "SELECT payload FROM state WHERE namespace = ?", ("test-ns",)
            ).fetchone()
            assert row is not None

    def test_rollback_on_error(self):
        try:
            with sqlite_transaction() as conn:
                conn.execute(
                    "INSERT INTO state(namespace, payload) VALUES (?, ?)",
                    ("rollback-test", '{"x": 1}'),
                )
                raise ValueError("Force rollback")
        except ValueError:
            pass

        with sqlite_connection() as conn:
            row = conn.execute(
                "SELECT payload FROM state WHERE namespace = ?",
                ("rollback-test",),
            ).fetchone()
            assert row is None


# ── DatabaseOps: Payload Encoding ─────────────────────────────────────


class TestPayloadEncoding:
    def test_decode_dict_returns_dict(self):
        assert DatabaseOps.decode_payload({"a": 1}) == {"a": 1}

    def test_decode_json_string(self):
        result = DatabaseOps.decode_payload('{"b": 2}')
        assert result == {"b": 2}

    def test_decode_none(self):
        assert DatabaseOps.decode_payload(None) is None

    def test_decode_invalid_json(self):
        assert DatabaseOps.decode_payload("not-json") is None

    def test_encode_dict(self):
        result = DatabaseOps.encode_payload({"c": 3})
        assert json.loads(result) == {"c": 3}

    def test_encode_string_passthrough(self):
        assert DatabaseOps.encode_payload('{"d": 4}') == '{"d": 4}'


# ── DatabaseOps: Job CRUD ─────────────────────────────────────────────


class TestJobCRUD:
    def _make_job(self, job_id="j-1", status="queued", priority=0, **extra):
        job = {
            "job_id": job_id,
            "name": f"test-{job_id}",
            "status": status,
            "priority": priority,
            "submitted_at": time.time(),
            "host_id": None,
            "vram_needed_gb": 8,
        }
        job.update(extra)
        return job

    def test_upsert_and_get_job(self):
        job = self._make_job()
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_job(conn, job)
        with sqlite_connection() as conn:
            loaded = DatabaseOps.get_job(conn, "j-1")
            assert loaded is not None
            assert loaded["job_id"] == "j-1"
            assert loaded["name"] == "test-j-1"

    def test_get_nonexistent_job(self):
        with sqlite_connection() as conn:
            assert DatabaseOps.get_job(conn, "nonexistent") is None

    def test_upsert_updates_existing(self):
        job = self._make_job()
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_job(conn, job)
        job["status"] = "running"
        job["host_id"] = "h-1"
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_job(conn, job)
        with sqlite_connection() as conn:
            loaded = DatabaseOps.get_job(conn, "j-1")
            assert loaded["status"] == "running"
            assert loaded["host_id"] == "h-1"

    def test_load_jobs_all(self):
        with sqlite_transaction() as conn:
            for i in range(3):
                DatabaseOps.upsert_job(conn, self._make_job(f"j-{i}"))
        with sqlite_connection() as conn:
            jobs = DatabaseOps.load_jobs(conn)
            assert len(jobs) == 3

    def test_load_jobs_by_status(self):
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_job(conn, self._make_job("j-q", status="queued"))
            DatabaseOps.upsert_job(conn, self._make_job("j-r", status="running"))
        with sqlite_connection() as conn:
            queued = DatabaseOps.load_jobs(conn, status="queued")
            assert len(queued) == 1
            assert queued[0]["job_id"] == "j-q"

    def test_delete_job(self):
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_job(conn, self._make_job("j-del"))
        with sqlite_transaction() as conn:
            DatabaseOps.delete_job(conn, "j-del")
        with sqlite_connection() as conn:
            assert DatabaseOps.get_job(conn, "j-del") is None

    def test_delete_all_jobs(self):
        with sqlite_transaction() as conn:
            for i in range(5):
                DatabaseOps.upsert_job(conn, self._make_job(f"j-{i}"))
        with sqlite_transaction() as conn:
            DatabaseOps.delete_all_jobs(conn)
        with sqlite_connection() as conn:
            assert DatabaseOps.load_jobs(conn) == []

    def test_empty_job_id_skipped(self):
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_job(conn, {"job_id": "", "status": "queued"})
        with sqlite_connection() as conn:
            assert DatabaseOps.load_jobs(conn) == []


# ── DatabaseOps: Host CRUD ────────────────────────────────────────────


class TestHostCRUD:
    def _make_host(self, host_id="h-1", status="active", **extra):
        host = {
            "host_id": host_id,
            "ip": "10.0.0.1",
            "gpu_model": "RTX 4090",
            "vram_gb": 24,
            "free_vram_gb": 24,
            "status": status,
            "registered_at": time.time(),
            "cost_per_hour": 0.30,
        }
        host.update(extra)
        return host

    def test_upsert_and_get_host(self):
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_host(conn, self._make_host())
        with sqlite_connection() as conn:
            loaded = DatabaseOps.get_host(conn, "h-1")
            assert loaded is not None
            assert loaded["gpu_model"] == "RTX 4090"

    def test_get_nonexistent_host(self):
        with sqlite_connection() as conn:
            assert DatabaseOps.get_host(conn, "nonexistent") is None

    def test_load_hosts_all(self):
        with sqlite_transaction() as conn:
            for i in range(3):
                DatabaseOps.upsert_host(conn, self._make_host(f"h-{i}"))
        with sqlite_connection() as conn:
            hosts = DatabaseOps.load_hosts(conn)
            assert len(hosts) == 3

    def test_load_hosts_active_only(self):
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_host(conn, self._make_host("h-a", status="active"))
            DatabaseOps.upsert_host(conn, self._make_host("h-d", status="dead"))
        with sqlite_connection() as conn:
            active = DatabaseOps.load_hosts(conn, active_only=True)
            assert len(active) == 1
            assert active[0]["host_id"] == "h-a"

    def test_delete_host(self):
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_host(conn, self._make_host("h-del"))
        with sqlite_transaction() as conn:
            DatabaseOps.delete_host(conn, "h-del")
        with sqlite_connection() as conn:
            assert DatabaseOps.get_host(conn, "h-del") is None

    def test_delete_all_hosts(self):
        with sqlite_transaction() as conn:
            for i in range(3):
                DatabaseOps.upsert_host(conn, self._make_host(f"h-{i}"))
        with sqlite_transaction() as conn:
            DatabaseOps.delete_all_hosts(conn)
        with sqlite_connection() as conn:
            assert DatabaseOps.load_hosts(conn) == []

    def test_empty_host_id_skipped(self):
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_host(conn, {"host_id": "", "status": "active"})
        with sqlite_connection() as conn:
            assert DatabaseOps.load_hosts(conn) == []


# ── DatabaseOps: State Namespace ──────────────────────────────────────


class TestStateNamespace:
    def test_upsert_and_get_state(self):
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_state(conn, "config", {"key": "value"})
        with sqlite_connection() as conn:
            state = DatabaseOps.get_state(conn, "config")
            assert state == {"key": "value"}

    def test_get_nonexistent_state(self):
        with sqlite_connection() as conn:
            assert DatabaseOps.get_state(conn, "missing") is None

    def test_upsert_overwrites_state(self):
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_state(conn, "counter", {"n": 1})
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_state(conn, "counter", {"n": 42})
        with sqlite_connection() as conn:
            state = DatabaseOps.get_state(conn, "counter")
            assert state["n"] == 42

    def test_state_with_list_payload(self):
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_state(conn, "items", [1, 2, 3])
        with sqlite_connection() as conn:
            state = DatabaseOps.get_state(conn, "items")
            assert state == [1, 2, 3]

    def test_state_with_string_payload(self):
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_state(conn, "raw", '{"raw": true}')
        with sqlite_connection() as conn:
            state = DatabaseOps.get_state(conn, "raw")
            assert state == {"raw": True}


# ── DatabaseOps: Query Hosts by GPU ───────────────────────────────────


class TestQueryHostsByGPU:
    def _seed_hosts(self):
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_host(conn, {
                "host_id": "h-4090",
                "gpu_model": "RTX 4090",
                "vram_gb": 24,
                "free_vram_gb": 20,
                "status": "active",
                "registered_at": time.time(),
            })
            DatabaseOps.upsert_host(conn, {
                "host_id": "h-a100",
                "gpu_model": "A100",
                "vram_gb": 80,
                "free_vram_gb": 60,
                "status": "active",
                "registered_at": time.time(),
            })
            DatabaseOps.upsert_host(conn, {
                "host_id": "h-3060",
                "gpu_model": "RTX 3060",
                "vram_gb": 12,
                "free_vram_gb": 8,
                "status": "active",
                "registered_at": time.time(),
            })

    def test_filter_by_gpu_model(self):
        self._seed_hosts()
        with sqlite_connection() as conn:
            results = DatabaseOps.query_hosts_by_gpu(conn, gpu_model="A100")
            assert len(results) == 1
            assert results[0]["host_id"] == "h-a100"

    def test_filter_by_min_vram(self):
        self._seed_hosts()
        with sqlite_connection() as conn:
            results = DatabaseOps.query_hosts_by_gpu(conn, min_vram_gb=20)
            assert len(results) == 2
            # Sorted by free_vram_gb descending
            assert results[0]["host_id"] == "h-a100"
            assert results[1]["host_id"] == "h-4090"

    def test_filter_by_both(self):
        self._seed_hosts()
        with sqlite_connection() as conn:
            results = DatabaseOps.query_hosts_by_gpu(
                conn, gpu_model="RTX 4090", min_vram_gb=15,
            )
            assert len(results) == 1
            assert results[0]["host_id"] == "h-4090"

    def test_no_match(self):
        self._seed_hosts()
        with sqlite_connection() as conn:
            results = DatabaseOps.query_hosts_by_gpu(conn, gpu_model="H100")
            assert results == []

    def test_no_filters_returns_all_active(self):
        self._seed_hosts()
        with sqlite_connection() as conn:
            results = DatabaseOps.query_hosts_by_gpu(conn)
            assert len(results) == 3


# ── DualWriteEngine (SQLite-only mode) ────────────────────────────────


class TestDualWriteEngineSQLite:
    """Test the DualWriteEngine in sqlite-only backend mode."""

    def test_engine_connection(self):
        from db import DualWriteEngine
        engine = DualWriteEngine()
        engine.backend = "sqlite"
        with engine.connection() as (conn, backend):
            assert backend == "sqlite"
            # Connection should be usable
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            assert len(tables) > 0

    def test_engine_transaction(self):
        from db import DualWriteEngine
        engine = DualWriteEngine()
        engine.backend = "sqlite"
        with engine.transaction() as (conn, backend):
            assert backend == "sqlite"
            DatabaseOps.upsert_state(conn, "engine-test", {"v": 1})

        with sqlite_connection() as conn:
            state = DatabaseOps.get_state(conn, "engine-test")
            assert state == {"v": 1}


# ── PgEventBus (in-memory mode for SQLite backend) ────────────────────


class TestPgEventBusInMemory:
    """Test event bus in-memory fallback when using SQLite backend."""

    def test_add_and_remove_listener(self):
        from db import PgEventBus
        bus = PgEventBus()
        events = []
        bus.add_listener(lambda e: events.append(e))
        bus._dispatch_inmemory("test", {"id": "1"})
        assert len(events) == 1
        assert events[0]["type"] == "test"

    def test_notify_inmemory(self, monkeypatch):
        from db import PgEventBus
        import db as db_mod
        monkeypatch.setattr(db_mod, "DB_BACKEND", "sqlite")
        bus = PgEventBus()
        received = []
        bus.add_listener(lambda e: received.append(e))
        bus.notify("job_started", {"job_id": "j-1"})
        # In-memory dispatch is synchronous
        assert len(received) == 1
        assert received[0]["data"]["job_id"] == "j-1"

    def test_remove_listener(self):
        from db import PgEventBus
        bus = PgEventBus()
        events = []
        cb = lambda e: events.append(e)
        bus.add_listener(cb)
        bus.remove_listener(cb)
        bus._dispatch_inmemory("test", {})
        assert len(events) == 0

    def test_remove_nonexistent_listener_no_error(self):
        from db import PgEventBus
        bus = PgEventBus()
        bus.remove_listener(lambda e: None)  # Should not raise
