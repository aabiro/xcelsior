"""Phase 7.5 — Database migration tests.

Tests table existence in PostgreSQL and DualWriteEngine lifecycle.
"""

import json
import os
import tempfile
import time

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

from db import (
    DatabaseOps,
    DualWriteEngine,
    _get_pg_pool,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _pg_tables():
    """Return set of table names in current PostgreSQL database."""
    pool = _get_pg_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public'"
        ).fetchall()
        # Handle both tuple_row and dict_row (pool connections may be contaminated)
        return {r["tablename"] if isinstance(r, dict) else r[0] for r in rows}


# ── 7.5.1 — PostgreSQL tables exist ─────────────────────────────────


class TestSQLiteAutoInit:
    """Verify all expected tables exist in PostgreSQL."""

    def test_db_module_creates_state_jobs_hosts(self):
        tables = _pg_tables()
        assert "state" in tables
        assert "jobs" in tables
        assert "hosts" in tables

    def test_events_module_creates_tables(self):
        tables = _pg_tables()
        assert "events" in tables
        assert "leases" in tables

    def test_billing_module_creates_tables(self):
        tables = _pg_tables()
        assert "usage_meters" in tables
        assert "invoices" in tables

    def test_reputation_module_creates_tables(self):
        tables = _pg_tables()
        assert "reputation_scores" in tables
        assert "reputation_events" in tables

    def test_verification_module_creates_tables(self):
        tables = _pg_tables()
        assert "host_verifications" in tables
        assert "verification_history" in tables
        assert "job_failure_log" in tables

    def test_privacy_module_creates_tables(self):
        tables = _pg_tables()
        assert "retention_records" in tables
        assert "consent_records" in tables

    def test_sla_module_creates_tables(self):
        tables = _pg_tables()
        assert "sla_downtime" in tables
        assert "sla_monthly" in tables
        assert "sla_violations" in tables


# ── 7.5.2 — PostgreSQL DatabaseOps lifecycle ─────────────────────────


class TestDatabaseOps:
    """DatabaseOps write/read round-trip via PostgreSQL."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        pool = _get_pg_pool()
        yield
        with pool.connection() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id LIKE 'dbmig-%%'")
            conn.execute("DELETE FROM hosts WHERE host_id LIKE 'dbmig-%%'")

    def test_write_job_read_back_matches(self):
        pool = _get_pg_pool()
        job = {
            "job_id": "dbmig-j1",
            "name": "gpu-training",
            "status": "queued",
            "priority": 5,
            "submitted_at": time.time(),
            "vram_needed_gb": 16,
            "tier": "premium",
        }
        with pool.connection() as conn:
            DatabaseOps.upsert_job(conn, job, backend="postgres")

        with pool.connection() as conn:
            loaded = DatabaseOps.get_job(conn, "dbmig-j1", backend="postgres")
        assert loaded is not None
        assert loaded["name"] == "gpu-training"
        assert loaded["priority"] == 5
        assert loaded["tier"] == "premium"

    def test_write_host_read_back_matches(self):
        pool = _get_pg_pool()
        host = {
            "host_id": "dbmig-h1",
            "ip": "10.0.0.5",
            "gpu_model": "RTX 4090",
            "total_vram_gb": 24,
            "free_vram_gb": 24,
            "status": "active",
            "registered_at": time.time(),
        }
        with pool.connection() as conn:
            DatabaseOps.upsert_host(conn, host, backend="postgres")

        with pool.connection() as conn:
            loaded = DatabaseOps.get_host(conn, "dbmig-h1", backend="postgres")
        assert loaded is not None
        assert loaded["gpu_model"] == "RTX 4090"
        assert loaded["free_vram_gb"] == 24

    def test_state_namespace_round_trip(self):
        pool = _get_pg_pool()
        data = {"last_run": time.time(), "count": 42}
        with pool.connection() as conn:
            DatabaseOps.upsert_state(conn, "dbmig_test_ns", data, backend="postgres")

        with pool.connection() as conn:
            loaded = DatabaseOps.get_state(conn, "dbmig_test_ns", backend="postgres")
        assert loaded["count"] == 42
        # cleanup
        with pool.connection() as conn:
            conn.execute("DELETE FROM state WHERE namespace = 'dbmig_test_ns'")

    def test_upsert_updates_existing_preserving_schema(self):
        """Upsert with changed status preserves other fields."""
        pool = _get_pg_pool()
        job = {
            "job_id": "dbmig-upsert-j1",
            "name": "ml-run",
            "status": "queued",
            "priority": 3,
            "submitted_at": time.time(),
            "vram_needed_gb": 8,
        }
        with pool.connection() as conn:
            DatabaseOps.upsert_job(conn, job, backend="postgres")

        job["status"] = "running"
        job["host_id"] = "rig-99"
        with pool.connection() as conn:
            DatabaseOps.upsert_job(conn, job, backend="postgres")

        with pool.connection() as conn:
            loaded = DatabaseOps.get_job(conn, "dbmig-upsert-j1", backend="postgres")
        assert loaded["status"] == "running"
        assert loaded["host_id"] == "rig-99"
        assert loaded["name"] == "ml-run"

    def test_delete_then_read_returns_none(self):
        pool = _get_pg_pool()
        with pool.connection() as conn:
            DatabaseOps.upsert_job(
                conn,
                {
                    "job_id": "dbmig-del-j1",
                    "status": "queued",
                    "priority": 0,
                    "submitted_at": time.time(),
                },
                backend="postgres",
            )
            DatabaseOps.delete_job(conn, "dbmig-del-j1", backend="postgres")

        with pool.connection() as conn:
            assert DatabaseOps.get_job(conn, "dbmig-del-j1", backend="postgres") is None

    def test_load_jobs_filter_by_status(self):
        pool = _get_pg_pool()
        with pool.connection() as conn:
            for i, status in enumerate(["queued", "running", "completed", "queued"]):
                DatabaseOps.upsert_job(
                    conn,
                    {
                        "job_id": f"dbmig-filter-j{i}",
                        "status": status,
                        "priority": 1,
                        "submitted_at": time.time() + i,
                    },
                    backend="postgres",
                )

        with pool.connection() as conn:
            queued = DatabaseOps.load_jobs(conn, status="queued", backend="postgres")
        # At least 2 from this test (may be more from other tests)
        our_queued = [j for j in queued if j["job_id"].startswith("dbmig-filter-")]
        assert len(our_queued) == 2

    def test_gpu_query_filters(self):
        """query_hosts_by_gpu filters by model and min_vram."""
        pool = _get_pg_pool()
        with pool.connection() as conn:
            for hid, model, vram in [
                ("dbmig-g1", "RTX 4090", 24),
                ("dbmig-g2", "A100", 80),
                ("dbmig-g3", "RTX 4090", 12),
            ]:
                DatabaseOps.upsert_host(
                    conn,
                    {
                        "host_id": hid,
                        "gpu_model": model,
                        "free_vram_gb": vram,
                        "status": "active",
                        "registered_at": time.time(),
                    },
                    backend="postgres",
                )

        with pool.connection() as conn:
            results = DatabaseOps.query_hosts_by_gpu(conn, gpu_model="RTX 4090", backend="postgres")
        our = [h for h in results if h["host_id"].startswith("dbmig-")]
        assert len(our) == 2

        with pool.connection() as conn:
            results = DatabaseOps.query_hosts_by_gpu(conn, min_vram_gb=20, backend="postgres")
        our = [h for h in results if h["host_id"].startswith("dbmig-")]
        assert len(our) == 2
        assert all(h["free_vram_gb"] >= 20 for h in our)


# ── 7.5.4 — Event bus (SQLite fallback) ─────────────────────────────


class TestEventBusSQLiteFallback:
    """PgEventBus in-memory fallback when running SQLite."""

    def test_inmemory_listener_receives_events(self, monkeypatch):
        import db as db_mod

        monkeypatch.setattr(db_mod, "DB_BACKEND", "sqlite")
        bus = db_mod.PgEventBus()
        received = []
        bus.add_listener(lambda ev: received.append(ev))
        bus.notify("test_event", {"key": "value"})
        assert len(received) == 1
        assert received[0]["type"] == "test_event"
        assert received[0]["data"]["key"] == "value"

    def test_remove_listener(self, monkeypatch):
        import db as db_mod

        monkeypatch.setattr(db_mod, "DB_BACKEND", "sqlite")
        bus = db_mod.PgEventBus()
        received = []
        cb = lambda ev: received.append(ev)
        bus.add_listener(cb)
        bus.remove_listener(cb)
        bus.notify("test_event", {"key": "value"})
        assert len(received) == 0

    def test_remove_nonexistent_listener_noop(self, monkeypatch):
        import db as db_mod

        monkeypatch.setattr(db_mod, "DB_BACKEND", "sqlite")
        bus = db_mod.PgEventBus()
        bus.remove_listener(lambda ev: None)  # Should not raise

    def test_emit_event_convenience(self, monkeypatch):
        import db as db_mod

        monkeypatch.setattr(db_mod, "DB_BACKEND", "sqlite")
        received = []
        db_mod.event_bus._listeners.clear()
        db_mod.event_bus.add_listener(lambda ev: received.append(ev))
        db_mod.emit_event("job_completed", {"job_id": "j-123"})
        assert len(received) == 1
        assert received[0]["data"]["job_id"] == "j-123"
        db_mod.event_bus._listeners.clear()


# ── 7.5.5 — Alembic migration file validation ───────────────────────


class TestAlembicMigrationFiles:
    """Validate migration scripts are importable and properly chained."""

    def _load_migration(self, name):
        import importlib.util
        import pathlib

        path = pathlib.Path(__file__).resolve().parent.parent / "migrations" / "versions" / name
        spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_001_initial_schema_importable(self):
        m = self._load_migration("001_initial_schema.py")
        assert hasattr(m, "revision")
        assert hasattr(m, "upgrade")
        assert hasattr(m, "downgrade")
        assert m.down_revision is None

    def test_002_spot_pricing_importable(self):
        m = self._load_migration("002_spot_pricing_and_security.py")
        assert m.down_revision == "001"
        assert hasattr(m, "upgrade")
        assert hasattr(m, "downgrade")

    def test_migration_chain_is_linear(self):
        """Verify there's a single linear chain: None → 001 → 002."""
        m1 = self._load_migration("001_initial_schema.py")
        m2 = self._load_migration("002_spot_pricing_and_security.py")
        assert m1.down_revision is None
        assert m2.down_revision == m1.revision
