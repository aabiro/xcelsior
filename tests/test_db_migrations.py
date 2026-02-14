"""Phase 7.5 — Database migration tests.

Tests SQLite auto-init across all modules, dual-write consistency,
and DualWriteEngine lifecycle without requiring a live PostgreSQL server.
"""

import json
import os
import sqlite3
import tempfile
import time

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_DB_BACKEND", "sqlite")

from db import (
    DatabaseOps,
    DualWriteEngine,
    _ensure_sqlite_tables,
    sqlite_connection,
    sqlite_transaction,
)


# ── Helpers ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_file = str(tmp_path / "migration_test.db")
    monkeypatch.setenv("XCELSIOR_DB_PATH", db_file)
    import db as db_mod
    monkeypatch.setattr(db_mod, "DEFAULT_DB_FILE", db_file)
    # Reset engine singleton so each test gets fresh config
    monkeypatch.setattr(db_mod, "_engine", None)
    yield db_file


# ── 7.5.1 — SQLite auto-init: core tables ───────────────────────────


class TestSQLiteAutoInit:
    """Each module's SQLite tables are created on first access."""

    def test_db_module_creates_state_jobs_hosts(self):
        """db.py auto-creates state, jobs, hosts tables."""
        with sqlite_connection() as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "state" in tables
        assert "jobs" in tables
        assert "hosts" in tables

    def test_db_creates_queue_index(self):
        with sqlite_connection() as conn:
            indexes = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()}
        assert "idx_jobs_queue" in indexes
        assert "idx_hosts_status" in indexes

    def test_events_module_creates_tables(self):
        """events.py auto-creates events + leases tables."""
        from events import EventStore
        store = EventStore()
        tables = set()
        with sqlite3.connect(store.db_path) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "events" in tables
        assert "leases" in tables

    def test_billing_module_creates_tables(self, tmp_path):
        """billing.py auto-creates usage_meters, invoices, wallets, etc."""
        db = str(tmp_path / "billing_init.db")
        from billing import BillingEngine
        be = BillingEngine(db_path=db)
        with sqlite3.connect(be.db_path) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "usage_meters" in tables
        assert "invoices" in tables

    def test_reputation_module_creates_tables(self, tmp_path):
        """reputation.py auto-creates reputation_scores + reputation_events."""
        from reputation import ReputationStore
        store = ReputationStore(str(tmp_path / "rep_init.db"))
        with sqlite3.connect(store.db_path) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "reputation_scores" in tables
        assert "reputation_events" in tables

    def test_verification_module_creates_tables(self, tmp_path):
        """verification.py auto-creates host_verifications, verification_history, job_failure_log."""
        from verification import VerificationStore
        store = VerificationStore(str(tmp_path / "verif_init.db"))
        with sqlite3.connect(store.db_path) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "host_verifications" in tables
        assert "verification_history" in tables
        assert "job_failure_log" in tables

    def test_privacy_module_creates_tables(self, tmp_path, monkeypatch):
        """privacy.py auto-creates retention_records, consent_records, privacy_configs."""
        db = str(tmp_path / "privacy_init.db")
        monkeypatch.setenv("XCELSIOR_PRIVACY_DB", db)
        from privacy import DataLifecycleManager
        dlm = DataLifecycleManager(db_path=db)
        with sqlite3.connect(db) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "retention_records" in tables
        assert "consent_records" in tables

    def test_sla_module_creates_tables(self, tmp_path, monkeypatch):
        """sla.py auto-creates sla_downtime, sla_monthly, sla_violations."""
        db = str(tmp_path / "sla_init.db")
        monkeypatch.setenv("XCELSIOR_SLA_DB", db)
        from sla import SLAEngine
        engine = SLAEngine(db_path=db)
        with sqlite3.connect(db) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "sla_downtime" in tables
        assert "sla_monthly" in tables
        assert "sla_violations" in tables


# ── 7.5.2 — Dual-write engine lifecycle ─────────────────────────────


class TestDualWriteEngine:
    """DualWriteEngine in SQLite-only mode (no Postgres required)."""

    def test_engine_defaults_to_sqlite(self, monkeypatch):
        import db as db_mod
        monkeypatch.setattr(db_mod, "DB_BACKEND", "sqlite")
        engine = DualWriteEngine()
        assert engine.backend == "sqlite"

    def test_connection_yields_sqlite(self, monkeypatch):
        import db as db_mod
        monkeypatch.setattr(db_mod, "DB_BACKEND", "sqlite")
        engine = DualWriteEngine()
        with engine.connection() as (conn, backend):
            assert backend == "sqlite"
            # Verify we can query
            conn.execute("SELECT 1")

    def test_transaction_yields_sqlite(self, monkeypatch):
        import db as db_mod
        monkeypatch.setattr(db_mod, "DB_BACKEND", "sqlite")
        engine = DualWriteEngine()
        with engine.transaction() as (conn, backend):
            assert backend == "sqlite"
            DatabaseOps.upsert_job(conn, {
                "job_id": "dual-j1",
                "status": "queued",
                "priority": 1,
                "submitted_at": time.time(),
                "name": "dual-test",
            }, backend="sqlite")

        # Verify written
        with engine.connection() as (conn, backend):
            job = DatabaseOps.get_job(conn, "dual-j1", backend="sqlite")
            assert job is not None
            assert job["name"] == "dual-test"

    def test_mirror_noop_in_sqlite_mode(self, monkeypatch):
        """mirror_to_secondary does nothing when not in dual mode."""
        import db as db_mod
        monkeypatch.setattr(db_mod, "DB_BACKEND", "sqlite")
        engine = DualWriteEngine()
        # Should not raise
        engine.mirror_to_secondary(DatabaseOps.upsert_job, {"job_id": "x"})

    def test_dual_mode_sets_backend(self, monkeypatch):
        import db as db_mod
        monkeypatch.setattr(db_mod, "DB_BACKEND", "dual")
        monkeypatch.setattr(db_mod, "DUAL_READ_FROM", "sqlite")
        engine = DualWriteEngine()
        assert engine.backend == "dual"
        assert engine.read_from == "sqlite"

    def test_dual_transaction_uses_sqlite_primary(self, monkeypatch):
        import db as db_mod
        monkeypatch.setattr(db_mod, "DB_BACKEND", "dual")
        monkeypatch.setattr(db_mod, "DUAL_READ_FROM", "sqlite")
        engine = DualWriteEngine()
        with engine.transaction() as (conn, backend):
            assert backend == "sqlite"


# ── 7.5.3 — Dual-write consistency (SQLite-only simulation) ─────────


class TestDualWriteConsistency:
    """Write via unified ops → verify data integrity across read/write paths."""

    def test_write_job_read_back_matches(self):
        job = {
            "job_id": "consistency-j1",
            "name": "gpu-training",
            "status": "queued",
            "priority": 5,
            "submitted_at": time.time(),
            "vram_needed_gb": 16,
            "tier": "premium",
        }
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_job(conn, job, backend="sqlite")

        with sqlite_connection() as conn:
            loaded = DatabaseOps.get_job(conn, "consistency-j1", backend="sqlite")
        assert loaded is not None
        assert loaded["name"] == "gpu-training"
        assert loaded["priority"] == 5
        assert loaded["tier"] == "premium"

    def test_write_host_read_back_matches(self):
        host = {
            "host_id": "consistency-h1",
            "ip": "10.0.0.5",
            "gpu_model": "RTX 4090",
            "total_vram_gb": 24,
            "free_vram_gb": 24,
            "status": "active",
            "registered_at": time.time(),
        }
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_host(conn, host, backend="sqlite")

        with sqlite_connection() as conn:
            loaded = DatabaseOps.get_host(conn, "consistency-h1", backend="sqlite")
        assert loaded is not None
        assert loaded["gpu_model"] == "RTX 4090"
        assert loaded["free_vram_gb"] == 24

    def test_state_namespace_round_trip(self):
        data = {"last_run": time.time(), "count": 42}
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_state(conn, "test_ns", data, backend="sqlite")

        with sqlite_connection() as conn:
            loaded = DatabaseOps.get_state(conn, "test_ns", backend="sqlite")
        assert loaded["count"] == 42

    def test_upsert_updates_existing_preserving_schema(self):
        """Upsert with changed status preserves other fields."""
        job = {
            "job_id": "upsert-j1",
            "name": "ml-run",
            "status": "queued",
            "priority": 3,
            "submitted_at": time.time(),
            "vram_needed_gb": 8,
        }
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_job(conn, job, backend="sqlite")

        job["status"] = "running"
        job["host_id"] = "rig-99"
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_job(conn, job, backend="sqlite")

        with sqlite_connection() as conn:
            loaded = DatabaseOps.get_job(conn, "upsert-j1", backend="sqlite")
        assert loaded["status"] == "running"
        assert loaded["host_id"] == "rig-99"
        assert loaded["name"] == "ml-run"

    def test_delete_then_read_returns_none(self):
        with sqlite_transaction() as conn:
            DatabaseOps.upsert_job(conn, {
                "job_id": "del-j1", "status": "queued",
                "priority": 0, "submitted_at": time.time(),
            }, backend="sqlite")
            DatabaseOps.delete_job(conn, "del-j1", backend="sqlite")

        with sqlite_connection() as conn:
            assert DatabaseOps.get_job(conn, "del-j1", backend="sqlite") is None

    def test_load_jobs_filter_by_status(self):
        with sqlite_transaction() as conn:
            for i, status in enumerate(["queued", "running", "completed", "queued"]):
                DatabaseOps.upsert_job(conn, {
                    "job_id": f"filter-j{i}",
                    "status": status,
                    "priority": 1,
                    "submitted_at": time.time() + i,
                }, backend="sqlite")

        with sqlite_connection() as conn:
            queued = DatabaseOps.load_jobs(conn, status="queued", backend="sqlite")
        assert len(queued) == 2

    def test_gpu_query_sqlite_fallback(self):
        """query_hosts_by_gpu filters in Python for SQLite backend."""
        with sqlite_transaction() as conn:
            for hid, model, vram in [("g1", "RTX 4090", 24), ("g2", "A100", 80), ("g3", "RTX 4090", 12)]:
                DatabaseOps.upsert_host(conn, {
                    "host_id": hid, "gpu_model": model,
                    "free_vram_gb": vram, "status": "active",
                    "registered_at": time.time(),
                }, backend="sqlite")

        with sqlite_connection() as conn:
            results = DatabaseOps.query_hosts_by_gpu(conn, gpu_model="RTX 4090", backend="sqlite")
        assert len(results) == 2
        # Sorted by free_vram descending
        assert results[0]["free_vram_gb"] >= results[1]["free_vram_gb"]

    def test_gpu_query_min_vram_filter(self):
        with sqlite_transaction() as conn:
            for hid, vram in [("v1", 8), ("v2", 24), ("v3", 48)]:
                DatabaseOps.upsert_host(conn, {
                    "host_id": hid, "gpu_model": "A100",
                    "free_vram_gb": vram, "status": "active",
                    "registered_at": time.time(),
                }, backend="sqlite")

        with sqlite_connection() as conn:
            results = DatabaseOps.query_hosts_by_gpu(conn, min_vram_gb=20, backend="sqlite")
        assert len(results) == 2
        assert all(h["free_vram_gb"] >= 20 for h in results)


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
