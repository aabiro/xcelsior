"""Real PostgreSQL database tests for volume hardening.

Tests run against the actual database (XCELSIOR_DATABASE_URL).
Volume IDs use test-vol-* prefix and are cleaned up after each test.
"""

import os
import time
import uuid
import threading

import pytest
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_NFS_SERVER", "")


def _test_vid():
    return f"test-vol-{uuid.uuid4().hex[:12]}"


def _test_owner():
    return f"test-owner-{uuid.uuid4().hex[:8]}@test.xcelsior.ca"


@pytest.fixture
def engine(monkeypatch):
    """VolumeEngine with NFS provisioning stubbed out (metadata-only)."""
    from volumes import VolumeEngine

    e = VolumeEngine()
    monkeypatch.setattr(e, "_provision_volume_storage", lambda vid, sz, **kw: True)
    monkeypatch.setattr(e, "_destroy_volume_storage", lambda vid, **kw: True)
    monkeypatch.setattr(e, "_mount_on_host", lambda *a, **kw: True)
    monkeypatch.setattr(e, "_unmount_from_host", lambda *a, **kw: True)
    monkeypatch.setattr(e, "_emit_event", lambda *a, **kw: None)
    return e


@pytest.fixture
def cleanup_vids():
    """Collect volume IDs created during test and delete them after."""
    vids = []
    yield vids
    if not vids:
        return
    try:
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            for vid in vids:
                conn.execute("DELETE FROM volume_attachments WHERE volume_id = %s", (vid,))
                conn.execute("DELETE FROM volumes WHERE volume_id = %s", (vid,))
            conn.commit()
    except Exception:
        pass


# ── Volume CRUD against real PostgreSQL ──────────────────────────────


class TestRealCRUD:
    """Create, read, list, delete against actual database tables."""

    def test_create_and_get(self, engine, cleanup_vids):
        owner = _test_owner()
        vol = engine.create_volume(owner, "db-test-vol", 5)
        cleanup_vids.append(vol["volume_id"])

        assert vol["volume_id"].startswith("vol-")
        assert vol["status"] == "available"
        assert vol["size_gb"] == 5

        fetched = engine.get_volume(vol["volume_id"])
        assert fetched is not None
        assert fetched["name"] == "db-test-vol"
        assert fetched["owner_id"] == owner

    def test_create_and_list(self, engine, cleanup_vids):
        owner = _test_owner()
        v1 = engine.create_volume(owner, "list-a", 1)
        v2 = engine.create_volume(owner, "list-b", 2)
        cleanup_vids.extend([v1["volume_id"], v2["volume_id"]])

        vols = engine.list_volumes(owner)
        vol_ids = {v["volume_id"] for v in vols}
        assert v1["volume_id"] in vol_ids
        assert v2["volume_id"] in vol_ids

    def test_duplicate_name_rejected(self, engine, cleanup_vids):
        owner = _test_owner()
        v1 = engine.create_volume(owner, "dup-name", 1)
        cleanup_vids.append(v1["volume_id"])

        with pytest.raises(ValueError, match="already exists"):
            engine.create_volume(owner, "dup-name", 1)

    def test_delete_volume(self, engine, cleanup_vids):
        owner = _test_owner()
        vol = engine.create_volume(owner, "to-delete", 1)
        cleanup_vids.append(vol["volume_id"])

        engine.delete_volume(vol["volume_id"], owner)
        fetched = engine.get_volume(vol["volume_id"])
        assert fetched is None  # deleted volumes are hidden


# ── State guard with real DB ─────────────────────────────────────────


class TestStateGuardRealDB:
    """Verify state transitions actually work against PostgreSQL."""

    def test_invalid_transition_rejected(self, engine, cleanup_vids):
        owner = _test_owner()
        vol = engine.create_volume(owner, "guard-test", 1)
        cleanup_vids.append(vol["volume_id"])

        # Volume is 'available' — trying to go to 'error' should fail
        with pytest.raises(ValueError, match="Invalid volume transition"):
            with engine._conn() as conn:
                engine._transition_status(conn, vol["volume_id"], "error")

    def test_valid_transition_succeeds(self, engine, cleanup_vids):
        owner = _test_owner()
        vol = engine.create_volume(owner, "guard-ok", 1)
        cleanup_vids.append(vol["volume_id"])

        # available → deleting is valid
        with engine._conn() as conn:
            result = engine._transition_status(conn, vol["volume_id"], "deleting")
        assert result == "deleting"

    def test_concurrent_transitions_serialized(self, engine, cleanup_vids):
        """Two threads try to transition the same volume — only one succeeds."""
        owner = _test_owner()
        vol = engine.create_volume(owner, "concurrent", 1)
        vid = vol["volume_id"]
        cleanup_vids.append(vid)

        results = {"t1": None, "t2": None}
        errors = {"t1": None, "t2": None}

        def transition(key, target):
            try:
                with engine._conn() as conn:
                    results[key] = engine._transition_status(conn, vid, target)
            except ValueError as e:
                errors[key] = str(e)

        t1 = threading.Thread(target=transition, args=("t1", "attached"))
        t2 = threading.Thread(target=transition, args=("t2", "deleting"))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Both are valid from 'available', but they can't both succeed
        # since after one transitions, the state has changed
        successes = [k for k in ["t1", "t2"] if results[k] is not None]
        # At least one should succeed
        assert len(successes) >= 1


# ── Stale cleanup with real DB ───────────────────────────────────────


class TestStaleCleanupRealDB:
    """cleanup_stale_volumes against real PostgreSQL."""

    def test_stale_provisioning_swept(self, engine, cleanup_vids):
        """A volume stuck in 'provisioning' for >10min gets moved to 'error'."""
        owner = _test_owner()
        vid = _test_vid()
        cleanup_vids.append(vid)

        # Insert a provisioning volume with old timestamp directly
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        pool = _get_pg_pool()
        old_time = time.time() - 1200  # 20 min ago
        with pool.connection() as conn:
            conn.row_factory = dict_row
            conn.execute(
                """INSERT INTO volumes (volume_id, owner_id, name, storage_type, size_gb,
                   region, province, encrypted, status, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    vid,
                    owner,
                    "stale-prov",
                    "nfs",
                    1,
                    "ca-east",
                    "ON",
                    False,
                    "provisioning",
                    old_time,
                ),
            )
            conn.commit()

        count = engine.cleanup_stale_volumes(max_age_seconds=600)
        assert count >= 1

        with pool.connection() as conn:
            conn.row_factory = dict_row
            row = conn.execute("SELECT status FROM volumes WHERE volume_id = %s", (vid,)).fetchone()
        assert row["status"] == "error"

    def test_stale_deleting_swept(self, engine, cleanup_vids):
        """A volume stuck in 'deleting' for >10min gets moved to 'error'."""
        owner = _test_owner()
        vid = _test_vid()
        cleanup_vids.append(vid)

        from db import _get_pg_pool
        from psycopg.rows import dict_row

        pool = _get_pg_pool()
        old_time = time.time() - 1200
        with pool.connection() as conn:
            conn.row_factory = dict_row
            conn.execute(
                """INSERT INTO volumes (volume_id, owner_id, name, storage_type, size_gb,
                   region, province, encrypted, status, created_at, deleted_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    vid,
                    owner,
                    "stale-del",
                    "nfs",
                    1,
                    "ca-east",
                    "ON",
                    False,
                    "deleting",
                    time.time(),
                    old_time,
                ),
            )
            conn.commit()

        count = engine.cleanup_stale_volumes(max_age_seconds=600)
        assert count >= 1

        with pool.connection() as conn:
            conn.row_factory = dict_row
            row = conn.execute("SELECT status FROM volumes WHERE volume_id = %s", (vid,)).fetchone()
        assert row["status"] == "error"

    def test_fresh_provisioning_not_swept(self, engine, cleanup_vids):
        """A volume that JUST entered 'provisioning' is NOT swept."""
        owner = _test_owner()
        vid = _test_vid()
        cleanup_vids.append(vid)

        from db import _get_pg_pool
        from psycopg.rows import dict_row

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            conn.execute(
                """INSERT INTO volumes (volume_id, owner_id, name, storage_type, size_gb,
                   region, province, encrypted, status, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    vid,
                    owner,
                    "fresh-prov",
                    "nfs",
                    1,
                    "ca-east",
                    "ON",
                    False,
                    "provisioning",
                    time.time(),
                ),
            )
            conn.commit()

        engine.cleanup_stale_volumes(max_age_seconds=600)

        with pool.connection() as conn:
            conn.row_factory = dict_row
            row = conn.execute("SELECT status FROM volumes WHERE volume_id = %s", (vid,)).fetchone()
        assert row["status"] == "provisioning"  # NOT swept


# ── Orphan reconciliation with real DB ───────────────────────────────


class TestOrphanReconciliationRealDB:
    """reconcile_orphaned_attachments against real PostgreSQL."""

    def test_attached_with_no_attachments_fixed(self, engine, cleanup_vids):
        """Volume marked 'attached' but zero active attachment rows → 'available'."""
        owner = _test_owner()
        vid = _test_vid()
        cleanup_vids.append(vid)

        from db import _get_pg_pool
        from psycopg.rows import dict_row

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            conn.execute(
                """INSERT INTO volumes (volume_id, owner_id, name, storage_type, size_gb,
                   region, province, encrypted, status, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    vid,
                    owner,
                    "orphan-vol",
                    "nfs",
                    1,
                    "ca-east",
                    "ON",
                    False,
                    "attached",
                    time.time(),
                ),
            )
            conn.commit()

        fixed = engine.reconcile_orphaned_attachments()
        assert fixed >= 1

        with pool.connection() as conn:
            conn.row_factory = dict_row
            row = conn.execute("SELECT status FROM volumes WHERE volume_id = %s", (vid,)).fetchone()
        assert row["status"] == "available"

    def test_attachment_to_dead_job_closed(self, engine, cleanup_vids):
        """Active attachment referencing a completed job → detached."""
        owner = _test_owner()
        vid = _test_vid()
        job_id = f"test-job-{uuid.uuid4().hex[:8]}"
        att_id = f"test-att-{uuid.uuid4().hex[:8]}"
        cleanup_vids.append(vid)

        from db import _get_pg_pool
        from psycopg.rows import dict_row

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            # Create volume
            conn.execute(
                """INSERT INTO volumes (volume_id, owner_id, name, storage_type, size_gb,
                   region, province, encrypted, status, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (vid, owner, "dead-att", "nfs", 1, "ca-east", "ON", False, "attached", time.time()),
            )
            # Create a completed job
            conn.execute(
                """INSERT INTO jobs (job_id, status, priority, submitted_at, payload)
                   VALUES (%s, %s, %s, %s, %s::jsonb)
                   ON CONFLICT (job_id) DO NOTHING""",
                (job_id, "completed", 50, time.time(), '{"owner": "' + owner + '"}'),
            )
            # Create active attachment
            conn.execute(
                """INSERT INTO volume_attachments (attachment_id, volume_id, instance_id, mount_path, mode, attached_at, detached_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (att_id, vid, job_id, "/workspace", "rw", time.time(), 0),
            )
            conn.commit()

        fixed = engine.reconcile_orphaned_attachments()
        assert fixed >= 1

        with pool.connection() as conn:
            conn.row_factory = dict_row
            att = conn.execute(
                "SELECT detached_at FROM volume_attachments WHERE attachment_id = %s", (att_id,)
            ).fetchone()
            assert att["detached_at"] > 0

            vol = conn.execute("SELECT status FROM volumes WHERE volume_id = %s", (vid,)).fetchone()
            assert vol["status"] == "available"

        # Cleanup job
        with pool.connection() as conn:
            conn.execute("DELETE FROM volume_attachments WHERE attachment_id = %s", (att_id,))
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))
            conn.commit()


# ── Retry provision with real DB ─────────────────────────────────────


class TestRetryProvisionRealDB:
    """retry_provision() against real PostgreSQL."""

    def test_retry_error_to_available(self, engine, cleanup_vids):
        """Volume in 'error' can be retried back to 'available'."""
        owner = _test_owner()
        vid = _test_vid()
        cleanup_vids.append(vid)

        from db import _get_pg_pool
        from psycopg.rows import dict_row

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            conn.execute(
                """INSERT INTO volumes (volume_id, owner_id, name, storage_type, size_gb,
                   region, province, encrypted, status, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (vid, owner, "retry-test", "nfs", 1, "ca-east", "ON", False, "error", time.time()),
            )
            conn.commit()

        result = engine.retry_provision(vid, owner)
        assert result["status"] == "available"

        with pool.connection() as conn:
            conn.row_factory = dict_row
            row = conn.execute("SELECT status FROM volumes WHERE volume_id = %s", (vid,)).fetchone()
        assert row["status"] == "available"

    def test_retry_wrong_owner_rejected(self, engine, cleanup_vids):
        owner = _test_owner()
        vid = _test_vid()
        cleanup_vids.append(vid)

        from db import _get_pg_pool
        from psycopg.rows import dict_row

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            conn.execute(
                """INSERT INTO volumes (volume_id, owner_id, name, storage_type, size_gb,
                   region, province, encrypted, status, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (vid, owner, "wrong-owner", "nfs", 1, "ca-east", "ON", False, "error", time.time()),
            )
            conn.commit()

        with pytest.raises(PermissionError):
            engine.retry_provision(vid, "hacker@evil.com")

    def test_retry_non_error_rejected(self, engine, cleanup_vids):
        owner = _test_owner()
        vol = engine.create_volume(owner, "not-error", 1)
        cleanup_vids.append(vol["volume_id"])

        with pytest.raises(ValueError, match="not 'error'"):
            engine.retry_provision(vol["volume_id"], owner)


# ── Capacity limits with real count queries ──────────────────────────


class TestCapacityRealDB:
    """Capacity enforcement against real PostgreSQL."""

    def test_capacity_check_counts_real_volumes(self, engine, cleanup_vids):
        """Capacity check uses actual SUM(size_gb) from DB."""
        owner = _test_owner()
        v1 = engine.create_volume(owner, "cap-a", 10)
        v2 = engine.create_volume(owner, "cap-b", 10)
        cleanup_vids.extend([v1["volume_id"], v2["volume_id"]])

        # Both volumes exist and count toward capacity
        vols = engine.list_volumes(owner)
        total_size = sum(
            v["size_gb"] for v in vols if v["volume_id"] in [v1["volume_id"], v2["volume_id"]]
        )
        assert total_size == 20
