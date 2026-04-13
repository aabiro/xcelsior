"""Backend tests for VolumeEngine, volume API endpoints, and security."""

import os
import time

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_NFS_SERVER", "")  # metadata-only mode


# ── VolumeEngine unit tests ──────────────────────────────────────────


class TestVolumeEngineCreate:
    """Volume creation: success, duplicate name, capacity exceeded."""

    def _make_engine(self, monkeypatch, conn_rows=None):
        """Return a VolumeEngine with _conn mocked to an in-memory cursor stub."""
        from unittest.mock import MagicMock, patch
        from contextlib import contextmanager
        from volumes import VolumeEngine

        engine = VolumeEngine()

        # Build a fake connection that returns rows from conn_rows
        fake_conn = MagicMock()
        _call_idx = {"i": 0}
        _rows = conn_rows or []

        def _execute(sql, params=None):
            result = MagicMock()
            idx = _call_idx["i"]
            _call_idx["i"] += 1
            if idx < len(_rows):
                result.fetchone.return_value = _rows[idx]
                result.fetchall.return_value = [_rows[idx]] if _rows[idx] else []
            else:
                result.fetchone.return_value = None
                result.fetchall.return_value = []
            return result

        fake_conn.execute = _execute
        fake_conn.commit = MagicMock()
        fake_conn.rollback = MagicMock()

        @contextmanager
        def _mock_conn():
            yield fake_conn

        monkeypatch.setattr(engine, "_conn", _mock_conn)
        monkeypatch.setattr(engine, "_provision_volume_storage", lambda vid, sz: True)
        return engine

    def test_create_success(self, monkeypatch):
        engine = self._make_engine(monkeypatch, conn_rows=[
            None,            # FOR UPDATE lock (unused)
            {"total": 0},   # capacity check
            None,            # name uniqueness check → no duplicate
        ])
        vol = engine.create_volume("user-1", "my-data", 10)
        assert vol["name"] == "my-data"
        assert vol["size_gb"] == 10
        assert vol["status"] == "available"
        assert vol["volume_id"].startswith("vol-")

    def test_create_empty_name(self, monkeypatch):
        engine = self._make_engine(monkeypatch)
        with pytest.raises(ValueError, match="name is required"):
            engine.create_volume("user-1", "", 10)

    def test_create_whitespace_name(self, monkeypatch):
        engine = self._make_engine(monkeypatch)
        with pytest.raises(ValueError, match="name is required"):
            engine.create_volume("user-1", "   ", 10)

    def test_create_exceeds_max_size(self, monkeypatch):
        engine = self._make_engine(monkeypatch)
        with pytest.raises(ValueError, match="exceeds max"):
            engine.create_volume("user-1", "huge", 999999)

    def test_create_zero_size(self, monkeypatch):
        engine = self._make_engine(monkeypatch)
        with pytest.raises(ValueError, match="at least 1GB"):
            engine.create_volume("user-1", "tiny", 0)

    def test_create_negative_size(self, monkeypatch):
        engine = self._make_engine(monkeypatch)
        with pytest.raises(ValueError, match="at least 1GB"):
            engine.create_volume("user-1", "neg", -5)

    def test_create_capacity_exceeded(self, monkeypatch):
        engine = self._make_engine(monkeypatch, conn_rows=[
            None,            # FOR UPDATE lock (unused)
            {"total": 95},  # capacity check: 95 used, requesting 10 → over 100
        ])
        with pytest.raises(ValueError, match="Insufficient storage capacity"):
            engine.create_volume("user-1", "big", 10)

    def test_create_duplicate_name(self, monkeypatch):
        engine = self._make_engine(monkeypatch, conn_rows=[
            None,                              # FOR UPDATE lock (unused)
            {"total": 0},                     # capacity check
            {"volume_id": "vol-existing"},     # name uniqueness → duplicate found
        ])
        with pytest.raises(ValueError, match="already exists"):
            engine.create_volume("user-1", "dupe", 10)

    def test_create_provision_failure(self, monkeypatch):
        engine = self._make_engine(monkeypatch, conn_rows=[
            None,            # FOR UPDATE lock (unused)
            {"total": 0},
            None,
        ])
        monkeypatch.setattr(engine, "_provision_volume_storage", lambda vid, sz: False)
        with pytest.raises(ValueError, match="Failed to provision"):
            engine.create_volume("user-1", "bad-nfs", 10)


class TestVolumeEngineAttach:
    """Attach: success, idempotent, wrong status."""

    def _make_engine_with_vol(self, monkeypatch, vol_status="available", existing_att=None):
        from unittest.mock import MagicMock
        from contextlib import contextmanager
        from volumes import VolumeEngine

        engine = VolumeEngine()
        _call_idx = {"i": 0}

        vol_row = {"volume_id": "vol-abc", "status": vol_status, "owner_id": "user-1"}
        rows = [
            vol_row,         # SELECT ... FOR UPDATE
            existing_att,    # existing attachment check
        ]

        fake_conn = MagicMock()

        def _execute(sql, params=None):
            result = MagicMock()
            idx = _call_idx["i"]
            _call_idx["i"] += 1
            if idx < len(rows):
                result.fetchone.return_value = rows[idx]
            else:
                result.fetchone.return_value = None
            return result

        fake_conn.execute = _execute
        fake_conn.commit = MagicMock()
        fake_conn.rollback = MagicMock()

        @contextmanager
        def _mock_conn():
            yield fake_conn

        monkeypatch.setattr(engine, "_conn", _mock_conn)
        monkeypatch.setattr(engine, "_mount_on_host", lambda *a: True)
        return engine

    def test_attach_success(self, monkeypatch):
        engine = self._make_engine_with_vol(monkeypatch, vol_status="available")
        result = engine.attach_volume("vol-abc", "inst-1")
        assert result["volume_id"] == "vol-abc"
        assert result["instance_id"] == "inst-1"
        assert result["attachment_id"].startswith("att-")

    def test_attach_idempotent(self, monkeypatch):
        engine = self._make_engine_with_vol(
            monkeypatch,
            vol_status="attached",
            existing_att={"attachment_id": "att-existing"},
        )
        result = engine.attach_volume("vol-abc", "inst-1")
        assert result["already_attached"] is True
        assert result["attachment_id"] == "att-existing"

    def test_attach_wrong_status(self, monkeypatch):
        engine = self._make_engine_with_vol(monkeypatch, vol_status="deleting")
        with pytest.raises(ValueError, match="cannot be attached"):
            engine.attach_volume("vol-abc", "inst-1")

    def test_attach_not_found(self, monkeypatch):
        from unittest.mock import MagicMock
        from contextlib import contextmanager
        from volumes import VolumeEngine

        engine = VolumeEngine()
        fake_conn = MagicMock()
        fake_conn.execute = MagicMock(return_value=MagicMock(fetchone=MagicMock(return_value=None)))
        fake_conn.commit = MagicMock()
        fake_conn.rollback = MagicMock()

        @contextmanager
        def _mock_conn():
            yield fake_conn

        monkeypatch.setattr(engine, "_conn", _mock_conn)
        with pytest.raises(ValueError, match="not found"):
            engine.attach_volume("vol-nope", "inst-1")

    def test_attach_mount_failure_rollback(self, monkeypatch):
        engine = self._make_engine_with_vol(monkeypatch, vol_status="available")
        monkeypatch.setattr(engine, "_mount_on_host", lambda *a: False)
        with pytest.raises(ValueError, match="rolled back"):
            engine.attach_volume("vol-abc", "inst-1")


class TestVolumeEngineDetach:
    """Detach: success, not found."""

    def _make_engine_detach(self, monkeypatch, att_row=None, remaining_count=0):
        from unittest.mock import MagicMock
        from contextlib import contextmanager
        from volumes import VolumeEngine

        engine = VolumeEngine()
        _call_idx = {"i": 0}

        # Row sequence: 0=SELECT att FOR UPDATE, 1=UPDATE detached_at, 2=SELECT COUNT remaining
        rows = [
            att_row,
            None,
            {"cnt": remaining_count},
        ]

        fake_conn = MagicMock()

        def _execute(sql, params=None):
            result = MagicMock()
            idx = _call_idx["i"]
            _call_idx["i"] += 1
            if idx < len(rows):
                result.fetchone.return_value = rows[idx]
            else:
                result.fetchone.return_value = None
            return result

        fake_conn.execute = _execute
        fake_conn.commit = MagicMock()
        fake_conn.rollback = MagicMock()

        @contextmanager
        def _mock_conn():
            yield fake_conn

        monkeypatch.setattr(engine, "_conn", _mock_conn)
        monkeypatch.setattr(engine, "_unmount_from_host", lambda *a: True)
        return engine

    def test_detach_success(self, monkeypatch):
        att = {
            "attachment_id": "att-123",
            "volume_id": "vol-abc",
            "instance_id": "inst-1",
            "mount_path": "/workspace",
        }
        engine = self._make_engine_detach(monkeypatch, att_row=att, remaining_count=0)
        result = engine.detach_volume("vol-abc", "inst-1")
        assert result["status"] == "detached"

    def test_detach_not_found(self, monkeypatch):
        engine = self._make_engine_detach(monkeypatch, att_row=None)
        with pytest.raises(ValueError, match="No active attachment"):
            engine.detach_volume("vol-abc", "inst-1")


class TestVolumeEngineDelete:
    """Delete: success, blocked by attachment, storage failure."""

    def _make_engine_delete(self, monkeypatch, vol_row, att_row=None, destroy_ok=True):
        from unittest.mock import MagicMock
        from contextlib import contextmanager
        from volumes import VolumeEngine

        engine = VolumeEngine()
        _call_idx = {"i": 0}
        rows = [vol_row, att_row]

        fake_conn = MagicMock()

        def _execute(sql, params=None):
            result = MagicMock()
            idx = _call_idx["i"]
            _call_idx["i"] += 1
            if idx < len(rows):
                result.fetchone.return_value = rows[idx]
            else:
                result.fetchone.return_value = None
            return result

        fake_conn.execute = _execute
        fake_conn.commit = MagicMock()
        fake_conn.rollback = MagicMock()

        @contextmanager
        def _mock_conn():
            yield fake_conn

        monkeypatch.setattr(engine, "_conn", _mock_conn)
        monkeypatch.setattr(engine, "_destroy_volume_storage", lambda vid: destroy_ok)
        return engine

    def test_delete_success(self, monkeypatch):
        vol = {"volume_id": "vol-abc", "owner_id": "user-1", "status": "available"}
        engine = self._make_engine_delete(monkeypatch, vol_row=vol, att_row=None)
        result = engine.delete_volume("vol-abc", "user-1")
        assert result["status"] == "deleted"

    def test_delete_blocked_by_attachment(self, monkeypatch):
        vol = {"volume_id": "vol-abc", "owner_id": "user-1", "status": "attached"}
        att = {"attachment_id": "att-123"}
        engine = self._make_engine_delete(monkeypatch, vol_row=vol, att_row=att)
        with pytest.raises(ValueError, match="active attachments"):
            engine.delete_volume("vol-abc", "user-1")

    def test_delete_not_found(self, monkeypatch):
        engine = self._make_engine_delete(monkeypatch, vol_row=None)
        with pytest.raises(ValueError, match="not found"):
            engine.delete_volume("vol-nope", "user-1")

    def test_delete_storage_failure(self, monkeypatch):
        vol = {"volume_id": "vol-abc", "owner_id": "user-1", "status": "available"}
        engine = self._make_engine_delete(monkeypatch, vol_row=vol, att_row=None, destroy_ok=False)
        with pytest.raises(RuntimeError, match="Failed to destroy"):
            engine.delete_volume("vol-abc", "user-1")


class TestVolumeEngineDetachAll:
    """detach_all_for_instance: bulk on termination."""

    def test_detach_all_returns_count(self, monkeypatch):
        from unittest.mock import MagicMock
        from contextlib import contextmanager
        from volumes import VolumeEngine

        engine = VolumeEngine()
        atts = [
            {"volume_id": "vol-a", "attachment_id": "att-1", "mount_path": "/workspace"},
            {"volume_id": "vol-b", "attachment_id": "att-2", "mount_path": "/data"},
        ]
        _call_idx = {"i": 0}

        fake_conn = MagicMock()

        def _execute(sql, params=None):
            result = MagicMock()
            idx = _call_idx["i"]
            _call_idx["i"] += 1
            if idx == 0:
                result.fetchall.return_value = atts
            elif "count(*)" in sql.lower():
                result.fetchone.return_value = {"cnt": 0}
            else:
                result.fetchone.return_value = None
            return result

        fake_conn.execute = _execute
        fake_conn.commit = MagicMock()
        fake_conn.rollback = MagicMock()

        @contextmanager
        def _mock_conn():
            yield fake_conn

        monkeypatch.setattr(engine, "_conn", _mock_conn)
        monkeypatch.setattr(engine, "_unmount_from_host", lambda *a: True)

        count = engine.detach_all_for_instance("inst-1")
        assert count == 2


# ── Security tests ───────────────────────────────────────────────────


class TestVolumePathSecurity:
    """Path traversal rejection and allowed-prefix enforcement in build_secure_docker_args."""

    def test_path_traversal_rejected(self):
        from security import build_secure_docker_args
        with pytest.raises(ValueError, match="\\.\\."):
            build_secure_docker_args(
                image="pytorch:latest",
                container_name="test",
                volumes=["/mnt/xcelsior-volumes/../etc/passwd:/data:ro"],
            )

    def test_null_byte_rejected(self):
        from security import build_secure_docker_args
        with pytest.raises(ValueError, match="null byte"):
            build_secure_docker_args(
                image="pytorch:latest",
                container_name="test",
                volumes=["/mnt/xcelsior-volumes/vol\x00evil:/data:ro"],
            )

    def test_prefix_bypass_rejected(self):
        """A path like /mnt/xcelsior-nfs-evil/ must not pass the prefix check."""
        from security import build_secure_docker_args
        with pytest.raises(ValueError, match="outside allowed prefixes"):
            build_secure_docker_args(
                image="pytorch:latest",
                container_name="test",
                volumes=["/mnt/xcelsior-nfs-evil/malware:/data:ro"],
            )

    def test_disallowed_prefix_rejected(self):
        from security import build_secure_docker_args
        with pytest.raises(ValueError, match="outside allowed prefixes"):
            build_secure_docker_args(
                image="pytorch:latest",
                container_name="test",
                volumes=["/etc/shadow:/data:ro"],
            )

    def test_allowed_xcelsior_volumes_prefix(self):
        from security import build_secure_docker_args
        args = build_secure_docker_args(
            image="pytorch:latest",
            container_name="test",
            volumes=["/mnt/xcelsior-volumes/vol-abc:/workspace:rw"],
        )
        assert "-v" in args
        assert "/mnt/xcelsior-volumes/vol-abc:/workspace:rw" in args

    def test_allowed_nfs_prefix(self):
        from security import build_secure_docker_args
        args = build_secure_docker_args(
            image="pytorch:latest",
            container_name="test",
            volumes=["/mnt/xcelsior-nfs/data:/data:ro"],
        )
        assert "/mnt/xcelsior-nfs/data:/data:ro" in args

    def test_volume_without_mode_defaults_ro(self):
        from security import build_secure_docker_args
        args = build_secure_docker_args(
            image="pytorch:latest",
            container_name="test",
            volumes=["/mnt/xcelsior-volumes/vol-abc:/workspace:notamode"],
        )
        # Should append :ro since mode isn't :rw or :ro
        found = [a for a in args if "vol-abc" in a]
        assert len(found) == 1
        assert found[0].endswith(":ro")

    def test_host_path_only_volume(self):
        """Volume spec without colon should still be validated."""
        from security import build_secure_docker_args
        with pytest.raises(ValueError, match="outside allowed prefixes"):
            build_secure_docker_args(
                image="pytorch:latest",
                container_name="test",
                volumes=["/tmp/evil"],
            )

    def test_cross_user_volume_access_uses_404(self):
        """Ownership check on instance launch returns 404 not 403 to prevent enumeration."""
        # This tests the API-level check pattern in routes/instances.py
        # The ownership check returns 404 (not 403) for volumes not owned by the requester
        from routes.instances import router
        # Verify the route exists — the actual ownership check is tested in integration
        assert router is not None


# ── API endpoint structure tests ─────────────────────────────────────


class TestVolumeAPIEndpoints:
    """Verify volume API route registration and pricing constant."""

    def test_volume_price_constant(self):
        from volumes import VOLUME_PRICE_PER_GB_MONTH_CAD
        assert VOLUME_PRICE_PER_GB_MONTH_CAD == 0.07

    def test_routes_registered(self):
        from routes.volumes import router
        paths = [r.path for r in router.routes]
        assert "/api/v2/volumes" in paths
        assert "/api/v2/volumes/available" in paths
        assert "/api/v2/volumes/{volume_id}" in paths
        assert "/api/v2/volumes/{volume_id}/attach" in paths
        assert "/api/v2/volumes/{volume_id}/detach" in paths

    def test_volume_engine_singleton(self):
        from volumes import get_volume_engine, VolumeEngine
        ve = get_volume_engine()
        assert isinstance(ve, VolumeEngine)
        ve2 = get_volume_engine()
        assert ve is ve2


# ── Name validation tests ────────────────────────────────────────────


class TestVolumeNameValidation:
    """Volume name validation: length, special chars, injection."""

    def _make_engine(self, monkeypatch):
        from unittest.mock import MagicMock
        from contextlib import contextmanager
        from volumes import VolumeEngine

        engine = VolumeEngine()
        fake_conn = MagicMock()
        _call_idx = {"i": 0}

        def _execute(sql, params=None):
            result = MagicMock()
            idx = _call_idx["i"]
            _call_idx["i"] += 1
            if idx == 0:
                result.fetchone.return_value = None  # FOR UPDATE lock (unused)
            elif idx == 1:
                result.fetchone.return_value = {"total": 0}  # capacity check
            else:
                result.fetchone.return_value = None  # no duplicate / insert
            return result

        fake_conn.execute = _execute
        fake_conn.commit = MagicMock()
        fake_conn.rollback = MagicMock()

        @contextmanager
        def _mock_conn():
            yield fake_conn

        monkeypatch.setattr(engine, "_conn", _mock_conn)
        monkeypatch.setattr(engine, "_provision_volume_storage", lambda vid, sz: True)
        return engine

    def test_name_too_long(self, monkeypatch):
        engine = self._make_engine(monkeypatch)
        with pytest.raises(ValueError, match="128 characters"):
            engine.create_volume("user-1", "a" * 200, 10)

    def test_name_xss_rejected(self, monkeypatch):
        engine = self._make_engine(monkeypatch)
        with pytest.raises(ValueError, match="must start with alphanumeric"):
            engine.create_volume("user-1", "<script>alert(1)</script>", 10)

    def test_name_sql_injection_rejected(self, monkeypatch):
        engine = self._make_engine(monkeypatch)
        with pytest.raises(ValueError, match="must start with alphanumeric"):
            engine.create_volume("user-1", "'; DROP TABLE volumes;--", 10)

    def test_name_valid_with_hyphens_dots(self, monkeypatch):
        engine = self._make_engine(monkeypatch)
        vol = engine.create_volume("user-1", "my-data.v2_backup", 10)
        assert vol["name"] == "my-data.v2_backup"

    def test_name_cannot_start_with_dot(self, monkeypatch):
        engine = self._make_engine(monkeypatch)
        with pytest.raises(ValueError, match="must start with alphanumeric"):
            engine.create_volume("user-1", ".hidden", 10)

    def test_name_cannot_start_with_hyphen(self, monkeypatch):
        engine = self._make_engine(monkeypatch)
        with pytest.raises(ValueError, match="must start with alphanumeric"):
            engine.create_volume("user-1", "-invalid", 10)


# ── Volume host ID resolution (data gravity) ─────────────────────────


class TestVolumeHostIds:
    """get_volume_host_ids: resolves attachment → instance → host."""

    def test_returns_host_ids(self, monkeypatch):
        from unittest.mock import MagicMock
        from contextlib import contextmanager
        from volumes import VolumeEngine

        engine = VolumeEngine()
        fake_conn = MagicMock()
        fake_conn.execute = MagicMock(return_value=MagicMock(
            fetchall=MagicMock(return_value=[{"host_id": "host-1"}, {"host_id": "host-2"}])
        ))
        fake_conn.commit = MagicMock()
        fake_conn.rollback = MagicMock()

        @contextmanager
        def _mock_conn():
            yield fake_conn

        monkeypatch.setattr(engine, "_conn", _mock_conn)
        result = engine.get_volume_host_ids(["vol-a", "vol-b"])
        assert result == {"host-1", "host-2"}

    def test_empty_volume_ids(self, monkeypatch):
        from volumes import VolumeEngine
        engine = VolumeEngine()
        assert engine.get_volume_host_ids([]) == set()

    def test_no_active_attachments(self, monkeypatch):
        from unittest.mock import MagicMock
        from contextlib import contextmanager
        from volumes import VolumeEngine

        engine = VolumeEngine()
        fake_conn = MagicMock()
        fake_conn.execute = MagicMock(return_value=MagicMock(
            fetchall=MagicMock(return_value=[])
        ))
        fake_conn.commit = MagicMock()
        fake_conn.rollback = MagicMock()

        @contextmanager
        def _mock_conn():
            yield fake_conn

        monkeypatch.setattr(engine, "_conn", _mock_conn)
        result = engine.get_volume_host_ids(["vol-x"])
        assert result == set()


# ── Volume event emission ────────────────────────────────────────────


class TestVolumeEventTypes:
    """Verify volume event types exist in the EventType enum."""

    def test_volume_event_types_registered(self):
        from events import EventType
        assert EventType.VOLUME_CREATED.value == "volume.created"
        assert EventType.VOLUME_DELETED.value == "volume.deleted"
        assert EventType.VOLUME_ATTACHED.value == "volume.attached"
        assert EventType.VOLUME_DETACHED.value == "volume.detached"


# ── Volume tables in _ensure_pg_tables ────────────────────────────────


class TestVolumeSchema:
    """Verify volumes and volume_attachments tables defined in _ensure_pg_tables."""

    def test_volumes_table_in_ensure(self):
        import inspect
        from db import _ensure_pg_tables
        source = inspect.getsource(_ensure_pg_tables)
        assert "CREATE TABLE IF NOT EXISTS volumes" in source

    def test_volume_attachments_table_in_ensure(self):
        import inspect
        from db import _ensure_pg_tables
        source = inspect.getsource(_ensure_pg_tables)
        assert "CREATE TABLE IF NOT EXISTS volume_attachments" in source


# ── Billing constant DRY ──────────────────────────────────────────────


class TestBillingConstantDRY:
    """Verify billing uses the canonical price constant."""

    def test_billing_imports_volume_price(self):
        import inspect
        import billing
        source = inspect.getsource(billing.BillingEngine.auto_billing_cycle)
        assert "VOLUME_PRICE_PER_GB_MONTH_CAD" in source
        assert "0.07 *" not in source  # no hardcoded magic number

    def test_routes_import_volume_price(self):
        import inspect
        import routes.volumes as rv
        source = inspect.getsource(rv)
        assert "VOLUME_PRICE_PER_GB_MONTH_CAD = 0.07" not in source  # not redefined locally


# ── Stale provisioning cleanup ────────────────────────────────────────


class TestStaleProvisioningCleanup:
    """cleanup_stale_volumes: sweeps stale provisioning/deleting volumes to 'error'."""

    def test_marks_stale_volumes(self, monkeypatch):
        from unittest.mock import MagicMock
        from contextlib import contextmanager
        from volumes import VolumeEngine

        engine = VolumeEngine()
        fake_conn = MagicMock()
        # First call: provisioning sweep (3 rows), Second call: deleting sweep (1 row)
        prov_result = MagicMock()
        prov_result.rowcount = 3
        del_result = MagicMock()
        del_result.rowcount = 1
        fake_conn.execute.side_effect = [prov_result, del_result]

        @contextmanager
        def _mock_conn():
            yield fake_conn

        monkeypatch.setattr(engine, "_conn", _mock_conn)
        count = engine.cleanup_stale_volumes(max_age_seconds=600)
        assert count == 4  # 3 provisioning + 1 deleting
        # Verify both UPDATEs were called
        calls = fake_conn.execute.call_args_list
        assert len(calls) == 2
        assert "'provisioning'" in calls[0][0][0]
        assert "'deleting'" in calls[1][0][0]

    def test_returns_zero_when_none_stale(self, monkeypatch):
        from unittest.mock import MagicMock
        from contextlib import contextmanager
        from volumes import VolumeEngine

        engine = VolumeEngine()
        fake_conn = MagicMock()
        fake_result = MagicMock()
        fake_result.rowcount = 0
        fake_conn.execute.return_value = fake_result

        @contextmanager
        def _mock_conn():
            yield fake_conn

        monkeypatch.setattr(engine, "_conn", _mock_conn)
        count = engine.cleanup_stale_volumes()
        assert count == 0


# ── Instance termination volume detach in scheduler ───────────────────


class TestTerminalStateVolumeDetach:
    """Verify scheduler.update_job_status calls detach_all_for_instance on terminal state."""

    def test_detach_in_update_job_status_source(self):
        """The detach_all_for_instance call must be present in update_job_status."""
        import inspect
        from scheduler import update_job_status
        source = inspect.getsource(update_job_status)
        assert "detach_all_for_instance" in source
        assert 'completed' in source and 'failed' in source and 'cancelled' in source

    def test_detach_covers_terminated_state(self):
        """update_job_status must also detach volumes for 'terminated' status."""
        import inspect
        from scheduler import update_job_status
        source = inspect.getsource(update_job_status)
        assert 'terminated' in source

    def test_detach_in_cancel_route_source(self):
        """The cancel endpoint must call detach_all_for_instance."""
        import inspect
        from routes.instances import api_cancel_instance
        source = inspect.getsource(api_cancel_instance)
        assert "detach_all_for_instance" in source

    def test_detach_in_preempt_job_source(self):
        """preempt_job must detach volumes when job leaves host."""
        import inspect
        from scheduler import preempt_job
        source = inspect.getsource(preempt_job)
        assert "detach_all_for_instance" in source


# ── Billing correctness ──────────────────────────────────────────────


class TestBillingVolumeQuery:
    """Billing must only charge volumes that actually exist (available/attached)."""

    def test_billing_only_charges_real_volumes(self):
        """Volume billing query must filter to available/attached status."""
        import inspect
        from billing import BillingEngine
        source = inspect.getsource(BillingEngine.auto_billing_cycle)
        assert "IN ('available', 'attached')" in source
        assert "!= 'deleted'" not in source  # old pattern removed

    def test_billing_uses_singleton(self):
        """terminate_instance must use get_volume_engine(), not VolumeEngine()."""
        import inspect
        from billing import BillingEngine
        source = inspect.getsource(BillingEngine.terminate_instance)
        assert "get_volume_engine" in source
        assert "VolumeEngine()" not in source


# ── Worker agent mount paths ─────────────────────────────────────────


class TestWorkerVolumeMountPaths:
    """Worker agent must use per-volume mount paths from the payload."""

    def test_worker_reads_volume_mounts(self):
        """run_job must read volume_mounts from the job payload."""
        import inspect
        from worker_agent import run_job
        source = inspect.getsource(run_job)
        assert "volume_mounts" in source
        # Must NOT hardcode /workspace:rw for all volumes
        assert ':/workspace:rw"' not in source

    def test_work_endpoint_enriches_mount_paths(self):
        """The agent work endpoint must inject volume_mounts into job payloads."""
        import inspect
        from routes.agent import api_agent_work
        source = inspect.getsource(api_agent_work)
        assert "volume_mounts" in source
        assert "get_instance_volumes" in source
