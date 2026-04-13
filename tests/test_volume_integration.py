"""Integration tests for volume lifecycle, data gravity, and billing."""

import os
import time

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_NFS_SERVER", "")  # metadata-only mode


class TestVolumeLifecycle:
    """Full lifecycle: create → attach → detach → delete."""

    def _make_engine(self, monkeypatch):
        from unittest.mock import MagicMock
        from contextlib import contextmanager
        from volumes import VolumeEngine

        engine = VolumeEngine()

        # Track DB state in-memory
        self._volumes = {}
        self._attachments = {}
        self._seq = 0

        outer = self

        class FakeConn:
            def __init__(self):
                self._pending_ops = []

            def execute(self, sql, params=None):
                result = MagicMock()
                sql_lower = sql.strip().lower()

                if "coalesce(sum(size_gb)" in sql_lower:
                    total = sum(v["size_gb"] for v in outer._volumes.values() if v["status"] != "deleted")
                    result.fetchone.return_value = {"total": total}

                elif "select volume_id from volumes where owner_id" in sql_lower and "name" in sql_lower:
                    owner_id, name = params[0], params[1]
                    found = next(
                        (v for v in outer._volumes.values()
                         if v["owner_id"] == owner_id and v["name"] == name and v["status"] != "deleted"),
                        None,
                    )
                    result.fetchone.return_value = {"volume_id": found["volume_id"]} if found else None

                elif "insert into volumes" in sql_lower:
                    vid = params[0]
                    outer._volumes[vid] = {
                        "volume_id": vid, "owner_id": params[1], "name": params[2],
                        "storage_type": params[3], "size_gb": params[4],
                        "region": params[5], "province": params[6],
                        "encrypted": params[7], "status": "provisioning",
                        "created_at": params[8],
                    }

                elif "update volumes set status" in sql_lower and "where volume_id" in sql_lower:
                    # Status may be hardcoded in SQL or passed as param
                    import re as _re
                    m = _re.search(r"status\s*=\s*'(\w+)'", sql)
                    if m:
                        status = m.group(1)
                        vid = params[-1]
                    else:
                        status = params[0]
                        vid = params[-1]
                    if vid in outer._volumes:
                        outer._volumes[vid]["status"] = status

                elif "select * from volumes where volume_id" in sql_lower and "for update" in sql_lower:
                    vid = params[0]
                    if len(params) > 1:
                        # delete_volume: volume_id AND owner_id
                        v = outer._volumes.get(vid)
                        if v and v.get("owner_id") == params[1]:
                            result.fetchone.return_value = v
                        else:
                            result.fetchone.return_value = None
                    else:
                        result.fetchone.return_value = outer._volumes.get(vid)

                elif "from volumes where volume_id" in sql_lower and "status != 'deleted'" in sql_lower:
                    # get_volume (explicit columns or SELECT *)
                    vid = params[0]
                    v = outer._volumes.get(vid)
                    if v and v["status"] != "deleted":
                        result.fetchone.return_value = v
                    else:
                        result.fetchone.return_value = None

                elif "select attachment_id from volume_attachments where volume_id" in sql_lower and "detached_at = 0" in sql_lower:
                    vid = params[0]
                    found = next(
                        (a for a in outer._attachments.values()
                         if a["volume_id"] == vid and a["detached_at"] == 0),
                        None,
                    )
                    result.fetchone.return_value = {"attachment_id": found["attachment_id"]} if found else None

                elif "select * from volume_attachments" in sql_lower and "for update" in sql_lower:
                    vid = params[0]
                    if len(params) > 1:
                        inst = params[1]
                        found = next(
                            (a for a in outer._attachments.values()
                             if a["volume_id"] == vid and a["instance_id"] == inst and a["detached_at"] == 0),
                            None,
                        )
                    else:
                        found = next(
                            (a for a in outer._attachments.values()
                             if a["volume_id"] == vid and a["detached_at"] == 0),
                            None,
                        )
                    result.fetchone.return_value = found

                elif "insert into volume_attachments" in sql_lower:
                    aid = params[0]
                    outer._attachments[aid] = {
                        "attachment_id": aid, "volume_id": params[1],
                        "instance_id": params[2], "mount_path": params[3],
                        "mode": params[4], "attached_at": params[5], "detached_at": 0,
                    }

                elif "update volume_attachments set detached_at" in sql_lower:
                    ts = params[0]
                    aid = params[1]
                    if aid in outer._attachments:
                        outer._attachments[aid]["detached_at"] = ts

                elif "select count(*) as cnt from volume_attachments" in sql_lower:
                    vid = params[0]
                    cnt = sum(
                        1 for a in outer._attachments.values()
                        if a["volume_id"] == vid and a["detached_at"] == 0
                    )
                    result.fetchone.return_value = {"cnt": cnt}

                elif "select volume_id, attachment_id, mount_path from volume_attachments" in sql_lower:
                    inst = params[0]
                    found = [
                        a for a in outer._attachments.values()
                        if a["instance_id"] == inst and a["detached_at"] == 0
                    ]
                    result.fetchall.return_value = found

                elif "from volumes where owner_id" in sql_lower and "status != 'deleted'" in sql_lower:
                    owner = params[0]
                    rows = [v for v in outer._volumes.values() if v["owner_id"] == owner and v["status"] != "deleted"]
                    result.fetchall.return_value = rows

                elif "select instance_id from volume_attachments where volume_id" in sql_lower and "detached_at = 0" in sql_lower:
                    # attached_to enrichment for get_volume
                    vid = params[0]
                    found = next(
                        (a for a in outer._attachments.values()
                         if a["volume_id"] == vid and a["detached_at"] == 0),
                        None,
                    )
                    result.fetchone.return_value = {"instance_id": found["instance_id"]} if found else None

                elif "select volume_id, instance_id from volume_attachments where volume_id = any" in sql_lower:
                    # attached_to batch enrichment for list_volumes
                    vol_ids = params[0]
                    found = [
                        {"volume_id": a["volume_id"], "instance_id": a["instance_id"]}
                        for a in outer._attachments.values()
                        if a["volume_id"] in vol_ids and a["detached_at"] == 0
                    ]
                    result.fetchall.return_value = found

                elif "select distinct j.host_id" in sql_lower and "volume_attachments" in sql_lower:
                    # get_volume_host_ids
                    vol_ids = params[0]
                    result.fetchall.return_value = []

                else:
                    result.fetchone.return_value = None
                    result.fetchall.return_value = []

                return result

            def commit(self):
                pass

            def rollback(self):
                pass

        @contextmanager
        def _mock_conn():
            yield FakeConn()

        monkeypatch.setattr(engine, "_conn", _mock_conn)
        monkeypatch.setattr(engine, "_provision_volume_storage", lambda vid, sz: True)
        monkeypatch.setattr(engine, "_destroy_volume_storage", lambda vid: True)
        monkeypatch.setattr(engine, "_mount_on_host", lambda *a: True)
        monkeypatch.setattr(engine, "_unmount_from_host", lambda *a: True)
        return engine

    def test_full_lifecycle(self, monkeypatch):
        """create → attach → detach → delete."""
        engine = self._make_engine(monkeypatch)

        # 1. Create volume
        vol = engine.create_volume("user-1", "training-data", 50)
        assert vol["volume_id"].startswith("vol-")
        assert vol["status"] == "available"
        vid = vol["volume_id"]

        # Verify in internal state
        assert self._volumes[vid]["status"] == "available"

        # 2. Attach to instance
        att = engine.attach_volume(vid, "inst-run-1")
        assert att["volume_id"] == vid
        assert att["instance_id"] == "inst-run-1"
        assert self._volumes[vid]["status"] == "attached"

        # 3. Detach
        det = engine.detach_volume(vid, "inst-run-1")
        assert det["status"] == "detached"
        assert self._volumes[vid]["status"] == "available"

        # 4. Delete
        deleted = engine.delete_volume(vid, "user-1")
        assert deleted["status"] == "deleted"
        assert self._volumes[vid]["status"] == "deleted"

    def test_attach_detach_all_on_termination(self, monkeypatch):
        """Simulate instance termination detaching all volumes."""
        engine = self._make_engine(monkeypatch)

        # Create two volumes and attach both
        v1 = engine.create_volume("user-1", "data-1", 10)
        v2 = engine.create_volume("user-1", "data-2", 20)
        engine.attach_volume(v1["volume_id"], "inst-term")
        engine.attach_volume(v2["volume_id"], "inst-term")

        assert self._volumes[v1["volume_id"]]["status"] == "attached"
        assert self._volumes[v2["volume_id"]]["status"] == "attached"

        # Simulate termination
        count = engine.detach_all_for_instance("inst-term")
        assert count == 2
        assert self._volumes[v1["volume_id"]]["status"] == "available"
        assert self._volumes[v2["volume_id"]]["status"] == "available"

    def test_cannot_delete_attached_volume(self, monkeypatch):
        """Delete is blocked while volume has active attachments."""
        engine = self._make_engine(monkeypatch)

        vol = engine.create_volume("user-1", "pinned", 10)
        vid = vol["volume_id"]
        engine.attach_volume(vid, "inst-x")

        with pytest.raises(ValueError, match="active attachments"):
            engine.delete_volume(vid, "user-1")

    def test_list_volumes_excludes_deleted(self, monkeypatch):
        """list_volumes should not return deleted volumes."""
        engine = self._make_engine(monkeypatch)

        v1 = engine.create_volume("user-1", "keep", 10)
        v2 = engine.create_volume("user-1", "gone", 10)
        engine.delete_volume(v2["volume_id"], "user-1")

        vols = engine.list_volumes("user-1")
        ids = [v["volume_id"] for v in vols]
        assert v1["volume_id"] in ids
        assert v2["volume_id"] not in ids


class TestDataGravity:
    """Data gravity: bin-packer prefers hosts where volumes are located."""

    def test_binpack_gravity_prefers_volume_host(self):
        """Hosts with data-local volumes should score 1.3x higher."""
        from scheduler import allocate_binpack

        hosts = [
            {
                "host_id": "remote",
                "free_vram_gb": 24,
                "total_vram_gb": 24,
                "gpu_count": 1,
                "admitted": True,
                "gpu_model": "RTX 4090",
                "cost_per_hour": 0.5,
            },
            {
                "host_id": "local",
                "free_vram_gb": 24,
                "total_vram_gb": 24,
                "gpu_count": 1,
                "admitted": True,
                "gpu_model": "RTX 4090",
                "cost_per_hour": 0.5,
            },
        ]
        job = {"name": "train", "vram_needed_gb": 20, "num_gpus": 1}
        # volume_host_ids tells the allocator where the data lives
        best = allocate_binpack(job, hosts, volume_host_ids={"local"})
        assert best["host_id"] == "local"

    def test_binpack_gravity_no_volumes(self):
        """Without volume preference, picks tighter fit."""
        from scheduler import allocate_binpack

        hosts = [
            {
                "host_id": "big",
                "free_vram_gb": 80,
                "total_vram_gb": 80,
                "gpu_count": 1,
                "admitted": True,
                "gpu_model": "A100",
                "cost_per_hour": 2.0,
            },
            {
                "host_id": "small",
                "free_vram_gb": 24,
                "total_vram_gb": 24,
                "gpu_count": 1,
                "admitted": True,
                "gpu_model": "RTX 4090",
                "cost_per_hour": 0.5,
            },
        ]
        job = {"name": "infer", "vram_needed_gb": 20, "num_gpus": 1}
        best = allocate_binpack(job, hosts)
        # Without gravity, tighter fit (small) should win
        assert best["host_id"] == "small"


class TestVolumeBillingConstants:
    """Verify volume billing configuration is correct."""

    def test_billing_rate_per_gb_month(self):
        """Volume price constant should be $0.07/GB/month."""
        from routes.volumes import VOLUME_PRICE_PER_GB_MONTH_CAD
        assert VOLUME_PRICE_PER_GB_MONTH_CAD == 0.07

    def test_billing_uses_storage_model(self):
        """Billing should categorize volumes as gpu_model='storage', tier='volume'."""
        # Verify the billing engine's auto_billing_cycle method references 'storage'
        import inspect
        from billing import BillingEngine
        source = inspect.getsource(BillingEngine.auto_billing_cycle)
        assert "storage" in source
        assert "volume" in source
