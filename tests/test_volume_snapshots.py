"""P2.5 — instant CoW volume snapshots via reflink.

Critical invariants the user explicitly asked for:
- NEVER use rsync (RunPod/Vast use CoW instead)
- Snapshot must be O(1)-style — single `cp --reflink` call
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import volumes as volumes_mod  # noqa: E402


class _FakeCursor:
    def __init__(self, rows):
        self._rows = list(rows) if isinstance(rows, list) else [rows]

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class _FakeConn:
    """Each execute() pops the next response; response is a list of row-dicts."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.executed: list[tuple] = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        if not self._responses:
            return _FakeCursor([])
        resp = self._responses.pop(0)
        return _FakeCursor(resp if resp is not None else [])

    def commit(self):
        pass

    def rollback(self):
        pass


def _patch_conn(engine, blocks):
    """``blocks`` is a list of response-queues, one per ``with engine._conn()``."""
    queue = list(blocks)

    class _Cm:
        def __enter__(self_inner):
            self_inner._fc = _FakeConn(queue.pop(0)) if queue else _FakeConn([])
            return self_inner._fc

        def __exit__(self_inner, *a):
            return False

    return patch.object(engine, "_conn", side_effect=lambda: _Cm())


def test_snapshot_uses_reflink_not_rsync(monkeypatch):
    monkeypatch.setattr(volumes_mod, "NFS_SERVER", "nfs.example")
    monkeypatch.setattr(volumes_mod, "NFS_EXPORT_BASE", "/exports/volumes")
    engine = volumes_mod.VolumeEngine()

    captured: list[str] = []

    def fake_ssh(ip, cmd, **_):
        captured.append(cmd)
        if "stat -c%s" in cmd or "du -sb" in cmd:
            return (0, "12345\n", "")
        return (0, "", "")

    blocks = [
        [[{"volume_id": "vol1", "owner_id": "u1", "status": "available", "encrypted": True}]],
        [[]],
    ]
    with _patch_conn(engine, blocks), patch.object(
        engine, "_ssh_exec_with_retry", side_effect=fake_ssh
    ), patch.object(engine, "_emit_event"):
        snap = engine.create_snapshot("vol1", "u1", label="nightly")

    assert snap["status"] == "ready"
    assert snap["size_bytes"] == 12345
    assert snap["label"] == "nightly"
    cp_cmds = [c for c in captured if c.startswith("cp ")]
    assert cp_cmds
    for c in cp_cmds:
        assert "--reflink" in c
    assert not any("rsync" in c for c in captured)


def test_snapshot_rejects_attached_volume(monkeypatch):
    monkeypatch.setattr(volumes_mod, "NFS_SERVER", "nfs.example")
    engine = volumes_mod.VolumeEngine()

    blocks = [
        [[{"volume_id": "vol1", "owner_id": "u1", "status": "attached", "encrypted": True}]],
    ]
    with _patch_conn(engine, blocks):
        try:
            engine.create_snapshot("vol1", "u1")
        except ValueError as e:
            assert "detached" in str(e).lower()
            return
    raise AssertionError("Expected ValueError for attached volume")


def test_snapshot_rejects_wrong_owner(monkeypatch):
    monkeypatch.setattr(volumes_mod, "NFS_SERVER", "nfs.example")
    engine = volumes_mod.VolumeEngine()

    blocks = [
        [[{"volume_id": "vol1", "owner_id": "alice", "status": "available", "encrypted": True}]],
    ]
    with _patch_conn(engine, blocks):
        try:
            engine.create_snapshot("vol1", "bob")
        except ValueError as e:
            assert "not found" in str(e).lower()
            return
    raise AssertionError("Expected ValueError for wrong owner")


def test_restore_uses_reflink(monkeypatch):
    monkeypatch.setattr(volumes_mod, "NFS_SERVER", "nfs.example")
    monkeypatch.setattr(volumes_mod, "NFS_EXPORT_BASE", "/exports/volumes")
    engine = volumes_mod.VolumeEngine()

    captured: list[str] = []

    def fake_ssh(ip, cmd, **_):
        captured.append(cmd)
        return (0, "", "")

    blocks = [
        [
            [{"volume_id": "vol1", "owner_id": "u1", "status": "available", "encrypted": True}],
            [{"snapshot_id": "snap-abc"}],
        ],
    ]
    with _patch_conn(engine, blocks), patch.object(
        engine, "_ssh_exec_with_retry", side_effect=fake_ssh
    ), patch.object(engine, "_emit_event"):
        res = engine.restore_snapshot("vol1", "u1", "snap-abc")

    assert res["status"] == "restored"
    assert any("--reflink" in c for c in captured)
    assert not any("rsync" in c for c in captured)
    assert any(".pre-restore-" in c for c in captured)


def test_delete_snapshot_guards_path(monkeypatch):
    monkeypatch.setattr(volumes_mod, "NFS_SERVER", "nfs.example")
    monkeypatch.setattr(volumes_mod, "NFS_EXPORT_BASE", "/exports/volumes")
    engine = volumes_mod.VolumeEngine()

    captured: list[str] = []

    def fake_ssh(ip, cmd, **_):
        captured.append(cmd)
        return (0, "", "")

    blocks = [
        [[{"snapshot_id": "snap-abc", "vol_owner": "u1", "encrypted": False}]],
        [[]],
    ]
    with _patch_conn(engine, blocks), patch.object(
        engine, "_ssh_exec_with_retry", side_effect=fake_ssh
    ), patch.object(engine, "_emit_event"):
        res = engine.delete_snapshot("vol1", "u1", "snap-abc")

    assert res["status"] == "deleted"
    assert any("_snapshots" in c for c in captured)


def test_list_snapshots_returns_rows(monkeypatch):
    engine = volumes_mod.VolumeEngine()

    rows = [
        {
            "snapshot_id": "snap-2",
            "volume_id": "vol1",
            "label": "later",
            "size_bytes": 200,
            "status": "ready",
            "created_at": time.time(),
        },
        {
            "snapshot_id": "snap-1",
            "volume_id": "vol1",
            "label": "earlier",
            "size_bytes": 100,
            "status": "ready",
            "created_at": time.time() - 100,
        },
    ]

    blocks = [
        [
            [{"owner_id": "u1"}],
            rows,
        ],
    ]
    with _patch_conn(engine, blocks):
        out = engine.list_snapshots("vol1", "u1")
    assert len(out) == 2
    assert out[0]["snapshot_id"] == "snap-2"
