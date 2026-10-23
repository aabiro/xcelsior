"""Real journal/flock recovery with privileged operations and Docker isolated."""

import fcntl
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import worker_agent as worker


@pytest.fixture
def resources(tmp_path, monkeypatch):
    auth = dict(
        job_id="job",
        attempt_id="attempt",
        lease_id="lease",
        fencing_token=7,
        container_name="xcl-job-attempt",
        stop=threading.Event(),
    )
    monkeypatch.setattr(worker, "HOST_ID", "host")
    monkeypatch.setattr(worker, "_V2_JOURNAL_PATH", str(tmp_path / "journal.json"))
    monkeypatch.setattr(worker, "GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))
    monkeypatch.setattr(worker, "_v2_attempts", {"job": auth})
    monkeypatch.setattr(worker, "_v2_cleanup_pending", {})
    monkeypatch.setattr(worker, "_v2_completed_commands", {})
    monkeypatch.setattr(worker, "_v2_enabled", True)
    monkeypatch.setattr(worker, "_shutdown", threading.Event())
    monkeypatch.setattr(worker, "_active_containers", {})
    monkeypatch.setattr(worker, "_adopted_containers", set())
    monkeypatch.setattr(worker, "LogForwarder", mock.Mock())
    monkeypatch.setattr(worker, "_v2_post", mock.Mock(return_value=(200, {})))
    monkeypatch.setattr(
        worker, "_v2_container_state", mock.Mock(return_value=("exited", 0, "attempt"))
    )
    monkeypatch.setattr(
        worker.requests, "post", mock.Mock(return_value=SimpleNamespace(status_code=200))
    )
    monkeypatch.setattr(worker.worker_privileged, "call", mock.Mock(return_value={"ok": True}))
    for name in ["_unmount_nfs", "detach_encrypted_volume", "destroy_encrypted_volume"]:
        monkeypatch.setattr(worker, name, mock.Mock(return_value=True))
    yield auth
    state = auth.get("_exclusive_state") or {}
    if state.get("lock_fd") is not None:
        os.close(state["lock_fd"])


def own(auth, *, exclusive=False):
    worker._v2_resource_intent(auth, "nfs_mounts", "/mnt/xcelsior-volumes/volume")
    worker._v2_resource_intent(auth, "encrypted_volumes", "volume")
    worker._v2_resource_intent(auth, "workspaces", "ws-job-attempt")
    if exclusive:
        worker._v2_resource_intent(auth, "exclusive", {"ollama_pids": [123]})
        assert worker._v2_restore_resource_lock(auth)


def test_restart_restores_flock_and_terminal_cleanup_ownership(resources):
    auth = resources
    own(auth, exclusive=True)
    state = auth["_exclusive_state"]
    os.close(state["lock_fd"])
    state["lock_fd"] = None
    worker._v2_attempts.clear()
    worker._v2_container_state.return_value = ("running", 0, "attempt")
    with mock.patch.object(worker.threading, "Thread"):
        assert worker.v2_adopt_attempts() == 1
    restored = worker._v2_attempts["job"]
    assert restored["resources"]["exclusive"]["ollama_pids"] == [123]
    competitor = os.open(worker.GPU_LOCK_PATH, os.O_WRONLY)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker._v2_forget_attempt("job", restored)
        record = json.loads(Path(worker._V2_JOURNAL_PATH).read_text())
        assert "attempt" in record[worker._V2_CLEANUP_KEY]
        assert "lock_fd" not in json.dumps(record)
        worker._v2_container_state.return_value = ("exited", 0, "attempt")
        worker._v2_retry_resource_cleanup()
        assert worker._v2_cleanup_pending == {}
        worker._unmount_nfs.assert_called_once_with("/mnt/xcelsior-volumes/volume")
        worker.detach_encrypted_volume.assert_called_once_with("volume")
        worker.destroy_encrypted_volume.assert_called_once_with("ws-job-attempt")
        worker.worker_privileged.call.assert_called_once_with(
            "gpu_tenants", action="resume", pids=[123]
        )
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(competitor)


@pytest.mark.parametrize(
    "state,label", [("running", "attempt"), ("unknown", ""), ("exited", "other")]
)
def test_cleanup_requires_matching_quiescent_container(resources, state, label):
    own(resources)
    worker._v2_forget_attempt("job", resources)
    worker._v2_container_state.return_value = (state, 0, label)
    worker._v2_retry_resource_cleanup()
    worker._unmount_nfs.assert_not_called()
    worker.detach_encrypted_volume.assert_not_called()
    worker.destroy_encrypted_volume.assert_not_called()
    assert "attempt" in worker._v2_cleanup_pending


def test_failed_cleanup_is_journaled_and_only_unfinished_operations_repeat(resources):
    own(resources, exclusive=True)
    worker._v2_forget_attempt("job", resources)
    worker._unmount_nfs.return_value = False
    worker._v2_retry_resource_cleanup()
    worker.destroy_encrypted_volume.assert_called_once()
    worker.detach_encrypted_volume.assert_called_once()
    worker.requests.post.assert_not_called()
    saved = json.loads(Path(worker._V2_JOURNAL_PATH).read_text())[worker._V2_CLEANUP_KEY]["attempt"]
    assert saved["resources"]["nfs_mounts"]
    assert not saved["resources"]["workspaces"]
    worker._unmount_nfs.return_value = True
    worker._v2_retry_resource_cleanup()
    worker.destroy_encrypted_volume.assert_called_once()
    worker.detach_encrypted_volume.assert_called_once()
    assert worker._unmount_nfs.call_count == 2
    assert worker._v2_cleanup_pending == {}


def test_shared_resources_transfer_to_last_owner(resources):
    own(resources)
    other = dict(
        resources,
        job_id="next",
        attempt_id="next",
        lease_id="next",
        fencing_token=8,
        container_name="xcl-next",
        resources={},
    )
    worker._v2_attempts["next"] = other
    own(other)
    worker._v2_forget_attempt("job", resources)
    worker._v2_retry_resource_cleanup()
    worker._unmount_nfs.assert_not_called()
    worker.detach_encrypted_volume.assert_not_called()
    worker.destroy_encrypted_volume.assert_not_called()
    worker._v2_forget_attempt("next", other)
    worker._v2_container_state.return_value = ("exited", 0, "next")
    worker._v2_retry_resource_cleanup()
    worker._unmount_nfs.assert_called_once()
    worker.detach_encrypted_volume.assert_called_once()
    worker.destroy_encrypted_volume.assert_called_once()


def test_stale_attempt_cannot_record_resource_intent(resources):
    worker._v2_attempts["job"] = dict(resources, attempt_id="new", fencing_token=8)
    with pytest.raises(RuntimeError, match="lost its local fenced authority"):
        worker._v2_resource_intent(resources, "nfs_mounts", "/mnt/xcelsior-nfs")
    assert "resources" not in resources


def test_resource_intent_requires_durable_journal_before_io(resources, monkeypatch):
    monkeypatch.setattr(worker, "_v2_journal_save", lambda: False)
    with pytest.raises(OSError, match="persist fenced resource ownership"):
        worker._v2_resource_intent(resources, "workspaces", "ws-job-attempt")
    worker.destroy_encrypted_volume.assert_not_called()


def test_another_host_cannot_cleanup_a_copied_journal(resources, monkeypatch):
    own(resources)
    worker._v2_forget_attempt("job", resources)
    monkeypatch.setattr(worker, "HOST_ID", "different")
    worker._v2_retry_resource_cleanup()
    worker._unmount_nfs.assert_not_called()
    assert worker._v2_cleanup_pending


def test_exclusive_lock_contention_defers_adoption(resources):
    own(resources, exclusive=True)
    fd = resources["_exclusive_state"]["lock_fd"]
    worker._v2_attempts.clear()
    worker._v2_container_state.return_value = ("running", 0, "attempt")
    try:
        with mock.patch.object(worker.threading, "Thread") as thread:
            assert worker.v2_adopt_attempts() == 0
        worker.LogForwarder.assert_not_called()
        assert thread.call_args.kwargs["target"] is worker._v2_pending_adoption_loop
    finally:
        os.close(fd)
        resources["_exclusive_state"]["lock_fd"] = None


def test_exclusive_pause_intent_is_durable_before_signals(resources, monkeypatch):
    monkeypatch.setattr(worker.subprocess, "run", mock.Mock())

    def helper(op, **args):
        if args["action"] == "inspect":
            return {"ok": True, "stopped": [111, 222]}
        assert args == {"action": "pause", "pids": [111, 222]}
        saved = json.loads(Path(worker._V2_JOURNAL_PATH).read_text())["job"]["resources"]
        assert saved["exclusive"]["ollama_pids"] == [111, 222]
        assert saved["exclusive"]["drain_requested"]
        return {"ok": True, "stopped": [111, 222]}

    monkeypatch.setattr(worker.worker_privileged, "call", mock.Mock(side_effect=helper))

    def checkpoint(state):
        resources["_exclusive_state"] = state
        worker._v2_resource_intent(
            resources,
            "exclusive",
            {
                "ollama_pids": list(state["ollama_pids"]),
                "drain_requested": state["drain_requested"],
            },
        )

    state = worker._acquire_exclusive_gpu("job", checkpoint=checkpoint)
    assert state["ok"]
    assert worker.worker_privileged.call.call_count == 2


def test_failed_checkpoint_cannot_drain_or_pause_gpu_tenants(resources):
    def failed_checkpoint(state):
        resources["_exclusive_state"] = state
        raise OSError("journal unavailable")

    with pytest.raises(OSError, match="journal unavailable"):
        worker._acquire_exclusive_gpu("job", checkpoint=failed_checkpoint)
    worker.requests.post.assert_not_called()
    worker.worker_privileged.call.assert_not_called()


def test_privileged_pause_signals_only_the_recorded_plan(monkeypatch):
    helper = worker.worker_privileged
    monkeypatch.setattr(helper, "_is_ollama", lambda pid: pid in [111, 333])
    monkeypatch.setattr(
        helper, "_pids", lambda predicate: [111, 222, 333] if predicate is helper._is_ollama else []
    )
    signal = mock.Mock()
    monkeypatch.setattr(helper.os, "kill", signal)
    assert helper.op_gpu_tenants({"action": "inspect"}, {}) == {"stopped": [111, 222, 333]}
    signal.assert_not_called()
    assert helper.op_gpu_tenants({"action": "pause", "pids": [111, 222]}, {}) == {"stopped": [111]}
    signal.assert_called_once_with(111, 19)
