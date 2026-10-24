"""Optional local coordination must not silently skip uncertain side effects."""

import fcntl
import os
import socket
from types import SimpleNamespace
from unittest import mock

import pytest
import requests

import worker_agent as worker
from tests.test_worker_resource_recovery import resources  # noqa: F401


def checkpoint(auth):
    def remember(state):
        auth["_exclusive_state"] = state

    return remember


def test_absent_optional_coordinator_does_not_wedge_exclusive_lock(resources, monkeypatch):
    # Keep the port bound without listening: connect is definitively refused,
    # and no competing process can acquire it or receive the drain request.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        monkeypatch.setattr(
            worker, "TOWER_SERVERLESS_URL", f"http://127.0.0.1:{listener.getsockname()[1]}"
        )
        monkeypatch.setattr(worker.requests, "post", requests.api.post)
        monkeypatch.setattr(worker.subprocess, "run", mock.Mock())
        worker.worker_privileged.call.side_effect = [
            {"ok": True, "stopped": []},
            {"ok": True},
        ]
        state = worker._acquire_exclusive_gpu("job", checkpoint=checkpoint(resources))
        assert state["ok"] and state["drain_requested"] is False
        assert worker._release_exclusive_gpu("job", state) is True
        assert state["lock_fd"] is None
        fd = os.open(worker.GPU_LOCK_PATH, os.O_WRONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)


@pytest.mark.parametrize("failure", ["timeout", "http_error"])
def test_uncertain_drain_cannot_start_exclusive_work(resources, failure):
    if failure == "timeout":
        worker.requests.post.side_effect = requests.exceptions.ReadTimeout("response lost")
    else:
        worker.requests.post.return_value = SimpleNamespace(status_code=500)
    with pytest.raises(RuntimeError, match="drain"):
        worker._acquire_exclusive_gpu("job", checkpoint=checkpoint(resources))
    worker.worker_privileged.call.assert_not_called()
    state = resources["_exclusive_state"]
    assert state["drain_requested"] and state["lock_fd"] is not None
    # An uncertain accepted drain still needs compensation before releasing.
    worker.requests.post.side_effect = None
    worker.requests.post.return_value = SimpleNamespace(status_code=200)
    assert worker._release_exclusive_gpu("job", state) is True


def test_failed_resume_retains_ownership_until_confirmed(resources, monkeypatch):
    monkeypatch.setattr(worker.subprocess, "run", mock.Mock())
    worker.worker_privileged.call.side_effect = [
        {"ok": True, "stopped": []},
        {"ok": True},
    ]
    state = worker._acquire_exclusive_gpu("job", checkpoint=checkpoint(resources))
    worker.requests.post.side_effect = requests.exceptions.ReadTimeout("resume response lost")
    assert worker._release_exclusive_gpu("job", state) is False
    assert state["drain_requested"] and state["lock_fd"] is not None
    fd = os.open(worker.GPU_LOCK_PATH, os.O_WRONLY)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker.requests.post.side_effect = None
        assert worker._release_exclusive_gpu("job", state) is True
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)
