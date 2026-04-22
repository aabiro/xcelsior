"""P3 A1 — regression guard: worker agent snapshot callback URL.

The `snapshot_container` handler in worker_agent.py originally posted to
`/api/v2/user-images/{id}/complete`, but the API mounts the router
without any prefix, so the correct path is `/user-images/{id}/complete`.
The original bug left every snapshot hanging in `pending` forever. This
test ensures we don't silently regress to the broken path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import worker_agent  # noqa: E402


def _fake_cmd(image_id: str, container: str, image_ref: str) -> dict:
    # Match the cmd shape consumed by worker_agent.drain_agent_commands:
    # `name = cmd.get("command")`, args under `"args"`.
    return {
        "id": "cmd-test-1",
        "command": "snapshot_container",
        "args": {
            "image_id": image_id,
            "container_name": container,
            "image_ref": image_ref,
        },
        "created_by": "pytest",
    }


@pytest.fixture
def _no_network(monkeypatch):
    # Stop drain_agent_commands from actually hitting the API for the
    # GET /agent/commands/... part of its loop.
    get_mock = MagicMock()
    get_mock.return_value = MagicMock(
        status_code=200,
        json=lambda: {"commands": [_fake_cmd("img_abc", "xcl-job1", "xcl-alice-demo:latest")]},
    )
    post_mock = MagicMock()
    post_mock.return_value = MagicMock(status_code=200, text="ok")
    monkeypatch.setattr(worker_agent.requests, "get", get_mock)
    monkeypatch.setattr(worker_agent.requests, "post", post_mock)
    return {"get": get_mock, "post": post_mock}


def _fake_run(returncode=0, stdout="0", stderr=""):
    def _r(*a, **kw):
        return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)
    return _r


def test_snapshot_callback_uses_unprefixed_path(_no_network, monkeypatch):
    monkeypatch.setattr(worker_agent.subprocess, "run", _fake_run(0, "0", ""))
    # Host identity / auth
    monkeypatch.setattr(worker_agent, "HOST_ID", "host-1", raising=False)
    worker_agent.drain_agent_commands()

    # Callback URL must NOT contain /api/v2 and MUST end with /user-images/<id>/complete.
    called_urls = [c.args[0] for c in _no_network["post"].call_args_list]
    callback = next((u for u in called_urls if "/user-images/" in u), None)
    assert callback is not None, f"no callback posted; urls={called_urls}"
    assert "/api/v2/" not in callback, f"regressed to legacy prefix: {callback}"
    assert callback.endswith("/user-images/img_abc/complete"), callback


def test_snapshot_callback_logs_http_error(_no_network, monkeypatch, caplog):
    # Simulate the API rejecting the callback (e.g. bad host binding).
    _no_network["post"].return_value = MagicMock(status_code=403, text="forbidden")
    monkeypatch.setattr(worker_agent.subprocess, "run", _fake_run(0, "0", ""))
    monkeypatch.setattr(worker_agent, "HOST_ID", "host-1", raising=False)
    with caplog.at_level("WARNING"):
        worker_agent.drain_agent_commands()
    # Must surface the HTTP status rather than swallow silently.
    assert any("403" in r.getMessage() for r in caplog.records), caplog.text
