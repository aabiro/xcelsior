"""C7 — subprocess-mocked lifecycle tests for worker_agent.drain_agent_commands.

Covers the four container-lifecycle command handlers introduced in P3.2 / A3:
    - snapshot_container  (docker commit + docker push)
    - stop_container      (docker stop -t 30  +  docker rm -f)
    - start_container     (docker start)
    - pause_container     (docker stop -t 30 ONLY — no rm)

Tests mock both ``worker_agent.requests`` (to feed a crafted commands list
without touching the real API) and ``worker_agent.subprocess.run`` (to
capture the docker argv and return canned CompletedProcess results).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

# Ensure project root on path (conftest already does this, but be defensive
# when pytest is invoked against this single file).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# worker_agent at import time wants XCELSIOR_HOST_ID — supply one.
os.environ.setdefault("XCELSIOR_HOST_ID", "test-host-lifecycle")
os.environ.setdefault("XCELSIOR_API_URL", "http://localhost:9500")

import worker_agent  # noqa: E402


def _ok(stdout: str = "", stderr: str = "", rc: int = 0):
    """Build a CompletedProcess-like object."""
    return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


def _mock_requests_get(commands: list[dict]):
    """Return a callable that mimics ``requests.get`` for /agent/commands/*."""
    def _fake_get(url, headers=None, timeout=None):
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"commands": commands},
        )
    return _fake_get


def _mock_requests_post_noop():
    """Swallow the snapshot /complete callback — we don't assert on it here."""
    def _fake_post(url, headers=None, json=None, timeout=None):
        return SimpleNamespace(status_code=200, text="ok")
    return _fake_post


# ---------------------------------------------------------------------------
# stop_container
# ---------------------------------------------------------------------------

def test_stop_container_runs_docker_stop_then_rm(monkeypatch):
    cmds = [{
        "id": 101, "command": "stop_container",
        "args": {"job_id": "job-aaa", "container_name": "xcl-aaa"},
        "created_by": "billing_terminate",
    }]
    monkeypatch.setattr(worker_agent, "requests",
                        SimpleNamespace(get=_mock_requests_get(cmds),
                                        RequestException=Exception))

    calls: list[list[str]] = []
    def _fake_run(argv, *a, **kw):
        calls.append(list(argv))
        return _ok()
    monkeypatch.setattr(worker_agent.subprocess, "run", _fake_run)

    n = worker_agent.drain_agent_commands()

    assert n == 1
    assert ["docker", "stop", "-t", "30", "xcl-aaa"] in calls
    assert ["docker", "rm", "-f", "xcl-aaa"] in calls


def test_stop_container_missing_container_name_is_skipped(monkeypatch):
    cmds = [{
        "id": 102, "command": "stop_container",
        "args": {},  # no job_id, no container_name → helper can't derive
        "created_by": "billing_terminate",
    }]
    monkeypatch.setattr(worker_agent, "requests",
                        SimpleNamespace(get=_mock_requests_get(cmds),
                                        RequestException=Exception))
    called = []
    monkeypatch.setattr(worker_agent.subprocess, "run",
                        lambda *a, **kw: (called.append(a), _ok())[1])

    n = worker_agent.drain_agent_commands()
    assert n == 0
    assert called == []


# ---------------------------------------------------------------------------
# pause_container — must NOT call docker rm
# ---------------------------------------------------------------------------

def test_pause_container_does_not_remove_container(monkeypatch):
    """A3 regression: pause must preserve the container for later resume."""
    cmds = [{
        "id": 201, "command": "pause_container",
        "args": {"job_id": "job-bbb", "container_name": "xcl-bbb"},
        "created_by": "billing_pause",
    }]
    monkeypatch.setattr(worker_agent, "requests",
                        SimpleNamespace(get=_mock_requests_get(cmds),
                                        RequestException=Exception))

    calls: list[list[str]] = []
    def _fake_run(argv, *a, **kw):
        calls.append(list(argv))
        return _ok()
    monkeypatch.setattr(worker_agent.subprocess, "run", _fake_run)

    n = worker_agent.drain_agent_commands()

    assert n == 1
    assert ["docker", "stop", "-t", "30", "xcl-bbb"] in calls
    # Critical: pause MUST NOT `docker rm` — otherwise the container is
    # destroyed and resume would need to re-run the image.
    rm_calls = [c for c in calls if len(c) >= 2 and c[0] == "docker" and c[1] == "rm"]
    assert rm_calls == [], f"pause_container unexpectedly issued docker rm: {rm_calls}"


def test_pause_container_nonzero_rc_is_not_counted(monkeypatch):
    cmds = [{
        "id": 202, "command": "pause_container",
        "args": {"job_id": "job-ccc", "container_name": "xcl-ccc"},
        "created_by": "billing_pause",
    }]
    monkeypatch.setattr(worker_agent, "requests",
                        SimpleNamespace(get=_mock_requests_get(cmds),
                                        RequestException=Exception))
    monkeypatch.setattr(worker_agent.subprocess, "run",
                        lambda *a, **kw: _ok(stderr="no such container", rc=1))

    n = worker_agent.drain_agent_commands()
    assert n == 0  # failed rc → not counted dispatched


# ---------------------------------------------------------------------------
# start_container
# ---------------------------------------------------------------------------

def test_start_container_runs_docker_start(monkeypatch):
    cmds = [{
        "id": 301, "command": "start_container",
        "args": {"job_id": "job-ddd", "container_name": "xcl-ddd"},
        "created_by": "billing_resume",
    }]
    monkeypatch.setattr(worker_agent, "requests",
                        SimpleNamespace(get=_mock_requests_get(cmds),
                                        RequestException=Exception))

    calls: list[list[str]] = []
    def _fake_run(argv, *a, **kw):
        calls.append(list(argv))
        return _ok()
    monkeypatch.setattr(worker_agent.subprocess, "run", _fake_run)

    n = worker_agent.drain_agent_commands()

    assert n == 1
    assert calls == [["docker", "start", "xcl-ddd"]]


def test_start_container_failure_reports_user_paused(monkeypatch):
    """B8 regression: docker start rc!=0 must flip job back to user_paused."""
    cmds = [{
        "id": 302, "command": "start_container",
        "args": {"job_id": "job-eee", "container_name": "xcl-eee"},
        "created_by": "billing_resume",
    }]
    monkeypatch.setattr(worker_agent, "requests",
                        SimpleNamespace(get=_mock_requests_get(cmds),
                                        RequestException=Exception))
    monkeypatch.setattr(worker_agent.subprocess, "run",
                        lambda *a, **kw: _ok(stderr="Error: no such container: xcl-eee", rc=1))

    reports: list[tuple] = []
    monkeypatch.setattr(
        worker_agent, "report_job_status",
        lambda job_id, status, error_message=None, **kw: reports.append(
            (job_id, status, error_message)
        ),
    )

    n = worker_agent.drain_agent_commands()

    assert n == 0
    assert len(reports) == 1
    job_id, status, err = reports[0]
    assert job_id == "job-eee"
    assert status == "user_paused"
    assert err and "resume failed" in err


# ---------------------------------------------------------------------------
# snapshot_container — commit + push
# ---------------------------------------------------------------------------

def test_snapshot_container_commits_and_pushes(monkeypatch):
    image_ref = "ghcr.io/aabiro/user/demo:v1"
    monkeypatch.setenv("XCELSIOR_REGISTRY_URL", "ghcr.io/aabiro")

    cmds = [{
        "id": 401, "command": "snapshot_container",
        "args": {
            "image_id": "img-fff",
            "container_name": "xcl-fff",
            "image_ref": image_ref,
        },
        "created_by": "snapshot",
    }]
    monkeypatch.setattr(worker_agent, "requests",
                        SimpleNamespace(get=_mock_requests_get(cmds),
                                        post=_mock_requests_post_noop(),
                                        RequestException=Exception))

    calls: list[list[str]] = []
    def _fake_run(argv, *a, **kw):
        calls.append(list(argv))
        # "docker image inspect … {{.Size}}" → must produce an int-parseable stdout
        if len(argv) >= 3 and argv[0] == "docker" and argv[1] == "image" and argv[2] == "inspect":
            return _ok(stdout="12345678\n")
        return _ok()
    monkeypatch.setattr(worker_agent.subprocess, "run", _fake_run)

    callback_bodies: list[dict] = []
    def _fake_post(url, headers=None, json=None, timeout=None):
        callback_bodies.append({"url": url, "body": json})
        return SimpleNamespace(status_code=200, text="ok")
    monkeypatch.setattr(worker_agent.requests, "post", _fake_post, raising=False)
    # Monkeypatch.setattr with raising=False may not hit the SimpleNamespace;
    # so rebuild requests namespace with the capturing post.
    monkeypatch.setattr(
        worker_agent, "requests",
        SimpleNamespace(
            get=_mock_requests_get(cmds),
            post=_fake_post,
            RequestException=Exception,
        ),
    )

    n = worker_agent.drain_agent_commands()

    assert n == 1
    # Commit + push must both have run with the exact image_ref.
    assert ["docker", "commit", "xcl-fff", image_ref] in calls
    assert ["docker", "push", image_ref] in calls
    # Callback MUST hit /user-images/{id}/complete (A1 regression guard — it
    # used to be /api/v2/user-images/{id}/complete which 404s).
    assert callback_bodies, "snapshot handler did not issue /complete callback"
    cb = callback_bodies[0]
    assert cb["url"].endswith("/user-images/img-fff/complete")
    assert cb["body"]["status"] == "ready"
    assert cb["body"]["size_bytes"] == 12345678
    assert cb["body"].get("error", "") == ""


def test_snapshot_container_no_registry_reports_failure(monkeypatch):
    """E2: XCELSIOR_REGISTRY_URL unset → must callback with status=failed +
    error='registry_not_configured', AND must have rmi'd the local tag."""
    monkeypatch.delenv("XCELSIOR_REGISTRY_URL", raising=False)

    cmds = [{
        "id": 402, "command": "snapshot_container",
        "args": {
            "image_id": "img-ggg",
            "container_name": "xcl-ggg",
            "image_ref": "ghcr.io/aabiro/user/demo:v2",
        },
        "created_by": "snapshot",
    }]

    calls: list[list[str]] = []
    def _fake_run(argv, *a, **kw):
        calls.append(list(argv))
        if len(argv) >= 3 and argv[0] == "docker" and argv[1] == "image" and argv[2] == "inspect":
            return _ok(stdout="999\n")
        return _ok()

    callback_bodies: list[dict] = []
    def _fake_post(url, headers=None, json=None, timeout=None):
        callback_bodies.append({"url": url, "body": json})
        return SimpleNamespace(status_code=200, text="ok")

    monkeypatch.setattr(
        worker_agent, "requests",
        SimpleNamespace(
            get=_mock_requests_get(cmds),
            post=_fake_post,
            RequestException=Exception,
        ),
    )
    monkeypatch.setattr(worker_agent.subprocess, "run", _fake_run)

    worker_agent.drain_agent_commands()

    # MUST have rmi'd the locally-committed tag (B6 cleanup).
    rmi_calls = [c for c in calls if len(c) >= 2 and c[0] == "docker" and c[1] == "rmi"]
    assert rmi_calls, f"expected docker rmi cleanup when registry unset, got calls={calls}"

    assert callback_bodies
    body = callback_bodies[0]["body"]
    assert body["status"] == "failed"
    assert body["error"] == "registry_not_configured"


# ---------------------------------------------------------------------------
# allowlist defence-in-depth
# ---------------------------------------------------------------------------

def test_unknown_command_is_rejected_not_dispatched(monkeypatch):
    cmds = [{
        "id": 501, "command": "rm_rf_slash",  # not in allowlist
        "args": {}, "created_by": "attacker",
    }]
    monkeypatch.setattr(worker_agent, "requests",
                        SimpleNamespace(get=_mock_requests_get(cmds),
                                        RequestException=Exception))
    called = []
    monkeypatch.setattr(worker_agent.subprocess, "run",
                        lambda *a, **kw: (called.append(a), _ok())[1])

    n = worker_agent.drain_agent_commands()

    assert n == 0
    assert called == [], "unknown command must never reach subprocess.run"
