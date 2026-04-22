"""P2.3 — Jupyter / VSCode auto-launch tests.

These tests verify the worker-side `_run_auto_launch` dispatcher:

* A job with no ``auto_launch`` field is a no-op (no docker exec).
* ``auto_launch=["jupyter"]`` issues exactly one `docker exec` whose
  shell payload installs + starts JupyterLab on :8888 with the
  per-instance token.
* ``auto_launch=["vscode"]`` issues exactly one `docker exec` whose
  shell payload starts code-server on :8443 with PASSWORD.
* Non-interactive jobs are never auto-launched.
* Unknown services are rejected (warning log, no exec).

The `subprocess.run` boundary is monkeypatched so the tests never
actually fork docker.
"""
from __future__ import annotations

import subprocess
from typing import Any, List

import pytest


@pytest.fixture
def captured_calls(monkeypatch: pytest.MonkeyPatch) -> List[List[str]]:
    import worker_agent as wa

    calls: List[List[str]] = []

    class _FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(cmd: Any, *a: Any, **kw: Any) -> _FakeCompleted:  # type: ignore[override]
        calls.append(list(cmd))
        return _FakeCompleted()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    # Silence the log-shipping helper so tests don't hit the network.
    monkeypatch.setattr(wa, "_push_log_lines", lambda *a, **kw: None)

    # Don't call the report endpoint during tests.
    monkeypatch.setenv("XCELSIOR_API_URL", "")
    return calls


def test_no_auto_launch_is_noop(captured_calls):
    import worker_agent as wa

    wa._run_auto_launch(
        "jobA", "container1",
        {"interactive": True},
    )
    assert captured_calls == []


def test_auto_launch_requires_interactive(captured_calls):
    import worker_agent as wa

    wa._run_auto_launch(
        "jobA", "container1",
        {"interactive": False, "auto_launch": ["jupyter"]},
    )
    assert captured_calls == []


def test_jupyter_launch(captured_calls):
    import worker_agent as wa

    wa._run_auto_launch(
        "jobA", "container1",
        {"interactive": True, "auto_launch": ["jupyter"]},
    )
    assert len(captured_calls) == 1
    cmd = captured_calls[0]
    # docker exec -d <container> bash -lc "<shell>"
    assert cmd[:3] == ["docker", "exec", "-d"]
    assert cmd[3] == "container1"
    assert cmd[4:6] == ["bash", "-lc"]
    shell = cmd[6]
    assert "jupyter lab" in shell
    assert "--ip=0.0.0.0" in shell
    assert "--port=8888" in shell
    assert "--ServerApp.token=" in shell
    assert "--allow-root" in shell


def test_vscode_launch(captured_calls):
    import worker_agent as wa

    wa._run_auto_launch(
        "jobA", "container1",
        {"interactive": True, "auto_launch": ["vscode"]},
    )
    assert len(captured_calls) == 1
    shell = captured_calls[0][6]
    assert "code-server" in shell
    assert "0.0.0.0:8443" in shell
    assert "PASSWORD=" in shell


def test_both_services_launch(captured_calls):
    import worker_agent as wa

    wa._run_auto_launch(
        "jobA", "container1",
        {"interactive": True, "auto_launch": ["jupyter", "vscode"]},
    )
    assert len(captured_calls) == 2
    shells = [c[6] for c in captured_calls]
    assert any("jupyter lab" in s for s in shells)
    assert any("code-server" in s for s in shells)


def test_unknown_service_skipped(captured_calls):
    import worker_agent as wa

    wa._run_auto_launch(
        "jobA", "container1",
        {"interactive": True, "auto_launch": ["ssh-tunnel"]},
    )
    assert captured_calls == []


def test_token_is_deterministic(monkeypatch: pytest.MonkeyPatch):
    import worker_agent as wa

    monkeypatch.setenv("HOST_SECRET", "s3cret")
    t1 = wa._auto_launch_token("jobA")
    t2 = wa._auto_launch_token("jobA")
    t3 = wa._auto_launch_token("jobB")

    assert t1 == t2
    assert t1 != t3
    assert len(t1) == 32


def test_csv_auto_launch_string_accepted(captured_calls):
    # scheduler sometimes passes a csv string instead of a list
    import worker_agent as wa

    wa._run_auto_launch(
        "jobA", "container1",
        {"interactive": True, "auto_launch": "jupyter,vscode"},
    )
    assert len(captured_calls) == 2
