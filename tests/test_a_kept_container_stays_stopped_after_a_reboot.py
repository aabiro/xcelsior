"""A container the agent stops and keeps stays stopped after a reboot.

Job containers run with `--restart on-failure:3` (auth/security.py). `docker
stop` and `docker kill` leave exit code 137, and Docker starts a stopped
on-failure container with a non-zero code again the next time its daemon
starts. A reboot does that too. Shown on Docker 29.1.3 on 2026-11-27: a stopped
container came back up after the daemon restarted, and one given restart policy
"no" after its stop stayed exited.

So an instance its owner stopped, a container the reconciler stopped and kept,
or a container fenced after its authority was lost came back after the host's
next reboot. It ran outside any attempt, unbilled, holding its GPU and its SSH
port. Two prod containers were in that state when this was found. From agent
2.4.3, every container the agent keeps after a stop gets restart policy "no".
"""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("XCELSIOR_HOST_ID", "test-host-kept-stopped")
os.environ.setdefault("XCELSIOR_API_URL", "http://localhost:9500")

import worker_agent  # noqa: E402

CLEAR = ["docker", "update", "--restart=no"]


def _done(rc: int = 0, stderr: str = ""):
    return SimpleNamespace(returncode=rc, stdout="", stderr=stderr)


def _stop_attempt(preserve: bool) -> dict:
    return {
        "command_id": "cmd-stop-kept",
        "command": "stop_attempt",
        "job_id": "job-k",
        "attempt_id": "att-k",
        "fencing_token": 3,
        "args": {
            "job_id": "job-k",
            "lease_id": "lease-k",
            "attempt_id": "att-k",
            "fencing_token": 3,
            "preserve": preserve,
        },
    }


def _run_owner_stop(tmp_path, preserve: bool, state: str = "running") -> list[list[str]]:
    auth = {
        "lease_id": "lease-k",
        "job_id": "job-k",
        "attempt_id": "att-k",
        "fencing_token": 3,
        "renewal_ttl_sec": 300,
        "container_name": "xcl-job-k-att-k",
        "stop": threading.Event(),
    }
    calls: list[list[str]] = []

    def run(argv, **_kwargs):
        calls.append(list(argv))
        return _done()

    with worker_agent._v2_attempts_lock:
        worker_agent._v2_attempts.clear()
        worker_agent._v2_completed_commands.clear()
        worker_agent._v2_attempts["job-k"] = auth
    try:
        with (
            mock.patch.object(worker_agent, "_V2_JOURNAL_PATH", str(tmp_path / "v2.json")),
            mock.patch.object(worker_agent, "_v2_renew", return_value=(200, {})),
            mock.patch.object(
                worker_agent, "_v2_container_state", return_value=(state, 137, "att-k")
            ),
            mock.patch.object(worker_agent, "v2_report_attempt_status", return_value="ok"),
            mock.patch.object(worker_agent, "_v2_record_and_ack") as ack,
            mock.patch.object(worker_agent.subprocess, "run", run),
        ):
            worker_agent.handle_stop_attempt(_stop_attempt(preserve))
        assert ack.called, "the stop was acknowledged"
    finally:
        with worker_agent._v2_attempts_lock:
            worker_agent._v2_attempts.clear()
            worker_agent._v2_completed_commands.clear()
    return calls


def test_an_owner_stop_keeps_the_container_with_restart_policy_no(tmp_path):
    calls = _run_owner_stop(tmp_path, preserve=True)
    stop = calls.index(["docker", "stop", "-t", "30", "xcl-job-k-att-k"])
    clear = calls.index(CLEAR + ["xcl-job-k-att-k"])
    assert stop < clear, calls
    assert not any(argv[:2] == ["docker", "rm"] for argv in calls)


def test_a_kept_container_that_had_already_exited_also_loses_its_restart_policy(tmp_path):
    calls = _run_owner_stop(tmp_path, preserve=True, state="exited")
    assert CLEAR + ["xcl-job-k-att-k"] in calls
    assert not any(argv[:2] == ["docker", "stop"] for argv in calls)


def test_a_terminate_removes_the_container_instead(tmp_path):
    calls = _run_owner_stop(tmp_path, preserve=False)
    assert ["docker", "rm", "-f", "xcl-job-k-att-k"] in calls
    assert not any(argv[:3] == CLEAR for argv in calls)


def _dispatch_keep_stop(update_rc: int = 0) -> tuple[int, list[list[str]]]:
    calls: list[list[str]] = []

    def run(argv, *_a, **_kw):
        calls.append(list(argv))
        if argv[:2] == ["docker", "update"]:
            return _done(update_rc, "" if update_rc == 0 else "Error response from daemon")
        return _done(stderr="")

    command = {
        "id": 7001,
        "command": "stop_container",
        "args": {"container_name": "xcl-raven-att4", "remove": False},
        "created_by": "reconciler",
    }
    with (
        mock.patch.object(worker_agent, "_protocol_negotiated", True),
        mock.patch.object(worker_agent, "_v2_attempts", {}),
        mock.patch.object(worker_agent, "_report_agent_degraded", mock.Mock()),
        mock.patch.object(worker_agent, "requests", SimpleNamespace(RequestException=Exception)),
        mock.patch.object(worker_agent.subprocess, "run", run),
    ):
        handled = worker_agent._dispatch_agent_commands([command])
    return handled, calls


def test_a_reconciler_stop_that_keeps_the_container_clears_its_restart_policy():
    handled, calls = _dispatch_keep_stop()
    assert handled == 1
    stop = calls.index(["docker", "stop", "-t", "30", "xcl-raven-att4"])
    clear = calls.index(CLEAR + ["xcl-raven-att4"])
    assert stop < clear, calls
    assert not any(argv[:2] == ["docker", "rm"] for argv in calls)


def test_a_failed_policy_change_does_not_undo_the_stop():
    handled, calls = _dispatch_keep_stop(update_rc=1)
    assert handled == 1, "the stop is still done and acknowledged"
    assert CLEAR + ["xcl-raven-att4"] in calls


def _fence_kill(state: str, label: str, kill_rc: int = 0, kill_err: str = "") -> list[list[str]]:
    calls: list[list[str]] = []

    def run(argv, **_kwargs):
        calls.append(list(argv))
        if argv[:2] == ["docker", "kill"]:
            return _done(kill_rc, kill_err)
        return _done()

    with (
        mock.patch.object(worker_agent, "_v2_container_state", return_value=(state, 137, label)),
        mock.patch.object(worker_agent.subprocess, "run", run),
    ):
        assert worker_agent._v2_fence_kill("job-f", "lease lost", "xcl-job-f-att-f", "att-f")
    return calls


def test_a_fenced_container_is_killed_and_kept_stopped():
    calls = _fence_kill("running", "att-f")
    assert calls == [["docker", "kill", "xcl-job-f-att-f"], CLEAR + ["xcl-job-f-att-f"]]


def test_a_fenced_container_that_stopped_on_its_own_is_kept_stopped():
    assert _fence_kill("exited", "att-f") == [CLEAR + ["xcl-job-f-att-f"]]
    racing = _fence_kill("running", "att-f", kill_rc=1, kill_err="container is not running")
    assert racing[-1] == CLEAR + ["xcl-job-f-att-f"]


def test_a_container_of_another_attempt_is_left_alone():
    assert _fence_kill("exited", "att-other") == []
