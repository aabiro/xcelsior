"""An exited fenced workload survives report loss until authority resolves."""

import json
import threading
from types import SimpleNamespace
from unittest import mock

import pytest
import requests

import worker_agent as worker


@pytest.fixture
def execution(tmp_path, monkeypatch):
    auth = {
        "job_id": "job-recovery",
        "attempt_id": "attempt-recovery",
        "lease_id": "lease-1",
        "fencing_token": 7,
        "renewal_ttl_sec": 300,
        "container_name": "xcl-recovery-attempt",
        "stop": threading.Event(),
    }
    journal = tmp_path / "attempts.json"
    monkeypatch.setattr(worker, "_V2_JOURNAL_PATH", str(journal))
    monkeypatch.setattr(worker, "_v2_attempts", {auth["job_id"]: auth})
    monkeypatch.setattr(worker, "_v2_completed_commands", {})
    monkeypatch.setattr(worker, "_v2_enabled", True)
    monkeypatch.setattr(worker, "_protocol_negotiated", True)
    monkeypatch.setattr(worker, "_shutdown", threading.Event())
    monkeypatch.setattr(worker, "_active_containers", {auth["job_id"]: {}})
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)
    worker._v2_journal_save()

    def docker(argv, **kwargs):
        if ".State.Status" in " ".join(argv) and ".State.ExitCode" in " ".join(argv):
            stdout = "exited 0 attempt-recovery"
        elif ".State.Status" in " ".join(argv):
            stdout = "exited"
        elif ".State.ExitCode" in " ".join(argv):
            stdout = "0"
        else:
            stdout = ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(worker.subprocess, "run", docker)
    monkeypatch.setattr(
        worker, "_api_request", mock.Mock(return_value=SimpleNamespace(status_code=200))
    )
    cleanup = mock.Mock()
    monkeypatch.setattr(worker, "_remove_container", cleanup)
    return auth, journal, cleanup


def test_transient_terminal_failure_retries_before_removing_container(execution, monkeypatch):
    auth, journal, cleanup = execution
    posts = []

    def post(path, body):
        assert path == "/agent/v2/attempts/status"
        posts.append(body)
        assert cleanup.call_count == 0
        assert json.loads(journal.read_text())[auth["job_id"]]["attempt_id"] == auth["attempt_id"]
        return (503 if len(posts) == 1 else 200), {}

    monkeypatch.setattr(worker, "_v2_post", post)
    worker._monitor_container(auth["job_id"], auth["container_name"])
    assert len(posts) == 2
    assert posts[0] == posts[1]
    cleanup.assert_called_once_with(auth["container_name"])
    assert auth["job_id"] not in json.loads(journal.read_text())


def test_shutdown_retains_exit_and_restart_retries(execution, monkeypatch):
    auth, journal, cleanup = execution

    def unavailable(path, body):
        worker._shutdown.set()
        return 503, {}

    monkeypatch.setattr(worker, "_v2_post", unavailable)
    worker._monitor_container(auth["job_id"], auth["container_name"])
    cleanup.assert_not_called()
    assert auth["job_id"] in json.loads(journal.read_text())
    worker._v2_attempts.clear()
    worker._shutdown.clear()
    post = mock.Mock(return_value=(200, {}))
    monkeypatch.setattr(worker, "_v2_post", post)
    worker.v2_adopt_attempts()
    assert post.call_args.args[1]["status"] == "succeeded"
    cleanup.assert_called_once_with(auth["container_name"])
    assert auth["job_id"] not in json.loads(journal.read_text())


def test_lost_reply_then_fencing_never_falls_back_to_legacy(execution, monkeypatch):
    auth, _, cleanup = execution
    post = mock.Mock(side_effect=[(0, {}), (409, {})])
    monkeypatch.setattr(worker, "_v2_post", post)
    worker._monitor_container(auth["job_id"], auth["container_name"])
    assert post.call_count == 2
    worker._api_request.assert_not_called()
    cleanup.assert_called_once_with(auth["container_name"])


def test_wrong_attempt_label_is_never_reported_or_removed(execution, monkeypatch):
    auth, _, cleanup = execution
    monkeypatch.setattr(worker, "_v2_container_state", lambda *args: ("exited", 0, "new-attempt"))
    post = mock.Mock(return_value=(200, {}))
    monkeypatch.setattr(worker, "_v2_post", post)
    worker._monitor_container(auth["job_id"], auth["container_name"])
    post.assert_not_called()
    cleanup.assert_not_called()


def test_confirmed_terminal_report_does_not_retry_legacy_patch(execution, monkeypatch):
    auth, _, cleanup = execution
    post = mock.Mock(return_value=(200, {}))
    monkeypatch.setattr(worker, "_v2_post", post)
    worker._api_request.side_effect = requests.ConnectionError("compatibility path unavailable")
    assert worker.report_job_status(auth["job_id"], "completed") is True
    assert post.call_count == 1
    assert auth["job_id"] not in worker._v2_attempts


def test_restart_failure_arms_retry_without_another_restart(execution, monkeypatch):
    auth, journal, cleanup = execution
    worker._v2_attempts.clear()
    monkeypatch.setattr(worker, "_v2_post", mock.Mock(return_value=(503, {})))
    with mock.patch.object(worker.threading, "Thread") as thread:
        worker.v2_adopt_attempts()
    cleanup.assert_not_called()
    assert auth["job_id"] in json.loads(journal.read_text())
    assert thread.call_count == 1
    invocation = thread.call_args
    monkeypatch.setattr(worker, "_v2_post", mock.Mock(return_value=(200, {})))
    invocation.kwargs["target"](*invocation.kwargs.get("args", ()))
    cleanup.assert_called_once_with(auth["container_name"])


@pytest.mark.parametrize("response", [200, 409, 503])
def test_old_terminal_response_preserves_replacement_authority(execution, monkeypatch, response):
    auth, journal, cleanup = execution
    replacement = dict(auth, attempt_id="replacement", fencing_token=8, stop=threading.Event())

    def post(path, body):
        assert body["attempt_id"] == auth["attempt_id"]
        worker._v2_attempts[auth["job_id"]] = replacement
        worker._v2_journal_save()
        return response, {}

    monkeypatch.setattr(worker, "_v2_post", post)
    worker._monitor_container(auth["job_id"], auth["container_name"])
    assert worker._v2_attempts[auth["job_id"]] is replacement
    assert not replacement["stop"].is_set()
    assert json.loads(journal.read_text())[auth["job_id"]]["attempt_id"] == "replacement"
    cleanup.assert_called_once_with(auth["container_name"])
    if response != 200:
        worker._api_request.assert_not_called()


def test_inflight_renewal_rejection_does_not_erase_pending_terminal(execution, monkeypatch):
    auth, journal, cleanup = execution
    stop = mock.Mock()
    stop.wait.return_value = False
    stop.is_set.return_value = False
    auth["stop"] = stop

    def renewal(path, body):
        # The container exits and reporting takes over while renewal is in flight.
        stop.is_set.return_value = True
        return 409, {}

    monkeypatch.setattr(worker, "_v2_post", renewal)
    worker._v2_renewal_loop(auth)
    assert worker._v2_attempts[auth["job_id"]] is auth
    assert auth["job_id"] in json.loads(journal.read_text())
    cleanup.assert_not_called()


def test_preserved_stop_retries_original_intent_and_ack(execution, monkeypatch):
    auth, journal, cleanup = execution
    result = {"stopped": True, "preserved": True}
    auth["pending_command"] = {
        "command_id": "stop-recovery",
        "terminal_status": "stopped",
        "result": result,
    }
    worker._v2_journal_save()
    post = mock.Mock(side_effect=[(503, {}), (200, {})])
    ack = mock.Mock(return_value=True)
    monkeypatch.setattr(worker, "_v2_post", post)
    monkeypatch.setattr(worker, "v2_ack_command", ack)
    worker._monitor_container(auth["job_id"], auth["container_name"])
    assert [call.args[1]["status"] for call in post.call_args_list] == ["stopped", "stopped"]
    ack.assert_called_once_with("stop-recovery", result)
    assert worker._v2_completed_command_result("stop-recovery") == result
    assert auth["job_id"] not in json.loads(journal.read_text())
    cleanup.assert_not_called()
    worker._api_request.assert_not_called()


def test_running_restart_rearms_exit_monitor(execution, monkeypatch):
    auth, _, _ = execution
    worker._v2_attempts.clear()
    monkeypatch.setattr(
        worker, "_v2_container_state", lambda *args: ("running", 0, auth["attempt_id"])
    )
    monkeypatch.setattr(worker, "_v2_post", mock.Mock(return_value=(200, {})))
    with mock.patch.object(worker.threading, "Thread") as thread:
        assert worker.v2_adopt_attempts() == 1
    assert {call.kwargs["target"] for call in thread.call_args_list} == {
        worker._v2_renewal_loop,
        worker._v2_monitor_container,
    }
    assert worker._active_containers[auth["job_id"]] == auth["container_name"]
    assert auth["job_id"] in worker._adopted_containers
    worker._v2_forget_attempt(auth["job_id"], worker._v2_attempts[auth["job_id"]])
    assert auth["job_id"] not in worker._active_containers


@pytest.mark.parametrize(
    "inspection",
    [
        SimpleNamespace(returncode=1, stdout="", stderr="Cannot connect to the Docker daemon"),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout="exited invalid attempt-recovery", stderr=""),
        worker.subprocess.TimeoutExpired("docker inspect", 10),
    ],
)
def test_inspect_uncertainty_cannot_fabricate_terminal_failure(execution, monkeypatch, inspection):
    auth, journal, cleanup = execution
    worker._v2_attempts.clear()
    inspected = mock.Mock()
    if isinstance(inspection, Exception):
        inspected.side_effect = inspection
    else:
        inspected.return_value = inspection
    monkeypatch.setattr(worker.subprocess, "run", inspected)
    post = mock.Mock(return_value=(200, {}))
    monkeypatch.setattr(worker, "_v2_post", post)
    with mock.patch.object(worker.threading, "Thread") as thread:
        assert worker.v2_adopt_attempts() == 0
    post.assert_not_called()
    cleanup.assert_not_called()
    assert auth["job_id"] in json.loads(journal.read_text())
    assert thread.call_args.kwargs["target"] is worker._v2_pending_adoption_loop


@pytest.mark.parametrize(
    "exit_code,status",
    [(0, "stopped"), (130, "stopped"), (137, "stopped"), (143, "stopped"), (7, "failed")],
)
def test_interactive_report_loss_preserves_exit_meaning(execution, monkeypatch, exit_code, status):
    auth, journal, cleanup = execution
    monkeypatch.setattr(
        worker, "_v2_container_state", lambda *args: ("exited", exit_code, auth["attempt_id"])
    )
    post = mock.Mock(side_effect=[(503, {}), (200, {})])
    monkeypatch.setattr(worker, "_v2_post", post)
    worker._monitor_interactive(auth["job_id"], auth["container_name"])
    assert [call.args[1]["status"] for call in post.call_args_list] == [status, status]
    assert auth["job_id"] not in json.loads(journal.read_text())
    cleanup.assert_called_once_with(auth["container_name"])


def test_interactive_restart_uses_journaled_exit_meaning(execution, monkeypatch):
    auth, journal, cleanup = execution
    auth["interactive"] = True
    worker._v2_journal_save()
    worker._v2_attempts.clear()
    post = mock.Mock(return_value=(200, {}))
    monkeypatch.setattr(worker, "_v2_post", post)
    worker.v2_adopt_attempts()
    assert post.call_args.args[1]["status"] == "stopped"
    assert auth["job_id"] not in json.loads(journal.read_text())
    cleanup.assert_called_once_with(auth["container_name"])


@pytest.mark.parametrize("state", ["created", "paused", "restarting"])
def test_incomplete_execution_is_not_reported_as_terminal(execution, monkeypatch, state):
    auth, _, cleanup = execution

    def inspection(argv, **kwargs):
        worker._shutdown.set()  # One monitor cycle suffices; workload has not exited.
        return SimpleNamespace(returncode=0, stdout=f"{state} 0 {auth['attempt_id']}", stderr="")

    monkeypatch.setattr(worker.subprocess, "run", inspection)
    post = mock.Mock(return_value=(200, {}))
    monkeypatch.setattr(worker, "_v2_post", post)
    worker._monitor_container(auth["job_id"], auth["container_name"])
    post.assert_not_called()
    cleanup.assert_not_called()
