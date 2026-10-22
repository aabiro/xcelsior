"""The active worker uses one fenced status and durable host-command protocol."""

from types import SimpleNamespace
from unittest import mock
import threading
import pytest
import worker_agent as worker


@pytest.fixture
def cutover(monkeypatch, tmp_path):
    auth = {
        "job_id": "j",
        "attempt_id": "a",
        "fencing_token": 7,
        "lease_id": "l",
        "stop": threading.Event(),
    }
    monkeypatch.setattr(worker, "_v2_attempts", {"j": auth})
    monkeypatch.setattr(worker, "_v2_completed_commands", {})
    monkeypatch.setattr(worker, "_V2_JOURNAL_PATH", str(tmp_path / "journal.json"))
    monkeypatch.setattr(worker, "_v2_enabled", True)
    monkeypatch.setattr(worker, "_protocol_negotiated", True)
    monkeypatch.setattr(worker, "_legacy_execution_allowed", False)
    return auth


def test_status_and_connection_metadata_never_use_legacy_patch(cutover, monkeypatch):
    post = mock.Mock(return_value=(200, {}))
    transport = mock.Mock(side_effect=AssertionError("legacy HTTP path used"))
    monkeypatch.setattr(worker, "_v2_post", post)
    monkeypatch.setattr(worker, "_api_request", transport)
    assert worker.report_job_status(
        "j", "running", container_name="xcl-j-a", ssh_port=12345, interactive=True
    )
    assert post.call_args.args[0] == "/agent/v2/attempts/status"
    assert post.call_args.args[1]["runtime_metadata"] == {
        "container_name": "xcl-j-a",
        "public_ssh_port": 12345,
        "interactive": True,
    }
    assert worker.report_job_status("j", "completed")
    assert "j" not in worker._v2_attempts
    transport.assert_not_called()


def test_lost_host_command_ack_replays_durable_result_after_restart(cutover, monkeypatch):
    cmd = {"command_id": "host-1", "command": "reverify", "args": {}, "attempt_id": None}
    monkeypatch.setattr(worker, "v2_claim_commands", lambda: [cmd])
    execute = mock.Mock(return_value=1)
    monkeypatch.setattr(worker, "_dispatch_agent_commands", execute)
    ack = mock.Mock(side_effect=[False, True])
    monkeypatch.setattr(worker, "v2_ack_command", ack)
    assert worker.drain_v2_commands() == 0
    monkeypatch.setattr(worker, "_v2_completed_commands", {})
    worker._v2_journal_load()
    assert worker.drain_v2_commands() == 1
    execute.assert_called_once()


def test_host_command_failure_is_nacked_and_retained(cutover, monkeypatch):
    monkeypatch.setattr(
        worker,
        "v2_claim_commands",
        lambda: [{"command_id": "host-2", "command": "mount_volume", "args": {}}],
    )
    monkeypatch.setattr(worker, "_dispatch_agent_commands", lambda _: 0)
    nack = mock.Mock()
    monkeypatch.setattr(worker, "v2_nack_command", nack)
    assert worker.drain_v2_commands() == 0
    nack.assert_called_once_with("host-2", "host_command_failed", retryable=True)
    assert worker._v2_completed_command_result("host-2") is None


def test_v2_worker_never_destructively_drains_host_commands(cutover, monkeypatch):
    transport = mock.Mock(side_effect=AssertionError("legacy drain called"))
    monkeypatch.setattr(worker, "_api_request", transport)
    assert worker.drain_agent_commands() == 0
    transport.assert_not_called()


def test_active_negotiation_requires_complete_server_contract(cutover, monkeypatch):
    response = SimpleNamespace(
        status_code=200,
        json=lambda: {"v2": True, "protocol": 2, "legacy_execution_allowed": False, "features": []},
    )
    monkeypatch.setattr(worker, "_api_request", lambda *a, **k: response)
    assert worker.negotiate_protocol_v2() is False
    assert not worker._protocol_negotiated


def test_host_command_never_acks_an_unpersisted_result(cutover, monkeypatch):
    monkeypatch.setattr(
        worker,
        "v2_claim_commands",
        lambda: [{"command_id": "host-3", "command": "reverify", "args": {}}],
    )
    monkeypatch.setattr(worker, "_dispatch_agent_commands", lambda _: 1)
    monkeypatch.setattr(worker, "_v2_journal_save", lambda: False)
    ack, nack = mock.Mock(), mock.Mock()
    monkeypatch.setattr(worker, "v2_ack_command", ack)
    monkeypatch.setattr(worker, "v2_nack_command", nack)
    assert worker.drain_v2_commands() == 0
    ack.assert_not_called()
    nack.assert_called_once_with("host-3", "journal_unavailable", retryable=True)
    assert worker._v2_completed_command_result("host-3") is None
