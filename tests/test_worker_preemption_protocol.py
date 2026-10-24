"""A preemption command needs durable intent and never implies a timed KILL."""

import threading
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest import mock

import pytest
import worker_agent as agent
from tests.test_worker_agent_v2 import _stop_cmd


@pytest.mark.parametrize(
    "force,allow_force,elapsed,valid",
    [
        (False, False, False, True),
        (False, False, True, True),
        (True, False, True, False),
        (True, True, False, False),
        (True, True, True, True),
    ],
)
def test_signal_policy_retains_allocation_until_observed_exit(
    tmp_path, force, allow_force, elapsed, valid
):
    command = _stop_cmd()
    deadline = datetime.now(timezone.utc) + timedelta(seconds=-60 if elapsed else 60)
    command["args"].update(
        preemption=True,
        preemption_plan_id="plan-1",
        preserve=False,
        checkpoint_policy="none",
        force=force,
        allow_force=allow_force,
        grace_deadline=deadline.isoformat(),
    )
    auth = dict(
        job_id="job-1",
        attempt_id="att-1",
        fencing_token=7,
        lease_id="lease-1",
        container_name="xcl-job-1-att-1",
        stop=threading.Event(),
    )
    with (
        mock.patch.object(agent, "_v2_attempts", {"job-1": auth}),
        mock.patch.object(agent, "_V2_JOURNAL_PATH", str(tmp_path / "journal.json")),
        mock.patch.object(agent, "_v2_post", return_value=(200, {})),
        mock.patch.object(agent, "_v2_container_state", return_value=("running", -1, "att-1")),
        mock.patch.object(
            agent.subprocess, "run", return_value=SimpleNamespace(returncode=0)
        ) as docker,
        mock.patch.object(agent, "v2_nack_command") as nack,
        mock.patch.object(agent, "v2_report_attempt_status") as report,
    ):
        agent.handle_stop_attempt(command)
    report.assert_not_called()
    assert not auth["stop"].is_set()
    if valid:
        expected = ["docker", "kill", *([] if force else ["--signal", "TERM"]), "xcl-job-1-att-1"]
        assert docker.call_args.args[0] == expected
        assert auth["pending_command"]["terminal_status"] == "stopped"
        assert nack.call_args.args[1] == "preemption_grace_pending"
        assert (tmp_path / "journal.json").exists()
    else:
        docker.assert_not_called()
        assert "pending_command" not in auth
        assert nack.call_args.args[1] == "preemption_policy_invalid"


def test_failed_journal_never_signals_and_does_not_install_terminal_intent():
    auth = dict(
        job_id="job-1",
        attempt_id="att-1",
        fencing_token=7,
        lease_id="lease-1",
        container_name="xcl-job-1-att-1",
        stop=threading.Event(),
    )
    with (
        mock.patch.object(agent, "_v2_attempts", {"job-1": auth}),
        mock.patch.object(agent, "_v2_post", return_value=(200, {})),
        mock.patch.object(agent, "_v2_container_state", return_value=("running", -1, "att-1")),
        mock.patch.object(agent, "_v2_journal_save", return_value=False),
        mock.patch.object(agent.subprocess, "run") as docker,
        mock.patch.object(agent, "v2_nack_command") as nack,
    ):
        agent.handle_stop_attempt(_stop_cmd())
    docker.assert_not_called()
    assert "pending_command" not in auth
    assert nack.call_args.args[1] == "stop_journal_unavailable"
