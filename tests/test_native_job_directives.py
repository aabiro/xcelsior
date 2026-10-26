"""Job-scoped directives never gain authority from historical JSONB names."""

from unittest.mock import Mock
import threading

import pytest
from fastapi import HTTPException

from db import _get_pg_pool
from routes.agent import enqueue_agent_command
from tests.fixtures.native_runtime import running_instance


@pytest.mark.parametrize(
    "command",
    [
        "reinject_shell",
        "snapshot_container",
        "mount_volume",
        "unmount_volume",
        "fingerprint_environment",
    ],
)
def test_each_job_directive_persists_the_exact_current_authority(command):
    with running_instance() as instance:
        command_id = enqueue_agent_command(
            instance["host_id"],
            command,
            {"job_id": instance["job_id"]},
            created_by="native-directive-proof",
        )
        with _get_pg_pool().connection() as conn:
            row = conn.execute(
                "SELECT c.job_id,c.attempt_id,c.fencing_token,c.spec_hash,c.args,a.spec_hash FROM agent_commands c JOIN job_attempts a ON a.attempt_id=c.attempt_id WHERE c.id=%s",
                (command_id,),
            ).fetchone()
        assert row[0] == instance["job_id"]
        assert str(row[1]) == instance["authority"]["attempt_id"]
        assert row[2] == instance["authority"]["fencing_token"]
        assert row[3] == row[5]
        assert row[4]["container_name"] == f"xcl-{row[0]}-{str(row[1])[:8]}"


@pytest.mark.parametrize(
    "invalid", ["expired", "wrong_host", "legacy_container", "stopping", "stop_intent"]
)
def test_invalid_authority_cannot_enqueue_a_container_directive(invalid):
    with running_instance() as instance:
        args = {"job_id": instance["job_id"]}
        host = instance["host_id"]
        with _get_pg_pool().connection() as conn:
            if invalid == "expired":
                conn.execute(
                    "UPDATE placement_leases SET expires_at=clock_timestamp()-interval '1 second' WHERE attempt_id=%s",
                    (instance["authority"]["attempt_id"],),
                )
            if invalid == "wrong_host":
                host = "unrelated-host"
            if invalid == "legacy_container":
                args["container_name"] = "xcl-" + instance["job_id"]
            if invalid == "stopping":
                conn.execute(
                    "UPDATE jobs SET desired_state='stopped' WHERE job_id=%s",
                    (instance["job_id"],),
                )
            if invalid == "stop_intent":
                conn.execute(
                    "UPDATE jobs SET lifecycle_intent='stop' WHERE job_id=%s",
                    (instance["job_id"],),
                )
        with pytest.raises(HTTPException) as error:
            enqueue_agent_command(host, "reinject_shell", args)
        assert error.value.status_code == 409
        with _get_pg_pool().connection() as conn:
            assert conn.execute(
                "SELECT count(*) FROM agent_commands WHERE job_id=%s AND command='reinject_shell'",
                (instance["job_id"],),
            ).fetchone() == (0,)


@pytest.mark.parametrize(
    "invalid", [None, "missing_attempt", "wrong_fence", "legacy_container", "expired"]
)
def test_worker_checks_native_command_authority_before_dispatch(monkeypatch, invalid):
    import worker_agent as worker

    auth = dict(
        job_id="job",
        attempt_id="attempt-1234",
        fencing_token=7,
        spec_hash="hash",
        lease_id="lease",
        container_name="xcl-job-attempt-",
        stop=threading.Event(),
    )
    command = dict(
        command_id="directive",
        job_id="job",
        attempt_id=auth["attempt_id"],
        fencing_token=7,
        spec_hash="hash",
        command="reinject_shell",
        args=dict(job_id="job", container_name=auth["container_name"]),
    )
    if invalid == "missing_attempt":
        command.pop("attempt_id")
    elif invalid == "wrong_fence":
        command["fencing_token"] = 6
    elif invalid == "legacy_container":
        command["args"]["container_name"] = "xcl-job"
    dispatch, nack = Mock(return_value=1), Mock()
    monkeypatch.setattr(worker, "_v2_attempts", {"job": auth})
    monkeypatch.setattr(worker, "_v2_completed_command_result", lambda _: None)
    monkeypatch.setattr(worker, "_v2_mark_command_complete", lambda *a, **k: None)
    monkeypatch.setattr(
        worker, "_v2_post", lambda *a, **k: (409 if invalid == "expired" else 200, {})
    )
    monkeypatch.setattr(
        worker, "_v2_container_state", lambda *a: ("running", 0, auth["attempt_id"])
    )
    monkeypatch.setattr(worker, "_dispatch_agent_commands", dispatch)
    monkeypatch.setattr(worker, "v2_ack_command", lambda *a: True)
    monkeypatch.setattr(worker, "v2_nack_command", nack)
    assert worker._dispatch_v2_command(command) == (1 if invalid is None else 0)
    assert dispatch.call_count == (1 if invalid is None else 0)
    assert nack.call_count == (0 if invalid is None else 1)


def test_worker_runs_a_promotion_as_a_host_directive(monkeypatch):
    """A promotion names no attempt; treating it as a job directive NACKed every one."""
    import worker_agent as worker

    command = dict(command_id="promote", command="promote_artifacts", args={"promotion_id": "p-1"})
    dispatch, nack = Mock(return_value=1), Mock()
    monkeypatch.setattr(worker, "_v2_completed_command_result", lambda _: None)
    monkeypatch.setattr(worker, "_v2_mark_command_complete", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_dispatch_agent_commands", dispatch)
    monkeypatch.setattr(worker, "v2_ack_command", lambda *a: True)
    monkeypatch.setattr(worker, "v2_nack_command", nack)

    assert worker._dispatch_v2_command(command) == 1
    assert nack.call_count == 0
    assert dispatch.call_args[0][0][0]["args"] == {"promotion_id": "p-1"}
