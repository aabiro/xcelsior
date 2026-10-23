"""Old start/stop directives cannot bypass active-mode or attempt fencing."""

import json
import time
import uuid
from types import SimpleNamespace
from unittest import mock

import pytest
from fastapi import HTTPException

import worker_agent as worker
from db import _get_pg_pool
from routes.agent import enqueue_agent_command


@pytest.fixture
def legacy_job():
    marker = uuid.uuid4().hex[:10]
    job_id, host_id, cname = f"legacy-{marker}", f"host-{marker}", f"xcl-legacy-{marker}"
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs(job_id,status,host_id,priority,submitted_at,payload, phase, desired_state, effective_priority, queued_at) VALUES(%s,'stopped',%s,0,%s,%s, 'stopped', 'stopped', 0, NULL)""",
            (job_id, host_id, time.time(), json.dumps({"container_name": cname})),
        )
    yield job_id, host_id, cname
    with pool.connection() as conn:
        conn.execute("DELETE FROM agent_commands WHERE host_id=%s", (host_id,))
        conn.execute("DELETE FROM job_attempts WHERE job_id=%s", (job_id,))
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (job_id,))


@pytest.mark.parametrize("command", ["start_container", "pause_container"])
def test_enqueue_refuses_fenced_history(legacy_job, command):
    job_id, host_id, cname = legacy_job
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "INSERT INTO job_attempts(attempt_id,job_id,attempt_number,status,host_id,"
            "fencing_token,job_generation) VALUES(%s,%s,1,'cancelled',%s,1,1)",
            (str(uuid.uuid4()), job_id, host_id),
        )
    with pytest.raises(HTTPException) as exc:
        enqueue_agent_command(host_id, command, {"job_id": job_id, "container_name": cname})
    assert exc.value.status_code == 400
    with _get_pg_pool().connection() as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM agent_commands WHERE host_id=%s", (host_id,)
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("mode", ["active", "paused", "canary"])
@pytest.mark.parametrize("command", ["start_container", "pause_container", "reset_container"])
def test_all_modes_refuse_retired_directives(legacy_job, monkeypatch, mode, command):
    job_id, host_id, cname = legacy_job
    monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", mode)
    with pytest.raises(HTTPException) as exc:
        enqueue_agent_command(host_id, command, {"job_id": job_id, "container_name": cname})
    assert exc.value.status_code == 400


@pytest.mark.parametrize(
    "field,value", [("host_id", "other-host"), ("container_name", "other-container")]
)
def test_enqueue_requires_current_host_and_container(legacy_job, field, value):
    job_id, host_id, cname = legacy_job
    if field == "host_id":
        host_id = value
    else:
        cname = value
    with pytest.raises(HTTPException) as exc:
        enqueue_agent_command(
            host_id, "start_container", {"job_id": job_id, "container_name": cname}
        )
    assert exc.value.status_code == 400


@pytest.mark.parametrize(
    "labels,negotiated,current",
    [
        ({"ai.xcelsior.attempt_id": "old-attempt"}, True, False),
        ({}, False, False),
        ({}, True, False),
        ({}, True, True),
    ],
)
def test_worker_refuses_all_unfenced_starts(monkeypatch, labels, negotiated, current):
    job_id, cname = "guard-job", "xcl-guard-job"
    monkeypatch.setattr(worker, "_protocol_negotiated", negotiated)
    monkeypatch.setattr(worker, "_v2_attempts", {job_id: {"attempt_id": "live"}} if current else {})
    monkeypatch.setattr(worker, "_report_agent_degraded", mock.Mock())
    docker = mock.Mock(return_value=SimpleNamespace(returncode=0, stdout=json.dumps(labels)))
    monkeypatch.setattr(worker.subprocess, "run", docker)
    assert (
        worker._dispatch_agent_commands(
            [
                {
                    "id": 1,
                    "command": "start_container",
                    "args": {"job_id": job_id, "container_name": cname},
                }
            ]
        )
        == 0
    )
    docker.assert_not_called()


@pytest.mark.parametrize("monitor", ["_monitor_container", "_monitor_interactive"])
def test_monitor_does_not_adopt_an_unfenced_container(monkeypatch, monitor):
    monkeypatch.setattr(worker, "_v2_attempts", {})
    docker = mock.Mock(side_effect=AssertionError("legacy container inspection"))
    monkeypatch.setattr(worker.subprocess, "run", docker)
    report = mock.Mock()
    monkeypatch.setattr(worker, "report_job_status", report)
    getattr(worker, monitor)("job-1", "xcl-job-1")
    report.assert_not_called()
    docker.assert_not_called()


@pytest.mark.parametrize("command", ["pause_container", "reset_container"])
def test_worker_has_no_historical_container_lifecycle(monkeypatch, command):
    docker = mock.Mock(side_effect=AssertionError("unfenced container IO"))
    monkeypatch.setattr(worker.subprocess, "run", docker)
    assert (
        worker._dispatch_agent_commands(
            [{"id": 1, "command": command, "args": {"job_id": "old", "container_name": "old"}}]
        )
        == 0
    )
    docker.assert_not_called()
    assert not hasattr(worker, "_historical_container_stop_allowed")
