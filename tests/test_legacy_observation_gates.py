"""Retirement evidence counts real compatibility effects and refuses false zeroes."""

import json
import os
import time
import uuid

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from api import app
from db import _get_pg_pool


client = TestClient(app, raise_server_exceptions=False)


def counts(path):
    return {
        outcome: REGISTRY.get_sample_value(
            "xcelsior_legacy_execution_total", {"path": path, "outcome": outcome}
        )
        or 0
        for outcome in ("requested", "used", "blocked", "failed")
    }


@pytest.fixture
def job(monkeypatch):
    monkeypatch.setenv("XCELSIOR_AGENT_HOST_TOKENS", "off")
    marker = uuid.uuid4().hex
    job_id, host_id = f"observed-{marker}", f"host-{marker}"
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "INSERT INTO jobs(job_id,status,host_id,priority,submitted_at,payload) "
            "VALUES(%s,'assigned',%s,0,%s,%s)",
            (job_id, host_id, time.time(), json.dumps({"name": job_id})),
        )
    yield job_id, host_id
    with _get_pg_pool().connection() as conn:
        conn.execute("DELETE FROM agent_commands WHERE host_id=%s", (host_id,))
        conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (job_id,))
        conn.execute("DELETE FROM job_attempts WHERE job_id=%s", (job_id,))
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (job_id,))


def patch(job, **extra):
    job_id, host_id = job
    return client.patch(
        f"/instance/{job_id}",
        headers={"Authorization": f"Bearer {os.environ['XCELSIOR_API_TOKEN']}"},
        json={"status": "starting", "host_id": host_id, **extra},
    )


def test_status_patch_counts_committed_status(job):
    before = counts("status_patch")
    response = patch(job)
    assert response.status_code == 200, response.text
    assert counts("status_patch") == {
        **before,
        "requested": before["requested"] + 1,
        "used": before["used"] + 1,
    }
    with _get_pg_pool().connection() as conn:
        assert (
            conn.execute("SELECT status FROM jobs WHERE job_id=%s", (job[0],)).fetchone()[0]
            == "starting"
        )


@pytest.mark.parametrize("attributed", [False, True])
def test_patch_counts_fence_refusal_and_attributed_compatibility_use(job, attributed):
    job_id, host_id = job
    attempt_id = str(uuid.uuid4())
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "INSERT INTO job_attempts(attempt_id,job_id,attempt_number,status,host_id,"
            "fencing_token,job_generation) VALUES(%s,%s,1,'running',%s,1,1)",
            (attempt_id, job_id, host_id),
        )
        conn.execute("UPDATE jobs SET active_attempt_id=%s WHERE job_id=%s", (attempt_id, job_id))
    before = counts("status_patch")
    authority = {"attempt_id": attempt_id, "fencing_token": 1} if attributed else {}
    assert patch(job, **authority).status_code == (200 if attributed else 409)
    outcome = "used" if attributed else "blocked"
    assert counts("status_patch") == {
        **before,
        "requested": before["requested"] + 1,
        outcome: before[outcome] + 1,
    }


def test_error_after_committed_patch_is_not_reported_as_blocked(job):
    before = counts("status_patch")
    # This field is validated after the status transaction commits. Observability
    # must not describe its HTTP 400 as a side-effect-free refusal.
    assert patch(job, ssh_port=22).status_code == 400
    assert counts("status_patch") == {
        **before,
        "requested": before["requested"] + 1,
        "used": before["used"] + 1,
        "failed": before["failed"] + 1,
    }
    with _get_pg_pool().connection() as conn:
        assert (
            conn.execute("SELECT status FROM jobs WHERE job_id=%s", (job[0],)).fetchone()[0]
            == "starting"
        )


def test_command_drain_counts_committed_rows_and_empty_polls(job):
    job_id, host_id = job
    with _get_pg_pool().connection() as conn:
        for _ in range(2):
            conn.execute(
                "INSERT INTO agent_commands(host_id,command,args,created_at,expires_at,status) "
                "VALUES(%s,'reverify','{}',%s,%s,'pending')",
                (host_id, time.time(), time.time() + 100),
            )
    before = counts("command_drain")
    response = client.get(f"/agent/commands/{host_id}")
    assert response.status_code == 200, response.text
    assert len(response.json()["commands"]) == 2
    assert counts("command_drain") == {
        **before,
        "requested": before["requested"] + 1,
        "used": before["used"] + 2,
    }
    response = client.get(f"/agent/commands/{host_id}")
    assert response.json()["commands"] == []
    assert counts("command_drain") == {
        **before,
        "requested": before["requested"] + 2,
        "used": before["used"] + 2,
    }


def test_drain_database_failure_cannot_look_like_an_empty_poll(monkeypatch):
    import routes.agent as agent

    def unavailable():
        raise RuntimeError("synthetic pool outage")

    monkeypatch.setattr(agent, "_get_pg_pool", unavailable)
    before = counts("command_drain")
    assert client.get("/agent/commands/observation-outage").status_code == 500
    assert counts("command_drain") == {
        **before,
        "requested": before["requested"] + 1,
        "failed": before["failed"] + 1,
    }


def test_error_with_uncertain_commit_is_failed_not_claimed_side_effect_free(job, monkeypatch):
    import routes.instances as instances

    original = instances.update_job_status

    def reply_lost(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("synthetic error after committed mutation")

    monkeypatch.setattr(instances, "update_job_status", reply_lost)
    before = counts("status_patch")
    assert patch(job).status_code == 500
    assert counts("status_patch") == {
        **before,
        "requested": before["requested"] + 1,
        "failed": before["failed"] + 1,
    }
    with _get_pg_pool().connection() as conn:
        assert (
            conn.execute("SELECT status FROM jobs WHERE job_id=%s", (job[0],)).fetchone()[0]
            == "starting"
        )


@pytest.mark.parametrize("path", ["status_patch", "command_drain"])
def test_auth_rejection_does_not_claim_an_authenticated_legacy_caller(job, monkeypatch, path):
    import routes.agent as agent
    import routes.instances as instances

    def denied(*args, **kwargs):
        raise HTTPException(401, "synthetic auth rejection")

    before = counts(path)
    if path == "status_patch":
        monkeypatch.setattr(instances, "_require_worker_status_update", denied)
        response = patch(job)
    else:
        monkeypatch.setattr(agent, "_require_agent_auth", denied)
        response = client.get(f"/agent/commands/{job[1]}")
    assert response.status_code == 401
    assert counts(path) == before


def test_retirement_policy_does_not_assume_active_mode_retires_status_or_commands(monkeypatch):
    from prometheus_client import generate_latest

    monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", "active")
    monkeypatch.delenv("XCELSIOR_RETIRED_LEGACY_PATHS", raising=False)
    generate_latest(REGISTRY)
    assert REGISTRY.get_sample_value("xcelsior_legacy_path_retired", {"path": "work_poll"}) == 1
    for path in ("status_patch", "command_drain"):
        assert REGISTRY.get_sample_value("xcelsior_legacy_path_retired", {"path": path}) == 0
    monkeypatch.setenv("XCELSIOR_RETIRED_LEGACY_PATHS", "status_patch,command_drain")
    generate_latest(REGISTRY)
    assert REGISTRY.get_sample_value("xcelsior_legacy_path_retired", {"path": "status_patch"}) == 1
    assert REGISTRY.get_sample_value("xcelsior_legacy_path_retired", {"path": "command_drain"}) == 1


def test_unknown_retirement_path_is_rejected(monkeypatch):
    from control_plane.legacy_execution import retired_paths

    monkeypatch.setenv("XCELSIOR_RETIRED_LEGACY_PATHS", "status_typo")
    with pytest.raises(ValueError, match="XCELSIOR_RETIRED_LEGACY_PATHS"):
        retired_paths()


def test_api_prometheus_surface_exports_both_legacy_observation_and_policy():
    response = client.get(
        "/metrics/prometheus",
        headers={"Authorization": f"Bearer {os.environ['XCELSIOR_API_TOKEN']}"},
    )
    assert response.status_code == 200
    assert (
        'xcelsior_legacy_execution_total{outcome="requested",path="status_patch"}' in response.text
    )
    assert (
        'xcelsior_legacy_execution_total{outcome="requested",path="command_drain"}' in response.text
    )
    assert 'xcelsior_legacy_path_retired{path="status_patch"}' in response.text
