"""Retired transports count authenticated refusals and never mutate state."""

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
            """INSERT INTO jobs(job_id,status,host_id,priority,submitted_at,payload, phase, desired_state, effective_priority, queued_at) VALUES(%s,'assigned',%s,0,%s,%s, 'scheduled', 'running', 0, NULL)""",
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


def assert_refused(path, before):
    assert counts(path) == {
        **before,
        "requested": before["requested"] + 1,
        "blocked": before["blocked"] + 1,
    }


def test_status_patch_counts_refusal_without_committed_status(job):
    before = counts("status_patch")
    response = patch(job)
    assert response.status_code == 410, response.text
    assert_refused("status_patch", before)
    with _get_pg_pool().connection() as conn:
        assert (
            conn.execute("SELECT status FROM jobs WHERE job_id=%s", (job[0],)).fetchone()[0]
            == "assigned"
        )


@pytest.mark.parametrize("attributed", [False, True])
def test_status_patch_is_retired_even_with_current_authority(job, attributed):
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
    assert patch(job, **authority).status_code == 410
    assert_refused("status_patch", before)


def test_patch_cannot_apply_runtime_fields_after_retirement(job):
    before = counts("status_patch")
    assert patch(job, ssh_port=22).status_code == 410
    assert_refused("status_patch", before)
    with _get_pg_pool().connection() as conn:
        assert conn.execute(
            "SELECT status, payload->>'ssh_port' FROM jobs WHERE job_id=%s", (job[0],)
        ).fetchone() == ("assigned", None)


def test_command_drain_counts_refusals_and_preserves_pending_rows(job):
    job_id, host_id = job
    with _get_pg_pool().connection() as conn:
        for _ in range(2):
            conn.execute(
                "INSERT INTO agent_commands(host_id,command,args,created_at,expires_at,status) "
                "VALUES(%s,'reverify','{}',%s,%s,'pending')",
                (host_id, time.time(), time.time() + 100),
            )
    for _ in range(2):
        before = counts("command_drain")
        response = client.get(f"/agent/commands/{host_id}")
        assert response.status_code == 410, response.text
        assert_refused("command_drain", before)
    with _get_pg_pool().connection() as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM agent_commands WHERE host_id=%s AND status='pending'",
                (host_id,),
            ).fetchone()[0]
            == 2
        )


def test_retired_drain_does_not_open_the_command_database(monkeypatch):
    import routes.agent as agent

    def forbidden():
        raise AssertionError("a retired transport must not read or drain commands")

    monkeypatch.setattr(agent, "_get_pg_pool", forbidden)
    before = counts("command_drain")
    assert client.get("/agent/commands/observation-outage").status_code == 410
    assert_refused("command_drain", before)


def test_retired_patch_never_enters_the_raw_lifecycle_writer(job, monkeypatch):
    import routes.instances as instances

    def forbidden(*args, **kwargs):
        raise AssertionError("a retired transport must not mutate jobs")

    monkeypatch.setattr(instances, "update_job_status", forbidden)
    before = counts("status_patch")
    assert patch(job).status_code == 410
    assert_refused("status_patch", before)


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


@pytest.mark.parametrize("mode", ["active", "canary", "paused", "shadow"])
def test_every_execution_transport_is_permanently_retired(monkeypatch, mode):
    from prometheus_client import generate_latest
    from control_plane.legacy_execution import LEGACY_PATHS, retired_paths

    monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", mode)
    monkeypatch.delenv("XCELSIOR_RETIRED_LEGACY_PATHS", raising=False)
    generate_latest(REGISTRY)
    assert retired_paths() == set(LEGACY_PATHS)
    for path in LEGACY_PATHS:
        assert REGISTRY.get_sample_value("xcelsior_legacy_path_retired", {"path": path}) == 1


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
