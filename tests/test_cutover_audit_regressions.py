"""Crash, concurrent-save and failed-execution cases from the cutover audit."""

import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from api import app
from control_plane.commands import redeliver_expired_claims
from db import _get_pg_pool
import worker_agent as worker


@pytest.fixture
def journal(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "_v2_attempts", {})
    monkeypatch.setattr(worker, "_v2_completed_commands", {})
    monkeypatch.setattr(worker, "_V2_JOURNAL_PATH", str(tmp_path / "journal.json"))
    monkeypatch.setattr(worker, "_v2_enabled", True)
    monkeypatch.setattr(worker, "_protocol_negotiated", True)
    monkeypatch.setattr(worker, "_report_agent_degraded", mock.Mock())
    return tmp_path / "journal.json"


def test_ack_waits_for_file_and_directory_fsync(journal, monkeypatch):
    synced = []
    real_sync = os.fsync

    def fsync(fd):
        synced.append(fd)
        return real_sync(fd)

    monkeypatch.setattr(worker.os, "fsync", fsync)
    worker._v2_mark_command_complete("durable", {"done": True}, strict=True)
    assert len(synced) >= 2, "rename without file/directory fsync is not a durable ACK"
    assert json.loads(journal.read_text())[worker._V2_COMPLETED_COMMANDS_KEY]["durable"][
        "result"
    ] == {"done": True}


def test_disk_sync_failure_cannot_leave_a_replayable_success(journal, monkeypatch):
    monkeypatch.setattr(worker.os, "fsync", mock.Mock(side_effect=OSError("disk unavailable")))
    with pytest.raises(OSError):
        worker._v2_mark_command_complete("unsynced", {"done": True}, strict=True)
    assert worker._v2_completed_command_result("unsynced") is None


def test_concurrent_journal_writes_do_not_collide_or_expose_uncommitted_results(
    journal, monkeypatch
):
    first_writing = threading.Event()
    release_first = threading.Event()
    overlapping_writer = threading.Event()
    real_dump = worker.json.dump

    def dump(snapshot, output):
        if threading.current_thread().name.endswith("_0"):
            first_writing.set()
            assert release_first.wait(5)
        else:
            overlapping_writer.set()
        return real_dump(snapshot, output)

    monkeypatch.setattr(worker.json, "dump", dump)
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="journal") as pool:
        first = pool.submit(worker._v2_mark_command_complete, "first", {"done": True}, strict=True)
        assert first_writing.wait(2)
        second = pool.submit(
            worker._v2_mark_command_complete, "second", {"done": True}, strict=True
        )
        try:
            assert not overlapping_writer.wait(0.15), "concurrent saves share a temporary journal"
        finally:
            release_first.set()
        first.result(timeout=5)
        second.result(timeout=5)
    records = json.loads(journal.read_text())[worker._V2_COMPLETED_COMMANDS_KEY]
    assert set(records) == {"first", "second"}


@pytest.mark.parametrize("failure_stage", ["inspect", "stop", "remove"])
def test_failed_docker_stop_is_nacked_without_discarding_local_ownership(
    journal, monkeypatch, failure_stage
):
    cmd = {
        "command_id": "stop",
        "command": "stop_container",
        "args": {"job_id": "old-job", "container_name": "xcl-old-job"},
    }
    monkeypatch.setattr(worker, "v2_claim_commands", lambda: [cmd])
    monkeypatch.setattr(worker, "_active_containers", {"old-job": "xcl-old-job"})
    calls = []

    def docker(argv, **kwargs):
        stage = {"inspect": "inspect", "stop": "stop", "rm": "remove"}[argv[1]]
        calls.append(stage)
        return SimpleNamespace(
            returncode=int(stage == failure_stage),
            stdout="container-id" if stage == "inspect" else "",
            stderr="permission denied" if stage == failure_stage else "",
        )

    monkeypatch.setattr(worker.subprocess, "run", docker)
    ack, nack = mock.Mock(), mock.Mock()
    monkeypatch.setattr(worker, "v2_ack_command", ack)
    monkeypatch.setattr(worker, "v2_nack_command", nack)
    assert worker.drain_v2_commands() == 0
    ack.assert_not_called()
    nack.assert_called_once_with("stop", "host_command_failed", retryable=True)
    assert worker._active_containers == {"old-job": "xcl-old-job"}
    assert worker._v2_completed_command_result("stop") is None
    if failure_stage == "stop":
        assert "remove" not in calls


@pytest.fixture
def command_row(monkeypatch):
    monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", "active")
    marker = uuid.uuid4().hex
    host = "claim-audit-" + marker
    pool = _get_pg_pool()
    with pool.connection() as conn:
        row = conn.execute(
            "INSERT INTO agent_commands(host_id,command,args,created_at,expires_at,status) VALUES(%s,'health_probe','{}',%s,%s,'pending') RETURNING command_id",
            (host, time.time(), time.time() + 300),
        ).fetchone()
    yield host, str(row[0]), pool
    with pool.connection() as conn:
        conn.execute("DELETE FROM agent_commands WHERE host_id=%s", (host,))


@pytest.mark.parametrize("mutation", ["ack", "nack"])
@pytest.mark.parametrize("new_session", ["new-session", "old-session"])
def test_previous_worker_session_cannot_complete_a_redelivered_command(
    command_row, mutation, new_session
):
    host, command_id, pool = command_row
    client = TestClient(app)
    headers = {"Authorization": "Bearer " + os.environ["XCELSIOR_API_TOKEN"]}

    def claim(session):
        response = client.post(
            "/agent/v2/commands/claim",
            headers=headers,
            json={"host_id": host, "worker_session_id": session, "host_commands": True},
        )
        assert response.status_code == 200, response.text
        return response.json()["commands"][0]

    first = claim("old-session")
    with pool.connection() as conn:
        conn.execute(
            "UPDATE agent_commands SET claim_expires_at=clock_timestamp()-interval '1 second' WHERE command_id=%s",
            (command_id,),
        )
        redeliver_expired_claims(conn)
    current = claim(new_session)
    response = client.post(
        f"/agent/v2/commands/{command_id}/{mutation}",
        headers=headers,
        json={
            "host_id": host,
            "worker_session_id": "old-session",
            "claim_attempt": first["attempt_count"],
            "result": {"stale": True},
            "error_code": "old_failure",
        },
    )
    assert response.status_code == 409, response.text
    with pool.connection() as conn:
        assert conn.execute(
            "SELECT status,claim_session,attempt_count FROM agent_commands WHERE command_id=%s",
            (command_id,),
        ).fetchone() == ("claimed", new_session, current["attempt_count"])
    current_body = {
        "host_id": host,
        "worker_session_id": new_session,
        "claim_attempt": current["attempt_count"],
        "result": {"current": True},
        "error_code": "current_failure",
    }
    accepted = client.post(
        f"/agent/v2/commands/{command_id}/{mutation}", headers=headers, json=current_body
    )
    assert accepted.status_code == 200, accepted.text
    if mutation == "ack":
        duplicate = client.post(
            f"/agent/v2/commands/{command_id}/ack", headers=headers, json=current_body
        )
        assert duplicate.json()["duplicate"] is True
        assert duplicate.json()["result"] == {"current": True}


def test_overlapping_handlers_retain_their_own_claim_identity(journal, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    sent = []

    def post(path, payload, **kwargs):
        sent.append(payload)
        return 200, {}

    monkeypatch.setattr(worker, "_v2_post", post)

    def handle(command):
        if command["attempt_count"] == 1:
            entered.set()
            assert release.wait(5)
        worker.v2_ack_command(command["command_id"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        old = pool.submit(
            worker._v2_run_claimed_command, handle, {"command_id": "same", "attempt_count": 1}
        )
        assert entered.wait(2)
        try:
            pool.submit(
                worker._v2_run_claimed_command, handle, {"command_id": "same", "attempt_count": 2}
            ).result(timeout=3)
        finally:
            release.set()
        old.result(timeout=3)
    assert [payload["claim_attempt"] for payload in sent] == [2, 1]
    assert all(payload["worker_session_id"] == worker.WORKER_SESSION_ID for payload in sent)


def test_active_mode_requires_command_claim_identity(command_row):
    host, command_id, _ = command_row
    client = TestClient(app)
    headers = {"Authorization": "Bearer " + os.environ["XCELSIOR_API_TOKEN"]}
    assert (
        client.post(
            "/agent/v2/commands/claim",
            headers=headers,
            json={"host_id": host, "worker_session_id": "current", "host_commands": True},
        ).status_code
        == 200
    )
    response = client.post(
        f"/agent/v2/commands/{command_id}/ack", headers=headers, json={"host_id": host}
    )
    assert response.status_code == 422, response.text
