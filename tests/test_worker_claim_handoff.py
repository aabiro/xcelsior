"""Real fenced settlement and restart handoff with Docker execution isolated."""

import json
import os
import signal
import threading
from unittest import mock

import pytest
from fastapi.testclient import TestClient

import worker_agent as worker
from control_plane.db import run_transaction
from control_plane.leases import claim_lease
from tests.test_serverless_native_lifecycle import native, start  # noqa: F401


@pytest.fixture
def agent(native, monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "HOST_ID", native.hid)
    monkeypatch.setattr(worker, "WORKER_SESSION_ID", "native-test")
    monkeypatch.setattr(worker, "_v2_enabled", True)
    monkeypatch.setattr(worker, "_shutdown", threading.Event())
    monkeypatch.setattr(worker, "_v2_attempts", {})
    monkeypatch.setattr(worker, "_v2_completed_commands", {})
    monkeypatch.setattr(worker, "_active_containers", {})
    monkeypatch.setattr(worker, "_adopted_containers", set())
    monkeypatch.setattr(worker, "_DEREGISTER_ON_EXIT", False)
    monkeypatch.setattr(worker, "_V2_JOURNAL_PATH", str(tmp_path / "attempts.json"))
    return native


def old_claim(body):
    grant = run_transaction(lambda conn: claim_lease(conn, **body))
    return 200, {
        "lease_id": grant.lease_id,
        "job_id": grant.job_id,
        "attempt_id": grant.attempt_id,
        "fencing_token": grant.fencing_token,
        "renewal_ttl_sec": grant.renewal_ttl_sec,
    }


def command(x):
    r = x.reserve()
    with x.pool.connection() as conn:
        row = conn.execute(
            "SELECT command_id,job_id,attempt_id,fencing_token,spec_hash,args FROM agent_commands WHERE job_id=%s AND command='start_attempt'",
            (x.jid,),
        ).fetchone()
    result = dict(
        zip(["command_id", "job_id", "attempt_id", "fencing_token", "spec_hash", "args"], row)
    )
    result["command_id"] = str(result["command_id"])
    result["attempt_id"] = str(result["attempt_id"])
    return result


def client_post(client, path, body):
    response = client.post(
        path, json=body, headers={"Authorization": f"Bearer {os.environ['XCELSIOR_API_TOKEN']}"}
    )
    return response.status_code, response.json()


def assert_settled(x):
    with x.pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT status FROM placement_leases WHERE lease_id=%s", (x.r.lease_id,)
            ).fetchone()[0]
            == "released"
        )
        assert conn.execute(
            "SELECT status,failure_code FROM job_attempts WHERE attempt_id=%s", (x.r.attempt_id,)
        ).fetchone() == ("failed", "callback_credential_missing")
        assert (
            conn.execute(
                "SELECT count(*) FROM gpu_device_allocations WHERE attempt_id=%s AND released_at IS NULL",
                (x.r.attempt_id,),
            ).fetchone()[0]
            == 0
        )
    assert x.repo.get_worker(x.wid)["released_at"] is not None


@pytest.mark.parametrize("outage", [False, True])
def test_tokenless_old_api_grant_settles_without_container_or_lease_expiry(
    agent, monkeypatch, outage
):
    import api

    x = agent
    cmd = command(x)
    threads = mock.Mock()
    run = mock.Mock()
    monkeypatch.setattr(worker, "run_job", run)
    nack = mock.Mock()
    monkeypatch.setattr(worker, "v2_nack_command", nack)
    posts = []
    with TestClient(api.app) as client:

        def post(path, body):
            posts.append((path, body))
            if path == "/agent/v2/leases/claim":
                return old_claim(body)
            return (503, {}) if outage else client_post(client, path, body)

        monkeypatch.setattr(worker, "_v2_post", post)
        with mock.patch.object(worker.threading, "Thread", threads):
            worker.handle_start_attempt(cmd)
        run.assert_not_called()
        nack.assert_called_once_with(
            cmd["command_id"], "callback_credential_missing", retryable=False
        )
        if outage:
            saved = json.loads(open(worker._V2_JOURNAL_PATH).read())
            assert saved[x.jid]["pending_start_failure"] == "callback_credential_missing"
            assert threads.call_args.kwargs["target"] is worker._v2_retry_aborted_start
            with x.pool.connection() as conn:
                assert (
                    conn.execute(
                        "SELECT status FROM placement_leases WHERE lease_id=%s", (x.r.lease_id,)
                    ).fetchone()[0]
                    == "active"
                )
            worker._v2_attempts.clear()
            monkeypatch.setattr(
                worker, "_v2_post", lambda path, body: client_post(client, path, body)
            )
            with mock.patch.object(worker, "_v2_container_state") as docker:
                assert worker.v2_adopt_attempts() == 0
            docker.assert_not_called()
        assert_settled(x)
        assert x.jid not in json.loads(open(worker._V2_JOURNAL_PATH).read())
        assert all(
            path in ["/agent/v2/leases/claim", "/agent/v2/attempts/status"] for path, _ in posts
        )


def test_sigterm_preserves_new_and_adopted_fenced_work_for_verified_restart(agent, monkeypatch):
    import api

    x = agent
    r = start(x)
    auth = {
        "job_id": x.jid,
        "attempt_id": r.attempt_id,
        "lease_id": r.lease_id,
        "fencing_token": r.fencing_token,
        "renewal_ttl_sec": 300,
        "container_name": worker._v2_container_name(x.jid, r.attempt_id),
        "stop": threading.Event(),
    }
    worker._v2_attempts[x.jid] = auth
    worker._active_containers[x.jid] = auth["container_name"]
    docker = mock.Mock()
    monkeypatch.setattr(worker.subprocess, "run", docker)
    cleanup = mock.Mock()
    monkeypatch.setattr(worker, "cleanup_orphaned_volume_mounts", cleanup)
    worker._signal_handler(signal.SIGTERM, None)
    worker.graceful_shutdown()
    docker.assert_not_called()
    cleanup.assert_not_called()
    assert json.loads(open(worker._V2_JOURNAL_PATH).read())[x.jid]["attempt_id"] == r.attempt_id
    worker._v2_attempts.clear()
    worker._active_containers.clear()
    worker._shutdown.clear()
    monkeypatch.setattr(worker, "_v2_container_state", lambda *args: ("running", 0, r.attempt_id))
    with TestClient(api.app) as client, mock.patch.object(worker.threading, "Thread"):
        monkeypatch.setattr(worker, "_v2_post", lambda path, body: client_post(client, path, body))
        assert worker.v2_adopt_attempts() == 1
    assert worker._v2_attempts[x.jid]["attempt_id"] == r.attempt_id
    assert worker._active_containers[x.jid] == auth["container_name"]
    assert x.jid in worker._adopted_containers
    worker.graceful_shutdown()  # Identical behavior for recovered work.
    docker.assert_not_called()
    with x.pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT status FROM placement_leases WHERE lease_id=%s", (r.lease_id,)
            ).fetchone()[0]
            == "active"
        )
        assert (
            conn.execute(
                "SELECT status FROM job_attempts WHERE attempt_id=%s", (r.attempt_id,)
            ).fetchone()[0]
            == "running"
        )


@pytest.mark.parametrize("handoff", [False, True])
@pytest.mark.parametrize("managed", [False, True])
def test_runner_finally_preserves_mounts_and_workspace_only_for_shutdown_handoff(
    agent, monkeypatch, handoff, managed
):
    from types import SimpleNamespace

    x = agent
    r = start(x)
    cname = worker._v2_container_name(x.jid, r.attempt_id)
    auth = {
        "job_id": x.jid,
        "attempt_id": r.attempt_id,
        "lease_id": r.lease_id,
        "fencing_token": r.fencing_token,
        "container_name": cname,
        "stop": threading.Event(),
    }
    worker._v2_attempts[x.jid] = auth
    worker._v2_journal_save()
    for name in ["get_gpu_info", "build_platform_env"]:
        monkeypatch.setattr(worker, name, mock.Mock(return_value={}))
    for name in ["_kill_container", "_remove_container", "_push_log_lines"]:
        monkeypatch.setattr(worker, name, mock.Mock())
    monkeypatch.setattr(worker.time, "sleep", mock.Mock())
    monkeypatch.setattr(worker, "report_job_status", mock.Mock(return_value=True))
    monkeypatch.setattr(worker, "_acquire_exclusive_gpu", mock.Mock(return_value={"ok": True}))
    monkeypatch.setattr(worker, "_mount_nfs", mock.Mock(return_value=True))
    monkeypatch.setattr(
        worker, "attach_encrypted_volume", mock.Mock(return_value="/encrypted/source")
    )
    monkeypatch.setattr(worker, "provision_encrypted_volume", mock.Mock(return_value=True))
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        mock.Mock(return_value=SimpleNamespace(returncode=0, stdout="", stderr="")),
    )
    monkeypatch.setattr(worker, "_image_cache_index", {})
    cleanup = {}
    for name in [
        "_unmount_nfs",
        "detach_encrypted_volume",
        "destroy_encrypted_volume",
        "_release_exclusive_gpu",
    ]:
        cleanup[name] = mock.Mock()
        monkeypatch.setattr(worker, name, cleanup[name])

    def exit_worker(**kwargs):
        if handoff:
            worker._shutdown.set()
        raise SystemExit("worker exits while execution authority remains journaled")

    monkeypatch.setattr(worker, "cache_evict_lru", exit_worker)
    job = {
        "job_id": x.jid,
        "image": "isolated/test:1",
        "_v2_attempt_id": r.attempt_id,
        "_v2_fencing_token": r.fencing_token,
        "_v2_container_name": cname,
        "nfs_server": "192.0.2.1",
        "nfs_path": "/exports/data",
        "encrypted_volumes": [{"volume_id": "source"}],
        "encrypted_workspace": True,
        "exclusive_gpu": True,
    }
    if managed:
        job["volume_ids"] = ["managed-source"]
    with pytest.raises(SystemExit):
        worker.run_job(job)
    assert (x.jid in worker._active_containers) == handoff
    for function in cleanup.values():
        assert function.call_count == (0 if handoff else 1)
