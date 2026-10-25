"""Physical reservations reach Docker unchanged; invalid grants never execute."""

import json
import os
from unittest import mock

import pytest
from fastapi.testclient import TestClient

import worker_agent as worker
from control_plane.db import run_transaction
from control_plane.leases import LeaseClaimRejected, claim_lease
from security import build_secure_docker_args
from tests.test_control_plane_reservation import (
    _claim,
    _mkhost,
    _mkjob,
    _reserve,
    _pool,
    fleet,  # noqa: F401
)
from tests.test_serverless_native_lifecycle import native  # noqa: F401
from tests.test_worker_claim_handoff import agent, client_post, command  # noqa: F401


def claim_body(r):
    return dict(
        lease_id=r.lease_id,
        job_id=r.job_id,
        attempt_id=r.attempt_id,
        host_id=r.host_id,
        fencing_token=r.fencing_token,
        worker_session_id="gpu-binding-test",
    )


def test_two_jobs_on_one_host_receive_only_their_reserved_device(fleet):
    import api

    host = _mkhost(fleet, gpus=2)
    reservations = [_reserve(_claim(_mkjob(fleet)), host) for _ in range(2)]
    grants = []
    with TestClient(api.app) as client:
        for r in reservations:
            response = client.post(
                "/agent/v2/leases/claim",
                json=claim_body(r),
                headers={"Authorization": f"Bearer {os.environ['XCELSIOR_API_TOKEN']}"},
            )
            assert response.status_code == 200, response.text
            grant = response.json()
            with _pool.connection() as conn:
                expected = [
                    row[0]
                    for row in conn.execute(
                        "SELECT d.gpu_uuid FROM gpu_device_allocations a "
                        "JOIN host_gpu_devices d USING(gpu_device_id) "
                        "WHERE a.attempt_id=%s AND a.released_at IS NULL",
                        (r.attempt_id,),
                    ).fetchall()
                ]
            assert grant["gpu_device_uuids"] == expected
            args = build_secure_docker_args(
                "image:test",
                r.job_id,
                num_gpus=1,
                runtime="runc",
                gpu_device_uuids=grant["gpu_device_uuids"],
                environment={"NVIDIA_VISIBLE_DEVICES": "all"},
            )
            assert args[args.index("--gpus") + 1] == f'"device={expected[0]}"'
            assert f"NVIDIA_VISIBLE_DEVICES={expected[0]}" in args
            assert "NVIDIA_VISIBLE_DEVICES=all" not in args
            grants.append(expected)
    assert len(grants[0]) == len(grants[1]) == 1
    assert set(grants[0]).isdisjoint(grants[1])


def test_multi_gpu_grant_contains_every_reserved_uuid(fleet):
    host = _mkhost(fleet, gpus=3)
    r = _reserve(_claim(_mkjob(fleet, num_gpus=2)), host)
    grant = run_transaction(lambda conn: claim_lease(conn, **claim_body(r)))
    assert len(grant.gpu_device_uuids) == 2
    with _pool.connection() as conn:
        reserved = {
            row[0]
            for row in conn.execute(
                "SELECT d.gpu_uuid FROM gpu_device_allocations a JOIN host_gpu_devices d USING(gpu_device_id) "
                "WHERE a.attempt_id=%s AND a.released_at IS NULL",
                (r.attempt_id,),
            ).fetchall()
        }
    assert set(grant.gpu_device_uuids) == reserved


@pytest.mark.parametrize("fault", ["released", "missing", "synthetic", "retired"])
def test_invalid_allocation_rolls_back_lease_claim_and_projection(fleet, fault):
    host = _mkhost(fleet, gpus=1)
    r = _reserve(_claim(_mkjob(fleet)), host)
    with _pool.connection() as conn:
        if fault == "released":
            conn.execute(
                "UPDATE gpu_device_allocations SET released_at=now(), status='released' WHERE attempt_id=%s",
                (r.attempt_id,),
            )
        elif fault == "missing":
            conn.execute("DELETE FROM gpu_device_allocations WHERE attempt_id=%s", (r.attempt_id,))
        elif fault == "synthetic":
            conn.execute("UPDATE host_gpu_devices SET gpu_uuid='slot:0' WHERE host_id=%s", (host,))
        else:
            conn.execute("UPDATE host_gpu_devices SET retired_at=now() WHERE host_id=%s", (host,))
        before = conn.execute(
            "SELECT phase,version FROM jobs WHERE job_id=%s", (r.job_id,)
        ).fetchone()
        outbox_before = conn.execute(
            "SELECT count(*) FROM outbox_events WHERE aggregate_id=%s", (r.job_id,)
        ).fetchone()
    with pytest.raises(LeaseClaimRejected, match="gpu_allocation_authority_missing"):
        run_transaction(lambda conn: claim_lease(conn, **claim_body(r)))
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT status,claimed_at FROM placement_leases WHERE lease_id=%s", (r.lease_id,)
        ).fetchone() == ("offered", None)
        assert (
            conn.execute(
                "SELECT status FROM job_attempts WHERE attempt_id=%s", (r.attempt_id,)
            ).fetchone()[0]
            == "reserved"
        )
        assert (
            conn.execute("SELECT phase,version FROM jobs WHERE job_id=%s", (r.job_id,)).fetchone()
            == before
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM outbox_events WHERE aggregate_id=%s", (r.job_id,)
            ).fetchone()
            == outbox_before
        )


@pytest.mark.parametrize(
    "devices", [None, [], ["all"], ["0"], ["slot:0"], ["GPU-a", "GPU-a"], ["GPU-a", "GPU-b"]]
)
def test_invalid_api_grant_settles_claim_without_start(agent, monkeypatch, devices):
    import api

    cmd = command(agent)
    run = mock.Mock()
    nack = mock.Mock()
    monkeypatch.setattr(worker, "run_job", run)
    monkeypatch.setattr(worker, "v2_nack_command", nack)
    with TestClient(api.app) as client:

        def post(path, body):
            code, result = client_post(client, path, body)
            if path == "/agent/v2/leases/claim" and code == 200:
                result["gpu_device_uuids"] = devices
            return code, result

        monkeypatch.setattr(worker, "_v2_post", post)
        worker.handle_start_attempt(cmd)
    run.assert_not_called()
    nack.assert_called_once_with(
        cmd["command_id"], "gpu_allocation_authority_missing", retryable=False
    )
    with agent.pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT status FROM placement_leases WHERE lease_id=%s", (agent.r.lease_id,)
            ).fetchone()[0]
            == "released"
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM gpu_device_allocations WHERE attempt_id=%s AND released_at IS NULL",
                (agent.r.attempt_id,),
            ).fetchone()[0]
            == 0
        )


def test_valid_api_grant_is_journaled_and_passed_to_real_docker_builder(agent, monkeypatch):
    import api

    cmd = command(agent)
    recorded = []

    def run(job):
        auth = worker._v2_attempts[job["job_id"]]
        args = build_secure_docker_args(
            "image:test",
            auth["container_name"],
            num_gpus=1,
            gpu_device_uuids=auth["gpu_device_uuids"],
        )
        recorded.append((auth["gpu_device_uuids"], args))

    monkeypatch.setattr(worker, "run_job", run)
    monkeypatch.setattr(worker, "_v2_record_and_ack", mock.Mock(return_value=True))
    with TestClient(api.app) as client:
        monkeypatch.setattr(worker, "_v2_post", lambda path, body: client_post(client, path, body))
        with mock.patch.object(worker.threading, "Thread"):
            worker.handle_start_attempt(cmd)
    expected = [f"GPU-{agent.hid}"]
    assert recorded[0][0] == expected
    assert f'"device={expected[0]}"' in recorded[0][1]
    saved = json.loads(open(worker._V2_JOURNAL_PATH).read())
    assert saved[agent.jid]["gpu_device_uuids"] == expected
    worker._v2_attempts.clear()
    with (
        mock.patch.object(
            worker, "_v2_container_state", return_value=("running", 0, agent.r.attempt_id)
        ),
        mock.patch.object(worker, "_v2_post", return_value=(200, {})),
        mock.patch.object(worker, "_v2_start_exit_monitor"),
        mock.patch.object(worker.threading, "Thread"),
    ):
        assert worker.v2_adopt_attempts() == 1
    assert worker._v2_attempts[agent.jid]["gpu_device_uuids"] == expected


@pytest.mark.parametrize(
    "devices", [None, [], ["all"], ["0"], ["slot:0"], ["GPU-a", "GPU-a"], ["GPU-a,all"]]
)
def test_docker_builder_never_falls_back_to_count_or_all(devices):
    with pytest.raises(ValueError, match="reserved physical GPU UUIDs"):
        build_secure_docker_args(
            "image:test", "container-test", num_gpus=1, gpu_device_uuids=devices
        )


def test_multi_gpu_binding_uses_uuid_set_instead_of_first_indices():
    args = build_secure_docker_args(
        "image:test", "multi-test", num_gpus=2, gpu_device_uuids=["GPU-second", "GPU-fourth"]
    )
    assert args[args.index("--gpus") + 1] == '"device=GPU-second,GPU-fourth"'
    assert "NVIDIA_VISIBLE_DEVICES=GPU-second,GPU-fourth" in args
    with pytest.raises(ValueError):
        build_secure_docker_args(
            "image:test", "count-mismatch", num_gpus=2, gpu_device_uuids=["GPU-second"]
        )


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "lease_expired",
        "local_authority_lost",
        "renewal_unavailable",
        "authority_lost_during_renewal",
        "authority_lost_during_start",
        "shutdown_before_start",
        "stop_during_setup",
    ],
)
def test_real_runner_uses_claimed_devices_and_rechecks_start_authority(agent, monkeypatch, fault):
    import api
    import egress_isolation
    from serverless.env_secrets import encrypt_env_for_storage
    from tests.test_native_environment_delivery import bind_environment

    bind_environment(agent, encrypt_env_for_storage({"NVIDIA_VISIBLE_DEVICES": "all"}))
    cmd = command(agent)
    built = []
    for name in ["_kill_container", "_remove_container", "_push_log_lines", "cache_track_pull"]:
        monkeypatch.setattr(worker, name, mock.Mock())
    monkeypatch.setattr(worker.time, "sleep", mock.Mock())
    monkeypatch.setattr(worker, "get_gpu_info", lambda: {"gpu_model": "RTX 4090"})
    monkeypatch.setattr(worker, "build_platform_env", lambda *a: {})

    def image_available(*args):
        if fault == "lease_expired":
            with agent.pool.connection() as conn:
                conn.execute(
                    "UPDATE placement_leases SET expires_at=now()-interval '1 second' WHERE lease_id=%s",
                    (agent.r.lease_id,),
                )
        elif fault == "local_authority_lost":
            worker._v2_forget_attempt(agent.jid)
        elif fault == "shutdown_before_start":
            worker._shutdown.set()
        elif fault == "stop_during_setup":
            with agent.pool.connection() as conn:
                conn.execute(
                    "UPDATE jobs SET desired_state='stopped',lifecycle_intent='stop' WHERE job_id=%s",
                    (agent.jid,),
                )
        return True

    monkeypatch.setattr(worker, "cache_image_available", image_available)
    monkeypatch.setattr(worker, "PREFER_GVISOR", False)
    monkeypatch.setattr(egress_isolation, "launch_args", lambda job, env: ([], env))
    monkeypatch.setattr(worker, "_v2_record_and_ack", mock.Mock(return_value=True))
    stop = mock.Mock()
    monkeypatch.setattr(worker, "_v2_stop_container", stop)

    def subprocess_run(args, **kwargs):
        if args[:2] == ["docker", "run"]:
            built.append(args)
            if fault == "authority_lost_during_start":
                worker._v2_forget_attempt(agent.jid)
                return mock.Mock(returncode=0, stdout="isolated-container", stderr="")
            # No real Docker execution: exercise the real runner's failure
            # settlement after capturing its complete container request.
            return mock.Mock(returncode=1, stdout="", stderr="isolated test stop")
        raise AssertionError(f"unexpected subprocess: {args[:2]}")

    monkeypatch.setattr(worker.subprocess, "run", subprocess_run)
    with TestClient(api.app) as client:

        def post(path, body):
            if path == "/agent/v2/leases/renew" and fault == "renewal_unavailable":
                return 503, {}
            code, value = client_post(client, path, body)
            if path == "/agent/v2/leases/renew" and fault == "authority_lost_during_renewal":
                worker._v2_forget_attempt(agent.jid)
            return code, value

        monkeypatch.setattr(worker, "_v2_post", post)
        with mock.patch.object(worker.threading, "Thread"):
            worker.handle_start_attempt(cmd)
    if fault not in (None, "authority_lost_during_start"):
        assert not built
        stop.assert_not_called()
        if fault in {"lease_expired", "renewal_unavailable", "stop_during_setup"}:
            with agent.pool.connection() as conn:
                assert (
                    conn.execute(
                        "SELECT status FROM placement_leases WHERE lease_id=%s", (agent.r.lease_id,)
                    ).fetchone()[0]
                    == "released"
                )
                assert (
                    conn.execute(
                        "SELECT count(*) FROM gpu_device_allocations WHERE attempt_id=%s AND released_at IS NULL",
                        (agent.r.attempt_id,),
                    ).fetchone()[0]
                    == 0
                )
                if fault == "stop_during_setup":
                    assert conn.execute(
                        "SELECT status FROM jobs WHERE job_id=%s", (agent.jid,)
                    ).fetchone() == ("stopped",)
        return
    assert len(built) == 1
    expected = f"GPU-{agent.hid}"
    assert built[0][built[0].index("--gpus") + 1] == f'"device={expected}"'
    assert f"NVIDIA_VISIBLE_DEVICES={expected}" in built[0]
    assert "NVIDIA_VISIBLE_DEVICES=all" not in built[0]
    assert agent.jid not in worker._v2_attempts
    if fault == "authority_lost_during_start":
        stop.assert_called_once_with(
            agent.jid,
            "authority lost during container start",
            worker._v2_container_name(agent.jid, agent.r.attempt_id),
            agent.r.attempt_id,
        )
    else:
        stop.assert_not_called()
