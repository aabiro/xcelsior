"""Admission, mutable observations, reservation and worker hash parity on real PG."""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest
from psycopg.types.json import Jsonb

from db import DatabaseOps
from control_plane.job_specs import canonical_spec_hash, make_execution_spec
from control_plane.launch import canonicalize
from control_plane.scheduler.claim import claim_next_job
from control_plane.scheduler.reservation import CapacityConflict, JobNotSchedulable
from control_plane.scheduler.reservation import reserve_and_bind
from control_plane.db import run_transaction
from tests.test_serverless_native_lifecycle import native  # noqa: F401
from tests.test_worker_claim_handoff import agent, command, client_post  # noqa: F401


def row(native):
    with native.pool.connection() as conn:
        return conn.execute(
            "SELECT spec,spec_hash,status,host_id,payload FROM jobs WHERE job_id=%s",
            (native.jid,),
        ).fetchone()


def admit(native, **extra):
    with native.pool.connection() as conn:
        payload = conn.execute(
            "DELETE FROM jobs WHERE job_id=%s RETURNING payload", (native.jid,)
        ).fetchone()[0]
        payload.update(extra)
        approved = canonicalize(
            {
                **payload,
                "interactive": bool(payload.get("interactive", False)),
                "template_image_id": payload.get("source_template_id") or "",
            }
        )
        DatabaseOps.upsert_job(conn, payload, backend="postgres", admission_spec=approved)
    return row(native)


def test_execution_snapshot_separates_logical_workload_and_trusted_context():
    a = {"name": "work", "image": "python:3.12", "interactive": False, "owner": "alice"}
    b = {**a, "owner": "bob", "egress": "restricted", "job_id": "other", "status": "failed"}
    first, second = make_execution_spec(a), make_execution_spec(b)
    assert first["launch_spec_hash"] == second["launch_spec_hash"]
    assert canonical_spec_hash(first) != canonical_spec_hash(second)
    assert "job_id" not in second and "status" not in second
    assert second["egress"] == "restricted"


def test_snapshot_keeps_storage_runtime_and_serverless_constraints_without_aliasing():
    job = {
        "name": "native",
        "owner": "tenant",
        "exclusive_gpu": True,
        "required_runtime": "nvidia",
        "environment": {"MODEL": "model"},
        "volume_ids": ["volume"],
        "volume_mounts": {"volume": "/models"},
        "encrypted_workspace": True,
        "require_image_signature": True,
        "serverless_worker_id": "worker",
        "serverless_endpoint_id": "endpoint",
        "_serverless_callback_token": "never-store-this",
        "container_id": "observation",
    }
    spec = make_execution_spec(job)
    for key in job.keys() - {"_serverless_callback_token", "container_id"}:
        assert spec[key] == job[key]
    job["environment"]["MODEL"] = "mutated"
    assert spec["environment"] == {"MODEL": "model"}
    assert "never-store-this" not in json.dumps(spec)


def test_status_upserts_never_replace_admitted_workload(native):
    initial = admit(native, owner=native.owner)
    with native.pool.connection() as conn:
        DatabaseOps.upsert_job(
            conn,
            {"job_id": native.jid, "status": "running", "host_id": native.hid, "image": "other"},
            backend="postgres",
        )
    current = row(native)
    assert current[:2] == initial[:2]
    assert current[2:4] == ("running", native.hid)


def test_observation_insert_and_update_never_construct_execution_authority(native):
    with native.pool.connection() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (native.jid,))
        with mock.patch(
            "control_plane.job_specs.make_execution_spec",
            side_effect=AssertionError("observation attempted admission"),
        ):
            for status in ("queued", "running"):
                DatabaseOps.upsert_job(
                    conn,
                    {
                        "job_id": native.jid,
                        "status": status,
                        "owner": native.owner,
                        "image": "untrusted/observation",
                    },
                    backend="postgres",
                )
    assert row(native)[:2] == (None, None)


def test_submission_replay_does_not_recompute_changed_server_policy(native, monkeypatch):
    import scheduler
    import abuse_policy

    with native.pool.connection() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (native.jid,))
    policy = mock.Mock(return_value="restricted")
    monkeypatch.setattr(abuse_policy, "egress_for_owner", policy)
    args = dict(name="request", vram_needed_gb=0, owner=native.owner, job_id=native.jid)
    scheduler.submit_job(**args)
    original = row(native)
    assert original[0]["egress"] == "restricted"
    policy.side_effect = AssertionError("a replay recomputed server policy")
    with native.pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET status='running',phase='running',desired_state='stopped' WHERE job_id=%s",
            (native.jid,),
        )
    retry = scheduler.submit_job(**args)
    assert retry["status"] == "running"
    assert row(native)[:2] == original[:2]
    assert policy.call_count == 1
    with native.pool.connection() as conn:
        assert conn.execute(
            "SELECT desired_state FROM jobs WHERE job_id=%s", (native.jid,)
        ).fetchone() == ("stopped",)


@pytest.mark.parametrize(
    "change",
    [
        {"owner": "another-owner"},
        {"preferred_host_id": "other-host"},
        {"exclusive_gpu": True},
        {"priority": 9},
        {"placement_preference": {"require_verified": True}},
    ],
)
def test_submission_replay_refuses_changed_request_context(native, change):
    import scheduler

    with native.pool.connection() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (native.jid,))
    args = dict(name="request", vram_needed_gb=0, owner=native.owner, job_id=native.jid)
    scheduler.submit_job(**args)
    original = row(native)
    with pytest.raises(ValueError, match="different execution spec or request"):
        scheduler.submit_job(**{**args, **change})
    assert row(native) == original


def test_reservation_uses_admission_for_pin_worker_binding_and_command(native):
    initial = admit(native, owner=native.owner, preferred_host_id=native.hid)
    with native.pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET payload=payload || %s WHERE job_id=%s",
            (
                Jsonb(
                    {
                        "image": "mutated/image:1",
                        "preferred_host_id": "wrong-host",
                        "serverless_worker_id": "wrong-worker",
                        "serverless_endpoint_id": "wrong-endpoint",
                        "num_gpus": 9,
                    }
                ),
                native.jid,
            ),
        )
    reservation = native.reserve()
    with native.pool.connection() as conn:
        command = conn.execute(
            "SELECT args FROM agent_commands WHERE command_id=%s", (reservation.command_id,)
        ).fetchone()[0]
        attempt = conn.execute(
            "SELECT spec_hash,placement_explanation FROM job_attempts WHERE attempt_id=%s",
            (reservation.attempt_id,),
        ).fetchone()
    assert command["spec"] == initial[0]
    assert command["spec_hash"] == attempt[0] == initial[1]
    assert attempt[1]["billing_snapshot"]["num_gpus"] == 1
    assert str(native.repo.get_worker(native.wid)["attempt_id"]) == reservation.attempt_id
    from worker_agent import compute_spec_hash

    assert compute_spec_hash(command["spec"]) == initial[1]


@pytest.mark.parametrize("corruption", ["empty", "changed", "missing_hash", "unsupported"])
def test_corrupt_admission_never_falls_back_to_payload_or_creates_authority(native, corruption):
    spec, digest, *_ = admit(native)
    if corruption == "empty":
        spec = {}
    elif corruption == "changed":
        spec["image"] = "changed"
    elif corruption == "missing_hash":
        digest = None
    else:
        spec["execution_spec_version"] = "future-unknown"
        digest = canonical_spec_hash(spec)
    with native.pool.connection() as conn:
        claim = claim_next_job(conn, replica_id="spec-test", scope_gpu_models=[native.jid])
        assert claim and claim.job_id == native.jid
        conn.execute(
            "UPDATE jobs SET spec=%s,spec_hash=%s WHERE job_id=%s",
            (Jsonb(spec), digest, native.jid),
        )
    with pytest.raises(JobNotSchedulable, match="unqualified"):
        run_transaction(
            lambda conn: reserve_and_bind(
                conn,
                job_id=native.jid,
                claim_token=claim.claim_token,
                replica_id="spec-test",
                host_id=native.hid,
            )
        )
    with native.pool.connection() as conn:
        for table in (
            "job_attempts",
            "placement_leases",
            "gpu_device_allocations",
            "agent_commands",
        ):
            assert (
                conn.execute(
                    f"SELECT count(*) FROM {table} WHERE job_id=%s", (native.jid,)
                ).fetchone()[0]
                == 0
            )


@pytest.mark.parametrize("evidence", ["missing", "recorded_execution"])
def test_unadmitted_execution_evidence_never_authorizes_a_new_attempt(native, evidence):
    spec = None if evidence == "missing" else {"image": "historical-image", "gpu_model": native.jid}
    digest = None if spec is None else canonical_spec_hash(spec)
    token = uuid.uuid4()
    with native.pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET spec=%s,spec_hash=%s WHERE job_id=%s",
            (Jsonb(spec) if spec is not None else None, digest, native.jid),
        )
        assert claim_next_job(conn, replica_id="strict-spec", scope_gpu_models=[native.jid]) is None
        # A stale or manually injected claim cannot bypass the reservation
        # transaction's independent admission check.
        conn.execute(
            """UPDATE jobs SET schedule_claim_token=%s,
            schedule_claim_expires_at=clock_timestamp()+interval '1 minute'
            WHERE job_id=%s""",
            (token, native.jid),
        )
    with pytest.raises(JobNotSchedulable, match="unqualified"):
        run_transaction(
            lambda conn: reserve_and_bind(
                conn,
                job_id=native.jid,
                claim_token=str(token),
                replica_id="strict-spec",
                host_id=native.hid,
            )
        )
    assert row(native)[:2] == (spec, digest)
    with native.pool.connection() as conn:
        for table in (
            "job_attempts",
            "placement_leases",
            "gpu_device_allocations",
            "agent_commands",
        ):
            assert conn.execute(
                f"SELECT count(*) FROM {table} WHERE job_id=%s", (native.jid,)
            ).fetchone() == (0,)


def test_historical_hash_is_never_replaced_to_hide_unqualified_evidence(native):
    with native.pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET spec=NULL,spec_hash='sha256:original' WHERE job_id=%s", (native.jid,)
        )
        assert claim_next_job(conn, replica_id="strict-spec", scope_gpu_models=[native.jid]) is None
    assert row(native)[:2] == (None, "sha256:original")


def test_canary_claim_scope_uses_admitted_model(native, monkeypatch):
    import scheduler
    from control_plane.scheduler.config import SchedulerConfig, SchedulerMode

    admit(native)
    with native.pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET payload=payload || %s WHERE job_id=%s",
            (Jsonb({"gpu_model": "mutated"}), native.jid),
        )
    with native.pool.connection() as conn:
        claimed = claim_next_job(conn, replica_id="spec-test", scope_gpu_models=[native.jid])
    assert claimed and claimed.job_id == native.jid
    assert claimed.spec["gpu_model"] == native.jid
    monkeypatch.setattr(
        scheduler,
        "_cp_config",
        SchedulerConfig(mode=SchedulerMode.CANARY, canary_gpu_models=frozenset({native.jid})),
    )
    visible = next(job for job in scheduler.list_jobs() if job["job_id"] == native.jid)
    assert scheduler._cp_config.owns_job(visible)


def test_scheduler_places_from_admission_despite_changed_payload_constraints(native):
    from control_plane.scheduler.config import SchedulerConfig, SchedulerMode
    from control_plane.scheduler.service import SchedulerService

    initial = admit(native, preferred_host_id=native.hid, trace_id="admission-trace")
    with native.pool.connection() as conn:
        conn.execute(
            "UPDATE hosts SET payload=payload || %s WHERE host_id=%s",
            (
                Jsonb(
                    {
                        "gpu_model": native.jid,
                        "last_seen": time.time(),
                        "capabilities": {"egress_isolation": True},
                    }
                ),
                native.hid,
            ),
        )
        conn.execute(
            "UPDATE jobs SET payload=payload || %s WHERE job_id=%s",
            (Jsonb({"gpu_model": "wrong", "num_gpus": 99}), native.jid),
        )
        claimed = claim_next_job(conn, replica_id="spec-test", scope_gpu_models=[native.jid])
    assert claimed
    service = SchedulerService(SchedulerConfig(mode=SchedulerMode.ACTIVE, replica_id="spec-test"))
    reservation = service._place_one(claimed)
    assert reservation.host_id == native.hid and reservation.spec_hash == initial[1]
    with native.pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT trace_id FROM job_attempts WHERE attempt_id=%s", (reservation.attempt_id,)
            ).fetchone()[0]
            == "admission-trace"
        )


def test_resources_are_rechecked_under_lock_before_allocating(native):
    admit(native, num_gpus=2)
    # A stale/default caller count of one cannot under-reserve the admitted
    # two-GPU workload on a host that only has one concrete device.
    with pytest.raises(CapacityConflict) as error:
        native.reserve()
    assert error.value.details["requested"] == 2
    with native.pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM gpu_device_allocations WHERE job_id=%s", (native.jid,)
            ).fetchone()[0]
            == 0
        )


def test_native_worker_gets_admitted_context_and_real_fenced_callback_credential(
    agent, monkeypatch
):
    import api
    import worker_agent as worker
    from fastapi.testclient import TestClient

    initial = admit(
        agent,
        owner=agent.owner,
        egress="restricted",
        exclusive_gpu=True,
        encrypted_workspace=True,
        environment={"MODEL": "pinned"},
        volume_mounts={"volume": "/models"},
    )
    cmd = command(agent)
    # Observation drift must not redefine the callback/stop/settlement identity
    # after a correct reservation has already bound the admitted worker.
    with agent.pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET payload=payload || %s WHERE job_id=%s",
            (
                Jsonb(
                    {
                        "job_type": "wrong",
                        "serverless_worker_id": "wrong",
                        "serverless_endpoint_id": "wrong",
                    }
                ),
                agent.jid,
            ),
        )
    run = mock.Mock()
    monkeypatch.setattr(worker, "run_job", run)
    monkeypatch.setattr(worker, "_v2_record_and_ack", lambda *args, **kwargs: True)
    with TestClient(api.app) as client:
        monkeypatch.setattr(worker, "_v2_post", lambda path, body: client_post(client, path, body))
        with mock.patch.object(worker.threading, "Thread"):
            worker.handle_start_attempt(cmd)
    run.assert_called_once()
    job = run.call_args.args[0]
    for key, value in initial[0].items():
        assert job[key] == value
    assert job["job_id"] == agent.jid
    assert job["_serverless_callback_token"].startswith("xsw_")
    assert job["_v2_spec_hash"] == initial[1]
    assert job["_serverless_callback_token"] not in open(worker._V2_JOURNAL_PATH).read()
    from control_plane.serverless_credentials import authenticate_callback
    from control_plane.attempts import report_attempt_status

    with agent.pool.connection() as conn:
        assert authenticate_callback(conn, job["_serverless_callback_token"], agent.wid)
    stopped = agent.svc.deprovision_worker(agent.wid)
    assert stopped["requested"] and stopped["pending"]
    run_transaction(
        lambda conn: report_attempt_status(
            conn,
            job_id=agent.jid,
            attempt_id=agent.r.attempt_id,
            host_id=agent.hid,
            fencing_token=agent.r.fencing_token,
            status="stopped",
        )
    )
    assert agent.repo.get_worker(agent.wid)["released_at"] is not None


def test_submission_retry_preserves_running_state_and_rejects_changed_spec(native):
    import scheduler

    with native.pool.connection() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (native.jid,))
    args = dict(name="admitted", vram_needed_gb=0, owner=native.owner, job_id=native.jid)
    scheduler.submit_job(**args)
    initial = row(native)
    with native.pool.connection() as conn:
        conn.execute(
            """UPDATE jobs SET status='running', phase='running', desired_state='running',host_id=%s WHERE job_id=%s""",
            (native.hid, native.jid),
        )
    retry = scheduler.submit_job(**args)
    assert retry["status"] == "running" and retry["host_id"] == native.hid
    assert row(native)[:2] == initial[:2]
    with pytest.raises(ValueError, match="different execution spec"):
        scheduler.submit_job(**{**args, "command": "changed"})
    assert row(native)[2:4] == ("running", native.hid)


def test_concurrent_submission_admits_one_immutable_job_and_event(native):
    import scheduler

    with native.pool.connection() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (native.jid,))

    def submit():
        return scheduler.submit_job("concurrent", 0, job_id=native.jid)

    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = list(executor.map(lambda _: submit(), range(2)))
    assert jobs[0]["job_id"] == jobs[1]["job_id"] == native.jid
    with native.pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM outbox_events WHERE aggregate_id=%s AND event_type='job.v1.submitted'",
                (native.jid,),
            ).fetchone()[0]
            == 1
        )


def test_submit_refuses_disagreement_with_approved_workload_before_writing(native):
    import scheduler

    approved = canonicalize({"name": "original", "interactive": False})
    before = row(native)
    with pytest.raises(ValueError, match="approved launch spec"):
        scheduler.submit_job("different", 0, job_id=native.jid, admitted_launch_spec=approved)
    assert row(native) == before
