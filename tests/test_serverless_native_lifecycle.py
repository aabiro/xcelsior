"""Real native placement, stop confirmation and retryable serverless settlement."""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import pytest
from psycopg.types.json import Jsonb

from billing import get_billing_engine
from control_plane.attempts import report_attempt_status
from control_plane.db import run_transaction
from control_plane.leases import claim_lease, FencingViolation, LeaseClaimRejected
from control_plane.scheduler.claim import claim_next_job
from control_plane.scheduler.reservation import reserve_and_bind
from db import _get_pg_pool
from serverless.metering import bill_active_serverless_workers, charge_serverless_worker
from serverless.repo import EndpointCreate, ServerlessRepo
from serverless.service import ServerlessService


@pytest.fixture
def native():
    repo = ServerlessRepo()
    jid, hid, owner = [f"native-slv-{uuid.uuid4().hex[:12]}" for _ in range(3)]
    ep = repo.create_endpoint(
        EndpointCreate(
            owner_id=owner,
            name=jid,
            mode="preset",
            model_ref="Qwen/Qwen3-8B",
            gpu_tier="RTX 4090",
            min_workers=0,
        )
    )
    w = repo.create_worker(ep["endpoint_id"], scheduler_job_id=jid, allocated_at=0)
    payload = {
        "job_id": jid,
        "name": jid,
        "image": "test/image:1",
        "gpu_model": jid,
        "job_type": "serverless_worker",
        "serverless_worker_id": w["worker_id"],
        "serverless_endpoint_id": ep["endpoint_id"],
        "num_gpus": 1,
    }
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO jobs(job_id,status,priority,submitted_at,payload) VALUES(%s,'queued',0,%s,%s)",
            (jid, time.time(), Jsonb(payload)),
        )
        conn.execute(
            "INSERT INTO hosts(host_id,status,registered_at,payload,administrative_state,availability_state,admission_state) VALUES(%s,'active',%s,%s,'admitted','ready','admitted')",
            (hid, time.time(), Jsonb({"admitted": True, "cost_per_hour": 3.6})),
        )
        conn.execute(
            "INSERT INTO host_gpu_devices(host_id,gpu_uuid,model,total_vram_mb,allocatable_vram_mb,health) VALUES(%s,%s,'RTX 4090',24576,24576,'healthy')",
            (hid, f"GPU-{hid}"),
        )
    x = SimpleNamespace(
        repo=repo,
        svc=ServerlessService(repo),
        jid=jid,
        hid=hid,
        ep=ep,
        wid=w["worker_id"],
        pool=pool,
        owner=owner,
    )

    def reserve():
        def tx(conn):
            claim = claim_next_job(conn, replica_id=jid, scope_gpu_models=[jid])
            assert claim and claim.job_id == jid
            return reserve_and_bind(
                conn, job_id=jid, claim_token=claim.claim_token, replica_id=jid, host_id=hid
            )

        x.r = run_transaction(tx)
        return x.r

    x.reserve = reserve
    yield x
    with pool.connection() as conn:
        conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (jid,))
        conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (jid,))
        conn.execute("DELETE FROM serverless_jobs WHERE endpoint_id=%s", (ep["endpoint_id"],))
        conn.execute("DELETE FROM serverless_workers WHERE worker_id=%s", (w["worker_id"],))
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.execute("DELETE FROM serverless_endpoints WHERE endpoint_id=%s", (ep["endpoint_id"],))


def report(x, status, **kw):
    r = x.r
    return run_transaction(
        lambda conn: report_attempt_status(
            conn,
            job_id=r.job_id,
            attempt_id=r.attempt_id,
            host_id=r.host_id,
            fencing_token=kw.pop("fencing_token", r.fencing_token),
            status=status,
            **kw,
        )
    )


def start(x):
    r = x.reserve()
    run_transaction(
        lambda conn: claim_lease(
            conn,
            lease_id=r.lease_id,
            job_id=r.job_id,
            attempt_id=r.attempt_id,
            host_id=r.host_id,
            fencing_token=r.fencing_token,
            worker_session_id="native-test",
        )
    )
    report(x, "running")
    return r


def test_unplaced_stop_settles_without_inventing_runtime(native):
    result = native.svc.deprovision_worker(native.wid)
    assert result == {"requested": True, "pending": False}
    w = native.repo.get_worker(native.wid)
    assert w["state"] == "terminated" and w["allocated_at"] == 0
    with native.pool.connection() as conn:
        assert (
            conn.execute("SELECT status FROM jobs WHERE job_id=%s", (native.jid,)).fetchone()[0]
            == "cancelled"
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM job_attempts WHERE job_id=%s", (native.jid,)
            ).fetchone()[0]
            == 0
        )


def test_unclaimed_offer_stop_revokes_grant_before_any_container_can_start(native):
    r = native.reserve()
    assert native.svc.deprovision_worker(native.wid)["pending"] is False
    with pytest.raises(LeaseClaimRejected):
        run_transaction(
            lambda conn: claim_lease(
                conn,
                lease_id=r.lease_id,
                job_id=r.job_id,
                attempt_id=r.attempt_id,
                host_id=r.host_id,
                fencing_token=r.fencing_token,
                worker_session_id="late",
            )
        )
    assert native.repo.get_worker(native.wid)["state"] == "terminated"


@pytest.mark.parametrize("phase", ["queued", "offered", "claimed"])
@pytest.mark.parametrize("field", ["serverless_endpoint_id", "job_type"])
def test_stop_refuses_mismatched_execution_metadata(native, phase, field):
    if phase == "offered":
        native.reserve()
    elif phase == "claimed":
        start(native)
    before = native.repo.get_worker(native.wid)
    with native.pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET payload=payload||%s,"
            "spec=CASE WHEN spec IS NULL THEN NULL ELSE spec||%s END WHERE job_id=%s",
            (Jsonb({field: "mismatched"}), Jsonb({field: "mismatched"}), native.jid),
        )
        commands = conn.execute(
            "SELECT count(*) FROM agent_commands WHERE job_id=%s", (native.jid,)
        ).fetchone()[0]
    assert native.svc.deprovision_worker(native.wid)["requested"] is False
    after = native.repo.get_worker(native.wid)
    assert (after["state"], after["released_at"]) == (before["state"], before["released_at"])
    with native.pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM agent_commands WHERE job_id=%s", (native.jid,)
            ).fetchone()[0]
            == commands
        )


def test_claimed_stop_retains_release_and_inflight_until_native_confirmation(native):
    start(native)
    request = native.repo.enqueue_job(native.ep["endpoint_id"], native.owner, {"task": 1})
    native.repo.claim_next_job(native.ep["endpoint_id"], native.wid)
    result = native.svc.deprovision_worker(native.wid)
    assert result["requested"] and result["pending"]
    assert native.repo.get_worker(native.wid)["state"] == "draining"
    assert native.repo.get_worker(native.wid)["released_at"] is None
    assert native.repo.get_job(request["job_id"])["status"] == "IN_PROGRESS"
    again = native.svc.deprovision_worker(native.wid)
    assert again["command_id"] == result["command_id"]
    report(native, "stopped")
    w = native.repo.get_worker(native.wid)
    assert w["state"] == "terminated" and w["released_at"] > 0
    assert native.repo.get_job(request["job_id"])["status"] == "IN_QUEUE"


def test_projection_failure_rolls_back_terminal_and_worker_together(native, monkeypatch):
    start(native)
    import control_plane.attempts as attempts

    original = attempts.sync_serverless_attempt

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("after serverless projection")

    monkeypatch.setattr(attempts, "sync_serverless_attempt", fail)
    with pytest.raises(RuntimeError):
        report(native, "failed")
    assert native.repo.get_worker(native.wid)["released_at"] is None
    with native.pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT status FROM job_attempts WHERE attempt_id=%s", (native.r.attempt_id,)
            ).fetchone()[0]
            == "running"
        )
        assert (
            conn.execute(
                "SELECT status FROM placement_leases WHERE lease_id=%s", (native.r.lease_id,)
            ).fetchone()[0]
            == "active"
        )


def test_stale_fence_cannot_close_or_requeue_serverless_capacity(native):
    start(native)
    with pytest.raises(FencingViolation):
        report(native, "failed", fencing_token=native.r.fencing_token + 1)
    assert native.repo.get_worker(native.wid)["state"] == "ready"
    assert native.repo.get_worker(native.wid)["released_at"] is None


def test_free_warm_readiness_and_heartbeat_cannot_bypass_native_health(native):
    native.repo.update_worker(native.wid, billing_exempt=True, warm_expires_at=time.time() - 1)
    native.svc.worker_heartbeat(native.wid)
    assert native.repo.get_worker(native.wid)["state"] == "booting"
    start(native)
    assert native.repo.get_worker(native.wid)["warm_expires_at"] > time.time()


def test_native_terminal_charge_retries_atomically_and_survives_soft_delete(native, monkeypatch):
    start(native)
    billing = get_billing_engine()
    billing.deposit(native.owner, 50)
    native.repo.update_worker(native.wid, allocated_at=time.time() - 130)
    native.repo.accrue_endpoint_token_cost(native.ep["endpoint_id"], 1.25)
    report(native, "failed")
    w = native.repo.get_worker(native.wid)
    released = w["released_at"]
    native.repo.soft_delete_endpoint(native.ep["endpoint_id"], native.owner)
    before = billing.get_wallet(native.owner)["balance_cad"]
    original = native.repo.consume_endpoint_token_cost
    monkeypatch.setattr(
        native.repo,
        "consume_endpoint_token_cost",
        mock.Mock(side_effect=RuntimeError("ledger fault")),
    )
    with pytest.raises(RuntimeError):
        charge_serverless_worker(native.repo, w, endpoint=native.ep, released_at=released)
    assert billing.get_wallet(native.owner)["balance_cad"] == before
    assert native.repo.peek_endpoint_token_cost(native.ep["endpoint_id"]) == pytest.approx(1.25)
    monkeypatch.setattr(native.repo, "consume_endpoint_token_cost", original)
    bill_active_serverless_workers(billing, now=released + 1000)
    assert before - billing.get_wallet(native.owner)["balance_cad"] == pytest.approx(1.25)
    bill_active_serverless_workers(billing, now=released + 2000)
    assert before - billing.get_wallet(native.owner)["balance_cad"] == pytest.approx(1.25)
    with native.pool.connection() as conn:
        row = conn.execute(
            "SELECT count(*),max(period_end) FROM billing_cycles WHERE job_id=%s AND status='charged'",
            (native.jid,),
        ).fetchone()
        assert row[0] == 1 and row[1] == released


def test_overlapping_final_billing_cannot_charge_twice(native):
    start(native)
    billing = get_billing_engine()
    billing.deposit(native.owner, 50)
    native.repo.update_worker(native.wid, allocated_at=time.time() - 130)
    native.repo.accrue_endpoint_token_cost(native.ep["endpoint_id"], 2.5)
    report(native, "succeeded")
    w = native.repo.get_worker(native.wid)

    def charge(_):
        return charge_serverless_worker(
            native.repo, w, endpoint=native.ep, released_at=w["released_at"]
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(charge, range(3)))
    assert sum(bool(r.get("charged")) for r in results) == 1
    assert billing.get_wallet(native.owner)["balance_cad"] == pytest.approx(47.5)


def test_stop_intent_survives_enqueue_failure_and_does_not_force_autoscaler_drain(native):
    start(native)
    native.repo.update_worker(native.wid, state="draining")
    assert native.repo.list_requested_stops() == []
    with mock.patch(
        "control_plane.lifecycle.request_fenced_stop_remove",
        side_effect=RuntimeError("enqueue offline"),
    ):
        with pytest.raises(RuntimeError):
            native.svc.deprovision_worker(native.wid)
    assert native.repo.get_worker(native.wid)["released_at"] is None
    assert native.wid in [w["worker_id"] for w in native.repo.list_requested_stops()]
    assert native.svc.deprovision_worker(native.wid)["requested"]


def test_wrong_worker_binding_is_never_promoted_or_closed(native):
    native.reserve()
    native.repo.update_worker(native.wid, host_id="other-host")
    report(native, "running")
    assert native.repo.get_worker(native.wid)["state"] == "booting"
    report(native, "failed")
    assert native.repo.get_worker(native.wid)["released_at"] is None


@pytest.mark.parametrize("suffix", ["ready", "exited"])
@pytest.mark.parametrize("prefix", ["/api/v2", "/agent/v2"])
def test_old_capacity_callbacks_are_authenticated_tombstones(native, suffix, prefix):
    import os
    from fastapi.testclient import TestClient
    from api import app

    with TestClient(app) as client:
        response = client.post(
            f"{prefix}/serverless/workers/{native.wid}/{suffix}",
            json={},
            headers={"Authorization": f"Bearer {os.environ['XCELSIOR_API_TOKEN']}"},
        )
    assert response.status_code == 410
    w = native.repo.get_worker(native.wid)
    assert w["state"] == "booting" and w["released_at"] is None


def test_late_heartbeats_and_concurrency_callbacks_cannot_revive_released_worker(native):
    start(native)
    report(native, "failed")
    released = native.repo.get_worker(native.wid)["released_at"]
    native.repo.decrement_worker_concurrency(native.wid)
    native.repo.increment_worker_concurrency(native.wid)
    assert native.svc.worker_heartbeat(native.wid) is None
    assert native.svc.worker_claim_job(native.wid) is None
    w = native.repo.get_worker(native.wid)
    assert w["state"] == "error" and w["released_at"] == released
