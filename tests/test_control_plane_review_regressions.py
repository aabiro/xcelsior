"""Review findings reproduced against real native placement and billing transactions."""

import json
import os
import time
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb

from billing import get_billing_engine
from serverless.metering import charge_serverless_execution
from tests.test_serverless_native_lifecycle import native, start  # noqa: F401


def test_stop_reresolves_attempt_after_cancel_loses_placement_race(native):
    from control_plane.serverless_workers import cancel_unstarted_worker

    def lost_race(worker_id, job_id):
        start(native)
        return cancel_unstarted_worker(worker_id, job_id)

    with mock.patch(
        "control_plane.serverless_workers.cancel_unstarted_worker", side_effect=lost_race
    ):
        result = native.svc.deprovision_worker(native.wid)
    assert result["requested"] and result["pending"]
    worker = native.repo.get_worker(native.wid)
    assert worker["state"] == "draining" and worker["released_at"] is None
    assert native.repo.list_requested_stops()[0]["worker_id"] == native.wid
    with native.pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM agent_commands WHERE job_id=%s AND attempt_id=%s AND command='stop_attempt'",
                (native.jid, native.r.attempt_id),
            ).fetchone()[0]
            == 1
        )


@pytest.mark.parametrize("evidence", ["none", "allocated", "heartbeat", "host", "ready"])
def test_only_provably_unstarted_orphans_are_released(native, evidence):
    with native.pool.connection() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (native.jid,))
    fields = {
        "allocated": {"allocated_at": time.time()},
        "heartbeat": {"last_heartbeat_at": time.time()},
        "host": {"host_id": native.hid},
        "ready": {"state": "ready"},
    }
    if evidence in fields:
        native.repo.update_worker(native.wid, **fields[evidence])
    result = native.svc.deprovision_worker(native.wid)
    worker = native.repo.get_worker(native.wid)
    if evidence == "none":
        assert result == {"requested": True, "pending": False}
        assert worker["state"] == "terminated" and worker["released_at"] is not None
        assert worker["allocated_at"] == 0
    else:
        assert not result["requested"] and worker["released_at"] is None


@pytest.mark.parametrize("after_insert", [False, True])
def test_provisioning_failure_rolls_back_worker_job_and_submission_event(
    native, monkeypatch, after_insert
):
    import scheduler

    real_submit = scheduler.submit_job
    attempted = []

    def fail_after_insert(**kwargs):
        attempted.append(kwargs["job_id"])
        if after_insert:
            real_submit(**kwargs)
        # A second replica cannot see either half of in-flight provisioning.
        with native.pool.connection() as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM serverless_workers WHERE scheduler_job_id=%s",
                    (kwargs["job_id"],),
                ).fetchone()[0]
                == 0
            )
            assert (
                conn.execute(
                    "SELECT count(*) FROM jobs WHERE job_id=%s", (kwargs["job_id"],)
                ).fetchone()[0]
                == 0
            )
        raise RuntimeError("crash after job insert before commit")

    monkeypatch.setattr(scheduler, "submit_job", fail_after_insert)
    assert native.svc.provision_worker(native.ep["endpoint_id"], billable=False) is None
    assert len(attempted) == 1
    with native.pool.connection() as conn:
        for table, column in [
            ("jobs", "job_id"),
            ("serverless_workers", "scheduler_job_id"),
            ("outbox_events", "aggregate_id"),
        ]:
            assert (
                conn.execute(
                    f"SELECT count(*) FROM {table} WHERE {column}=%s", (attempted[0],)
                ).fetchone()[0]
                == 0
            )


def test_callback_mint_miss_rolls_back_claim_and_can_retry_same_offer(native, monkeypatch):
    import api

    r = native.reserve()
    body = {
        "lease_id": r.lease_id,
        "job_id": r.job_id,
        "attempt_id": r.attempt_id,
        "host_id": r.host_id,
        "fencing_token": r.fencing_token,
        "worker_session_id": "review-test",
    }
    headers = {"Authorization": f"Bearer {os.environ['XCELSIOR_API_TOKEN']}"}
    with TestClient(api.app) as client:
        with mock.patch(
            "control_plane.serverless_credentials.mint_callback_credential", return_value=None
        ):
            response = client.post("/agent/v2/leases/claim", json=body, headers=headers)
        assert response.status_code == 409, response.text
        with native.pool.connection() as conn:
            assert (
                conn.execute(
                    "SELECT status FROM placement_leases WHERE lease_id=%s", (r.lease_id,)
                ).fetchone()[0]
                == "offered"
            )
            assert (
                conn.execute(
                    "SELECT status FROM job_attempts WHERE attempt_id=%s", (r.attempt_id,)
                ).fetchone()[0]
                == "reserved"
            )
            assert (
                conn.execute(
                    "SELECT count(*) FROM serverless_attempt_credentials WHERE attempt_id=%s",
                    (r.attempt_id,),
                ).fetchone()[0]
                == 0
            )
        response = client.post("/agent/v2/leases/claim", json=body, headers=headers)
        assert response.status_code == 200, response.text
        assert response.json()["callback_token"].startswith("xsw_")


def notices(native):
    with native.pool.connection() as conn:
        return conn.execute(
            "SELECT type FROM notifications WHERE user_email=%s ORDER BY type", (native.owner,)
        ).fetchall()


def test_wallet_notices_rollback_with_cycle_then_commit_once_and_push_after_commit(
    native, monkeypatch
):
    from db import NotificationStore

    billing = get_billing_engine()
    billing.deposit(native.owner, 1)
    native.repo.update_worker(native.wid, allocated_at=1000)
    worker = native.repo.get_worker(native.wid)
    monkeypatch.setattr("serverless.metering.get_gpu_rate_per_hour", lambda *a: 3.6)
    original = native.repo.increment_endpoint_totals
    monkeypatch.setattr(
        native.repo,
        "increment_endpoint_totals",
        mock.Mock(side_effect=RuntimeError("cycle rollback")),
    )
    pushes = []

    def delivered(nid):
        with native.pool.connection() as conn:
            assert conn.execute("SELECT id FROM notifications WHERE id=%s", (nid,)).fetchone()
        pushes.append(nid)

    monkeypatch.setattr(NotificationStore, "deliver", delivered)
    with pytest.raises(RuntimeError, match="cycle rollback"):
        charge_serverless_execution(
            billing, native.repo, worker, native.ep, period_end=1100, final=True
        )
    assert billing.get_wallet(native.owner)["balance_cad"] == 1
    assert notices(native) == [] and pushes == []
    monkeypatch.setattr(native.repo, "increment_endpoint_totals", original)
    result = charge_serverless_execution(
        billing, native.repo, worker, native.ep, period_end=1100, final=True
    )
    assert result["charged"] and notices(native) == [("billing_low_balance",)]
    assert len(pushes) == 1
    charge_serverless_execution(
        billing, native.repo, worker, native.ep, period_end=1100, final=True
    )
    assert len(pushes) == 1
    result = charge_serverless_execution(
        billing, native.repo, worker, native.ep, period_end=2000, final=True
    )
    assert result["charged"] and result["balance_cad"] == 0
    assert notices(native) == [("billing_low_balance",), ("billing_suspended",)]
    assert len(pushes) == 2
    charge_serverless_execution(
        billing, native.repo, worker, native.ep, period_end=2100, final=True
    )
    assert len(pushes) == 2


def test_control_plane_dashboard_keeps_shared_heartbeat_counts_per_service():
    dashboard = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "api/infra/observability/grafana/dashboards/xcelsior-control-plane.json"
        ).read_text()
    )
    counts = [
        target["expr"]
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
        if "xcelsior_service_heartbeat_fresh_replicas" in target.get("expr", "")
    ]
    assert counts == ['xcelsior_service_heartbeat_fresh_replicas{job="xcelsior-api"}']
    assert not any(panel["title"] == "Services" for panel in dashboard["panels"])


@pytest.mark.parametrize("state", ["booting", "error"])
def test_reaper_releases_old_missing_submissions_and_replenishes_capacity(
    native, monkeypatch, state
):
    from serverless.reaper import BOOTING_TTL_SEC, reap_all

    with native.pool.connection() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (native.jid,))
    native.repo.update_worker(native.wid, state=state)
    cutoff = time.time() - BOOTING_TTL_SEC
    assert native.wid not in [
        w["worker_id"] for w in native.repo.list_stuck_booting_workers(booting_before=cutoff)
    ]
    with native.pool.connection() as conn:
        conn.execute(
            "UPDATE serverless_workers SET created_at=%s WHERE worker_id=%s",
            (cutoff - 1, native.wid),
        )
        conn.execute(
            "UPDATE serverless_endpoints SET min_workers=1 WHERE endpoint_id=%s",
            (native.ep["endpoint_id"],),
        )
    replenish = mock.Mock(return_value=None)
    monkeypatch.setattr(native.svc, "provision_worker", replenish)
    result = reap_all(native.svc)
    assert result["reprovisioned"] == 1
    replenish.assert_called_once_with(native.ep["endpoint_id"])
    assert native.repo.get_worker(native.wid)["released_at"] is not None
    assert native.repo.count_active_workers(native.ep["endpoint_id"]) == 0


@pytest.mark.parametrize("initial_balance", [0, 1])
def test_insufficient_wallet_warns_without_debiting_and_suspends_only_when_empty(
    native, monkeypatch, initial_balance
):
    billing = get_billing_engine()
    billing.get_wallet(native.owner)
    if initial_balance:
        billing.deposit(native.owner, initial_balance)
    native.repo.update_worker(native.wid, allocated_at=1000)
    worker = native.repo.get_worker(native.wid)
    monkeypatch.setattr("serverless.metering.get_gpu_rate_per_hour", lambda *a: 3.6)
    monkeypatch.setattr("db.NotificationStore.deliver", mock.Mock())
    for _ in range(2):
        result = charge_serverless_execution(
            billing, native.repo, worker, native.ep, period_end=3000, final=True
        )
        assert not result["charged"]
        wallet = billing.get_wallet(native.owner)
        assert wallet["balance_cad"] == initial_balance
        assert (wallet["status"] == "suspended") == (initial_balance == 0)
        expected = [("billing_low_balance",)]
        if initial_balance == 0:
            expected.append(("billing_suspended",))
        assert notices(native) == expected
