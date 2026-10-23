"""Approved native operator eviction, real PostgreSQL and existing HTTP auth."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import uuid

import pytest
import psycopg
from sqlalchemy import create_engine
from psycopg.conninfo import make_conninfo

from tests.test_serverless_native_lifecycle import native, start  # noqa: F401
from tests.test_db_service_roles import provisioned_roles  # noqa: F401
from control_plane import host_evictions as evictions
from control_plane.attempts import report_attempt_status
from control_plane.db import run_transaction
from control_plane.launch import action_plans
from control_plane.launch.service import PlanConflict
from tests.test_control_plane_reservation import fleet, _mkhost, _mkjob, _claim, _pool  # noqa: F401
from tests.test_durable_preemption import running
from tests.test_control_plane_v1_drain_evict import (
    _auth_env,
    _admin_headers,
    _OWNED,
    client,
)  # noqa: F401


def setup(fleet, *, count=1, fields=None):
    host = _mkhost(fleet, gpus=count)
    active = [running(_claim(_mkjob(fleet, admission_fields=fields)), host) for _ in range(count)]
    return host, active, approved_plan(host)


def approved_plan(host, *, advance_version=True):
    with _pool.connection() as conn:
        if advance_version:
            conn.execute(
                "UPDATE hosts SET status='draining',version=version+1 WHERE host_id=%s", (host,)
            )
        version = conn.execute("SELECT version FROM hosts WHERE host_id=%s", (host,)).fetchone()[0]
        canonical = dict(
            host_id=host,
            expected_version=version,
            reason="maintenance",
            policy=evictions.POLICY,
            grace_seconds=60,
            allow_force=False,
            victims=evictions.snapshot(conn, host_id=host, expected_version=version),
        )
        digest = evictions.canonical_hash(canonical)
        plan = action_plans.create_quoted_plan(
            conn,
            action_type=evictions.ACTION,
            principal_id="operator",
            client_id=None,
            tenant_id="operator-tenant",
            team_id=None,
            canonical_args=canonical,
            canonical_args_hash=digest,
            spec_hash=digest,
            quote_id="test-eviction",
            pricing_version="not-applicable",
            estimate_micros=0,
            currency="CAD",
            price_tolerance_bps=0,
            required_scopes=["hosts:evict"],
            approval_mode="human",
            ttl_sec=300,
        )
        _OWNED["plans"].add(str(plan["plan_id"]))
        plan = action_plans.mark_approved(
            conn,
            str(plan["plan_id"]),
            approved_by="human",
            approval_method="human",
            approval_session_id=None,
        )
    return plan


def execute(plan, host):
    return run_transaction(
        lambda conn: evictions.execute(
            conn,
            action_plans.get_plan_for_update(conn, str(plan["plan_id"])),
            host_id=host,
            created_by="operator",
        ),
        isolation_level="SERIALIZABLE",
    )


def settle(active, *, intent=None):
    def transaction(conn):
        if intent:
            conn.execute(
                "UPDATE jobs SET desired_state='stopped',lifecycle_intent=%s WHERE job_id=%s",
                (intent, active.job_id),
            )
        return report_attempt_status(
            conn,
            job_id=active.job_id,
            attempt_id=active.attempt_id,
            host_id=active.host_id,
            fencing_token=active.fencing_token,
            status="stopped",
        )

    return run_transaction(transaction)


def commands(active):
    with _pool.connection() as conn:
        return conn.execute(
            "SELECT command_id,args FROM agent_commands WHERE attempt_id=%s AND command='stop_attempt'",
            (active.attempt_id,),
        ).fetchall()


def test_exact_native_targets_pending_until_terminal_release_and_restart_sweep(fleet):
    host, active, plan = setup(fleet, count=2)
    result = execute(plan, host)
    assert result["pending"] and len(result["command_ids"]) == 2
    assert execute(plan, host)["command_ids"] == result["command_ids"]
    for attempt in active:
        rows = commands(attempt)
        assert len(rows) == 1
        assert rows[0][1]["force"] is False and rows[0][1]["allow_force"] is False
        assert rows[0][1]["preemption"] is True
    with _pool.connection() as conn:
        # ACK, elapsed grace and a scheduler restart are not release evidence.
        conn.execute(
            "UPDATE agent_commands SET status='acknowledged' WHERE command_id=ANY(%s::uuid[])",
            (result["command_ids"],),
        )
        conn.execute(
            "UPDATE action_plans SET expires_at=now()-interval '1 day' WHERE plan_id=%s",
            (plan["plan_id"],),
        )
        assert evictions.sweep(conn) >= 1
        assert conn.execute(
            "SELECT count(*) FROM gpu_device_allocations WHERE host_id=%s AND released_at IS NULL",
            (host,),
        ).fetchone() == (2,)
        assert conn.execute(
            "SELECT status FROM action_plans WHERE plan_id=%s", (plan["plan_id"],)
        ).fetchone() == ("executing",)
    settle(active[0])
    assert execute(plan, host)["pending"]
    settle(active[1])
    run_transaction(lambda conn: evictions.sweep(conn))
    done = execute(plan, host)
    assert done["status"] == "succeeded" and not done["pending"] and done["idempotent"]
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT status,host_id,active_attempt_id,lifecycle_intent FROM jobs WHERE job_id=%s",
            (active[0].job_id,),
        ).fetchone() == ("queued", None, None, None)
        assert conn.execute(
            "SELECT job_id,resulting_resource_id FROM action_plans WHERE plan_id=%s",
            (plan["plan_id"],),
        ).fetchone() == (None, host)
        assert conn.execute(
            "SELECT count(*) FROM outbox_events WHERE idempotency_key=%s",
            (f'host-eviction-settled:{plan["plan_id"]}',),
        ).fetchone() == (1,)


@pytest.mark.parametrize(
    "intent,expected", [("stop", "stopped"), ("cancel", "cancelled"), ("terminate", "terminated")]
)
def test_later_user_intent_takes_precedence(fleet, intent, expected):
    host, active, plan = setup(fleet)
    execute(plan, host)
    settle(active[0], intent=intent)
    assert not execute(plan, host)["pending"]
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT status,lifecycle_intent FROM jobs WHERE job_id=%s", (active[0].job_id,)
        ).fetchone() == (expected, None)


@pytest.mark.parametrize(
    "change,code",
    [
        ("expired", "plan_expired"),
        ("unapproved", "approval_required"),
        ("hash", "argument_hash_mismatch"),
        ("undrain", "host_not_draining"),
        ("version", "version_conflict"),
        ("generation", "eviction_authority_unavailable"),
        ("lease_expired", "eviction_authority_unavailable"),
        ("user_stop", "eviction_authority_unavailable"),
    ],
)
def test_refusals_leave_no_stop_command_or_consumption(fleet, change, code):
    host, active, plan = setup(fleet)
    with _pool.connection() as conn:
        if change == "expired":
            conn.execute(
                "UPDATE action_plans SET expires_at=now()-interval '1 minute' WHERE plan_id=%s",
                (plan["plan_id"],),
            )
        elif change == "unapproved":
            conn.execute(
                "UPDATE action_plans SET status='quoted' WHERE plan_id=%s", (plan["plan_id"],)
            )
        elif change == "hash":
            conn.execute(
                "UPDATE action_plans SET canonical_args=canonical_args||'{\"grace_seconds\":1}' WHERE plan_id=%s",
                (plan["plan_id"],),
            )
        elif change == "undrain":
            conn.execute("UPDATE hosts SET status='active' WHERE host_id=%s", (host,))
        elif change == "version":
            conn.execute("UPDATE hosts SET version=version+1 WHERE host_id=%s", (host,))
        elif change == "generation":
            conn.execute(
                "UPDATE jobs SET generation=generation+1 WHERE job_id=%s", (active[0].job_id,)
            )
        elif change == "lease_expired":
            conn.execute(
                "UPDATE placement_leases SET expires_at=now()-interval '1 minute' WHERE lease_id=%s",
                (active[0].lease_id,),
            )
        elif change == "user_stop":
            conn.execute(
                "UPDATE jobs SET lifecycle_intent='stop',desired_state='stopped' WHERE job_id=%s",
                (active[0].job_id,),
            )
    with pytest.raises(PlanConflict) as error:
        execute(plan, host)
    assert error.value.code == code
    assert not commands(active[0])
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT consumed_at FROM action_plans WHERE plan_id=%s", (plan["plan_id"],)
        ).fetchone() == (None,)


def test_partial_command_failure_rolls_back_every_effect(fleet, monkeypatch):
    host, active, plan = setup(fleet, count=2)
    original = evictions.enqueue_current_attempt_command
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(kwargs["job_id"])
        if len(calls) == 2:
            raise RuntimeError("injected second enqueue failure")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(evictions, "enqueue_current_attempt_command", fail_second)
        with pytest.raises(RuntimeError, match="injected"):
            execute(plan, host)
    assert all(not commands(a) for a in active)
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT status,consumed_at FROM action_plans WHERE plan_id=%s", (plan["plan_id"],)
        ).fetchone() == ("approved", None)
        assert conn.execute(
            "SELECT count(*) FROM jobs WHERE host_id=%s AND reason_code='operator_eviction_requested'",
            (host,),
        ).fetchone() == (0,)
    assert execute(plan, host)["pending"]


def test_two_executors_collapse_on_one_durable_command_set(fleet):
    host, active, plan = setup(fleet)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: execute(plan, host), range(2)))
    assert outcomes[0]["command_ids"] == outcomes[1]["command_ids"]
    assert len(commands(active[0])) == 1


def test_changed_victim_set_requires_new_approval(fleet):
    host, active, plan = setup(fleet)
    settle(active[0])
    with pytest.raises(PlanConflict) as error:
        execute(plan, host)
    assert error.value.code == "eviction_snapshot_changed"
    assert not commands(active[0])


def test_completed_host_result_downgrade_refuses_evidence_loss(fleet):
    host, active, plan = setup(fleet)
    execute(plan, host)
    settle(active[0])
    execute(plan, host)
    path = (
        Path(__file__).resolve().parents[1]
        / "api/migrations/versions/137_allow_native_host_eviction_results.py"
    )
    spec = importlib.util.spec_from_file_location("eviction_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = create_engine(
        os.environ["XCELSIOR_TEST_POSTGRES_DSN"].replace("postgresql:", "postgresql+psycopg:", 1)
    )
    try:
        with pytest.raises(RuntimeError, match="downgrade refused"):
            with engine.begin() as conn:
                migration._constraint(conn, allow_eviction=False)
    finally:
        engine.dispose()
    assert execute(plan, host)["status"] == "succeeded"


def test_http_preview_approval_execute_and_scoped_progress(fleet):
    admin = _admin_headers(f"native-evict-{uuid.uuid4().hex[:8]}@xcelsior.ca")
    host = _mkhost(fleet)
    active = running(_claim(_mkjob(fleet)), host)
    assert client.post(f"/api/v1/hosts/{host}/drain", headers=admin, json={}).status_code == 200
    with _pool.connection() as conn:
        version = conn.execute("SELECT version FROM hosts WHERE host_id=%s", (host,)).fetchone()[0]
    preview = client.post(
        f"/api/v1/hosts/{host}/eviction-plans",
        headers=admin,
        json=dict(expected_version=version, reason="maintenance", grace_seconds=30),
    )
    assert preview.status_code == 200, preview.text
    pid = preview.json()["plan_id"]
    _OWNED["plans"].add(pid)
    assert preview.json()["canonical_spec"]["victims"][0]["attempt_id"] == active.attempt_id
    url = f"/api/v1/hosts/{host}/eviction-plans/{pid}"
    assert client.post(url + "/execute", headers=admin).json()["code"] == "approval_required"
    assert (
        client.post(
            f"/api/v1/launch-plans/{pid}/approve", headers=admin, json={"confirm": True}
        ).status_code
        == 200
    )
    # An operator approval cannot be consumed through the generic launch API.
    wrong_action = client.post(
        f"/api/v1/launch-plans/{pid}/execute", headers=admin, json={"confirm": True}
    )
    assert wrong_action.status_code == 409 and wrong_action.json()["code"] == "wrong_action_type"
    response = client.post(url + "/execute", headers=admin)
    assert response.status_code == 200 and response.json()["pending"], response.text
    assert client.get(url, headers=admin).json()["execution"]["pending"]
    other = _admin_headers(f"native-evict-other-{uuid.uuid4().hex[:8]}@xcelsior.ca")
    assert client.get(url, headers=other).status_code == 404
    assert client.post(url + "/execute", headers=other).status_code == 404
    wrong = f"/api/v1/hosts/other/eviction-plans/{pid}/execute"
    assert client.post(wrong, headers=admin).json()["code"] == "resource_mismatch"
    settle(active)
    done = client.post(url + "/execute", headers=admin)
    assert done.status_code == 200 and not done.json()["pending"], done.text
    replay_as_launch = client.post(
        f"/api/v1/launch-plans/{pid}/execute", headers=admin, json={"confirm": True}
    )
    assert (
        replay_as_launch.status_code == 409
        and replay_as_launch.json()["code"] == "wrong_action_type"
    )


def test_serverless_eviction_closes_worker_identity_without_requeuing(native):
    active = start(native)
    plan = approved_plan(native.hid)
    assert plan["canonical_args"]["victims"][0]["settlement"] == "terminate"
    assert execute(plan, native.hid)["pending"]
    assert native.repo.get_worker(native.wid)["released_at"] is None
    settle(active)
    assert not execute(plan, native.hid)["pending"]
    worker = native.repo.get_worker(native.wid)
    assert worker["state"] == "terminated" and worker["released_at"] is not None
    with native.pool.connection() as conn:
        assert conn.execute(
            "SELECT status,active_attempt_id FROM jobs WHERE job_id=%s", (native.jid,)
        ).fetchone() == ("terminated", None)


def test_restricted_scheduler_login_can_finish_settled_host_action(fleet, provisioned_roles):
    host, active, plan = setup(fleet)
    execute(plan, host)
    settle(active[0])
    with _pool.connection() as conn:
        conn.execute("ALTER ROLE xcelsior_scheduler LOGIN PASSWORD 'isolated-host-eviction-test'")
    dsn = make_conninfo(
        os.environ["XCELSIOR_TEST_POSTGRES_DSN"],
        user="xcelsior_scheduler",
        password="isolated-host-eviction-test",
    )
    try:
        with psycopg.connect(dsn) as conn:
            assert conn.execute("SELECT current_user,session_user").fetchone() == (
                "xcelsior_scheduler",
                "xcelsior_scheduler",
            )
            assert evictions.sweep(conn) >= 1
    finally:
        with _pool.connection() as conn:
            conn.execute("ALTER ROLE xcelsior_scheduler NOLOGIN PASSWORD NULL")
    assert execute(plan, host)["status"] == "succeeded"


def test_native_drain_uses_column_identity_and_locked_version(fleet):
    from control_plane.host_lifecycle import set_draining

    host = _mkhost(fleet)
    with _pool.connection() as conn:
        original = conn.execute(
            "SELECT payload,version FROM hosts WHERE host_id=%s", (host,)
        ).fetchone()
        assert "host_id" not in original[0]
    drained = run_transaction(
        lambda c: set_draining(c, host_id=host, draining=True, expected_version=original[1])
    )
    assert drained["host_id"] == host and drained["status"] == "draining"
    with pytest.raises(PlanConflict) as error:
        run_transaction(
            lambda c: set_draining(c, host_id=host, draining=False, expected_version=original[1])
        )
    assert error.value.code == "version_conflict"
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT status,administrative_state,version FROM hosts WHERE host_id=%s", (host,)
        ).fetchone() == ("draining", "draining", original[1] + 1)
    undrained = run_transaction(
        lambda c: set_draining(c, host_id=host, draining=False, expected_version=drained["version"])
    )
    assert undrained["status"] == "active" and undrained["admitted"]


def test_two_approved_plans_cannot_evict_the_same_host_snapshot(fleet):
    host, active, first = setup(fleet)
    second = approved_plan(host, advance_version=False)
    assert first["canonical_args"] == second["canonical_args"]
    assert execute(first, host)["pending"]
    with pytest.raises(PlanConflict) as error:
        execute(second, host)
    assert error.value.code == "version_conflict"
    with pytest.raises(PlanConflict) as error:
        approved_plan(host, advance_version=False)
    assert error.value.code == "eviction_in_progress"
    assert len(commands(active[0])) == 1
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT status,consumed_at FROM action_plans WHERE plan_id=%s", (second["plan_id"],)
        ).fetchone() == ("approved", None)
    settle(active[0])
    assert not execute(first, host)["pending"]
    with pytest.raises(PlanConflict):
        execute(second, host)


def test_concurrent_distinct_approved_plans_only_enqueue_one_stop_set(fleet):
    host, active, first = setup(fleet)
    second = approved_plan(host, advance_version=False)

    def attempt(plan):
        try:
            return execute(plan, host)
        except PlanConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, [first, second]))
    assert sum(r is not None for r in results) == 1
    assert len(commands(active[0])) == 1
