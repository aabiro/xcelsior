"""Wallet enforcement and estimates use native admission and captured prices."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from psycopg.types.json import Jsonb
from fastapi import HTTPException

from billing import get_billing_engine
from tests.test_serverless_native_lifecycle import native, start  # noqa: F401


@pytest.fixture
def running(native):
    engine = get_billing_engine()
    engine.deposit(native.owner, 50, description="native authority proof")
    with native.pool.connection() as conn:
        conn.execute(
            "UPDATE hosts SET payload=payload||%s WHERE host_id=%s",
            (Jsonb({"cost_per_hour": 1.8}), native.hid),
        )
    start(native)
    return native, engine


@pytest.mark.parametrize("native", [{"billable": True}], indirect=True)
def test_depletion_uses_captured_rate_and_admitted_owner_after_drift(running):
    x, engine = running
    with x.pool.connection() as conn:
        conn.execute(
            "UPDATE hosts SET payload=payload||%s WHERE host_id=%s",
            (Jsonb({"cost_per_hour": 999}), x.hid),
        )
        conn.execute(
            "UPDATE jobs SET payload=payload||%s WHERE job_id=%s",
            (Jsonb({"owner": "forged-owner", "gpu_model": "forged-gpu"}), x.jid),
        )
    result = engine.time_to_zero(x.owner)
    assert result["estimate_complete"] and result["unpriced_running_instances"] == 0
    assert result["running_instances"] == 1 and result["burn_rate_per_hour"] == 1.8
    assert result["instance_burns"][0]["gpu_model"] == x.jid
    assert engine.time_to_zero("forged-owner")["running_instances"] == 0


@pytest.mark.parametrize("native", [{"billable": True}], indirect=True)
def test_depletion_reports_missing_quote_instead_of_fabricating_rate(running):
    x, engine = running
    with x.pool.connection() as conn:
        conn.execute(
            "UPDATE job_attempts SET placement_explanation=placement_explanation-'billing_snapshot' WHERE attempt_id=%s",
            (x.r.attempt_id,),
        )
    result = engine.time_to_zero(x.owner)
    assert not result["estimate_complete"] and result["unpriced_running_instances"] == 1
    assert result["seconds_to_zero"] is None and result["instance_burns"] == []


def test_unadmitted_storage_observations_cannot_debit_wallet(native):
    engine = get_billing_engine()
    engine.deposit(native.owner, 50, description="storage observation refusal")
    with native.pool.connection() as conn:
        conn.execute(
            """UPDATE jobs SET spec=NULL,spec_hash=NULL,status='stopped',phase='stopped',
            desired_state='stopped',payload=payload||%s||jsonb_build_object('stopped_at',EXTRACT(EPOCH FROM clock_timestamp())-3600)
            WHERE job_id=%s""",
            (
                Jsonb(
                    {
                        "owner": native.owner,
                        "storage_gb": 10000,
                        "storage_type": "hdd",
                        "storage_rate_cad_per_gb_hr": 100,
                    }
                ),
                native.jid,
            ),
        )
    before = engine.get_wallet(native.owner)["balance_micros"]
    result = engine.auto_billing_cycle()
    assert result["storage_billed"] == 0
    assert engine.get_wallet(native.owner)["balance_micros"] == before
    with native.pool.connection() as conn:
        assert conn.execute(
            "SELECT count(*) FROM billing_cycles WHERE job_id=%s", (native.jid,)
        ).fetchone() == (0,)


@pytest.mark.parametrize("native", [{"billable": True}], indirect=True)
def test_suspended_wallet_enforcement_ignores_forged_observation_owner(running):
    x, engine = running
    other = x.owner + "-suspended"
    engine.get_wallet(other)
    with x.pool.connection() as conn:
        conn.execute("UPDATE wallets SET status='suspended' WHERE customer_id=%s", (other,))
        conn.execute(
            "UPDATE jobs SET payload=payload||%s WHERE job_id=%s", (Jsonb({"owner": other}), x.jid)
        )
    engine.stop_jobs_for_suspended_wallets()
    with x.pool.connection() as conn:
        assert conn.execute("SELECT status FROM jobs WHERE job_id=%s", (x.jid,)).fetchone() == (
            "running",
        )
        assert conn.execute(
            "SELECT count(*) FROM agent_commands WHERE job_id=%s AND command='stop_attempt'",
            (x.jid,),
        ).fetchone() == (0,)


@pytest.mark.parametrize("native", [{"billable": True}], indirect=True)
def test_reset_routes_through_fenced_restart_and_never_host_reset(running, monkeypatch):
    import routes.instances as routes
    import routes.agent as agent_routes

    x, _ = running
    monkeypatch.setattr(
        routes,
        "_authorize_instance_mutation",
        lambda *args: ({"customer_id": x.owner}, {"status": "running"}),
    )
    monkeypatch.setattr(routes, "_check_not_locked", lambda job: None)
    monkeypatch.setattr(routes, "_broadcast_instance_lifecycle_sse", Mock())
    monkeypatch.setattr(routes, "append_user_audit_event", Mock())
    host = Mock(side_effect=AssertionError("unfenced reset"))
    monkeypatch.setattr(agent_routes, "enqueue_agent_command", host)
    result = routes.api_reset_instance(x.jid, SimpleNamespace())
    assert result["ok"] and result["instance"]["status"] == "stopping"
    with x.pool.connection() as conn:
        command, args = conn.execute(
            "SELECT command,args FROM agent_commands WHERE job_id=%s AND command='stop_attempt'",
            (x.jid,),
        ).fetchone()
    assert command == "stop_attempt" and args["intent"] == "restart" and not args["preserve"]
    host.assert_not_called()
