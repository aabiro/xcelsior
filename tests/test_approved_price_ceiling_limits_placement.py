"""An approved launch plan's hourly rate is a ceiling on placement and billing.

A plan is quoted before any host is chosen, at the default on-demand rate, and
the customer approves that rate. Nothing wrote it into the admitted spec, so
the reservation price filter, the captured billing snapshot and every billed
slice had no ceiling to check: the job could land on any host and be billed
that host's provider-set price (up to 1000/h).

Execute now admits the plan's approved hourly rate as `max_price_per_hour`.
Placement refuses a host above it; a host below it is placed and billed at its
own, lower rate.
"""

from __future__ import annotations

import time
import uuid

import pytest
from psycopg.types.json import Jsonb

from billing import BillingEngine, get_billing_engine
from control_plane.attempts import report_attempt_status
from control_plane.db import run_transaction
from control_plane.launch.service import Principal, approve, execute, preview
from control_plane.leases import claim_lease
from control_plane.scheduler.claim import claim_next_job
from control_plane.scheduler.reservation import HostNotEligible, reserve_and_bind
from db import _get_pg_pool


@pytest.fixture
def made():
    rows: dict[str, list[str]] = {"jobs": [], "hosts": [], "plans": [], "tenants": []}
    yield rows
    with _get_pg_pool().connection() as conn:
        for job_id in rows["jobs"]:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (job_id,))
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (job_id,))
            conn.execute("DELETE FROM usage_meters WHERE job_id=%s", (job_id,))
            conn.execute("DELETE FROM billing_cycles WHERE job_id=%s", (job_id,))
            conn.execute("UPDATE jobs SET wallet_hold_id=NULL WHERE job_id=%s", (job_id,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (job_id,))
        for host_id in rows["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (host_id,))
        for plan_id in rows["plans"]:
            conn.execute("DELETE FROM action_plans WHERE plan_id=%s", (plan_id,))
        for tenant in rows["tenants"]:
            conn.execute("DELETE FROM wallet_holds WHERE customer_id=%s", (tenant,))
            conn.execute("DELETE FROM wallet_transactions WHERE customer_id=%s", (tenant,))
            conn.execute("DELETE FROM wallets WHERE customer_id=%s", (tenant,))


def _host(made, cost_per_hour: float) -> str:
    host_id = f"ceiling-host-{uuid.uuid4().hex[:8]}"
    with _get_pg_pool().connection() as conn:
        conn.execute(
            """INSERT INTO hosts(host_id,status,registered_at,payload,admission_state,
               administrative_state,availability_state,capabilities) VALUES(%s,'active',%s,%s,
               'admitted','admitted','ready','{"egress_isolation":true}')""",
            (host_id, time.time(), Jsonb({"host_id": host_id, "cost_per_hour": cost_per_hour})),
        )
        conn.execute(
            """INSERT INTO host_gpu_devices(host_id,gpu_uuid,model,total_vram_mb,
               allocatable_vram_mb,health) VALUES(%s,%s,'RTX 4090',24576,24576,'healthy')""",
            (host_id, f"GPU-{host_id}"),
        )
    made["hosts"].append(host_id)
    return host_id


def _place(job_id: str, gpu_model: str, host_id: str):
    def tx(conn):
        claim = claim_next_job(conn, replica_id=job_id, scope_gpu_models=[gpu_model])
        assert claim and claim.job_id == job_id
        return reserve_and_bind(
            conn, job_id=job_id, claim_token=claim.claim_token, replica_id=job_id, host_id=host_id
        )

    return run_transaction(tx)


def test_a_plan_approved_at_twenty_cents_places_and_bills_only_at_or_below_it(made):
    tenant = f"ceiling-{uuid.uuid4().hex[:10]}"
    made["tenants"].append(tenant)
    BillingEngine().deposit(tenant, 100)
    principal = Principal(principal_id="u1", tenant_id=tenant)
    gpu_model = f"ceiling-gpu-{uuid.uuid4().hex[:8]}"
    plan = preview(
        {"name": "ceiling", "num_gpus": 1, "interactive": True, "gpu_model": gpu_model},
        principal=principal,
    )
    made["plans"].append(plan["plan_id"])
    assert plan["estimate"]["hourly_burn_micros"] == 200_000  # approved at 0.20/h
    approve(plan["plan_id"], principal=principal, is_human=True)
    job_id = execute(plan["plan_id"], principal=principal)["job"]["job_id"]
    made["jobs"].append(job_id)

    with _get_pg_pool().connection() as conn:
        spec = conn.execute("SELECT spec FROM jobs WHERE job_id=%s", (job_id,)).fetchone()[0]
    assert spec["max_price_per_hour"] == pytest.approx(
        0.20
    ), "the approved rate must be admitted as the job's price ceiling"
    assert (
        "max_price_per_hour" not in spec["launch_spec"]
    ), "the ceiling sits beside the approved launch spec, so its hash is unchanged"

    expensive = _host(made, 5.00)
    with pytest.raises(HostNotEligible) as refused:
        _place(job_id, gpu_model, expensive)
    assert refused.value.details.get("filter_reason") == "price_exceeds_approved"

    cheap = _host(made, 0.15)
    placed = _place(job_id, gpu_model, cheap)
    auth = dict(
        job_id=job_id,
        host_id=cheap,
        attempt_id=placed.attempt_id,
        fencing_token=placed.fencing_token,
    )
    run_transaction(
        lambda conn: claim_lease(conn, lease_id=placed.lease_id, worker_session_id="c", **auth)
    )
    run_transaction(lambda conn: report_attempt_status(conn, status="running", **auth))
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE job_attempts SET started_at=clock_timestamp()-interval '1 hour' "
            "WHERE attempt_id=%s",
            (placed.attempt_id,),
        )
        started = conn.execute(
            "SELECT EXTRACT(EPOCH FROM started_at)::float8 FROM job_attempts WHERE attempt_id=%s",
            (placed.attempt_id,),
        ).fetchone()[0]
        snapshot = conn.execute(
            "SELECT placement_explanation->'billing_snapshot' FROM job_attempts "
            "WHERE attempt_id=%s",
            (placed.attempt_id,),
        ).fetchone()[0]
    assert snapshot["host"]["cost_per_hour"] == 0.15

    result = get_billing_engine().bill_running_period(job_id, period_end=started + 3600)
    assert result["billed"] is True, result
    with _get_pg_pool().connection() as conn:
        rate, micros = conn.execute(
            "SELECT rate_per_hour,amount_micros FROM billing_cycles WHERE job_id=%s "
            "AND status='charged'",
            (job_id,),
        ).fetchone()
    assert rate == 0.15 and micros == 150_000, "billed at the placed host's 0.15/h"


def test_spot_plans_carry_no_ceiling_and_a_plan_without_a_bound_rate_is_refused():
    from control_plane.launch.service import PlanConflict, approved_price_ceiling

    assert approved_price_ceiling({"canonical_args": {"pricing_mode": "spot"}}) is None
    assert approved_price_ceiling(
        {"canonical_args": {"pricing_mode": "on_demand"}, "pricing_version": "v1:450000:CAD"}
    ) == pytest.approx(0.45)
    with pytest.raises(PlanConflict):
        approved_price_ceiling(
            {"canonical_args": {"pricing_mode": "on_demand"}, "pricing_version": "garbage"}
        )
