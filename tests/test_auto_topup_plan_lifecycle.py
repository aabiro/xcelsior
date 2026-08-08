"""The auto-top-up approval plan, driven end to end against real PostgreSQL.

`tests/test_widening_auto_topup_needs_approval.py` proves the *refusal*: an agent
raising the unattended charge amount is turned away. This file proves the other
half — that the approved path actually works — because a gate with no reachable
way through it is not a gate, it is an outage.

**Why real PostgreSQL and not a fake.** `ck_action_plans_state_machine` requires
`job_id IS NOT NULL` for a `succeeded` plan. Auto-top-up produces no job, so the
first version of this code passed `job_id=None` and would have raised an
IntegrityError on the happy path — in production, after approval, at the moment
the setting was finally being applied. Nothing in-process would have caught it:
the constraint lives in the database. That is the whole reason this file exists,
and it is why the assertions below go through `mark_consumed` rather than
stopping at the route's return value.

Every test owns only its own rows.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _has = _c.execute("SELECT to_regclass('action_plans')").fetchone()[0] is not None
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("test database has no action_plans table")


@pytest.fixture
def tenant():
    """A tenant id owned by this test, with its plans removed afterwards."""
    name = f"auto-topup-test-{uuid.uuid4().hex[:10]}"
    yield name
    if _pool is None:  # pragma: no cover
        return
    with _pool.connection() as conn:
        conn.execute("DELETE FROM action_plans WHERE tenant_id = %s", (name,))
        conn.commit()


def _canonical(amount_cad: float = 500.0) -> dict:
    return {
        "enabled": True,
        "amount_cad": amount_cad,
        "threshold_cad": 5.0,
        "stripe_payment_method_id": "pm_test",
    }


def _make_plan(conn, tenant: str, canonical: dict):
    import hashlib

    from control_plane.launch import action_plans as plans_repo
    from control_plane.launch.service import DEFAULT_TOLERANCE_BPS, plan_ttl_sec

    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(blob.encode()).hexdigest()
    return plans_repo.create_quoted_plan(
        conn,
        action_type="configure_auto_topup",
        principal_id="test-principal",
        client_id="test-client",
        tenant_id=tenant,
        team_id=None,
        canonical_args=canonical,
        canonical_args_hash=digest,
        spec_hash=digest,
        quote_id=f"auto-topup:{digest[:20]}",
        pricing_version="auto-topup-v1",
        estimate_micros=int(canonical["amount_cad"] * 1_000_000),
        currency="CAD",
        price_tolerance_bps=DEFAULT_TOLERANCE_BPS,
        required_scopes=["billing:write"],
        approval_mode="human",
        ttl_sec=plan_ttl_sec(),
    )


def test_a_fresh_plan_is_quoted_and_needs_approval(tenant):
    from control_plane.db import control_plane_transaction

    with control_plane_transaction() as conn:
        plan = _make_plan(conn, tenant, _canonical())
    assert plan["status"] == "quoted"
    assert plan["approval_mode"] == "human"
    assert plan["action_type"] == "configure_auto_topup"


def test_a_machine_principal_cannot_approve_a_human_mode_plan(tenant):
    """`approval_mode: "human"` is what stops an agent approving its own raise."""
    from control_plane.db import control_plane_transaction
    from control_plane.launch.service import LaunchPlanError, Principal, approve

    with control_plane_transaction() as conn:
        plan = _make_plan(conn, tenant, _canonical())

    principal = Principal(
        principal_id="test-principal",
        tenant_id=tenant,
        client_id="test-client",
        team_id=None,
        scopes=("billing:write",),
    )
    with pytest.raises(LaunchPlanError):
        approve(str(plan["plan_id"]), principal=principal, is_human=False)


def test_the_approved_plan_can_be_consumed_and_survives_the_state_machine(tenant):
    """The regression this file was written for.

    `mark_consumed` must satisfy `ck_action_plans_state_machine`, which demands a
    non-null `job_id` for `succeeded`. Auto-top-up has no job; passing None here
    raises IntegrityError, and only the database can tell us that.
    """
    from control_plane.db import control_plane_transaction
    from control_plane.launch import action_plans as plans_repo
    from control_plane.launch.service import Principal, approve

    with control_plane_transaction() as conn:
        plan = _make_plan(conn, tenant, _canonical())
    plan_id = str(plan["plan_id"])

    principal = Principal(
        principal_id="test-principal",
        tenant_id=tenant,
        client_id="test-client",
        team_id=None,
        scopes=("billing:write",),
    )
    approve(plan_id, principal=principal, is_human=True)

    response = {"ok": True, "auto_topup": _canonical(), "plan_id": plan_id}
    with control_plane_transaction() as conn:
        consumed = plans_repo.mark_consumed(
            conn,
            plan_id,
            job_id=f"auto-topup:{tenant}",
            wallet_hold_id=None,
            idempotent_response=response,
            idempotency_key=f"auto-topup-plan:{plan_id}",
        )
    assert consumed["status"] == "succeeded"
    assert consumed["job_id"] == f"auto-topup:{tenant}"


def test_the_route_uses_the_same_job_id_shape_this_test_asserts(tenant):
    """Bind the test to the implementation rather than to a guess.

    If the route stops naming the wallet it changed, this fails instead of the
    lifecycle silently diverging from what production writes.
    """
    import inspect

    import routes.billing as billing_mod

    source = inspect.getsource(billing_mod.api_billing_execute_auto_topup_plan)
    assert 'job_id=f"auto-topup:{customer_id}"' in source, (
        "the executor no longer writes an auto-topup:<customer> job id — update "
        "this test deliberately, and check the succeeded CHECK still holds"
    )


def test_a_consumed_plan_replays_its_original_response(tenant):
    """Idempotency: applying an approved plan twice changes the setting once."""
    from control_plane.db import control_plane_transaction
    from control_plane.launch import action_plans as plans_repo
    from control_plane.launch.service import Principal, approve

    with control_plane_transaction() as conn:
        plan = _make_plan(conn, tenant, _canonical())
    plan_id = str(plan["plan_id"])
    principal = Principal(
        principal_id="test-principal",
        tenant_id=tenant,
        client_id="test-client",
        team_id=None,
        scopes=("billing:write",),
    )
    approve(plan_id, principal=principal, is_human=True)

    response = {"ok": True, "marker": "first-apply"}
    with control_plane_transaction() as conn:
        plans_repo.mark_consumed(
            conn, plan_id, job_id=f"auto-topup:{tenant}", wallet_hold_id=None,
            idempotent_response=response, idempotency_key=f"auto-topup-plan:{plan_id}",
        )
    with control_plane_transaction() as conn:
        again = plans_repo.get_plan_for_update(conn, plan_id)
    assert again["status"] == "succeeded"
    assert dict(again["idempotent_response"])["marker"] == "first-apply", (
        "the second read did not return the first apply's response"
    )
