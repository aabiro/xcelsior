"""A charge must lead back to the plan that approved it, in one query.

Gate P1: *"An envelope-funded charge is traceable to its approving plan in one
query."*

The schema already carries the links — `action_plans.job_id`,
`action_plans.wallet_hold_id`, `wallet_holds.job_id` — and
`control_plane/launch/service.py` populates them when it consumes a plan. What
was missing is anything that says so. An invariant nobody asserts is an
invariant that holds until someone changes a call site.

**The state machine does not cover this.** `ck_action_plans_state_machine`
requires `job_id` and `idempotent_response` for a `succeeded` plan, and says
nothing about `wallet_hold_id`. A plan can therefore succeed, spend money, and
record no link to the hold that paid for it, without violating any constraint.
Two of the three `mark_consumed` call sites already pass `wallet_hold_id=None` —
correctly, because eviction and serverless-endpoint creation place no hold — so
the value being `None` is normal elsewhere in the codebase and would not look
wrong in review.

That is what makes this worth asserting rather than assuming: the failure is a
`None` in one keyword argument, in a file that already passes `None` for it
twice, guarded by a constraint that permits it.

**Why traceability is the gate and not the ledger.** Money moving is already
guarded — holds, quotes, price tolerance, idempotency. What unattended spend
additionally needs is the ability to answer *"why was this charged?"* after the
fact, without reconstructing a session. One query, from the money to the
approval, with the quote and the approval mode attached.
"""

from __future__ import annotations

import json
import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest  # noqa: E402


@pytest.fixture
def plan_probe():
    """An approved plan and a real wallet hold, removed afterwards.

    The hold has to exist: `action_plans.wallet_hold_id` carries a foreign key
    to `wallet_holds`, so a plan cannot point at a hold that was never placed.
    That is a second, narrower guarantee than the one this file asserts — the
    key stops a plan naming a *fictional* hold; nothing stops it naming *no*
    hold.
    """
    import time

    from db import _get_pg_pool

    plan_id = str(uuid.uuid4())
    hold_id = str(uuid.uuid4())
    job_id = f"job-trace-{uuid.uuid4().hex[:10]}"
    now = time.time()
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.execute(
            # `amount_micros`, not `amount_cad`: migration 087 removed the
            # duplicate float-CAD columns. Reading the migration that created
            # this table would have given the wrong column name — the live
            # schema is the one to check.
            """INSERT INTO wallet_holds
                 (hold_id, customer_id, amount_micros, status, job_id,
                  created_at, expires_at, updated_at)
               VALUES (%s, 'cus_traceprobe', 1230000, 'held', %s, %s, %s, %s)""",
            (hold_id, job_id, now, now + 3600, now),
        )
        conn.execute(
            """INSERT INTO action_plans
                 (plan_id, action_type, principal_id, tenant_id,
                  canonical_args, canonical_args_hash, estimate_micros,
                  approval_mode, status, approved_at, expires_at)
               VALUES (%s, 'create_instance', 'probe-principal', 'probe-tenant',
                       %s, 'probehash', 12345678,
                       'standing_policy', 'approved', clock_timestamp(),
                       clock_timestamp() + interval '1 hour')""",
            (plan_id, json.dumps({"probe": True})),
        )
        conn.commit()
    yield plan_id, hold_id, job_id
    with pool.connection() as conn:
        # Plan first: it references the hold.
        conn.execute("DELETE FROM action_plans WHERE plan_id = %s", (plan_id,))
        conn.execute("DELETE FROM wallet_holds WHERE hold_id = %s", (hold_id,))
        conn.commit()


def test_a_consumed_plan_records_the_hold_that_paid_for_it(plan_probe):
    """`mark_consumed` is the only writer of the link; exercise it, not SQL."""
    from control_plane.launch.action_plans import mark_consumed
    from db import _get_pg_pool

    plan_id, hold_id, job_id = plan_probe
    pool = _get_pg_pool()
    with pool.connection() as conn:
        mark_consumed(
            conn,
            plan_id,
            job_id=job_id,
            wallet_hold_id=hold_id,
            idempotent_response={"job_id": job_id, "phase": "pending"},
            idempotency_key=f"probe:{plan_id}",
        )
        conn.commit()

        row = conn.execute(
            "SELECT status, job_id, wallet_hold_id::text AS hold "
            "FROM action_plans WHERE plan_id = %s",
            (plan_id,),
        ).fetchone()

    assert row is not None
    assert row[0] == "succeeded"
    assert row[1] == job_id
    assert row[2] == hold_id, (
        "the consumed plan did not record the wallet hold that paid for it; "
        "the spend is no longer traceable to its approval"
    )


def test_one_query_from_the_money_reaches_the_approval(plan_probe):
    """The gate, literally: start at the hold, end at the quote.

    One statement, no joins to reconstruct and no session to replay. If this
    needs two queries the property has been lost even if the data still exists.
    """
    from control_plane.launch.action_plans import mark_consumed
    from db import _get_pg_pool

    plan_id, hold_id, job_id = plan_probe
    pool = _get_pg_pool()
    with pool.connection() as conn:
        mark_consumed(
            conn,
            plan_id,
            job_id=job_id,
            wallet_hold_id=hold_id,
            idempotent_response={"job_id": job_id, "phase": "pending"},
            idempotency_key=f"probe:{plan_id}",
        )
        conn.commit()

        row = conn.execute(
            """SELECT plan_id::text, estimate_micros, approval_mode,
                      action_type, principal_id
                 FROM action_plans
                WHERE wallet_hold_id = %s""",
            (hold_id,),
        ).fetchone()

    assert row is not None, (
        "no plan is reachable from the wallet hold — a charge exists that "
        "cannot be explained without replaying the session that made it"
    )
    assert row[0] == plan_id
    assert row[1] == 12345678, "the quote did not survive to the audit"
    assert row[2] == "standing_policy", "the approval mode did not survive"


def test_one_query_from_the_instance_reaches_the_approval(plan_probe):
    """The other direction, which is the one an operator actually asks.

    "Why is this instance running and who agreed to pay for it?"
    """
    from control_plane.launch.action_plans import mark_consumed
    from db import _get_pg_pool

    plan_id, hold_id, job_id = plan_probe
    pool = _get_pg_pool()
    with pool.connection() as conn:
        mark_consumed(
            conn,
            plan_id,
            job_id=job_id,
            wallet_hold_id=hold_id,
            idempotent_response={"job_id": job_id, "phase": "pending"},
            idempotency_key=f"probe:{plan_id}",
        )
        conn.commit()

        row = conn.execute(
            "SELECT plan_id::text, approval_mode, estimate_micros "
            "FROM action_plans WHERE job_id = %s",
            (job_id,),
        ).fetchone()

    assert row is not None, "no plan is reachable from the job it launched"
    assert row[0] == plan_id


def test_the_launch_path_still_passes_a_real_hold():
    """Structural, because the runtime test cannot see the production call.

    The tests above prove the link works when `mark_consumed` is given a hold.
    They pass just as happily if the launch service starts passing `None` —
    which is the realistic regression, since two other call sites pass `None`
    legitimately and the state machine permits it.
    """
    import inspect

    from control_plane.launch import service

    source = inspect.getsource(service)
    assert "wallet_hold_id=str(hold[" in source, (
        "the launch path no longer passes the wallet hold to mark_consumed; "
        "instances would launch, money would move, and nothing would connect "
        "the charge to the plan that approved it"
    )


def test_the_constraint_does_not_cover_this_and_that_is_why_this_exists():
    """Names the reason this file is a test rather than a CHECK.

    If `ck_action_plans_state_machine` ever *does* require `wallet_hold_id` for
    a succeeded plan, the database enforces it and this suite can say so instead
    of asserting it from outside. Until then, this is the only thing standing
    between a silent `None` and an untraceable charge.
    """
    from db import _get_pg_pool

    pool = _get_pg_pool()
    with pool.connection() as conn:
        row = conn.execute(
            """SELECT pg_get_constraintdef(oid)
                 FROM pg_constraint
                WHERE conname = 'ck_action_plans_state_machine'"""
        ).fetchone()

    if row is None:
        pytest.skip("state machine constraint not present in this database")
    assert "wallet_hold_id" not in row[0], (
        "the state machine now constrains wallet_hold_id — the database "
        "enforces traceability, so update this file to assert that instead of "
        "guarding it from outside"
    )
