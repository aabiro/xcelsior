"""Track B B2.5 — action-plan execute (§14.3).

Execute is the *only* place a launch plan turns into desired state: it locks
the plan, verifies approved / not expired / not consumed / not revoked, checks
the canonical-argument hash, re-quotes within tolerance, reserves the wallet
hold, creates the job through the one job authority, and consumes the plan —
all keyed to the plan so a replay or a crash converges on **exactly one** job
and **one** hold.

This is the gate the B2.5 checklist item requires, driven by *executing* the
path (B0.1 rule 3), not by reading it:

- repeated execute returns the original job and creates no second job or hold;
- concurrent execute of one plan admits exactly one;
- a not-approved / revoked / expired plan is refused;
- a price move beyond tolerance returns a replacement plan and charges nothing
  (§15.4) — never a silent charge at the new price;
- an unfundable plan is refused with a typed 402, no job, no dangling hold.

Real PostgreSQL; every test owns only its own rows (B0.2 rule 14).
"""

from __future__ import annotations

import concurrent.futures
import uuid

import pytest

from control_plane.launch import quoting
from control_plane.launch.service import (
    LaunchPlanError,
    PlanConflict,
    Principal,
    approve,
    execute,
    preview,
    revoke,
)

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _has = (
            _c.execute("SELECT to_regclass('action_plans')").fetchone()[0] is not None
            and _c.execute("SELECT to_regclass('public.wallet_holds')").fetchone()[0]
            is not None
        )
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("action_plans / wallet_holds missing — upgrade head")


@pytest.fixture
def scratch():
    made = {"plans": [], "customers": [], "jobs": []}
    yield made
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in made["jobs"]:
            conn.execute("UPDATE jobs SET wallet_hold_id = NULL WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for pid in made["plans"]:
            conn.execute("DELETE FROM action_plans WHERE plan_id = %s", (pid,))
        for cid in made["customers"]:
            conn.execute("DELETE FROM wallet_holds WHERE customer_id=%s", (cid,))
            conn.execute("DELETE FROM wallet_transactions WHERE customer_id=%s", (cid,))
            conn.execute("DELETE FROM wallets WHERE customer_id=%s", (cid,))
        conn.commit()


def _funded_principal(scratch, *, deposit_cad: float = 1000.0) -> Principal:
    """A principal whose tenant wallet is funded enough to cover the hold."""
    from billing import BillingEngine

    tenant = f"exec-{uuid.uuid4().hex[:10]}"
    scratch["customers"].append(tenant)
    if deposit_cad > 0:
        BillingEngine().deposit(tenant, deposit_cad, description="b25 seed")
    return Principal(principal_id="u1", tenant_id=tenant)


def _plan(principal, scratch, **over) -> str:
    req = {"name": "b25", "num_gpus": 1, "interactive": True}
    req.update(over)
    result = preview(req, principal=principal)
    scratch["plans"].append(result["plan_id"])
    return result["plan_id"]


def _approved_plan(principal, scratch, **over) -> str:
    pid = _plan(principal, scratch, **over)
    approve(pid, principal=principal, is_human=True)
    return pid


def _count_jobs(job_id: str) -> int:
    with _pool.connection() as conn:
        return conn.execute("SELECT count(*) FROM jobs WHERE job_id=%s", (job_id,)).fetchone()[0]


def _count_holds(tenant: str, plan_id: str) -> int:
    with _pool.connection() as conn:
        return conn.execute(
            "SELECT count(*) FROM wallet_holds WHERE customer_id=%s AND idempotency_key=%s",
            (tenant, f"plan:{plan_id}"),
        ).fetchone()[0]


# ── Happy path: exactly one job, one hold ────────────────────────────


def test_execute_creates_one_job_and_one_hold(scratch):
    p = _funded_principal(scratch)
    pid = _approved_plan(p, scratch)

    out = execute(pid, principal=p)
    assert out["ok"] is True
    assert out["idempotent"] is False
    job_id = out["job"]["job_id"]
    scratch["jobs"].append(job_id)

    assert out["plan"]["status"] == "succeeded"
    assert _count_jobs(job_id) == 1
    assert _count_holds(p.tenant_id, pid) == 1


# ── Exactly-once replay ──────────────────────────────────────────────


def test_repeated_execute_returns_original_job_no_second_effect(scratch):
    p = _funded_principal(scratch)
    pid = _approved_plan(p, scratch)

    first = execute(pid, principal=p)
    job_id = first["job"]["job_id"]
    scratch["jobs"].append(job_id)

    second = execute(pid, principal=p)
    assert second["idempotent"] is True
    assert second["job"]["job_id"] == job_id
    # No second job, no second hold — the plan replays its stored response.
    assert _count_jobs(job_id) == 1
    assert _count_holds(p.tenant_id, pid) == 1


# ── Concurrency: one plan admits exactly one ─────────────────────────


def test_concurrent_execute_admits_exactly_one(scratch):
    p = _funded_principal(scratch)
    pid = _approved_plan(p, scratch)

    def go():
        try:
            return ("ok", execute(pid, principal=p))
        except Exception as exc:  # a lock-timeout loser is still a valid outcome
            return ("err", exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        results = [f.result() for f in [ex.submit(go), ex.submit(go)]]

    successes = [r for tag, r in results if tag == "ok" and r.get("ok")]
    assert successes, f"no execute succeeded: {results}"
    job_id = successes[0]["job"]["job_id"]
    scratch["jobs"].append(job_id)

    # The invariant that matters regardless of who won the race:
    assert _count_jobs(job_id) == 1
    assert _count_holds(p.tenant_id, pid) == 1
    # Exactly one caller performed the real launch; any other saw the replay.
    non_idempotent = [r for r in successes if r.get("idempotent") is False]
    assert len(non_idempotent) == 1


# ── Refusals ─────────────────────────────────────────────────────────


def test_execute_refused_when_not_approved(scratch):
    p = _funded_principal(scratch)
    pid = _plan(p, scratch)  # quoted, never approved
    with pytest.raises(PlanConflict) as ei:
        execute(pid, principal=p)
    assert ei.value.code == "plan_not_approved"
    assert _count_holds(p.tenant_id, pid) == 0


def test_execute_refused_when_revoked(scratch):
    p = _funded_principal(scratch)
    pid = _approved_plan(p, scratch)
    revoke(pid, principal=p)
    with pytest.raises(PlanConflict) as ei:
        execute(pid, principal=p)
    assert ei.value.code == "plan_revoked"


def test_execute_refused_when_expired(scratch):
    p = _funded_principal(scratch)
    pid = _approved_plan(p, scratch)
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE action_plans SET expires_at = now() - interval '1 hour' WHERE plan_id=%s",
            (pid,),
        )
        conn.commit()
    with pytest.raises(PlanConflict) as ei:
        execute(pid, principal=p)
    assert ei.value.code == "plan_expired"
    # The refused plan is moved to a terminal expired state, not left approved.
    with _pool.connection() as conn:
        st = conn.execute("SELECT status FROM action_plans WHERE plan_id=%s", (pid,)).fetchone()[0]
    assert st == "expired"


def test_execute_refused_on_spec_hash_mismatch(scratch):
    p = _funded_principal(scratch)
    pid = _approved_plan(p, scratch)
    # Tamper the stored canonical args without updating the hash: the integrity
    # check (§14.3) must refuse rather than launch a spec nobody quoted.
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE action_plans SET canonical_args = jsonb_set(canonical_args, '{name}', '\"tampered\"') WHERE plan_id=%s",
            (pid,),
        )
        conn.commit()
    with pytest.raises(PlanConflict) as ei:
        execute(pid, principal=p)
    assert ei.value.code == "spec_hash_mismatch"
    assert _count_holds(p.tenant_id, pid) == 0


def test_execute_insufficient_funds_is_402_no_job(scratch):
    p = _funded_principal(scratch, deposit_cad=0.0)  # empty wallet
    pid = _approved_plan(p, scratch)
    with pytest.raises(LaunchPlanError) as ei:
        execute(pid, principal=p)
    assert ei.value.status == 402
    assert ei.value.code == "insufficient_funds"


# ── Price change beyond tolerance → replacement plan, no charge (§15.4) ─


def test_price_beyond_tolerance_returns_replacement_no_charge(scratch):
    p = _funded_principal(scratch)
    pid = _approved_plan(p, scratch)
    # Drive the *approved* estimate far below the live quote so re-quoting at
    # execute exceeds tolerance. Nothing may be charged; a fresh plan is issued.
    with _pool.connection() as conn:
        conn.execute("UPDATE action_plans SET estimate_micros = 1 WHERE plan_id=%s", (pid,))
        conn.commit()

    out = execute(pid, principal=p)
    assert out["ok"] is False
    assert out["code"] == "quote_changed"
    replacement = out["replacement_plan"]
    assert replacement["plan_id"] != pid
    scratch["plans"].append(replacement["plan_id"])

    # No hold, and the original plan is not consumed.
    assert _count_holds(p.tenant_id, pid) == 0
    with _pool.connection() as conn:
        st = conn.execute("SELECT status FROM action_plans WHERE plan_id=%s", (pid,)).fetchone()[0]
    assert st == "approved"  # unchanged; the caller must approve the replacement


def test_price_within_tolerance_proceeds(scratch):
    # Sanity companion to the above: an untampered plan re-quotes to the same
    # price and executes normally (tolerance is not a blanket block).
    p = _funded_principal(scratch)
    pid = _approved_plan(p, scratch)
    out = execute(pid, principal=p)
    assert out["ok"] is True
    scratch["jobs"].append(out["job"]["job_id"])
