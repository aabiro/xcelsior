"""Track B B3.4 — price-change reapproval end to end (§15.4).

The tolerance bound recorded on the plan (B2.1) is enforced at execute (B2.5).
This gate drives the *real* pricing authority — `BillingEngine.
estimate_launch_hold_cad` — not the stored estimate: it quotes a plan at one
rate, then moves the rate and executes.

  * beyond tolerance  → execute is blocked with `quote_changed` and a fresh
    replacement plan; nothing is charged (§15.4 — never a silent charge at the
    new price);
  * within tolerance  → execute proceeds unchanged.
"""

from __future__ import annotations

import uuid

import pytest

from billing import BillingEngine
from control_plane.launch.service import Principal, approve, execute, preview

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
        pytestmark = pytest.mark.skip("action_plans missing — upgrade head")


@pytest.fixture
def scratch():
    made = {"plans": [], "customers": [], "jobs": []}
    yield made
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in made["jobs"]:
            conn.execute("UPDATE jobs SET wallet_hold_id=NULL WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for pid in made["plans"]:
            conn.execute("DELETE FROM action_plans WHERE plan_id=%s", (pid,))
        for cid in made["customers"]:
            conn.execute("DELETE FROM wallet_holds WHERE customer_id=%s", (cid,))
            conn.execute("DELETE FROM wallet_transactions WHERE customer_id=%s", (cid,))
            conn.execute("DELETE FROM wallets WHERE customer_id=%s", (cid,))
        conn.commit()


def _funded(scratch) -> Principal:
    tenant = f"b34-{uuid.uuid4().hex[:10]}"
    scratch["customers"].append(tenant)
    BillingEngine().deposit(tenant, 5000.0, description="b34 seed")
    return Principal(principal_id="u1", tenant_id=tenant)


def _approved_plan(principal, scratch) -> tuple[str, int]:
    result = preview(
        {"name": "b34", "interactive": False, "vram_needed_gb": 16, "gpu_model": "RTX 4090"},
        principal=principal,
    )
    scratch["plans"].append(result["plan_id"])
    assert result["ok"], result
    approve(result["plan_id"], principal=principal, is_human=True)
    return result["plan_id"], int(result["estimate"]["estimate_micros"])


def _scale_rate(monkeypatch, factor: float) -> None:
    """Move the live hourly rate by *factor* for subsequent quotes."""
    orig = BillingEngine.estimate_launch_hold_cad

    def scaled(self, **kwargs):
        return float(orig(self, **kwargs)) * factor

    monkeypatch.setattr(BillingEngine, "estimate_launch_hold_cad", scaled)


def test_pricing_bump_beyond_tolerance_blocks_execute(scratch, monkeypatch):
    p = _funded(scratch)
    plan_id, base_estimate = _approved_plan(p, scratch)
    assert base_estimate > 0  # a real, non-trivial quote

    _scale_rate(monkeypatch, 2.0)  # rate doubled after approval — far beyond 5%
    out = execute(plan_id, principal=p)

    assert out["ok"] is False
    assert out["code"] == "quote_changed"
    replacement = out["replacement_plan"]
    assert replacement["plan_id"] != plan_id
    scratch["plans"].append(replacement["plan_id"])
    # The replacement is quoted at the *new* (higher) rate.
    assert replacement["estimate"]["estimate_micros"] > base_estimate

    # Nothing was charged and the original plan is untouched (still approved).
    with _pool.connection() as conn:
        st = conn.execute("SELECT status FROM action_plans WHERE plan_id=%s", (plan_id,)).fetchone()[0]
        holds = conn.execute(
            "SELECT count(*) FROM wallet_holds WHERE idempotency_key=%s", (f"plan:{plan_id}",)
        ).fetchone()[0]
    assert st == "approved"
    assert holds == 0


def test_pricing_bump_within_tolerance_proceeds(scratch, monkeypatch):
    p = _funded(scratch)
    plan_id, _ = _approved_plan(p, scratch)

    _scale_rate(monkeypatch, 1.01)  # +1% — inside the 500 bps (5%) tolerance
    out = execute(plan_id, principal=p)

    assert out["ok"] is True
    assert out["idempotent"] is False
    scratch["jobs"].append(out["job"]["job_id"])
