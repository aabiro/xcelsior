"""Track B B2.4 — action-plan approval and revoke (§14.2).

Server-bound approval: a standing policy self-approves only inside its
ceilings, a human approves everything else, and ``confirm:true`` never
approves anything. The gate pins the three isolation/safety properties:

- a plan cannot be approved across a tenant boundary;
- a plan that has fallen outside its standing policy cannot auto-approve;
- approval (and revoke) is idempotent.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from api import app
from control_plane.launch.service import (
    Principal,
    PlanConflict,
    PlanNotFound,
    approve,
    preview,
    revoke,
)

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT to_regclass('action_plans')").fetchone()
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None

client = TestClient(app)


@pytest.fixture
def scratch():
    """Track plans and policies to delete after the test."""
    made = {"plans": [], "policies": []}
    yield made
    if _pool is None:
        return
    with _pool.connection() as conn:
        for pid in made["plans"]:
            conn.execute("DELETE FROM action_plans WHERE plan_id = %s", (pid,))
        for cid, tid in made["policies"]:
            conn.execute(
                "DELETE FROM mcp_client_policies WHERE client_id = %s AND tenant_id = %s",
                (cid, tid),
            )
        conn.commit()


def _tenant() -> str:
    return f"t-{uuid.uuid4().hex[:10]}"


def _preview(principal, scratch, **over):
    req = {"name": "b24", "num_gpus": 1, "interactive": True}
    req.update(over)
    result = preview(req, principal=principal)
    scratch["plans"].append(result["plan_id"])
    return result


def _add_policy(scratch, *, client_id, tenant_id, per_action_max_micros, auto_approve=True):
    with _pool.connection() as conn:
        conn.execute(
            "INSERT INTO mcp_client_policies "
            "(client_id, tenant_id, per_action_max_micros, auto_approve) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (client_id, tenant_id) DO UPDATE "
            "SET per_action_max_micros = EXCLUDED.per_action_max_micros, "
            "    auto_approve = EXCLUDED.auto_approve",
            (client_id, tenant_id, per_action_max_micros, auto_approve),
        )
        conn.commit()
    scratch["policies"].append((client_id, tenant_id))


# ── Isolation ────────────────────────────────────────────────────────


def test_cross_tenant_cannot_approve(scratch):
    owner = Principal(principal_id="u1", tenant_id=_tenant())
    result = _preview(owner, scratch)
    intruder = Principal(principal_id="u2", tenant_id=_tenant())
    with pytest.raises(PlanNotFound):
        approve(result["plan_id"], principal=intruder, is_human=True)


def test_cross_tenant_cannot_revoke(scratch):
    owner = Principal(principal_id="u1", tenant_id=_tenant())
    result = _preview(owner, scratch)
    intruder = Principal(principal_id="u2", tenant_id=_tenant())
    with pytest.raises(PlanNotFound):
        revoke(result["plan_id"], principal=intruder)


# ── Human approval ───────────────────────────────────────────────────


def test_human_approves_quoted_plan(scratch):
    owner = Principal(principal_id="u1", tenant_id=_tenant())
    result = _preview(owner, scratch)
    assert result["approval_mode"] == "human"
    out = approve(result["plan_id"], principal=owner, is_human=True)
    assert out["plan"]["status"] == "approved"
    assert out["plan"]["approval_method"] == "human"


def test_machine_cannot_do_human_approval(scratch):
    owner = Principal(principal_id="u1", tenant_id=_tenant())
    result = _preview(owner, scratch)
    with pytest.raises(PlanConflict) as ei:
        approve(result["plan_id"], principal=owner, is_human=False)
    assert ei.value.code == "human_approval_required"


def test_approval_is_idempotent(scratch):
    owner = Principal(principal_id="u1", tenant_id=_tenant())
    result = _preview(owner, scratch)
    first = approve(result["plan_id"], principal=owner, is_human=True)
    second = approve(result["plan_id"], principal=owner, is_human=True)
    assert first["idempotent"] is False
    assert second["idempotent"] is True
    assert second["plan"]["status"] == "approved"
    # Version did not advance on the idempotent re-approve.
    assert second["plan"]["version"] == first["plan"]["version"]


# ── Standing policy ──────────────────────────────────────────────────


def test_within_policy_auto_approves(scratch):
    tenant = _tenant()
    client_id = f"c-{uuid.uuid4().hex[:8]}"
    _add_policy(scratch, client_id=client_id, tenant_id=tenant, per_action_max_micros=10**9)
    principal = Principal(principal_id="u1", tenant_id=tenant, client_id=client_id)
    result = _preview(principal, scratch)
    assert result["approval_mode"] == "standing_policy"
    out = approve(result["plan_id"], principal=principal, is_human=False)
    assert out["plan"]["status"] == "approved"
    assert out["plan"]["approval_method"] == "standing_policy"


def test_out_of_policy_cannot_auto_approve(scratch):
    tenant = _tenant()
    client_id = f"c-{uuid.uuid4().hex[:8]}"
    # Grant auto-approve within a generous limit so the *plan* is created in
    # standing_policy mode …
    _add_policy(scratch, client_id=client_id, tenant_id=tenant, per_action_max_micros=10**9)
    principal = Principal(principal_id="u1", tenant_id=tenant, client_id=client_id)
    result = _preview(principal, scratch)
    assert result["approval_mode"] == "standing_policy"
    # … then tighten the ceiling below the estimate. Approval must now refuse
    # to self-approve and demand a human.
    _add_policy(scratch, client_id=client_id, tenant_id=tenant, per_action_max_micros=1)
    with pytest.raises(PlanConflict) as ei:
        approve(result["plan_id"], principal=principal, is_human=False)
    assert ei.value.code == "auto_approval_denied"


# ── Revoke ───────────────────────────────────────────────────────────


def test_revoke_then_idempotent(scratch):
    owner = Principal(principal_id="u1", tenant_id=_tenant())
    result = _preview(owner, scratch)
    first = revoke(result["plan_id"], principal=owner, reason="changed mind")
    assert first["plan"]["status"] == "revoked"
    second = revoke(result["plan_id"], principal=owner)
    assert second["idempotent"] is True


def test_cannot_approve_a_revoked_plan(scratch):
    owner = Principal(principal_id="u1", tenant_id=_tenant())
    result = _preview(owner, scratch)
    revoke(result["plan_id"], principal=owner)
    with pytest.raises(PlanConflict) as ei:
        approve(result["plan_id"], principal=owner, is_human=True)
    assert ei.value.code == "plan_not_approvable"


# ── HTTP surface smoke ───────────────────────────────────────────────


def test_http_approve_unknown_plan_is_404(scratch):
    reg = client.post(
        "/api/auth/register",
        json={"email": "b24-http@xcelsior.ca", "password": "Str0ngPass!abc"},
    ).json()
    token = reg["access_token"]
    resp = client.post(
        f"/api/v1/launch-plans/{uuid.uuid4()}/approve",
        headers={"Authorization": f"Bearer {token}"},
        json={"confirm": True},
    )
    assert resp.status_code == 404
