"""Versioned launch-plan API (Track B B2, blueprint §14, §18).

The `/api/v1/launch-plans` surface: preview (§14.1), approve/revoke (§14.2),
and execute (§14.3). Every launch surface funnels through the same
`control_plane.launch.service`, so MCP, dashboard, and REST submit
byte-identical canonical specs and share one approval/idempotency authority.

Principal resolution is here (transport concern); the service receives an
already-resolved :class:`Principal` and never touches request headers.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from pydantic import BaseModel

from control_plane.launch.service import (
    LaunchPlanError,
    Principal,
    approve,
    execute,
    preview,
    revoke,
)
from routes._deps import (
    _canonical_owner_id,
    _effective_billing_customer_id,
    _require_auth,
    _require_scope,
    _user_team_id,
)
from routes.instances import JobIn

router = APIRouter(tags=["Launch plans"])


def _is_human(user: dict) -> bool:
    """Interactive humans only — a machine client is never a human approver."""
    return str(user.get("auth_type", "")) != "client_credentials"


def _map_plan_error(exc: LaunchPlanError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail={"code": exc.code, "detail": exc.detail})


def _resolve_principal(request: Request) -> tuple[dict, Principal]:
    """Authenticate and resolve who is acting, in which tenant.

    Tenant is the effective billing customer — the same identity the wallet
    hold is created under (execute, B2.5) — so a plan and the funds it later
    reserves can never belong to different tenants.
    """
    user = _require_auth(request)
    principal = Principal(
        principal_id=_canonical_owner_id(user),
        tenant_id=_effective_billing_customer_id(user),
        client_id=user.get("client_id"),
        team_id=_user_team_id(user),
        scopes=tuple(user.get("scopes") or ()),
    )
    return user, principal


@router.post("/api/v1/launch-plans")
def api_create_launch_plan(j: JobIn, request: Request):
    """§14.1 preview. Creates an action plan; no attempt/allocation/lease/hold/job.

    Returns the plan id, versioned estimate, current placement availability,
    expiry, approval mode, and the next action. A confirmed launch happens
    only after approval, through the execute endpoint (B2.5).
    """
    user, principal = _resolve_principal(request)
    _require_scope(user, "instances:write")
    result = preview(j.model_dump(), principal=principal)
    if not result.get("ok"):
        # Structurally invalid spec — surface every problem at once.
        raise HTTPException(status_code=422, detail={"problems": result.get("problems")})
    return result


class _ApproveIn(BaseModel):
    # `confirm` is intent only and never constitutes approval (§14.2); it is
    # accepted for client symmetry and deliberately ignored here.
    confirm: bool = False


@router.post("/api/v1/launch-plans/{plan_id}/approve")
def api_approve_launch_plan(plan_id: str, request: Request, body: _ApproveIn | None = None):
    """§14.2 approval. Standing policy self-approves inside limits; else a human."""
    user, principal = _resolve_principal(request)
    _require_scope(user, "instances:write")
    try:
        return approve(plan_id, principal=principal, is_human=_is_human(user))
    except LaunchPlanError as exc:
        raise _map_plan_error(exc) from exc


class _RevokeIn(BaseModel):
    reason: str = ""


@router.post("/api/v1/launch-plans/{plan_id}/revoke")
def api_revoke_launch_plan(plan_id: str, request: Request, body: _RevokeIn | None = None):
    """§14.2 revoke — idempotent; a consumed or terminal plan cannot be revoked."""
    user, principal = _resolve_principal(request)
    _require_scope(user, "instances:write")
    try:
        return revoke(plan_id, principal=principal, reason=(body.reason if body else ""))
    except LaunchPlanError as exc:
        raise _map_plan_error(exc) from exc


class _ExecuteIn(BaseModel):
    # `confirm` expresses intent; approval is what authorizes (§14.2). Accepted
    # for client symmetry with create_instance's two-step flow.
    confirm: bool = False


@router.post("/api/v1/launch-plans/{plan_id}/execute")
def api_execute_launch_plan(plan_id: str, request: Request, body: _ExecuteIn | None = None):
    """§14.3 execute. Exactly-once; a price move beyond tolerance is 409 quote_changed."""
    user, principal = _resolve_principal(request)
    _require_scope(user, "instances:write")
    try:
        result = execute(plan_id, principal=principal)
    except LaunchPlanError as exc:
        raise _map_plan_error(exc) from exc
    if not result.get("ok") and result.get("code") == "quote_changed":
        # The approved price no longer holds; the caller must approve the
        # replacement plan (§15.4). Never a silent charge at the new price.
        raise HTTPException(status_code=409, detail=result)
    return result
