"""Versioned launch-plan API (Track B B2, blueprint §14, §18).

The `/api/v1/launch-plans` surface: preview (§14.1), approve/revoke (§14.2),
and execute (§14.3). Every launch surface funnels through the same
`control_plane.launch.service`, so MCP, dashboard, and REST submit
byte-identical canonical specs and share one approval/idempotency authority.

Principal resolution is here (transport concern); the service receives an
already-resolved :class:`Principal` and never touches request headers.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from pydantic import BaseModel, Field

from control_plane.launch.service import (
    Principal,
    approve,
    execute,
    get_owned_plan,
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
from routes.problem import problem_response

router = APIRouter(tags=["Launch plans"])


def _request_trace_id(request: Request) -> str | None:
    """Return a valid non-zero W3C trace id from the inbound request."""
    parts = request.headers.get("traceparent", "").split("-")
    if len(parts) != 4 or len(parts[1]) != 32:
        return None
    try:
        int(parts[1], 16)
    except ValueError:
        return None
    return parts[1].lower() if parts[1] != "0" * 32 else None


def _is_human(user: dict) -> bool:
    """Interactive humans only — a machine client is never a human approver."""
    return str(user.get("auth_type", "")) != "client_credentials"


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


@router.post("/api/v1/placements/simulate")
def api_simulate_placement(j: JobIn, request: Request):
    """§18 placement feasibility for a spec — read-only.

    Reuses the launch service's snapshot + Stage-C filter simulation: it creates
    no plan, no attempt, no allocation, and no lease. Answers "could this be
    placed right now, and if not, why".
    """
    from control_plane.launch.canonicalize import canonicalize, spec_hash
    from control_plane.launch.service import simulate_placement
    from control_plane.launch.validation import validate_canonical_spec

    user, _ = _resolve_principal(request)
    _require_scope(user, "instances:read")
    spec = canonicalize(j.model_dump())
    problems = validate_canonical_spec(spec)
    if problems:
        return problem_response(
            status=422,
            code="invalid_spec",
            detail="the launch spec failed validation",
            errors=[p.as_dict() for p in problems],
        )
    return {
        "ok": True,
        "spec_hash": spec_hash(spec),
        "availability": simulate_placement(spec),
    }


@router.post("/api/v1/launch-plans")
def api_create_launch_plan(j: JobIn, request: Request):
    """§14.1 preview. Creates an action plan; no attempt/allocation/lease/hold/job.

    Returns the plan id, versioned estimate, current placement availability,
    expiry, approval mode, and the next action. A confirmed launch happens
    only after approval, through the execute endpoint (B2.5).
    """
    user, principal = _resolve_principal(request)
    _require_scope(user, "instances:write")
    result = preview(
        j.model_dump(),
        principal=principal,
        trace_id=_request_trace_id(request),
    )
    if not result.get("ok"):
        # Structurally invalid spec — surface every problem at once as RFC 9457
        # field errors, and persist no plan.
        return problem_response(
            status=422,
            code="invalid_spec",
            detail="the launch spec failed validation",
            errors=result.get("problems"),
        )
    return result


@router.get("/api/v1/launch-plans/{plan_id}")
def api_get_launch_plan(plan_id: str, request: Request):
    """Return action status only to its tenant; foreign ids remain not-found."""
    user, principal = _resolve_principal(request)
    result = get_owned_plan(plan_id, principal=principal)
    if str(user.get("grant_type", "")) == "client_credentials":
        held = set(user.get("scopes") or ())
        required = set(result["plan"].get("required_scopes") or ())
        read_equivalents = {
            "instances:operate": "instances:read",
            "inference:write": "inference:read",
            "hosts:evict": "hosts:read",
        }
        accepted = required | {read_equivalents[s] for s in required if s in read_equivalents}
        if not held.intersection(accepted):
            _require_scope(user, next(iter(accepted), "instances:read"))
    return result


class _ApproveIn(BaseModel):
    # `confirm` is intent only and never constitutes approval (§14.2); it is
    # accepted for client symmetry and deliberately ignored here.
    confirm: bool = False
    expected_version: int | None = Field(default=None, ge=1)
    confirmation: str | None = Field(default=None, max_length=32)


@router.post("/api/v1/launch-plans/{plan_id}/approve")
def api_approve_launch_plan(plan_id: str, request: Request, body: _ApproveIn | None = None):
    """§14.2 approval. Standing policy self-approves inside limits; else a human."""
    user, principal = _resolve_principal(request)
    plan_view = get_owned_plan(plan_id, principal=principal)["plan"]
    required_scopes = list(plan_view.get("required_scopes") or ["instances:write"])
    _require_scope(user, *required_scopes)
    # LaunchPlanError propagates to the app-level RFC 9457 handler (B2.8).
    return approve(
        plan_id,
        principal=principal,
        is_human=_is_human(user),
        expected_version=body.expected_version if body else None,
    )


class _RevokeIn(BaseModel):
    reason: str = ""


@router.post("/api/v1/launch-plans/{plan_id}/revoke")
def api_revoke_launch_plan(plan_id: str, request: Request, body: _RevokeIn | None = None):
    """§14.2 revoke — idempotent; a consumed or terminal plan cannot be revoked."""
    user, principal = _resolve_principal(request)
    plan_view = get_owned_plan(plan_id, principal=principal)["plan"]
    _require_scope(user, *list(plan_view.get("required_scopes") or ["instances:write"]))
    return revoke(plan_id, principal=principal, reason=(body.reason if body else ""))


class _ExecuteIn(BaseModel):
    # `confirm` expresses intent; approval is what authorizes (§14.2). Accepted
    # for client symmetry with create_instance's two-step flow.
    confirm: bool = False


@router.post("/api/v1/launch-plans/{plan_id}/execute")
def api_execute_launch_plan(plan_id: str, request: Request, body: _ExecuteIn | None = None):
    """§14.3 execute. Exactly-once; a price move beyond tolerance is 409 quote_changed."""
    user, principal = _resolve_principal(request)
    _require_scope(user, "instances:write")
    # LaunchPlanError propagates to the app-level RFC 9457 handler (B2.8).
    result = execute(plan_id, principal=principal)
    if not result.get("ok") and result.get("code") == "quote_changed":
        # The approved price no longer holds; the caller must approve the
        # replacement plan (§15.4). Never a silent charge at the new price —
        # the replacement is carried as an RFC 9457 extension member.
        return problem_response(
            status=409,
            code="quote_changed",
            detail=result.get("detail", "the price moved beyond the approved tolerance"),
            extra={"replacement_plan": result.get("replacement_plan")},
        )
    return result


# ── P4 pipelines: one approval for a dependency graph ────────────────


class PipelineStageIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    action_type: str = Field(min_length=1, max_length=64)
    on_failure: str = Field(default="halt", max_length=16)
    max_attempts: int = Field(default=1, ge=1, le=10)
    estimate_micros: int = Field(default=0, ge=0)
    args: dict = Field(default_factory=dict)


class PipelineIn(BaseModel):
    name: str = Field(default="pipeline", max_length=120)
    stages: list[PipelineStageIn] = Field(min_length=1, max_length=20)


@router.post("/api/v1/pipelines")
def api_create_pipeline(body: PipelineIn, request: Request):
    """Quote a dependency graph and persist it as one approvable plan.

    Gate P4's "one approval, three stages". The graph lives in the plan's
    canonical args, so the existing `canonical_args_hash` check is what makes
    the approved graph server-bound — editing any stage afterwards invalidates
    it without a mechanism written for this phase.

    **The plan requires the union of its stages' scopes.** Approving
    `train → serve` on `instances:operate` alone would let the serve stage run
    without `inference:write`; one approval silently widening authority is the
    failure this phase could introduce if scopes came from the first stage.

    `approval_mode` is `"human"` and not the spend policy's decision. A standing
    policy pre-authorises spending inside ceilings; letting it approve a graph
    that sets its own ceiling is circular — the same reasoning that fixed
    `configure_auto_topup` to human-only.
    """
    import hashlib
    import json as _json
    import uuid as _uuid

    from control_plane.db import control_plane_transaction
    from control_plane.launch import action_plans as plans_repo
    from control_plane.launch.service import DEFAULT_TOLERANCE_BPS, plan_ttl_sec
    from control_plane.pipelines import (
        PipelineError,
        assert_graph_is_runnable,
        canonical_graph,
        required_scopes_for_graph,
    )

    user, principal = _resolve_principal(request)

    try:
        stages, graph_hash = canonical_graph([s.model_dump() for s in body.stages])
        assert_graph_is_runnable(stages)
    except PipelineError as exc:
        return problem_response(
            status=422, code=exc.code, detail=str(exc),
        )

    required = required_scopes_for_graph(stages)
    for scope in required:
        _require_scope(user, scope)

    total_micros = sum(int(s["estimate_micros"]) for s in stages)
    canonical = {"name": body.name, "stages": stages}
    canonical_json = _json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    canonical_hash = hashlib.sha256(canonical_json.encode()).hexdigest()

    with control_plane_transaction() as conn:
        plan = plans_repo.create_quoted_plan(
            conn,
            action_type="run_pipeline",
            principal_id=principal.principal_id,
            client_id=principal.client_id,
            tenant_id=principal.tenant_id,
            team_id=principal.team_id,
            canonical_args=canonical,
            canonical_args_hash=canonical_hash,
            spec_hash=graph_hash,
            quote_id=f"pipeline:{graph_hash[:20]}",
            pricing_version="pipeline-v1",
            estimate_micros=total_micros,
            currency="CAD",
            price_tolerance_bps=DEFAULT_TOLERANCE_BPS,
            required_scopes=required,
            approval_mode="human",
            ttl_sec=plan_ttl_sec(),
        )

    plan_id = str(plan["plan_id"])
    return {
        "ok": True,
        "preview": True,
        "plan_id": plan_id,
        "approval_state": plan["status"],
        "stages": [
            {"index": i, "name": s["name"], "action_type": s["action_type"],
             "on_failure": s["on_failure"], "max_attempts": s["max_attempts"],
             "estimate_micros": s["estimate_micros"]}
            for i, s in enumerate(stages)
        ],
        # The ceiling, stated before approval rather than after — a total the
        # user is agreeing to, not an estimate they will be reconciled against.
        "approved_max_micros": total_micros,
        "currency": "CAD",
        "required_scopes": required,
        "approval_url": f"/dashboard/launch-plans/{plan_id}",
        "expires_at": plan["expires_at"].isoformat(),
    }


@router.get("/api/v1/pipelines/{plan_id}")
def api_get_pipeline(plan_id: str, request: Request):
    """Stage-by-stage state. A foreign plan is not-found, never forbidden."""
    from control_plane.db import control_plane_transaction
    from control_plane.pipelines import pipeline_state

    user, principal = _resolve_principal(request)
    _require_scope(user, "instances:read")

    with control_plane_transaction() as conn:
        owner = conn.execute(
            "SELECT tenant_id, status FROM action_plans "
            " WHERE plan_id = %s AND action_type = 'run_pipeline'",
            (plan_id,),
        ).fetchone()
        if not owner or str(owner[0]) != principal.tenant_id:
            return problem_response(
                status=404, code="not_found", detail="no such pipeline"
            )
        rows = conn.execute(
            """SELECT stage_index, name, action_type, state, on_failure,
                      attempt_count, max_attempts, failure_code, result_ref,
                      spent_micros
                 FROM pipeline_stages
                WHERE plan_id = %s ORDER BY stage_index""",
            (plan_id,),
        ).fetchall()
        summary = pipeline_state(conn, plan_id)

    return {
        "ok": True,
        "plan_id": plan_id,
        "approval_state": owner[1],
        "finished": summary["finished"],
        "failed": summary["failed"],
        "stages": [
            {
                "index": r[0], "name": r[1], "action_type": r[2], "state": r[3],
                "on_failure": r[4], "attempt_count": r[5], "max_attempts": r[6],
                "failure_code": r[7], "result_ref": r[8], "spent_micros": r[9],
            }
            for r in rows
        ],
    }
