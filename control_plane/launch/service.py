"""The unified launch service (§14).

One entry point every surface calls. :func:`preview` implements §14.1 steps
1–8 and is strictly side-effect-free apart from persisting the action plan
itself: it creates *no* attempt, allocation, lease, wallet hold, or job row
(that is execute's job, B2.5). Approval (B2.4) and execute (B2.5) live here
too so a plan's whole lifecycle has one authority.

The service is transport-agnostic: it receives an already-resolved
``Principal`` (who is acting, in which tenant) and a raw request dict using
the REST ``JobIn`` field names, and returns plain dicts. The MCP tools, the
dashboard, and the REST routes all resolve their own principal and then call
the same functions, so they cannot diverge (§14, B2.7).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

from control_plane.db import control_plane_transaction, run_transaction
from control_plane.launch import action_plans as plans_repo
from control_plane.launch import quoting, spend_policy
# Import the functions from the submodule path directly: the package __init__
# re-exports a `canonicalize` *function*, which shadows the same-named
# submodule for attribute access — `from ...canonicalize import` sidesteps it.
from control_plane.launch.canonicalize import canonicalize, spec_hash as _spec_hash
from control_plane.launch.validation import validate_canonical_spec
from control_plane.scheduler.filters import FilterContext, aggregate_reason, filter_hosts
from control_plane.scheduler.snapshot import take_snapshot

# Plan expiry — B18 variable. A quoted price is only honoured for this long,
# after which the plan can no longer be approved or executed (§9.5 expiry).
_DEFAULT_PLAN_TTL_SEC = 600
# Default price tolerance (basis points) recorded on the plan; execute
# re-quotes and refuses beyond it (§15.4).
DEFAULT_TOLERANCE_BPS = 500

# The scope an action requires. Kept here so a new action type cannot exist
# without declaring one (B5.1 will drive the MCP scope check from the plan).
ACTION_REQUIRED_SCOPES: dict[str, list[str]] = {
    "create_instance": ["instances:operate"],
}


def _iso(value: Any) -> str:
    """Render a timestamptz (or anything) as an ISO string for JSON."""
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def plan_ttl_sec() -> int:
    try:
        return max(60, int(os.environ.get("XCELSIOR_MCP_ACTION_PLAN_TTL_SEC", "")))
    except (TypeError, ValueError):
        return _DEFAULT_PLAN_TTL_SEC


class LaunchPlanError(Exception):
    """A launch-plan operation the transport must map to an HTTP status.

    Carries a stable ``code`` (RFC 9457 ``code``, B2.8), a human ``detail``,
    and the HTTP ``status`` so every surface reports the same failure the
    same way.
    """

    def __init__(self, code: str, detail: str, *, status: int, retryable: bool = False):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.status = status
        self.retryable = retryable


class PlanNotFound(LaunchPlanError):
    def __init__(self, detail: str = "action plan not found"):
        super().__init__("plan_not_found", detail, status=404)


class PlanConflict(LaunchPlanError):
    def __init__(self, code: str, detail: str):
        super().__init__(code, detail, status=409)


@dataclass(frozen=True)
class Principal:
    """Who is acting, resolved by the transport before it calls the service."""

    principal_id: str
    tenant_id: str
    client_id: str | None = None
    team_id: str | None = None
    scopes: tuple[str, ...] = ()


def simulate_placement(spec: dict[str, Any]) -> dict[str, Any]:
    """Feasibility of placing this spec *right now* — read-only (§14.1 step 6).

    Reuses the shadow scheduler's consistent snapshot and the pure Stage-C
    hard filters. Creates no allocation and no lease; it only asks "is there
    an eligible host". Runs in its own REPEATABLE READ transaction and does
    not hold a lock while it thinks.
    """
    snapshot = run_transaction(
        lambda conn: take_snapshot(conn), what="launch_preview_snapshot"
    )
    job = {
        "gpu_model": spec.get("gpu_model"),
        "num_gpus": spec.get("num_gpus"),
        "vram_needed_gb": spec.get("vram_needed_gb"),
        "region": spec.get("region"),
    }
    ctx = FilterContext(stale_host_ids=snapshot.stale_host_ids)
    eligible, rejections = filter_hosts(job, snapshot.hosts, ctx)
    return {
        "feasible": bool(eligible),
        "eligible_hosts": len(eligible),
        "hosts_considered": len(snapshot.hosts),
        "reason": aggregate_reason(rejections) if not eligible else None,
    }


def preview(
    request: dict[str, Any],
    *,
    principal: Principal,
    action_type: str = "create_instance",
    runway_hours: float = quoting.DEFAULT_RUNWAY_HOURS,
) -> dict[str, Any]:
    """§14.1 steps 1–8. Returns a plan preview; no compute-side effects.

    On a structurally invalid spec it returns the problems and does *not*
    persist a plan — a plan for an unlaunchable spec is dead weight.
    """
    # 1–2. validate + canonicalize.
    spec = canonicalize(request)
    problems = validate_canonical_spec(spec)
    if problems:
        return {
            "ok": False,
            "problems": [p.as_dict() for p in problems],
            "plan_id": None,
        }

    spec_hash = _spec_hash(spec)

    # 4–5. versioned quote + expected burn / worst-case authorized amount.
    quote = quoting.quote_launch(spec, runway_hours=runway_hours)

    # 6. placement feasibility (read-only).
    availability = simulate_placement(spec)

    required_scopes = ACTION_REQUIRED_SCOPES.get(action_type, [])

    # 3 + 7. policy-driven approval mode, then persist the plan.
    with control_plane_transaction() as conn:
        policy = plans_repo.load_client_policy(
            conn, client_id=principal.client_id, tenant_id=principal.tenant_id
        )
        decision = spend_policy.evaluate(
            spec, estimate_micros=quote.estimate_micros, policy=policy
        )
        plan = plans_repo.create_quoted_plan(
            conn,
            action_type=action_type,
            principal_id=principal.principal_id,
            client_id=principal.client_id,
            tenant_id=principal.tenant_id,
            team_id=principal.team_id,
            canonical_args=spec,
            canonical_args_hash=spec_hash,
            spec_hash=spec_hash,
            quote_id=quote.quote_id,
            pricing_version=quote.pricing_version,
            estimate_micros=quote.estimate_micros,
            currency=quote.currency,
            price_tolerance_bps=DEFAULT_TOLERANCE_BPS,
            required_scopes=required_scopes,
            approval_mode=decision.approval_mode,
            ttl_sec=plan_ttl_sec(),
        )

    plan_id = str(plan["plan_id"])
    return {
        "ok": True,
        "plan_id": plan_id,
        "status": plan["status"],
        "action_type": action_type,
        "approval_mode": decision.approval_mode,
        "policy": decision.as_dict(),
        "spec_hash": spec_hash,
        "estimate": quote.as_dict(),
        "availability": availability,
        "expires_at": _iso(plan["expires_at"]),
        "next_action": "approve",
        "approval_url": f"/dashboard/launch-plans/{plan_id}"
        if decision.approval_mode == "human"
        else None,
        "required_scopes": required_scopes,
    }


def _load_owned_for_update(conn, plan_id: str, principal: Principal) -> dict[str, Any]:
    """Row-lock a plan the principal is allowed to act on, else :class:`PlanNotFound`.

    The isolation boundary is the *tenant* (workspace): a plan is invisible
    across tenants, which is why a cross-tenant id returns not-found rather
    than a permission hint (no existence leak). Within a tenant, §14.2
    deliberately lets a human approve a plan a machine client created, so
    ownership is not narrowed to the exact principal here.
    """
    plan = plans_repo.get_plan_for_update(conn, plan_id)
    if plan is None or str(plan["tenant_id"]) != principal.tenant_id:
        raise PlanNotFound()
    return plan


def _plan_view(plan: dict[str, Any]) -> dict[str, Any]:
    """The public shape of a plan for approve/revoke/execute responses."""
    return {
        "plan_id": str(plan["plan_id"]),
        "status": plan["status"],
        "approval_mode": plan.get("approval_mode"),
        "approved_by": plan.get("approved_by"),
        "approval_method": plan.get("approval_method"),
        "estimate_micros": plan.get("estimate_micros"),
        "expires_at": _iso(plan.get("expires_at")),
        "job_id": plan.get("job_id"),
        "version": plan.get("version"),
    }


def approve(plan_id: str, *, principal: Principal, is_human: bool = False) -> dict[str, Any]:
    """§14.2 approval. Server-bound; ``confirm:true`` never reaches here.

    A standing-policy plan re-checks its policy and self-approves only inside
    the ceilings; anything else needs a human. Idempotent: approving an
    already-approved plan is a no-op success.
    """
    deferred_expired = False
    updated: dict[str, Any] | None = None
    with control_plane_transaction() as conn:
        plan = _load_owned_for_update(conn, plan_id, principal)
        status = str(plan["status"])

        if status == "approved":
            return {"ok": True, "plan": _plan_view(plan), "idempotent": True}
        if status not in ("quoted", "awaiting_approval"):
            raise PlanConflict(
                "plan_not_approvable", f"a {status} plan cannot be approved"
            )
        if plans_repo.is_expired(plan):
            # Lazy expiry on access: persist the terminal transition durably
            # (it commits with this transaction), then refuse *after* the block
            # so the raise cannot roll the mark back.
            plans_repo.mark_expired(conn, plan_id)
            deferred_expired = True
        else:
            updated = _approve_locked(conn, plan, principal, is_human=is_human)

    if deferred_expired:
        raise PlanConflict("plan_expired", "the quote has expired; re-preview")
    assert updated is not None  # set on the non-expired branch
    return {"ok": True, "plan": _plan_view(updated), "idempotent": False}


def _approve_locked(
    conn, plan: dict[str, Any], principal: Principal, *, is_human: bool
) -> dict[str, Any]:
    """Apply the approval transition on an already-locked, non-expired plan.

    Returns the updated plan row. May raise :class:`PlanConflict`
    (``auto_approval_denied`` / ``human_approval_required``) inside the caller's
    transaction; those are pure refusals with no state change, so the rollback
    the raise triggers is correct.
    """
    approval_mode = str(plan["approval_mode"])
    plan_id = str(plan["plan_id"])
    if approval_mode == "standing_policy":
        policy = plans_repo.load_client_policy(
            conn, client_id=plan.get("client_id"), tenant_id=str(plan["tenant_id"])
        )
        decision = spend_policy.evaluate(
            plan["canonical_args"],
            estimate_micros=int(plan["estimate_micros"] or 0),
            policy=policy,
        )
        if decision.approval_mode != "standing_policy":
            # The policy no longer covers this plan (removed, tightened, or the
            # estimate moved). Auto-approval is refused; a human must decide.
            raise PlanConflict(
                "auto_approval_denied",
                "plan is no longer within standing policy; requires human approval",
            )
        return plans_repo.mark_approved(
            conn,
            plan_id,
            approved_by=str(plan.get("client_id") or plan["principal_id"]),
            approval_method="standing_policy",
        )
    # Human (or a no-approval action). The transport has already authenticated
    # the caller and confirmed it is an interactive human for the human path.
    if approval_mode == "human" and not is_human:
        raise PlanConflict(
            "human_approval_required",
            "this plan requires an interactive human approval",
        )
    return plans_repo.mark_approved(
        conn,
        plan_id,
        approved_by=principal.principal_id,
        approval_method="human" if approval_mode == "human" else "none",
        approval_session_id=None,
    )


def revoke(plan_id: str, *, principal: Principal, reason: str = "") -> dict[str, Any]:
    """§14.2 revoke. Idempotent; a consumed/terminal plan cannot be revoked."""
    with control_plane_transaction() as conn:
        plan = _load_owned_for_update(conn, plan_id, principal)
        status = str(plan["status"])
        if status == "revoked":
            return {"ok": True, "plan": _plan_view(plan), "idempotent": True}
        if status not in ("quoted", "awaiting_approval", "approved"):
            raise PlanConflict(
                "plan_not_revocable", f"a {status} plan cannot be revoked"
            )
        updated = plans_repo.mark_revoked(conn, plan_id, reason=reason or "revoked by owner")
        return {"ok": True, "plan": _plan_view(updated), "idempotent": False}


def _deterministic_job_id(plan_id: str) -> str:
    """A stable job id per plan, so a replayed execute upserts one job row."""
    return "job-" + hashlib.sha256(plan_id.encode()).hexdigest()[:12]


def _submit_from_spec(spec: dict[str, Any], *, job_id: str, owner: str):
    """Create the job through the one job-creation authority (B0.2 rule 11).

    Maps the canonical spec onto ``submit_job`` — the same call the REST
    ``/instance`` path makes — passing a deterministic ``job_id`` so a retry
    upserts rather than duplicates. Does *not* call ``process_queue``: §10.1
    enqueues durably and lets the scheduler claim it.
    """
    from scheduler import submit_job

    return submit_job(
        spec.get("name"),
        float(spec.get("vram_needed_gb") or 0),
        int(spec.get("priority") or 0),
        tier=spec.get("tier") or None,
        num_gpus=int(spec.get("num_gpus") or 1),
        gpu_model=spec.get("gpu_model") or None,
        nfs_server=spec.get("nfs_server") or None,
        nfs_path=spec.get("nfs_path") or None,
        nfs_mount_point=spec.get("nfs_mount_point") or None,
        image=spec.get("image") or None,
        interactive=bool(spec.get("interactive", True)),
        command=spec.get("command") or None,
        ssh_port=int(spec.get("ssh_port") or 22),
        owner=owner,
        volume_ids=list(spec.get("volume_ids") or []),
        encrypted_workspace=bool(spec.get("encrypted_workspace", False)),
        init_script=spec.get("init_script") or None,
        git_repo=spec.get("git_repo") or None,
        auto_launch=list(spec.get("auto_launch") or []),
        exposed_ports=list(spec.get("exposed_ports") or []),
        source_template_id=spec.get("template_image_id") or None,
        pricing_mode=spec.get("pricing_mode") or "on_demand",
        region=spec.get("region") or "",
        job_id=job_id,
    )


def execute(plan_id: str, *, principal: Principal) -> dict[str, Any]:
    """§14.3 execute. Exactly-once: repeated execute returns the original job.

    The plan is row-locked for the whole operation, so concurrent executes of
    one plan serialize and the loser sees ``succeeded`` and replays the stored
    response. The wallet hold and the job are both keyed to the plan
    (idempotency key / deterministic id), so even the rare crash between them
    replays without a second hold or job — the single-transaction version is
    deferred with the billing-engine ``conn=`` variant (B9.5), exactly as the
    Lightning credit path chose in B9.3b-2. A price move beyond the plan's
    tolerance returns a *replacement plan* and charges nothing (§15.4).
    """
    from billing import get_billing_engine
    from money import micros_to_cad

    replacement_spec: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    deferred_expired = False

    with control_plane_transaction() as conn:
        plan = _load_owned_for_update(conn, plan_id, principal)
        status = str(plan["status"])

        # Exactly-once replay.
        if status == "succeeded":
            return {
                "ok": True,
                "job": plan.get("idempotent_response") or {"job_id": plan.get("job_id")},
                "plan": _plan_view(plan),
                "idempotent": True,
            }
        if status in ("quoted", "awaiting_approval"):
            raise PlanConflict("plan_not_approved", "plan must be approved before execute")
        if status == "revoked":
            raise PlanConflict("plan_revoked", "a revoked plan cannot be executed")
        if status == "expired":
            raise PlanConflict("plan_expired", "the quote has expired; re-preview")
        if status != "approved":
            raise PlanConflict("plan_not_executable", f"a {status} plan cannot be executed")
        if plans_repo.is_expired(plan):
            # Lazy expiry on access: persist the terminal transition durably
            # (commits with this transaction) and refuse *after* the block, so
            # the raise cannot roll the mark back.
            plans_repo.mark_expired(conn, plan_id)
            deferred_expired = True
        else:
            spec = dict(plan["canonical_args"])
            # Integrity: the spec must still hash to what was quoted (§14.3).
            if _spec_hash(spec) != str(plan["canonical_args_hash"]):
                raise PlanConflict(
                    "spec_hash_mismatch", "plan arguments failed integrity check"
                )

            # Re-quote and enforce tolerance (§15.4).
            requote = quoting.quote_launch(spec)
            if quoting.price_moved_beyond_tolerance(
                int(plan["estimate_micros"] or 0),
                requote.estimate_micros,
                int(plan["price_tolerance_bps"] or 0),
            ):
                # Do not consume, do not charge; a replacement plan is built
                # after this transaction releases the lock (avoids nesting txns).
                replacement_spec = spec
            else:
                tenant_id = str(plan["tenant_id"])
                job_id = _deterministic_job_id(plan_id)
                hold = get_billing_engine().create_wallet_hold(
                    tenant_id,
                    micros_to_cad(int(plan["estimate_micros"] or 0)),
                    idempotency_key=f"plan:{plan_id}",
                    job_id=job_id,
                    reason="launch",
                )
                if not hold.get("held"):
                    raise LaunchPlanError(
                        "insufficient_funds",
                        "wallet balance minus active holds is below the estimate",
                        status=402,
                    )
                try:
                    _submit_from_spec(spec, job_id=job_id, owner=tenant_id)
                except Exception:
                    # Match /instance: a failed submit releases the hold rather
                    # than stranding funds until expiry.
                    try:
                        get_billing_engine().release_wallet_hold(
                            str(hold["hold_id"]), reason="launch_execute_failed"
                        )
                    except Exception:
                        pass
                    raise
                updated = plans_repo.mark_consumed(
                    conn,
                    plan_id,
                    job_id=job_id,
                    wallet_hold_id=str(hold["hold_id"]),
                    idempotent_response={"job_id": job_id, "phase": "pending"},
                    idempotency_key=f"plan:{plan_id}",
                )
                result = {
                    "ok": True,
                    "job": {"job_id": job_id, "phase": "pending"},
                    "plan": _plan_view(updated),
                    "idempotent": False,
                }

    if deferred_expired:
        raise PlanConflict("plan_expired", "the quote has expired; re-preview")
    if replacement_spec is not None:
        replacement = preview(replacement_spec, principal=principal)
        return {
            "ok": False,
            "code": "quote_changed",
            "detail": "the price moved beyond the approved tolerance; a new plan was created",
            "replacement_plan": replacement,
        }
    assert result is not None  # exactly one branch above ran
    return result
