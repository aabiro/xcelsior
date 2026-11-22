"""Routes: verification."""

import re
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from routes._deps import (
    _require_admin,
    log,
)
from scheduler import (
    list_hosts,
    log,
)
from db import emit_event
from verification import get_verification_engine
from reputation import VerificationType, get_reputation_engine

router = APIRouter()


# ── Model: VerifyHostRequest ──


class VerifyHostRequest(BaseModel):
    gpu_info: dict = Field(default_factory=dict)
    network_info: dict = Field(default_factory=dict)


@router.post("/api/verify/{host_id}", tags=["Verification"])
def api_verify_host(host_id: str, req: VerifyHostRequest, request: Request):
    """Run verification checks on a host against a caller-supplied report. Admin only.

    The caller supplies the measurements, so whoever may call this may verify
    any host with numbers they typed. It required only a signed-in user (the
    `verification:write` scope check is a no-op for browser sessions), which
    let any account mark any host verified — and with verified-only placement
    that decides where every job runs. Real reports come from the host's own
    agent through `/agent/verify`, which is bound to that host.
    """
    from routes._deps import _get_current_user, _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "verification:write")
    _require_admin(request)
    ve = get_verification_engine()
    result = ve.run_verification(host_id, {**req.gpu_info, **req.network_info})
    return {"ok": True, "host_id": host_id, "verification": result}


@router.get("/api/verify/{host_id}/status", tags=["Verification"])
def api_verification_status(host_id: str, request: Request):
    """Current verification status for one host. Requires a signed-in user.

    Three things were wrong here, and they compounded.

    **It was anonymous, and returned more than the listing beside it.**
    `v.__dict__` is the whole record: `deverify_reason`, `gpu_fingerprint`,
    `failure_count`, and `checks` — each check's expected-versus-actual values
    from the hardware report. `/api/verified-hosts` was gated for exposing a
    curated subset of exactly that; leaving this open made the gate bypassable
    one host at a time, and host ids come free from `/marketplace/search` and
    `/compute-scores`, both public.

    **`__dict__` is opt-out exposure.** Any field added to `HostVerification`
    later would be published by this route without anyone deciding to. The
    response is now an explicit shape, so publishing a new field takes an edit
    here.

    **The two branches returned different shapes.** A host with no record got
    `status` at the top level; a host *with* one got `verification` and no
    `status` at all. `dashboard/hosts/[id]/page.tsx` reads `verification.status`
    to colour its badge, so the badge read `undefined` for precisely the hosts
    that were verified — and `fetchVerificationStatus` in `lib/api.ts` types the
    response as `{ok, host_id, status}`, which matches only the empty branch, so
    TypeScript enforced the broken shape rather than catching it. `status` is
    now always present.
    """
    from routes._deps import _require_auth

    _require_auth(request)

    store = get_verification_engine().store
    v = store.get_verification(host_id)
    if not v:
        return {"ok": True, "host_id": host_id, "status": "unverified"}

    from verification import overdue_after, verification_freshness

    state = getattr(v.state, "value", None) or str(v.state)
    return {
        "ok": True,
        "host_id": host_id,
        "status": state,
        # `verified` past its next check plus the grace reads `overdue`; the
        # placement gate treats an overdue host as not verified.
        "freshness": verification_freshness(state, v.last_check_at, v.next_check_at),
        "overdue_after": (
            overdue_after(v.last_check_at, v.next_check_at) if state == "verified" else None
        ),
        "on_hold": v.hold_active(),
        "overall_score": v.overall_score,
        "verified_at": v.verified_at,
        "last_check_at": v.last_check_at,
        "next_check_at": v.next_check_at,
    }


@router.get("/api/verified-hosts", tags=["Verification"])
def api_verified_hosts(request: Request):
    """List hosts with their verification details. Requires a signed-in user.

    Returns host_id, state, gpu_model, country, last_check, overall_score for
    every host that has any verification record — not just 'verified'.

    It was anonymous, and it returns two fields that should not be: each host's
    `gpu_fingerprint`, and `deverify_reason` — the reason a provider's machine
    *failed* verification. Anyone could enumerate which providers had been
    deverified and why, which is commercially sensitive to them and was never
    the point of the endpoint.

    A signed-in user rather than an admin, because that is who consumes it:
    `/dashboard/trust` is an ordinary dashboard page, and the legacy console
    reaches it through a `window.fetch` wrapper that attaches the bearer token
    to every API call. Requiring admin would break the first; requiring nothing
    was the bug.
    """
    from routes._deps import _require_auth

    _require_auth(request)
    ve = get_verification_engine()
    store = ve.store
    # Return all hosts with verification records (any state)
    with store._conn() as conn:
        rows = conn.execute(
            "SELECT host_id, state, overall_score, last_check_at, next_check_at, "
            "gpu_fingerprint, deverify_reason FROM host_verifications ORDER BY state, host_id"
        ).fetchall()
    from verification import verification_freshness

    # Enrich with host data
    all_hosts = list_hosts(active_only=False)
    host_map = {h["host_id"]: h for h in all_hosts}
    result = []
    for r in rows:
        h = host_map.get(r["host_id"], {})
        result.append(
            {
                "host_id": r["host_id"],
                "status": r["state"],
                "freshness": verification_freshness(
                    r["state"], r["last_check_at"], r["next_check_at"]
                ),
                "overall_score": r["overall_score"],
                "last_check": r["last_check_at"],
                "next_check_at": r["next_check_at"],
                "gpu_fingerprint": r["gpu_fingerprint"],
                "deverify_reason": r["deverify_reason"] or "",
                "gpu_model": h.get("gpu_model", "—"),
                "country": h.get("country", ""),
                "province": h.get("province", ""),
            }
        )
    return {"ok": True, "count": len(result), "hosts": result}


@router.post("/api/verify/{host_id}/approve", tags=["Verification"])
def api_admin_approve_host(host_id: str, request: Request, notes: str = ""):
    """Admin manually approves a host, overriding automated checks.

    Sets host verification state to 'verified' regardless of check results and
    clears any hold. Useful when an admin has physically inspected hardware or
    reviewed logs. Clearing an identity-mismatch hold also clears the GPU
    identity baseline, so the next passing verification records the new card.
    """
    _require_admin(request)
    ve = get_verification_engine()
    store = ve.store
    existing = store.get_verification(host_id)
    if not existing:
        # Create a new verification record for this host
        from verification import HostVerification, HostVerificationState

        existing = HostVerification(
            verification_id=str(uuid.uuid4())[:12],
            host_id=host_id,
            state=HostVerificationState.UNVERIFIED,
        )
    from verification import HOLD_IDENTITY_MISMATCH, VERIFICATION_THRESHOLDS

    cleared_hold = existing.hold_reason
    now = time.time()
    existing.state = "verified"
    existing.verified_at = now
    existing.deverified_at = None
    existing.deverify_reason = ""
    existing.overall_score = 100.0
    existing.last_check_at = now
    existing.next_check_at = now + VERIFICATION_THRESHOLDS["reverify_interval_sec"]
    # An approval is the admin clearing whatever held the host. A held identity
    # mismatch means the card changed: the old baseline no longer describes the
    # machine, so the next passing verification records a new one.
    existing.hold_reason = ""
    existing.hold_until = None
    if cleared_hold == HOLD_IDENTITY_MISMATCH:
        existing.identity_baseline = {}
    store.save_verification(existing)
    log.info(
        "ADMIN APPROVED host=%s notes=%s cleared_hold=%s",
        host_id,
        notes or "(none)",
        cleared_hold or "(none)",
    )
    emit_event(
        "verification_override",
        {"host_id": host_id, "action": "approve", "notes": notes, "cleared_hold": cleared_hold},
    )
    return {
        "ok": True,
        "host_id": host_id,
        "status": "verified",
        "approved_by": "admin",
        "cleared_hold": cleared_hold,
    }


@router.post("/api/verify/{host_id}/reject", tags=["Verification"])
def api_admin_reject_host(host_id: str, request: Request, reason: str = "Admin rejection"):
    """Admin manually rejects/deverifies a host.

    Sets host verification state to 'deverified', held until an admin approves
    it again, so it cannot receive jobs while the placement gate is on.
    """
    _require_admin(request)
    ve = get_verification_engine()
    store = ve.store
    existing = store.get_verification(host_id)
    if not existing:
        from verification import HostVerification, HostVerificationState

        existing = HostVerification(
            verification_id=str(uuid.uuid4())[:12],
            host_id=host_id,
            state=HostVerificationState.UNVERIFIED,
        )
    from verification import HOLD_ADMIN_REJECTED

    existing.state = "deverified"
    existing.deverified_at = time.time()
    existing.deverify_reason = f"Admin: {reason}"
    existing.last_check_at = time.time()
    # Held until an admin approves: the reverification sweep must not undo a
    # rejection because a later benchmark happened to pass.
    existing.hold_reason = HOLD_ADMIN_REJECTED
    existing.hold_until = None
    existing.next_check_at = None
    store.save_verification(existing)
    log.warning("ADMIN REJECTED host=%s reason=%s", host_id, reason)
    emit_event("verification_override", {"host_id": host_id, "action": "reject", "reason": reason})
    return {"ok": True, "host_id": host_id, "status": "deverified", "reason": reason}


# ── Model: VerificationReportPayload ──


class VerificationReportPayload(BaseModel):
    host_id: str
    report: dict


@router.post("/agent/verify", tags=["Verification"])
def api_agent_verify(payload: VerificationReportPayload, request: Request):
    """Receive comprehensive benchmark report and run verification checks.

    Agent-authenticated, and bound to the host named in the body.

    It had no `Request` parameter at all, so there was nothing to authenticate
    *with*: any unauthenticated caller could POST a report naming any
    `host_id`. That is not a read — `run_verification` writes the host's
    verification state, and a passing result grants HARDWARE_AUDIT reputation
    below. So the endpoint let anyone verify a host they do not own with a
    forged benchmark, or push someone else's host toward deverification with a
    failing one, and reputation feeds provider scoring and earnings.

    Every other agent-reported endpoint already does this —
    `_require_agent_auth(request, host_id=report.host_id)` is the shape used
    throughout `routes/agent.py`, and binding to the body's host id is the part
    that stops one provider reporting as another. The sibling user-facing route
    `api_verify_host` above requires `verification:write`; this one is its
    agent twin and was the half without a gate.

    `/agent/v2/verify` is an alias onto this same function
    (`routes/agent_v2_aliases.py`), so it was open too and is closed by the
    same change.
    """
    from routes.agent import _require_agent_auth

    _require_agent_auth(request, host_id=payload.host_id)

    ve = get_verification_engine()
    result = ve.run_verification(payload.host_id, payload.report)

    # Wire verification → reputation: grant HARDWARE_AUDIT points on pass
    if result.state == "verified" or getattr(result.state, "value", None) == "verified":
        try:
            re = get_reputation_engine()
            re.add_verification(payload.host_id, VerificationType.HARDWARE_AUDIT)
            log.info("REPUTATION HARDWARE_AUDIT granted for verified host %s", payload.host_id)
        except Exception as e:
            log.exception("Non-fatal: could not update reputation for %s", payload.host_id)

    return {
        "ok": True,
        "host_id": payload.host_id,
        "state": getattr(result.state, "value", str(result.state)),
        "score": result.overall_score,
        "checks": result.checks,
        "gpu_fingerprint": result.gpu_fingerprint,
    }
