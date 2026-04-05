"""Routes: verification."""

import re
import time
import uuid

from fastapi import APIRouter, Request
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
def api_verify_host(host_id: str, req: VerifyHostRequest):
    """Run verification checks on a host."""
    ve = get_verification_engine()
    result = ve.run_verification(host_id, req.gpu_info, req.network_info)
    return {"ok": True, "host_id": host_id, "verification": result}

@router.get("/api/verify/{host_id}/status", tags=["Verification"])
def api_verification_status(host_id: str):
    """Get current verification status for a host."""
    store = get_verification_engine().store
    v = store.get_verification(host_id)
    if not v:
        return {"ok": True, "host_id": host_id, "status": "unverified"}
    return {"ok": True, "host_id": host_id, "verification": v.__dict__}

@router.get("/api/verified-hosts", tags=["Verification"])
def api_verified_hosts():
    """List all verified hosts with full verification details.

    Returns host_id, state, gpu_model, country, last_check, overall_score
    for every host that has any verification record (not just 'verified').
    """
    ve = get_verification_engine()
    store = ve.store
    # Return all hosts with verification records (any state)
    with store._conn() as conn:
        rows = conn.execute(
            "SELECT host_id, state, overall_score, last_check_at, gpu_fingerprint, deverify_reason FROM host_verifications ORDER BY state, host_id"
        ).fetchall()
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
                "overall_score": r["overall_score"],
                "last_check": r["last_check_at"],
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

    Sets host verification state to 'verified' regardless of check results.
    Useful when an admin has physically inspected hardware or reviewed logs.
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
    existing.state = "verified"
    existing.verified_at = time.time()
    existing.deverified_at = None
    existing.deverify_reason = ""
    existing.overall_score = 100.0
    existing.last_check_at = time.time()
    existing.next_check_at = time.time() + 86400
    store.save_verification(existing)
    log.info("ADMIN APPROVED host=%s notes=%s", host_id, notes or "(none)")
    emit_event("verification_override", {"host_id": host_id, "action": "approve", "notes": notes})
    return {"ok": True, "host_id": host_id, "status": "verified", "approved_by": "admin"}

@router.post("/api/verify/{host_id}/reject", tags=["Verification"])
def api_admin_reject_host(host_id: str, request: Request, reason: str = "Admin rejection"):
    """Admin manually rejects/deverifies a host.

    Sets host verification state to 'deverified' so it cannot receive jobs.
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
    existing.state = "deverified"
    existing.deverified_at = time.time()
    existing.deverify_reason = f"Admin: {reason}"
    existing.last_check_at = time.time()
    store.save_verification(existing)
    log.warning("ADMIN REJECTED host=%s reason=%s", host_id, reason)
    emit_event("verification_override", {"host_id": host_id, "action": "reject", "reason": reason})
    return {"ok": True, "host_id": host_id, "status": "deverified", "reason": reason}


# ── Model: VerificationReportPayload ──

class VerificationReportPayload(BaseModel):
    host_id: str
    report: dict

@router.post("/agent/verify", tags=["Verification"])
def api_agent_verify(payload: VerificationReportPayload):
    """Receive comprehensive benchmark report and run verification checks."""
    ve = get_verification_engine()
    result = ve.run_verification(payload.host_id, payload.report)

    # Wire verification → reputation: grant HARDWARE_AUDIT points on pass
    if result.state == "verified" or (hasattr(result.state, "value") and result.state.value == "verified"):
        try:
            re = get_reputation_engine()
            re.add_verification(payload.host_id, VerificationType.HARDWARE_AUDIT)
            log.info("REPUTATION HARDWARE_AUDIT granted for verified host %s", payload.host_id)
        except Exception as e:
            log.exception("Non-fatal: could not update reputation for %s", payload.host_id)

    return {
        "ok": True,
        "host_id": payload.host_id,
        "state": result.state.value if hasattr(result.state, "value") else str(result.state),
        "score": result.overall_score,
        "checks": result.checks,
        "gpu_fingerprint": result.gpu_fingerprint,
    }

