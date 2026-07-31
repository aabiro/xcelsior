"""Routes: authoritative host admission (migration 082).

Trust boundary, which is the whole point of this surface:

- A **provider** may open a compatibility session for a host they operate and
  submit hardware evidence for it. Everything they submit is *advisory*. A
  provider describing their own machine cannot be the authority that lets that
  machine take paid work.
- Only an **operator** (admin) may record authoritative evidence or decide
  admission. Those two paths are the only way a host becomes admitted.

Keeping the split at the routing layer means the service cannot be reached
through a path that quietly upgrades advisory evidence into an admission.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

import host_admission
from routes._deps import _require_admin, _require_auth, _require_scope
from routes.hosts import _require_host_operator, _resolve_host_id

router = APIRouter()


def _resolve_or_404(host_id: str) -> str:
    resolved, _ = _resolve_host_id(host_id)
    if not resolved:
        raise HTTPException(404, f"Host {host_id} not found")
    return resolved


def _translate(exc: host_admission.AdmissionError) -> HTTPException:
    """Surface the service's own code and status rather than flattening to 400.

    Callers need to tell "this host is not ready yet" (422) from "someone else
    just changed it" (409) from "your session token is wrong" (403), because
    only one of those is worth retrying.
    """
    return HTTPException(
        status_code=exc.http_status,
        detail={"code": exc.code, "message": exc.message, **exc.details},
    )


# ── Provider surface (advisory only) ──────────────────────────────────


class CreateSessionRequest(BaseModel):
    helper_public_key_spki: str = Field(..., min_length=1, max_length=4096)
    idempotency_key: str = Field(..., min_length=1, max_length=200)
    ttl_seconds: int = Field(default=host_admission.SESSION_TTL_SECONDS, ge=60, le=3600)


@router.post("/api/hosts/{host_id}/compatibility-sessions", tags=["Hosts"])
def api_create_compatibility_session(host_id: str, body: CreateSessionRequest, request: Request):
    """Open an expiring compatibility session for a host you operate."""
    user = _require_auth(request)
    _require_scope(user, "hosts:write")
    resolved = _resolve_or_404(host_id)
    _require_host_operator(user, resolved)
    try:
        return host_admission.create_compatibility_session(
            host_id=resolved,
            actor=user,
            helper_public_key_spki=body.helper_public_key_spki,
            idempotency_key=body.idempotency_key,
            ttl_seconds=body.ttl_seconds,
        )
    except host_admission.AdmissionError as exc:
        raise _translate(exc) from exc


class SubmitEvidenceRequest(BaseModel):
    submit_token: str = Field(..., min_length=1, max_length=512)
    report: dict[str, Any]
    signature: str = Field(..., min_length=1, max_length=4096)


@router.post("/api/hosts/compatibility-sessions/{session_id}/evidence", tags=["Hosts"])
def api_submit_compatibility_evidence(
    session_id: str, body: SubmitEvidenceRequest, request: Request
):
    """Submit signed helper output against an open session.

    Authorisation is the session's own proof-of-possession: the report must
    carry an Ed25519 signature from the key the session was opened with, and
    the submit token must match. The session is bound to a host, so no host id
    is accepted from the caller here.
    """
    user = _require_auth(request)
    _require_scope(user, "hosts:write")
    try:
        return host_admission.submit_compatibility_evidence(
            session_id=session_id,
            actor=user,
            submit_token=body.submit_token,
            report=body.report,
            signature=body.signature,
        )
    except host_admission.AdmissionError as exc:
        raise _translate(exc) from exc


class ProviderEvidenceRequest(BaseModel):
    report: dict[str, Any]


@router.post("/api/hosts/{host_id}/provider-evidence", tags=["Hosts"])
def api_record_provider_evidence(host_id: str, body: ProviderEvidenceRequest, request: Request):
    """Record advisory evidence reported by the provider's own agent."""
    user = _require_auth(request)
    _require_scope(user, "hosts:write")
    resolved = _resolve_or_404(host_id)
    _require_host_operator(user, resolved)
    try:
        return host_admission.record_provider_agent_evidence(
            host_id=resolved, actor=user, report=body.report
        )
    except host_admission.AdmissionError as exc:
        raise _translate(exc) from exc


@router.get("/api/hosts/{host_id}/admission", tags=["Hosts"])
def api_admission_status(host_id: str, request: Request):
    """Current admission state and what evidence is still outstanding."""
    user = _require_auth(request)
    _require_scope(user, "hosts:read")
    resolved = _resolve_or_404(host_id)
    _require_host_operator(user, resolved)
    try:
        return host_admission.admission_status(resolved)
    except host_admission.AdmissionError as exc:
        raise _translate(exc) from exc


@router.get("/api/admin/admission-queue", tags=["Hosts"])
def api_admission_queue(request: Request, limit: int = 100):
    """Hosts awaiting an operator decision, with evidence readiness."""
    _require_admin(request)
    return {"ok": True, "queue": host_admission.admission_queue(limit=limit)}


# ── Operator surface (the only path that can admit) ───────────────────


class AuthoritativeEvidenceRequest(BaseModel):
    evidence_type: str = Field(..., min_length=1, max_length=64)
    verdict: str = Field(..., pattern="^(pass|fail|inconclusive)$")
    summary: dict[str, Any]
    idempotency_key: str = Field(..., min_length=1, max_length=200)
    validity_seconds: int = Field(default=86400, ge=60, le=86400 * 30)


@router.post("/api/admin/hosts/{host_id}/authoritative-evidence", tags=["Hosts"])
def api_record_authoritative_evidence(
    host_id: str, body: AuthoritativeEvidenceRequest, request: Request
):
    """Record operator-signed evidence. Admin only — this is the trust anchor."""
    user = _require_admin(request)
    resolved = _resolve_or_404(host_id)
    try:
        return host_admission.record_authoritative_evidence(
            host_id=resolved,
            actor=user,
            evidence_type=body.evidence_type,
            verdict=body.verdict,
            summary=body.summary,
            idempotency_key=body.idempotency_key,
            validity_seconds=body.validity_seconds,
        )
    except host_admission.AdmissionError as exc:
        raise _translate(exc) from exc


class AdmissionDecisionRequest(BaseModel):
    action: str = Field(..., pattern="^(admit|reject|revoke)$")
    reason: str = Field(..., min_length=1, max_length=500)
    idempotency_key: str = Field(..., min_length=1, max_length=200)
    # Optimistic concurrency: two operators acting on the same host at once
    # must not silently overwrite each other, so a stale version is a 409.
    expected_version: int | None = Field(default=None, ge=0)


@router.post("/api/admin/hosts/{host_id}/admission-decisions", tags=["Hosts"])
def api_decide_admission(host_id: str, body: AdmissionDecisionRequest, request: Request):
    """Admit, reject, or revoke a host. Admin only."""
    user = _require_admin(request)
    resolved = _resolve_or_404(host_id)
    try:
        return host_admission.decide_admission(
            host_id=resolved,
            actor=user,
            action=body.action,
            reason=body.reason,
            idempotency_key=body.idempotency_key,
            expected_version=body.expected_version,
        )
    except host_admission.AdmissionError as exc:
        raise _translate(exc) from exc
