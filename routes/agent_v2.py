"""Routes: /agent/v2 — the fenced worker protocol (blueprint §11, Phase 5).

Execution authority becomes unambiguous here: work delivery is claim +
ACK (never destructive drain), the lease claim is a hard gate the worker
must pass before any container starts, and every status report carries
the full ``job/attempt/host/fence`` authority tuple. A 409 with
``code=fencing_violation`` (or a rejected renewal) is the worker's
definitive signal that authority is gone — it must stop the container.

Rollout is negotiated per host: ``XCELSIOR_AGENT_V2_HOSTS`` is a csv of
host_ids (or ``*``) allowed to speak v2; everyone else keeps the v1
poll/drain protocol untouched.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from control_plane.attempts import AttemptStatusRejected, report_attempt_status
from control_plane.observations import ingest_observation
from control_plane.commands import (
    CommandProtocolError,
    ack_command,
    claim_commands,
    nack_command,
)
from control_plane.db import run_transaction
from control_plane.leases import (
    LeaseClaimRejected,
    LeaseError,
    LeaseRenewRejected,
    claim_lease,
    release_lease,
    renew_lease,
)
from routes.agent import _require_agent_auth

router = APIRouter()

PROTOCOL_VERSION = 2
FEATURES = (
    "commands.claim_ack",
    "leases.fenced",
    "attempts.fenced_status",
)


def _v2_host_allowlist() -> frozenset[str]:
    raw = os.environ.get("XCELSIOR_AGENT_V2_HOSTS") or ""
    return frozenset(p.strip().lower() for p in raw.split(",") if p.strip())


def _v2_enabled_for(host_id: str) -> bool:
    allow = _v2_host_allowlist()
    return "*" in allow or host_id.strip().lower() in allow


def _lease_conflict(exc: LeaseError) -> HTTPException:
    # Shape matches api.py's HTTPException handler contract:
    # detail={"error": {...}} is preserved verbatim in the response body.
    return HTTPException(
        status_code=409,
        detail={"error": {"code": exc.code, "message": str(exc),
                          "details": exc.details}},
    )


# ── Protocol negotiation ─────────────────────────────────────────────


@router.get("/agent/v2/negotiate/{host_id}", tags=["AgentV2"])
def api_v2_negotiate(host_id: str, request: Request):
    """Which protocol should this host speak? (canary rollout control)"""
    _require_agent_auth(request, host_id=host_id)
    enabled = _v2_enabled_for(host_id)
    return {
        "ok": True,
        "protocol": PROTOCOL_VERSION if enabled else 1,
        "v2": enabled,
        "features": list(FEATURES) if enabled else [],
    }


# ── Command claim + ACK (§11.1 / §9.4) ───────────────────────────────


class CommandClaimRequest(BaseModel):
    host_id: str = Field(min_length=1, max_length=128)
    worker_session_id: str = Field(min_length=1, max_length=128)
    limit: int = Field(default=10, ge=1, le=50)


@router.post("/agent/v2/commands/claim", tags=["AgentV2"])
def api_v2_commands_claim(req: CommandClaimRequest, request: Request):
    """Claim attempt-bound commands (pending → claimed; redelivered on
    crash via claim expiry — never destructively drained)."""
    _require_agent_auth(request, host_id=req.host_id)
    claimed = run_transaction(
        lambda c: claim_commands(
            c,
            host_id=req.host_id,
            worker_session_id=req.worker_session_id,
            limit=req.limit,
            attempt_commands_only=True,
        ),
        what="v2_commands_claim",
    )
    return {
        "ok": True,
        "commands": [
            {
                "command_id": c.command_id,
                "command": c.command,
                "args": c.args,
                "job_id": c.job_id,
                "attempt_id": c.attempt_id,
                "fencing_token": c.fencing_token,
                "spec_hash": c.spec_hash,
                "idempotency_key": c.idempotency_key,
                "attempt_count": c.attempt_count,
                "claim_expires_at": str(c.claim_expires_at),
            }
            for c in claimed
        ],
    }


class CommandAckRequest(BaseModel):
    host_id: str = Field(min_length=1, max_length=128)
    result: dict[str, Any] | None = None


@router.post("/agent/v2/commands/{command_id}/ack", tags=["AgentV2"])
def api_v2_command_ack(command_id: str, req: CommandAckRequest, request: Request):
    """Once-only terminal ACK; duplicates replay the stored result."""
    _require_agent_auth(request, host_id=req.host_id)
    try:
        outcome = run_transaction(
            lambda c: ack_command(
                c, command_id=command_id, host_id=req.host_id, result=req.result
            ),
            what="v2_command_ack",
        )
    except CommandProtocolError as exc:
        raise HTTPException(
            409, detail={"error": {"code": "ack_rejected", "message": str(exc)}}
        )
    return {"ok": True, "duplicate": outcome.duplicate, "result": outcome.result}


class CommandNackRequest(BaseModel):
    host_id: str = Field(min_length=1, max_length=128)
    error_code: str = Field(min_length=1, max_length=64)
    error_details: dict[str, Any] | None = None
    retryable: bool = True


@router.post("/agent/v2/commands/{command_id}/nack", tags=["AgentV2"])
def api_v2_command_nack(command_id: str, req: CommandNackRequest, request: Request):
    """Typed failure report → bounded-backoff redelivery or dead-letter."""
    _require_agent_auth(request, host_id=req.host_id)
    try:
        status = run_transaction(
            lambda c: nack_command(
                c,
                command_id=command_id,
                host_id=req.host_id,
                error_code=req.error_code,
                error_details=req.error_details,
                retryable=req.retryable,
            ),
            what="v2_command_nack",
        )
    except CommandProtocolError as exc:
        raise HTTPException(
            409, detail={"error": {"code": "nack_rejected", "message": str(exc)}}
        )
    return {"ok": True, "status": status}


# ── Fenced leases (§11.2 hard gate) ──────────────────────────────────


class LeaseAuthorityRequest(BaseModel):
    lease_id: str = Field(min_length=1, max_length=64)
    job_id: str = Field(min_length=1, max_length=128)
    attempt_id: str = Field(min_length=1, max_length=64)
    host_id: str = Field(min_length=1, max_length=128)
    fencing_token: int = Field(ge=1)
    worker_session_id: str = Field(default="", max_length=128)


@router.post("/agent/v2/leases/claim", tags=["AgentV2"])
def api_v2_lease_claim(req: LeaseAuthorityRequest, request: Request):
    """THE hard gate: no 200 here → no container start, ever."""
    _require_agent_auth(request, host_id=req.host_id)
    try:
        grant = run_transaction(
            lambda c: claim_lease(
                c,
                lease_id=req.lease_id,
                job_id=req.job_id,
                attempt_id=req.attempt_id,
                host_id=req.host_id,
                fencing_token=req.fencing_token,
                worker_session_id=req.worker_session_id,
            ),
            what="v2_lease_claim",
        )
    except LeaseClaimRejected as exc:
        raise _lease_conflict(exc)
    return {
        "ok": True,
        "lease_id": grant.lease_id,
        "job_id": grant.job_id,
        "attempt_id": grant.attempt_id,
        "fencing_token": grant.fencing_token,
        "expires_at": str(grant.expires_at),
        "renewal_ttl_sec": grant.renewal_ttl_sec,
    }


@router.post("/agent/v2/leases/renew", tags=["AgentV2"])
def api_v2_lease_renew(req: LeaseAuthorityRequest, request: Request):
    """CAS renewal under the exact authority tuple. A 409 means the
    fence is lost for good — stop the container (§11.5)."""
    _require_agent_auth(request, host_id=req.host_id)
    try:
        expires_at = run_transaction(
            lambda c: renew_lease(
                c,
                lease_id=req.lease_id,
                attempt_id=req.attempt_id,
                host_id=req.host_id,
                fencing_token=req.fencing_token,
                worker_session_id=req.worker_session_id,
            ),
            what="v2_lease_renew",
        )
    except LeaseRenewRejected as exc:
        raise _lease_conflict(exc)
    return {"ok": True, "expires_at": str(expires_at)}


@router.post("/agent/v2/leases/release", tags=["AgentV2"])
def api_v2_lease_release(req: LeaseAuthorityRequest, request: Request):
    """Voluntary release by the current authority holder; idempotent."""
    _require_agent_auth(request, host_id=req.host_id)
    released = run_transaction(
        lambda c: release_lease(
            c,
            lease_id=req.lease_id,
            attempt_id=req.attempt_id,
            host_id=req.host_id,
            fencing_token=req.fencing_token,
        ),
        what="v2_lease_release",
    )
    return {"ok": True, "released": bool(released)}


# ── Host observations (§12.2) ────────────────────────────────────────


class ObservedWorkload(BaseModel):
    job_id: str | None = Field(default=None, max_length=128)
    attempt_id: str | None = Field(default=None, max_length=64)
    fencing_token: int | None = Field(default=None, ge=1)
    container_id: str | None = Field(default=None, max_length=128)
    container_name: str | None = Field(default=None, max_length=256)
    spec_hash: str | None = Field(default=None, max_length=128)
    state: str = Field(default="unknown", max_length=32)
    details: dict[str, Any] | None = None


class ObservationReport(BaseModel):
    host_id: str = Field(min_length=1, max_length=128)
    worker_session_id: str = Field(min_length=1, max_length=128)
    observation_generation: int = Field(ge=0)
    workloads: list[ObservedWorkload] = Field(default_factory=list, max_length=256)
    agent_version: str | None = Field(default=None, max_length=64)
    worker_reported_at: float | None = None


@router.post("/agent/v2/observations", tags=["AgentV2"])
def api_v2_observations(report: ObservationReport, request: Request):
    """Ingest one full-state worker observation (immutable snapshot);
    enqueues the host for desired-vs-observed reconciliation."""
    _require_agent_auth(request, host_id=report.host_id)
    result = run_transaction(
        lambda c: ingest_observation(
            c,
            host_id=report.host_id,
            session_id=report.worker_session_id,
            observation_generation=report.observation_generation,
            workloads=[w.model_dump() for w in report.workloads],
            agent_version=report.agent_version,
            worker_reported_at=report.worker_reported_at,
        ),
        what="v2_observation_ingest",
    )
    return {
        "ok": True,
        "duplicate": result.duplicate,
        "observation_id": result.observation_id,
        "workloads": result.workloads,
    }


# ── Fenced attempt status (§8.1 write gate) ──────────────────────────


class AttemptStatusRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=128)
    attempt_id: str = Field(min_length=1, max_length=64)
    host_id: str = Field(min_length=1, max_length=128)
    fencing_token: int = Field(ge=1)
    status: str = Field(min_length=1, max_length=32)
    failure_code: str | None = Field(default=None, max_length=64)
    detail: dict[str, Any] | None = None


@router.post("/agent/v2/attempts/status", tags=["AgentV2"])
def api_v2_attempt_status(req: AttemptStatusRequest, request: Request):
    """Fenced worker status report. Terminal reports settle allocations,
    lease, and job projection atomically. 409 fencing_violation = the
    caller's authority is gone; it must stop its container."""
    _require_agent_auth(request, host_id=req.host_id)
    try:
        result = run_transaction(
            lambda c: report_attempt_status(
                c,
                job_id=req.job_id,
                attempt_id=req.attempt_id,
                host_id=req.host_id,
                fencing_token=req.fencing_token,
                status=req.status,
                failure_code=req.failure_code,
                detail=req.detail,
            ),
            what="v2_attempt_status",
        )
    except LeaseError as exc:  # FencingViolation
        raise _lease_conflict(exc)
    except AttemptStatusRejected as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": exc.code, "message": str(exc)}},
        )
    return {
        "ok": True,
        "status": result.status,
        "changed": result.changed,
        "terminal": result.terminal,
    }
