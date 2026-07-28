"""Track B B2.8 — versioned control-plane host operations (§18, §3.3).

The §3.3 rule the blueprint is emphatic about: **draining never evicts.**
Draining a host only stops *new* placements; it must leave every running
workload running. Removing running workloads is a **separate**, separately
authorized, separately audited action (`/evictions`) that fences the workload
before it can be reassigned.

Track A's legacy `POST /host/{id}/drain` conflates the two — it calls
`run_drain_preemptions`, which "preempts all workloads on a draining host" — so
these v1 endpoints implement the correct split. Domain failures are RFC 9457
problem+json (B2.8); optimistic concurrency uses the host `version` so a stale
operator request is refused rather than racing.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from uuid import UUID
import hashlib
import json
import base64

from routes._deps import (
    _is_platform_admin,
    _require_auth,
    _require_scope,
    _user_owns_job,
    append_user_audit_event,
)
from routes.hosts import _resolve_host_id
from routes.problem import ProblemException

router = APIRouter(tags=["Control plane v1"])


class _McpAuditIn(BaseModel):
    tool_name: str = Field(min_length=1, max_length=128)
    tool_version: str = Field(max_length=32)
    transport: str = Field(max_length=32)
    client_id: str | None = Field(default=None, max_length=200)
    principal_id: str | None = Field(default=None, max_length=200)
    tenant_id: str | None = Field(default=None, max_length=200)
    team_id: str | None = Field(default=None, max_length=200)
    scopes_evaluated: list[str] = Field(default_factory=list, max_length=64)
    redacted_args_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    action_plan_id: UUID | None = None
    idempotency_key: str | None = Field(default=None, max_length=200)
    api_route: str | None = Field(default=None, max_length=300)
    api_status: int | None = None
    problem_type: str | None = Field(default=None, max_length=300)
    resource_id: str | None = Field(default=None, max_length=200)
    latency_ms: int = Field(ge=0)
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    approval_method: str | None = Field(default=None, max_length=64)
    outcome: str = Field(max_length=32)


@router.post("/api/v1/mcp/tool-audit", status_code=202)
def api_v1_mcp_tool_audit(body: _McpAuditIn, request: Request):
    """Persist a redacted MCP audit and its delivery intent atomically."""
    from control_plane.outbox import append_event
    from db import _get_pg_pool

    user = _require_auth(request)
    caller_tenant = str(
        user.get("customer_id")
        or user.get("workspace_id")
        or user.get("workspace_customer_id")
        or user.get("tenant_id")
        or ""
    )
    if not caller_tenant or body.tenant_id != caller_tenant:
        raise ProblemException(status=404, code="tenant_not_found", detail="tenant not found")
    values = body.model_dump()
    values["action_plan_id"] = str(body.action_plan_id) if body.action_plan_id else None
    with _get_pg_pool().connection() as conn:
        row = conn.execute(
            """
            INSERT INTO mcp_tool_audit (
                tool_name, tool_version, transport, client_id, principal_id,
                tenant_id, team_id, scopes_evaluated, redacted_args_hash,
                action_plan_id, idempotency_key, api_route, api_status,
                problem_type, resource_id, latency_ms, trace_id,
                approval_method, outcome
            ) VALUES (
                %(tool_name)s, %(tool_version)s, %(transport)s, %(client_id)s,
                %(principal_id)s, %(tenant_id)s, %(team_id)s,
                %(scopes_evaluated)s, %(redacted_args_hash)s,
                %(action_plan_id)s, %(idempotency_key)s, %(api_route)s,
                %(api_status)s, %(problem_type)s, %(resource_id)s,
                %(latency_ms)s, %(trace_id)s, %(approval_method)s, %(outcome)s
            ) RETURNING audit_id
            """,
            values,
        ).fetchone()
        audit_id = str(row[0])
        append_event(
            conn,
            aggregate_type="mcp_tool_audit",
            aggregate_id=audit_id,
            event_type="mcp.v1.tool_completed",
            payload={"audit_id": audit_id, "tool_name": body.tool_name, "outcome": body.outcome},
            headers={"trace_id": body.trace_id, "tenant_id": body.tenant_id},
            # Track A destination classes settle the immediate side effect.
            # Per-sink audit delivery is independently materialized by B4.4
            # from this same row; inventing an "audit" destination here leaves
            # the original dispatcher with no handler.
            destination_class="default",
            idempotency_key=f"mcp-audit:{audit_id}",
        )
        conn.commit()
    return {"ok": True, "audit_id": audit_id}


def _iso(value) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else (value if value is None else str(value))


@router.get("/api/v1/instances/{job_id}")
def api_v1_instance(job_id: str, request: Request):
    """Tenant-safe instance detail; cross-tenant identifiers are not-found."""
    job = dict(_job_for_caller(request, job_id))
    for field in (
        "init_script",
        "environment",
        "env",
        "registry_password",
        "ssh_private_key",
        "nfs_server",
        "nfs_path",
    ):
        job.pop(field, None)
    payload = job.get("payload")
    if isinstance(payload, dict):
        job["payload"] = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "init_script",
                "environment",
                "env",
                "registry_password",
                "ssh_private_key",
                "nfs_server",
                "nfs_path",
            }
        }
    return {"ok": True, "instance": job}


def _require_control_plane_read(request: Request) -> dict:
    """Operator read access: interactive admin, or `control_plane:read` machine."""
    user = _require_auth(request)
    scopes = set(user.get("scopes") or ())
    global_access = _is_platform_admin(user) or (
        str(user.get("grant_type", "")) == "client_credentials"
        and "control_plane:read" in scopes
    )
    if not global_access:
        raise ProblemException(
            status=403, code="forbidden",
            detail="platform admin or control_plane:read is required",
        )
    return user


def _job_for_caller(request: Request, job_id: str) -> dict:
    """Fetch a job the caller may see, else a not-found problem.

    Cross-tenant access returns **not-found**, not a permission hint (§B5.6 —
    no existence leak): a customer cannot probe which job ids exist.
    """
    from scheduler import get_job

    user = _require_auth(request)
    job = get_job(job_id)
    if not job or (not _is_platform_admin(user) and not _user_owns_job(user, job)):
        raise ProblemException(status=404, code="instance_not_found", detail=f"instance {job_id} not found")
    return job


def _attempts_for_job(job_id: str) -> list[dict]:
    from db import _get_pg_pool

    cols = [
        "attempt_id", "attempt_number", "status", "host_id", "spec_hash",
        "placement_score", "placement_explanation", "failure_code", "failure_details",
        "reserved_at", "command_created_at", "lease_claimed_at", "started_at",
        "ended_at", "trace_id",
    ]
    with _get_pg_pool().connection() as conn:
        rows = conn.execute(
            f"SELECT {', '.join(cols)} FROM job_attempts WHERE job_id = %s ORDER BY attempt_number ASC",
            (job_id,),
        ).fetchall()
    out = []
    for row in rows:
        rec = dict(zip(cols, row))
        for ts in ("reserved_at", "command_created_at", "lease_claimed_at", "started_at", "ended_at"):
            rec[ts] = _iso(rec[ts])
        out.append(rec)
    return out


def _require_host_operator(request: Request, scope: str) -> dict:
    """Interactive platform admin, or a machine principal holding *scope*.

    Scopes gate machine-to-machine callers (`_require_scope` no-ops for
    interactive sessions); admin gates humans. `drain`/`undrain` require
    `hosts:operate`; `evict` requires `hosts:evict` — a *distinct* scope, so a
    principal cleared to drain cannot evict (§3.3 "separately authorized").
    """
    user = _require_auth(request)
    if str(user.get("grant_type", "")) == "client_credentials":
        _require_scope(user, scope)
    elif not _is_platform_admin(user):
        raise HTTPException(403, f"admin access or the '{scope}' scope is required")
    return user


def _host_or_problem(host_id: str) -> tuple[str, dict]:
    resolved, hosts = _resolve_host_id(host_id)
    if not resolved:
        raise ProblemException(status=404, code="host_not_found", detail=f"host {host_id} not found")
    host = next(h for h in hosts if h["host_id"] == resolved)
    return resolved, host


def _host_version(host_id: str) -> int:
    """Authoritative host row version (0 if never versioned)."""
    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        row = conn.execute("SELECT version FROM hosts WHERE host_id = %s", (host_id,)).fetchone()
    if not row or row[0] is None:
        return 0
    return int(row[0])


def _check_version(host_id: str, expected: int | None) -> None:
    """Optimistic concurrency: refuse a stale operator request (§3.3 / B5.7)."""
    if expected is None:
        return
    current = _host_version(host_id)
    if current != int(expected):
        raise ProblemException(
            status=409,
            code="version_conflict",
            detail=f"host version is {current}, not the expected {expected}; re-read and retry",
            extra={"current_version": current, "expected_version": expected},
        )


class _OpIn(BaseModel):
    expected_version: int | None = None


class _EvictionPlanIn(BaseModel):
    expected_version: int
    reason: str = Field(min_length=1, max_length=500)


def _operator_plan_principal(user: dict):
    from control_plane.launch.service import Principal
    from routes._deps import _canonical_owner_id, _effective_billing_customer_id, _user_team_id
    return Principal(
        principal_id=_canonical_owner_id(user),
        tenant_id=_effective_billing_customer_id(user),
        client_id=user.get("client_id"),
        team_id=_user_team_id(user),
        scopes=tuple(user.get("scopes") or ()),
    )


@router.post("/api/v1/hosts/{host_id}/eviction-plans")
def api_v1_create_eviction_plan(host_id: str, request: Request, body: _EvictionPlanIn):
    """Create a human-approved plan; this never evicts workloads."""
    from control_plane.db import control_plane_transaction
    from control_plane.launch import action_plans as plans_repo
    from control_plane.launch.service import DEFAULT_TOLERANCE_BPS, plan_ttl_sec
    from routes.action_plans import _request_trace_id

    user = _require_host_operator(request, "hosts:evict")
    resolved, host = _host_or_problem(host_id)
    if host.get("status") != "draining":
        raise ProblemException(status=409, code="host_not_draining", detail="drain the host first")
    _check_version(resolved, body.expected_version)
    principal = _operator_plan_principal(user)
    canonical = {"host_id": resolved, "expected_version": body.expected_version, "reason": body.reason}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with control_plane_transaction() as conn:
        plan = plans_repo.create_quoted_plan(
            conn, action_type="evict_host_workloads",
            principal_id=principal.principal_id, client_id=principal.client_id,
            tenant_id=principal.tenant_id, team_id=principal.team_id,
            canonical_args=canonical, canonical_args_hash=digest, spec_hash=digest,
            quote_id=f"eviction:{digest[:20]}", pricing_version="not-applicable",
            estimate_micros=0, currency="CAD",
            price_tolerance_bps=DEFAULT_TOLERANCE_BPS,
            required_scopes=["hosts:evict"], approval_mode="human", ttl_sec=plan_ttl_sec(),
            trace_id=_request_trace_id(request),
        )
    pid = str(plan["plan_id"])
    return {
        "ok": True, "preview": True, "plan_id": pid,
        "approval_state": plan["status"], "impact": "running workloads will be fenced and evicted",
        "approval_url": f"/dashboard/launch-plans/{pid}",
        "expires_at": _iso(plan["expires_at"]),
    }


@router.post("/api/v1/hosts/{host_id}/eviction-plans/{plan_id}/execute")
def api_v1_execute_eviction_plan(host_id: str, plan_id: UUID, request: Request):
    from control_plane.db import control_plane_transaction
    from control_plane.launch import action_plans as plans_repo
    from control_plane.launch.service import PlanConflict, PlanNotFound
    from scheduler import run_drain_preemptions

    user = _require_host_operator(request, "hosts:evict")
    principal = _operator_plan_principal(user)
    with control_plane_transaction() as conn:
        plan = plans_repo.get_plan_for_update(conn, str(plan_id))
        if not plan or str(plan["tenant_id"]) != principal.tenant_id:
            raise PlanNotFound()
        if plan["action_type"] != "evict_host_workloads":
            raise PlanConflict("wrong_action_type", "the plan is not an eviction")
        if plan["status"] == "succeeded":
            return {**dict(plan["idempotent_response"]), "idempotent": True}
        if plan["status"] != "approved":
            raise PlanConflict("approval_required", "approve the eviction plan before execution")
        canonical = dict(plan["canonical_args"])
        if canonical["host_id"] != host_id:
            raise PlanConflict("resource_mismatch", "plan belongs to a different host")
        _check_version(host_id, int(canonical["expected_version"]))
        plans_repo.mark_executing(conn, str(plan_id))

    evicted = run_drain_preemptions(host_id)
    response = {"ok": True, "host_id": host_id, "evicted": [j["job_id"] for j in evicted], "plan_id": str(plan_id)}
    with control_plane_transaction() as conn:
        plans_repo.mark_consumed(
            conn, str(plan_id), job_id=host_id, wallet_hold_id=None,
            idempotent_response=response, idempotency_key=f"eviction-plan:{plan_id}",
        )
    return response


@router.post("/api/v1/hosts/{host_id}/drain")
def api_v1_drain_host(host_id: str, request: Request, body: _OpIn | None = None):
    """§3.3 drain — stop **new** placements only. Running workloads keep running.

    Unlike the legacy endpoint, this does NOT preempt. To remove workloads, call
    `/evictions` (a separate scope + audit trail).
    """
    from scheduler import set_host_draining

    user = _require_host_operator(request, "hosts:operate")
    resolved, host = _host_or_problem(host_id)
    if host.get("status") == "dead":
        raise ProblemException(status=409, code="host_dead", detail="cannot drain a dead host")
    _check_version(resolved, body.expected_version if body else None)

    updated = set_host_draining(resolved, draining=True)
    append_user_audit_event(
        "host.drained", "host", resolved, user, data={"evicted": False}
    )
    return {"ok": True, "host": updated, "evicted": [], "note": "new placements stopped; running workloads untouched"}


@router.post("/api/v1/hosts/{host_id}/undrain")
def api_v1_undrain_host(host_id: str, request: Request, body: _OpIn | None = None):
    """§3.3 undrain — return a drained host to service."""
    from scheduler import set_host_draining

    user = _require_host_operator(request, "hosts:operate")
    resolved, host = _host_or_problem(host_id)
    if host.get("status") == "dead":
        raise ProblemException(status=409, code="host_dead", detail="cannot undrain a dead host")
    _check_version(resolved, body.expected_version if body else None)

    updated = set_host_draining(resolved, draining=False)
    if not updated:
        raise ProblemException(status=404, code="host_not_found", detail=f"host {host_id} not found")
    append_user_audit_event("host.undrained", "host", resolved, user)
    return {"ok": True, "host": updated}


@router.post("/api/v1/hosts/{host_id}/evictions")
def api_v1_evict_host_workloads(host_id: str, request: Request, body: _OpIn | None = None):
    """§3.3 evict — remove running workloads from a host, distinct from drain.

    Requires the `hosts:evict` scope (a principal cleared only to drain cannot
    evict). Each workload is preempted (running → preempted → requeued), which
    abandons its placement so a fresh, fenced attempt is scheduled elsewhere.
    Records a separate `host.workloads_evicted` audit event.
    """
    from scheduler import run_drain_preemptions

    user = _require_host_operator(request, "hosts:evict")
    resolved, host = _host_or_problem(host_id)
    if host.get("status") == "dead":
        raise ProblemException(status=409, code="host_dead", detail="cannot evict a dead host")
    # §3.3: eviction is the *second* step. A host must be drained (no new
    # placements) before its running workloads may be removed, so draining and
    # evicting stay two distinct, separately-authorized actions.
    if host.get("status") != "draining":
        raise ProblemException(
            status=409,
            code="host_not_draining",
            detail="drain the host before evicting its workloads (§3.3)",
        )
    _check_version(resolved, body.expected_version if body else None)

    evicted = run_drain_preemptions(resolved)
    evicted_ids = [j["job_id"] for j in evicted]
    append_user_audit_event(
        "host.workloads_evicted", "host", resolved, user, data={"evicted": evicted_ids}
    )
    return {"ok": True, "host_id": resolved, "evicted": evicted_ids}


@router.get("/api/v1/instances/{job_id}/control-plane")
def api_v1_instance_control_plane(job_id: str, request: Request):
    """§18/§20.3 — a job's control-plane state: phase, desired state, current
    attempt. Tenant-scoped; a cross-tenant id is not-found (no existence leak)."""
    job = _job_for_caller(request, job_id)
    attempts = _attempts_for_job(job_id)
    current = attempts[-1] if attempts else None
    return {
        "ok": True,
        "job_id": job_id,
        "status": job.get("status"),
        "phase": job.get("phase"),
        "desired_state": job.get("desired_state"),
        "host_id": job.get("host_id"),
        "attempt_count": len(attempts),
        "current_attempt": current,
    }


@router.get("/api/v1/instances/{job_id}/timeline")
def api_v1_instance_timeline(job_id: str, request: Request):
    """§20.3 — the attempt timeline for a job (reserve → command → lease →
    start → end per attempt). Tenant-scoped; cross-tenant is not-found."""
    _job_for_caller(request, job_id)
    return {"ok": True, "job_id": job_id, "attempts": _attempts_for_job(job_id)}


@router.get("/api/v1/instances/{job_id}/events")
def api_v1_instance_events(
    job_id: str, request: Request, cursor: str | None = None, limit: int = 100
):
    """Durable, opaque-cursor event page for resumable MCP watches."""
    from control_plane.event_stream import resume_aggregate_after
    from db import _get_pg_pool

    _job_for_caller(request, job_id)
    with _get_pg_pool().connection() as conn:
        events = resume_aggregate_after(conn, "job", job_id, cursor, limit=limit)
    return {
        "ok": True,
        "job_id": job_id,
        "events": [
            {"cursor": event.cursor, "event_type": event.event_type, "payload": event.payload}
            for event in events
        ],
        "next_cursor": events[-1].cursor if events else cursor,
    }


@router.get("/api/v1/control-plane/reconciliation-findings")
def api_v1_reconciliation_findings(
    request: Request, status: str = "open", cursor: str | None = None, limit: int = 100
):
    """§18/§20.2 — reconciler findings feed (operator surface).

    Admin, or a machine principal with `control_plane:read`. Wraps the existing
    `reconciliation_findings` authority; read-only.
    """
    from db import _get_pg_pool

    user = _require_auth(request)
    scopes = set(user.get("scopes") or ())
    machine = str(user.get("grant_type", "")) == "client_credentials"
    global_access = _is_platform_admin(user) or (
        machine and "control_plane:read" in scopes
    )
    if not global_access:
        if not machine:
            raise ProblemException(
                status=403,
                code="forbidden",
                detail="platform admin or a scoped machine principal is required",
            )
        _require_scope(user, "instances:read")

    if status == "open":
        query = (
            "SELECT * FROM reconciliation_findings "
            "WHERE resolved_at IS NULL ORDER BY created_at DESC "
            "LIMIT %s OFFSET %s"
        )
    elif status == "resolved":
        query = (
            "SELECT * FROM reconciliation_findings "
            "WHERE resolved_at IS NOT NULL ORDER BY resolved_at DESC "
            "LIMIT %s OFFSET %s"
        )
    elif status == "all":
        query = (
            "SELECT * FROM reconciliation_findings "
            "ORDER BY created_at DESC LIMIT %s OFFSET %s"
        )
    else:
        raise ProblemException(
            status=422,
            code="invalid_status",
            detail="status must be open|resolved|all",
        )

    try:
        raw_cursor = cursor or "MA"
        offset = int(
            base64.urlsafe_b64decode(
                raw_cursor + "=" * ((4 - len(raw_cursor) % 4) % 4)
            ).decode()
        )
    except (ValueError, TypeError):
        raise ProblemException(status=422, code="invalid_cursor", detail="cursor is invalid")
    limit = max(1, min(int(limit), 200))
    with _get_pg_pool().connection() as conn:
        cur = conn.execute(query, (limit, offset))
        names = [c.name for c in cur.description]
        rows = cur.fetchall()
    findings = []
    for row in rows:
        rec = dict(zip(names, row))
        for k, v in list(rec.items()):
            rec[k] = _iso(v) if hasattr(v, "isoformat") else v
        findings.append(rec)
    if not global_access:
        from scheduler import get_job
        findings = [
            finding
            for finding in findings
            if finding.get("resource_type") == "job"
            and (job := get_job(str(finding.get("resource_id") or "")))
            and _user_owns_job(user, job)
        ]
    return {
        "ok": True, "status": status, "scope": "global" if global_access else "tenant",
        "findings": findings,
        "next_cursor": (
            base64.urlsafe_b64encode(str(offset + limit).encode()).decode().rstrip("=")
            if len(rows) == limit else None
        ),
    }


@router.get("/api/v1/instances/{job_id}/attempts")
def api_v1_instance_attempts(job_id: str, request: Request):
    """§18 — the raw attempt records for a job. Tenant-scoped."""
    _job_for_caller(request, job_id)
    return {"ok": True, "job_id": job_id, "attempts": _attempts_for_job(job_id)}


@router.get("/api/v1/instances/{job_id}/active-lease")
def api_v1_instance_active_lease(job_id: str, request: Request):
    """Current lease health with a tenant-safe host alias and no credentials."""
    from db import _get_pg_pool

    _job_for_caller(request, job_id)
    with _get_pg_pool().connection() as conn:
        row = conn.execute(
            """
            SELECT lease_id, attempt_id, host_id, status, offered_at,
                   claim_deadline, claimed_at, last_renewed_at, expires_at
              FROM placement_leases
             WHERE job_id=%s AND status IN ('offered','active')
             ORDER BY offered_at DESC LIMIT 1
            """,
            (job_id,),
        ).fetchone()
    if not row:
        return {"ok": True, "job_id": job_id, "lease": None}
    keys = [
        "lease_id", "attempt_id", "host_id", "status", "offered_at",
        "claim_deadline", "claimed_at", "last_renewed_at", "expires_at",
    ]
    lease = dict(zip(keys, row))
    host_id = str(lease.pop("host_id"))
    lease["host_alias"] = f"host-{hashlib.sha256(host_id.encode()).hexdigest()[:10]}"
    for key, value in list(lease.items()):
        lease[key] = _iso(value) if hasattr(value, "isoformat") else str(value) if isinstance(value, UUID) else value
    return {"ok": True, "job_id": job_id, "lease": lease}


@router.get("/api/v1/instances/{job_id}/placement-explanation")
def api_v1_instance_placement_explanation(job_id: str, request: Request):
    """§3.2/§18 — the persisted placement explanation for the current attempt.

    Returns the bounded, pre-computed explanation the scheduler stored (no LLM
    invents a reason). Tenant-scoped; not-found for a cross-tenant id.
    """
    _job_for_caller(request, job_id)
    attempts = _attempts_for_job(job_id)
    current = attempts[-1] if attempts else None
    explanation = current.get("placement_explanation") if current else None
    return {
        "ok": True,
        "job_id": job_id,
        "attempt_id": current.get("attempt_id") if current else None,
        "placement_score": current.get("placement_score") if current else None,
        "explanation": explanation,
        "explained": explanation is not None,
    }


@router.post("/api/v1/instances/{job_id}/retry")
def api_v1_instance_retry(job_id: str, request: Request, body: _OpIn | None = None):
    """§18 — re-enqueue a failed/stuck instance (does not run the queue inline).

    Tenant-scoped write. Delegates to the one requeue authority; the scheduler
    then claims and places it.
    """
    from scheduler import requeue_job

    job = _job_for_caller(request, job_id)
    expected_version = body.expected_version if body else None
    if expected_version is not None and int(job.get("version") or 0) != expected_version:
        raise ProblemException(
            status=409, code="version_conflict",
            detail="instance changed; re-read before retrying",
            extra={"current_version": int(job.get("version") or 0)},
        )
    status = str(job.get("status") or "")
    if status == "completed":
        raise ProblemException(status=409, code="already_completed", detail="a completed instance cannot be retried")
    if status == "queued":
        raise ProblemException(status=409, code="already_queued", detail="instance is already queued")
    result = requeue_job(job_id, user_initiated=True, expected_version=expected_version)
    if not result:
        raise ProblemException(status=409, code="retry_failed", detail="instance could not be requeued")
    return {"ok": True, "job_id": job_id, "status": "queued"}


@router.get("/api/v1/hosts/{host_id}/capacity")
def api_v1_host_capacity(host_id: str, request: Request):
    """§18/§20.4 — a host's GPU capacity snapshot. Operator read."""
    _require_control_plane_read(request)
    resolved, host = _host_or_problem(host_id)

    def _num(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    total = _num(host.get("total_vram_gb"))
    free = _num(host.get("free_vram_gb"))
    return {
        "ok": True,
        "host_id": resolved,
        "gpu_model": host.get("gpu_model"),
        "num_gpus": host.get("num_gpus"),
        "status": host.get("status"),
        "draining": host.get("status") == "draining",
        "total_vram_gb": total,
        "free_vram_gb": free,
        "allocated_vram_gb": round(max(0.0, total - free), 3),
    }


@router.get("/api/v1/hosts/{host_id}/observations")
def api_v1_host_observations(host_id: str, request: Request, limit: int = 20):
    """§18/§20.4 — recent worker-reported observations for a host. Operator read."""
    from db import _get_pg_pool

    _require_control_plane_read(request)
    resolved, _ = _host_or_problem(host_id)
    limit = max(1, min(int(limit), 200))
    cols = [
        "observation_id", "session_id", "inventory_generation", "agent_version",
        "capabilities", "conditions", "gpu_inventory", "observed_workload_count",
        "command_journal_watermark", "worker_reported_at", "received_at",
    ]
    with _get_pg_pool().connection() as conn:
        rows = conn.execute(
            f"SELECT {', '.join(cols)} FROM host_observations WHERE host_id = %s "
            "ORDER BY received_at DESC LIMIT %s",
            (resolved, limit),
        ).fetchall()
    out = []
    for row in rows:
        rec = dict(zip(cols, row))
        rec["worker_reported_at"] = _iso(rec["worker_reported_at"])
        rec["received_at"] = _iso(rec["received_at"])
        out.append(rec)
    return {"ok": True, "host_id": resolved, "observations": out}


@router.get("/api/v1/control-plane/queue")
def api_v1_control_plane_queue(request: Request):
    """§18/§20.2 — the queued instances awaiting placement, with reasons. Operator read."""
    from scheduler import list_jobs

    _require_control_plane_read(request)
    queued = list_jobs("queued")
    entries = [
        {
            "job_id": j.get("job_id"),
            "priority": j.get("priority"),
            "queue_reason": j.get("queue_reason") or j.get("queue_reason_code"),
            "queue_reason_detail": j.get("queue_reason_detail"),
            "gpu_model": j.get("gpu_model"),
            "num_gpus": j.get("num_gpus"),
            "vram_needed_gb": j.get("vram_needed_gb"),
            "submitted_at": _iso(j.get("submitted_at")),
            "scheduling_attempts": j.get("scheduling_attempts"),
        }
        for j in queued
    ]
    return {"ok": True, "depth": len(entries), "queue": entries}


@router.get("/api/v1/control-plane/health")
def api_v1_control_plane_health(request: Request):
    """§18/§20.2 — control-plane health aggregate: outbox, findings, tasks.

    A dashboard "0" from a broken pipeline must be distinguishable from a
    genuine zero (DA§17), so this reports live counts, not a single flag.
    Operator read.
    """
    from control_plane.projection_delivery import health_snapshot
    from db import _get_pg_pool

    _require_control_plane_read(request)
    with _get_pg_pool().connection() as conn:
        outbox_pending = conn.execute(
            "SELECT count(*) FROM outbox_events WHERE published_at IS NULL AND dead_lettered_at IS NULL"
        ).fetchone()[0]
        outbox_dead = conn.execute(
            "SELECT count(*) FROM outbox_events WHERE dead_lettered_at IS NOT NULL"
        ).fetchone()[0]
        findings_open = conn.execute(
            "SELECT count(*) FROM reconciliation_findings WHERE resolved_at IS NULL"
        ).fetchone()[0]
        task_rows = conn.execute(
            "SELECT task_name, enabled, last_status, last_run_at, next_run_at FROM scheduled_tasks"
        ).fetchall()
        try:
            projection = {"available": True, **health_snapshot(conn)}
        except Exception as exc:
            conn.rollback()
            projection = {
                "available": False,
                "error": type(exc).__name__,
                "unprepared": None,
                "orphaned": None,
                "sinks": [],
            }
    tasks = [
        {
            "task_name": r[0],
            "enabled": r[1],
            "last_status": r[2],
            "last_run_at": _iso(r[3]),
            "next_run_at": _iso(r[4]),
        }
        for r in task_rows
    ]
    failed_tasks = [t["task_name"] for t in tasks if t["last_status"] == "failed"]
    projection_dead = sum(
        int(sink.get("dead_lettered") or 0) for sink in projection["sinks"]
    )
    degraded = bool(
        outbox_dead
        or failed_tasks
        or not projection["available"]
        or projection_dead
        or projection.get("orphaned")
    )
    return {
        "ok": True,
        "status": "degraded" if degraded else "healthy",
        "outbox": {"pending": outbox_pending, "dead_lettered": outbox_dead},
        "projections": projection,
        "reconciliation": {"open_findings": findings_open},
        "scheduled_tasks": tasks,
        "failed_tasks": failed_tasks,
    }


@router.get("/api/v1/openapi.json")
def api_v1_openapi(request: Request):
    """The versioned OpenAPI schema for the `/api/v1` surface (§18.1).

    The app serves a *curated* public spec at `/openapi.json`; the generated MCP
    and dashboard clients (B5.2, B6.1) instead pin this — the live FastAPI schema
    filtered to `/api/v1/*` — so a client is always in lockstep with the routes
    actually mounted.
    """
    from fastapi.openapi.utils import get_openapi

    full = get_openapi(
        title="Xcelsior Control-Plane API v1",
        version="1.0.0",
        description="Versioned control-plane surface (§18). Errors are RFC 9457 problem+json.",
        routes=request.app.routes,
    )
    full["paths"] = {p: v for p, v in full.get("paths", {}).items() if p.startswith("/api/v1/")}
    return full


@router.post("/api/v1/instances/{job_id}/reconcile")
def api_v1_instance_reconcile(job_id: str, request: Request, body: _OpIn | None = None):
    """§3.3/§18 reconcile — **enqueue** a reconcile for this instance.

    It never performs direct repair (§3.3): it inserts a durable request into
    `reconciliation_queue`, coalesced to one pending entry per instance, and the
    reconciler claims and processes it out-of-band. Tenant-scoped.
    """
    from db import _get_pg_pool

    user = _require_auth(request)
    job = _job_for_caller(request, job_id)  # tenant-scope / not-found guard
    if body and body.expected_version is not None and int(job.get("version") or 0) != body.expected_version:
        raise ProblemException(status=409, code="version_conflict", detail="instance changed; re-read first")
    requested_by = str(user.get("email") or user.get("customer_id") or user.get("user_id") or "api")
    with _get_pg_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO reconciliation_queue (resource_type, resource_id, reason, requested_by)
            VALUES ('job', %s, 'manual_reconcile', %s)
            ON CONFLICT (resource_type, resource_id) DO UPDATE
               SET due_at = LEAST(reconciliation_queue.due_at, clock_timestamp()),
                   reason = EXCLUDED.reason,
                   updated_at = clock_timestamp()
            """,
            (job_id, requested_by),
        )
        conn.commit()
    return {
        "ok": True,
        "job_id": job_id,
        "enqueued": True,
        "note": "reconcile requested; the reconciler processes it out-of-band (never repaired inline)",
    }


@router.post("/api/v1/control-plane/commands/{command_id}/retry")
def api_v1_retry_agent_command(command_id: UUID, request: Request, body: _OpIn):
    """Redeliver only failed/dead-letter commands without changing identity."""
    from db import _get_pg_pool

    _require_host_operator(request, "control_plane:operate")
    with _get_pg_pool().connection() as conn:
        row = conn.execute(
            "SELECT status, attempt_count, idempotency_key FROM agent_commands "
            "WHERE command_id = %s FOR UPDATE",
            (str(command_id),),
        ).fetchone()
        if not row:
            raise ProblemException(status=404, code="command_not_found", detail="command not found")
        status, attempts, idempotency_key = row
        if int(attempts or 0) != body.expected_version:
            raise ProblemException(status=409, code="version_conflict", detail="command changed; re-read first")
        if status not in {"failed", "dead_letter"}:
            raise ProblemException(
                status=409, code="command_not_retryable",
                detail="only failed or dead-letter commands can be retried",
            )
        conn.execute(
            """
            UPDATE agent_commands
               SET status='pending', claim_owner=NULL, claim_session=NULL,
                   claim_expires_at=NULL, next_attempt_at=clock_timestamp(),
                   error_code=NULL, error_details=NULL
             WHERE command_id=%s
            """,
            (str(command_id),),
        )
        conn.commit()
    return {
        "ok": True, "command_id": str(command_id), "status": "pending",
        "idempotency_key": idempotency_key, "audit_preserved": True,
    }
