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
from pydantic import BaseModel

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


def _iso(value) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else (value if value is None else str(value))


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


@router.get("/api/v1/control-plane/reconciliation-findings")
def api_v1_reconciliation_findings(request: Request, status: str = "open"):
    """§18/§20.2 — reconciler findings feed (operator surface).

    Admin, or a machine principal with `control_plane:read`. Wraps the existing
    `reconciliation_findings` authority; read-only.
    """
    from db import _get_pg_pool

    user = _require_auth(request)
    if str(user.get("grant_type", "")) == "client_credentials":
        _require_scope(user, "control_plane:read")
    elif not _is_platform_admin(user):
        raise ProblemException(status=403, code="forbidden", detail="admin or control_plane:read required")

    if status == "open":
        where, order = "WHERE resolved_at IS NULL", "created_at DESC"
    elif status == "resolved":
        where, order = "WHERE resolved_at IS NOT NULL", "resolved_at DESC"
    elif status == "all":
        where, order = "", "created_at DESC"
    else:
        raise ProblemException(status=422, code="invalid_status", detail="status must be open|resolved|all")

    with _get_pg_pool().connection() as conn:
        cur = conn.execute(f"SELECT * FROM reconciliation_findings {where} ORDER BY {order} LIMIT 500")
        names = [c.name for c in cur.description]
        rows = cur.fetchall()
    findings = []
    for row in rows:
        rec = dict(zip(names, row))
        for k, v in list(rec.items()):
            rec[k] = _iso(v) if hasattr(v, "isoformat") else v
        findings.append(rec)
    return {"ok": True, "status": status, "findings": findings}
