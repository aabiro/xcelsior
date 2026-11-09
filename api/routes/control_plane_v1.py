"""Track B B2.8 — versioned control-plane host operations (§18, §3.3).

The §3.3 rule the blueprint is emphatic about: **draining never evicts.**
Draining a host only stops *new* placements; it must leave every running
workload running. Removing running workloads is a **separate**, separately
authorized, separately audited action (`/evictions`) that fences the workload
before it can be reassigned.

Both drain routes stop new placements without eviction. Approved eviction
commands wait for fenced terminal settlement before reporting completion. Domain failures are RFC 9457
problem+json (B2.8); optimistic concurrency uses the host `version` so a stale
operator request is refused rather than racing.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from uuid import UUID
import base64

from routes._deps import (
    _is_platform_admin,
    _job_access_owner_ids,
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
    return (
        value.isoformat()
        if hasattr(value, "isoformat")
        else (value if value is None else str(value))
    )


# Bounded so one export cannot become an unbounded table scan, and cursor-paged
# so a customer can still walk their whole history.
_AUDIT_PAGE_MAX = 500


@router.get("/api/v1/mcp/tool-audit")
def api_v1_mcp_tool_audit_export(request: Request):
    """Customer-visible export of this tenant's own MCP tool-call audit trail.

    The trail was already recorded per call — principal, tenant, scopes, a hash
    of the redacted arguments, the action plan, the upstream route and status,
    trace id, latency, and outcome. An audit trail a customer cannot read is a
    log file, not an assurance: this is what makes "every agent action is
    audited" checkable by the person whose account it is, which is what an
    enterprise security review actually asks for (adoption plan X6.30).

    Strictly tenant-scoped. There is no cross-tenant view here at any privilege
    level — an operator investigating an incident uses the operator surface,
    which is separately authorized and separately audited.
    """
    from db import _get_pg_pool

    user = _require_auth(request)
    tenant = str(
        user.get("customer_id")
        or user.get("workspace_id")
        or user.get("workspace_customer_id")
        or user.get("tenant_id")
        or ""
    )
    if not tenant:
        raise ProblemException(status=404, code="tenant_not_found", detail="tenant not found")

    params = request.query_params
    try:
        limit = min(max(int(params.get("limit", "100")), 1), _AUDIT_PAGE_MAX)
    except ValueError:
        limit = 100
    tool_name = str(params.get("tool_name", "")).strip()[:128] or None
    outcome = str(params.get("outcome", "")).strip()[:32] or None
    # Opaque cursor: the occurred_at of the last row of the previous page.
    # Keyset rather than OFFSET so deep pages stay cheap and a row written
    # mid-walk cannot shift the window.
    cursor = str(params.get("cursor", "")).strip()[:64] or None

    clauses = ["tenant_id = %(tenant)s"]
    values: dict = {"tenant": tenant, "limit": limit}
    if tool_name:
        clauses.append("tool_name = %(tool_name)s")
        values["tool_name"] = tool_name
    if outcome:
        clauses.append("outcome = %(outcome)s")
        values["outcome"] = outcome
    if cursor:
        clauses.append("occurred_at < %(cursor)s")
        values["cursor"] = cursor

    # Named explicitly rather than `SELECT *` so a column added to the table
    # later cannot silently become part of a customer-facing payload.
    columns = (
        "audit_id",
        "occurred_at",
        "tool_name",
        "tool_version",
        "transport",
        "client_id",
        "principal_id",
        "team_id",
        "scopes_evaluated",
        "redacted_args_hash",
        "action_plan_id",
        "idempotency_key",
        "api_route",
        "api_status",
        "problem_type",
        "resource_id",
        "latency_ms",
        "trace_id",
        "approval_method",
        "outcome",
    )
    with _get_pg_pool().connection() as conn:
        rows = conn.execute(
            f"SELECT {', '.join(columns)} FROM mcp_tool_audit "
            f" WHERE {' AND '.join(clauses)} "
            " ORDER BY occurred_at DESC "
            " LIMIT %(limit)s",
            values,
        ).fetchall()

    records = []
    for row in rows:
        # This pool yields tuples; zip against the column list rather than
        # assuming a dict row factory.
        record = dict(row) if isinstance(row, dict) else dict(zip(columns, row, strict=True))
        record["audit_id"] = str(record["audit_id"])
        record["occurred_at"] = _iso(record["occurred_at"])
        if record.get("action_plan_id"):
            record["action_plan_id"] = str(record["action_plan_id"])
        record["scopes_evaluated"] = list(record.get("scopes_evaluated") or [])
        records.append(record)

    return {
        "ok": True,
        "tenant_id": tenant,
        "records": records,
        # NULL when the page is not full: there is provably nothing older to
        # ask for, so a client can stop rather than probe once more.
        "next_cursor": records[-1]["occurred_at"] if len(records) == limit else None,
        "notice": (
            "Arguments are never stored — redacted_args_hash is a SHA-256 of the "
            "canonicalised arguments with secret-bearing fields removed first."
        ),
    }


@router.get("/api/v1/mcp/activation-funnel")
def api_v1_mcp_activation_funnel(request: Request):
    """Connector activation funnel, with per-surface attribution.

    Adoption plan X7.33/X7.34. The stages are the ones a connector actually
    passes through, computed from durable records rather than estimated:

      authorized       → a user approved the connector (`oauth_consent_grants`)
      first_tool_call  → that tenant made at least one MCP tool call
      first_success    → at least one of those calls succeeded
      first_write      → they used a tool that changes something
      first_paid       → they ran a workload that was billed

    **The two stages above `authorized` are deliberately not here.** Discovery
    hits and 401 challenges are counted at the MCP edge as Prometheus counters
    (`xcelsior_mcp_auth_failures_total`, `xcelsior_mcp_active_transports`) and
    read in Grafana. Persisting a database row per unauthenticated request would
    make an unauthenticated endpoint a write amplifier — the top of this funnel
    belongs in a metrics pipeline, not in Postgres, and saying so is more useful
    than a number we would have to fabricate.

    Platform-wide, so admin-only: a tenant's own activity is the audit export.
    """
    from db import _get_pg_pool, auth_connection

    user = _require_auth(request)
    _require_scope(user, "control_plane:read")
    if not _is_platform_admin(user):
        raise ProblemException(status=404, code="not_found", detail="not found")

    try:
        days = min(max(int(request.query_params.get("days", "30")), 1), 365)
    except ValueError:
        days = 30

    with auth_connection() as conn:
        grants = conn.execute(
            """
            SELECT surface, tenant_id, granted_at
              FROM oauth_consent_grants
             WHERE revoked_at IS NULL
               AND granted_at >= now() - make_interval(days => %s)
            """,
            (days,),
        ).fetchall()

    # tenant -> surface. A tenant that connected from two surfaces is counted
    # under its earliest, so the funnel's stages always sum to the same
    # population rather than double-counting a tenant that reconnected.
    surface_by_tenant: dict[str, str] = {}
    first_seen: dict[str, object] = {}
    for row in grants:
        record = (
            dict(row)
            if isinstance(row, dict)
            else dict(zip(("surface", "tenant_id", "granted_at"), row, strict=True))
        )
        tenant = str(record.get("tenant_id") or "")
        if not tenant:
            continue
        granted = record.get("granted_at")
        if tenant not in first_seen or (granted and first_seen[tenant] and granted < first_seen[tenant]):  # type: ignore[operator]
            first_seen[tenant] = granted
            surface_by_tenant[tenant] = str(record.get("surface") or "unknown")

    tenants = list(surface_by_tenant)
    called: set[str] = set()
    succeeded: set[str] = set()
    wrote: set[str] = set()
    if tenants:
        with _get_pg_pool().connection() as conn:
            rows = conn.execute(
                """
                SELECT tenant_id,
                       bool_or(outcome = 'success')                       AS any_success,
                       bool_or(tool_name IN (
                           'create_instance', 'cancel_instance', 'terminate_instance',
                           'run_training_job', 'schedule_under_budget',
                           'create_serverless_endpoint', 'run_serverless_job',
                           'retry_instance', 'reconcile_instance'
                       ))                                                 AS any_write
                  FROM mcp_tool_audit
                 WHERE tenant_id = ANY(%s)
                   AND occurred_at >= now() - make_interval(days => %s)
                 GROUP BY tenant_id
                """,
                (tenants, days),
            ).fetchall()
        for row in rows:
            tenant, any_success, any_write = (
                (row["tenant_id"], row["any_success"], row["any_write"])
                if isinstance(row, dict)
                else (row[0], row[1], row[2])
            )
            called.add(str(tenant))
            if any_success:
                succeeded.add(str(tenant))
            if any_write:
                wrote.add(str(tenant))

    paid: set[str] = set()
    try:
        from billing import get_billing_engine

        engine = get_billing_engine()
        for tenant in tenants:
            history = engine.get_wallet_history(tenant, limit=200) or []
            if any(
                str(entry.get("type") or entry.get("kind") or "") == "charge" for entry in history
            ):
                paid.add(tenant)
    except Exception as exc:  # pragma: no cover - billing is not the gate here
        log_billing_gap = str(exc)
        paid = set()
    else:
        log_billing_gap = ""

    def by_surface(population: set[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for tenant in population:
            surface = surface_by_tenant.get(tenant, "unknown")
            counts[surface] = counts.get(surface, 0) + 1
        return dict(sorted(counts.items()))

    stages = [
        ("authorized", set(tenants)),
        ("first_tool_call", called),
        ("first_success", succeeded),
        ("first_write", wrote),
        ("first_paid_workload", paid),
    ]
    funnel = []
    previous = len(tenants)
    for name, population in stages:
        count = len(population)
        funnel.append(
            {
                "stage": name,
                "tenants": count,
                # Conversion from the *previous* stage, which is where a cliff shows
                # up. Conversion from the top hides which step lost people.
                "conversion_from_previous": round(count / previous, 4) if previous else None,
                "by_surface": by_surface(population),
            }
        )
        previous = count or previous

    biggest_drop = None
    for index in range(1, len(funnel)):
        lost = funnel[index - 1]["tenants"] - funnel[index]["tenants"]
        if lost > 0 and (biggest_drop is None or lost > biggest_drop["tenants_lost"]):
            biggest_drop = {
                "between": f"{funnel[index - 1]['stage']} → {funnel[index]['stage']}",
                "tenants_lost": lost,
            }

    return {
        "ok": True,
        "window_days": days,
        "funnel": funnel,
        "biggest_drop": biggest_drop,
        "notes": {
            "top_of_funnel": (
                "Discovery hits and 401 challenges are Prometheus counters on the MCP "
                "edge, not database rows — persisting one row per unauthenticated "
                "request would make an unauthenticated endpoint a write amplifier."
            ),
            **({"billing_unavailable": log_billing_gap} if log_billing_gap else {}),
        },
    }


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
        str(user.get("grant_type", "")) == "client_credentials" and "control_plane:read" in scopes
    )
    if not global_access:
        raise ProblemException(
            status=403,
            code="forbidden",
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
        raise ProblemException(
            status=404, code="instance_not_found", detail=f"instance {job_id} not found"
        )
    return job


def _attempts_for_job(job_id: str) -> list[dict]:
    from db import _get_pg_pool

    cols = [
        "attempt_id",
        "attempt_number",
        "status",
        "host_id",
        "spec_hash",
        "placement_score",
        "placement_explanation",
        "failure_code",
        "failure_details",
        "reserved_at",
        "command_created_at",
        "lease_claimed_at",
        "started_at",
        "ended_at",
        "trace_id",
    ]
    with _get_pg_pool().connection() as conn:
        rows = conn.execute(
            f"SELECT {', '.join(cols)} FROM job_attempts WHERE job_id = %s ORDER BY attempt_number ASC",
            (job_id,),
        ).fetchall()
    out = []
    for row in rows:
        rec = dict(zip(cols, row))
        for ts in (
            "reserved_at",
            "command_created_at",
            "lease_claimed_at",
            "started_at",
            "ended_at",
        ):
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
        raise ProblemException(
            status=404, code="host_not_found", detail=f"host {host_id} not found"
        )
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
    grace_seconds: int = Field(default=60, ge=1, le=300)


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
    from control_plane.db import run_transaction
    from control_plane.launch import action_plans as plans_repo
    from control_plane.launch.service import DEFAULT_TOLERANCE_BPS, plan_ttl_sec
    from routes.action_plans import _request_trace_id

    user = _require_host_operator(request, "hosts:evict")
    resolved, _host = _host_or_problem(host_id)
    _check_version(resolved, body.expected_version)
    principal = _operator_plan_principal(user)
    from control_plane import host_evictions

    def transaction(conn):
        canonical = {
            "host_id": resolved,
            "expected_version": body.expected_version,
            "reason": body.reason,
            "policy": host_evictions.POLICY,
            "grace_seconds": body.grace_seconds,
            "allow_force": False,
            "victims": host_evictions.snapshot(
                conn, host_id=resolved, expected_version=body.expected_version
            ),
        }
        digest = host_evictions.canonical_hash(canonical)
        return plans_repo.create_quoted_plan(
            conn,
            action_type="evict_host_workloads",
            principal_id=principal.principal_id,
            client_id=principal.client_id,
            tenant_id=principal.tenant_id,
            team_id=principal.team_id,
            canonical_args=canonical,
            canonical_args_hash=digest,
            spec_hash=digest,
            quote_id=f"eviction:{digest[:20]}",
            pricing_version="not-applicable",
            estimate_micros=0,
            currency="CAD",
            price_tolerance_bps=DEFAULT_TOLERANCE_BPS,
            required_scopes=["hosts:evict"],
            approval_mode="human",
            ttl_sec=plan_ttl_sec(),
            trace_id=_request_trace_id(request),
        )

    plan = run_transaction(
        transaction, isolation_level="SERIALIZABLE", what="host_eviction_preview"
    )
    pid = str(plan["plan_id"])
    return {
        "ok": True,
        "preview": True,
        "plan_id": pid,
        "approval_state": plan["status"],
        "impact": "compute workloads requeue and serverless workers terminate after fenced settlement",
        "canonical_spec": dict(plan["canonical_args"]),
        "approval_url": f"/dashboard/launch-plans/{pid}",
        "expires_at": _iso(plan["expires_at"]),
    }


@router.post("/api/v1/hosts/{host_id}/eviction-plans/{plan_id}/execute")
def api_v1_execute_eviction_plan(host_id: str, plan_id: UUID, request: Request):
    from control_plane.db import run_transaction
    from control_plane.launch import action_plans as plans_repo
    from control_plane.launch.service import PlanConflict, PlanNotFound

    from control_plane import host_evictions

    user = _require_host_operator(request, "hosts:evict")
    principal = _operator_plan_principal(user)

    def transaction(conn):
        plan = plans_repo.get_plan_for_update(conn, str(plan_id))
        if not plan or str(plan["tenant_id"]) != principal.tenant_id:
            raise PlanNotFound()
        if plan["action_type"] != host_evictions.ACTION:
            raise PlanConflict("wrong_action_type", "the plan is not an eviction")
        return host_evictions.execute(
            conn, plan, host_id=host_id, created_by=principal.principal_id
        )

    return run_transaction(
        transaction, isolation_level="SERIALIZABLE", what="host_eviction_execute"
    )


@router.get("/api/v1/hosts/{host_id}/eviction-plans/{plan_id}")
def api_v1_read_eviction_plan(host_id: str, plan_id: UUID, request: Request):
    """Read the approved targets and current release proof without side effects."""
    from control_plane import host_evictions
    from control_plane.db import control_plane_transaction
    from control_plane.launch import action_plans as plans_repo
    from control_plane.launch.service import PlanNotFound

    user = _require_host_operator(request, "hosts:evict")
    principal = _operator_plan_principal(user)
    with control_plane_transaction() as conn:
        plan = plans_repo.get_plan(conn, str(plan_id))
        if (
            not plan
            or str(plan["tenant_id"]) != principal.tenant_id
            or plan["action_type"] != host_evictions.ACTION
        ):
            raise PlanNotFound()
        canonical = host_evictions.approved_canonical(plan, host_id)
        response = {
            "ok": True,
            "plan_id": str(plan_id),
            "status": plan["status"],
            "canonical_spec": canonical,
        }
        if plan["status"] in ("executing", "succeeded"):
            response["execution"] = host_evictions.progress(conn, plan)
        return response


@router.post("/api/v1/hosts/{host_id}/drain")
def api_v1_drain_host(host_id: str, request: Request, body: _OpIn | None = None):
    """§3.3 drain — stop **new** placements only. Running workloads keep running.

    Eviction requires a separate scope and the durable preemption controller.
    """
    from scheduler import set_host_draining

    user = _require_host_operator(request, "hosts:operate")
    resolved, host = _host_or_problem(host_id)
    if host.get("status") == "dead":
        raise ProblemException(status=409, code="host_dead", detail="cannot drain a dead host")
    _check_version(resolved, body.expected_version if body else None)

    updated = set_host_draining(
        resolved, draining=True, expected_version=body.expected_version if body else None
    )
    append_user_audit_event("host.drained", "host", resolved, user, data={"evicted": False})
    return {
        "ok": True,
        "host": updated,
        "evicted": [],
        "note": "new placements stopped; running workloads untouched",
    }


@router.post("/api/v1/hosts/{host_id}/undrain")
def api_v1_undrain_host(host_id: str, request: Request, body: _OpIn | None = None):
    """§3.3 undrain — return a drained host to service."""
    from scheduler import set_host_draining

    user = _require_host_operator(request, "hosts:operate")
    resolved, host = _host_or_problem(host_id)
    if host.get("status") == "dead":
        raise ProblemException(status=409, code="host_dead", detail="cannot undrain a dead host")
    _check_version(resolved, body.expected_version if body else None)

    updated = set_host_draining(
        resolved, draining=False, expected_version=body.expected_version if body else None
    )
    if not updated:
        raise ProblemException(
            status=404, code="host_not_found", detail=f"host {host_id} not found"
        )
    append_user_audit_event("host.undrained", "host", resolved, user)
    return {"ok": True, "host": updated}


@router.post("/api/v1/hosts/{host_id}/evictions")
def api_v1_evict_host_workloads(host_id: str, request: Request, body: _OpIn | None = None):
    """Require an approved plan; this route never bypasses human approval."""

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

    raise ProblemException(
        status=409,
        code="approval_required",
        detail="create and approve an eviction plan, then execute that plan",
        extra={"preview_url": f"/api/v1/hosts/{resolved}/eviction-plans"},
    )


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
    job_id: str, request: Request, cursor: str | None = None, limit: int = Query(100, ge=1, le=1000)
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
    global_access = _is_platform_admin(user) or (machine and "control_plane:read" in scopes)
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
            "SELECT * FROM reconciliation_findings " "ORDER BY created_at DESC LIMIT %s OFFSET %s"
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
            base64.urlsafe_b64decode(raw_cursor + "=" * ((4 - len(raw_cursor) % 4) % 4)).decode()
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
        "ok": True,
        "status": status,
        "scope": "global" if global_access else "tenant",
        "findings": findings,
        "next_cursor": (
            base64.urlsafe_b64encode(str(offset + limit).encode()).decode().rstrip("=")
            if len(rows) == limit
            else None
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

    _require_scope(_require_auth(request), "instances:read")
    _job_for_caller(request, job_id)
    from control_plane.lease_views import current_lease_view

    with _get_pg_pool().connection() as conn:
        lease = current_lease_view(conn, job_id)
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
            status=409,
            code="version_conflict",
            detail="instance changed; re-read before retrying",
            extra={"current_version": int(job.get("version") or 0)},
        )
    if str(job.get("job_type") or "") == "serverless_worker":
        # One worker identity is one attempt; its endpoint provisions a new one.
        raise ProblemException(
            status=409,
            code="serverless_worker",
            detail="a serverless endpoint worker is provisioned by its endpoint, not retried",
        )
    status = str(job.get("status") or "")
    if status == "completed":
        raise ProblemException(
            status=409, code="already_completed", detail="a completed instance cannot be retried"
        )
    if status == "queued":
        raise ProblemException(
            status=409, code="already_queued", detail="instance is already queued"
        )
    result = requeue_job(job_id, user_initiated=True, expected_version=expected_version)
    if not result:
        raise ProblemException(
            status=409, code="retry_failed", detail="instance could not be requeued"
        )
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
def api_v1_host_observations(host_id: str, request: Request, limit: int = Query(20, ge=1, le=1000)):
    """§18/§20.4 — recent worker-reported observations for a host. Operator read."""
    from db import _get_pg_pool

    _require_control_plane_read(request)
    resolved, _ = _host_or_problem(host_id)
    limit = max(1, min(int(limit), 200))
    cols = [
        "observation_id",
        "session_id",
        "inventory_generation",
        "agent_version",
        "capabilities",
        "conditions",
        "gpu_inventory",
        "observed_workload_count",
        "worker_reported_at",
        "received_at",
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
    projection_dead = sum(int(sink.get("dead_lettered") or 0) for sink in projection["sinks"])
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
    if (
        body
        and body.expected_version is not None
        and int(job.get("version") or 0) != body.expected_version
    ):
        raise ProblemException(
            status=409, code="version_conflict", detail="instance changed; re-read first"
        )
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
            raise ProblemException(
                status=409, code="version_conflict", detail="command changed; re-read first"
            )
        if status not in {"failed", "dead_letter"}:
            raise ProblemException(
                status=409,
                code="command_not_retryable",
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
        "ok": True,
        "command_id": str(command_id),
        "status": "pending",
        "idempotency_key": idempotency_key,
        "audit_preserved": True,
    }


# ── HPC (Slurm) read model ───────────────────────────────────────────
# The Slurm reconciler (workers/slurm_reconciler.py) is the only code that
# polls a cluster. These endpoints read its copy, so a dashboard or an agent
# can watch as often as it likes without adding a single call to the
# cluster (hpc/IMPLEMENTATION_PLAN.md §5a).

#: An open job unconfirmed for this many poll intervals is stale.
HPC_STALE_INTERVALS = 3
_HPC_JOB_SELECT = """
    SELECT m.xcelsior_job_id, m.cluster_id, m.slurm_job_id, m.observed_state,
           m.slurm_state, m.state_reason, m.exit_code, m.desired_state,
           m.submitted_at, m.started_at, m.ended_at, m.terminal_at,
           m.last_observed_at, m.last_changed_at, m.missing_since,
           EXTRACT(EPOCH FROM clock_timestamp() - m.last_observed_at) AS observed_age,
           COALESCE(c.poll_interval_seconds, 60) AS poll_interval,
           m.derived_exit_code, m.time_limit_seconds, m.left_controller_at,
           EXTRACT(EPOCH FROM COALESCE(m.ended_at, clock_timestamp()) - m.started_at) AS ran
      FROM slurm_job_mappings m
      LEFT JOIN slurm_clusters c ON c.cluster_id = m.cluster_id
"""


def _hpc_reader(request: Request) -> tuple[bool, set[str]]:
    """(global_access, owner_ids): operators read every job, others their own."""
    user = _require_auth(request)
    scopes = set(user.get("scopes") or ())
    machine = str(user.get("grant_type", "")) == "client_credentials"
    if _is_platform_admin(user) or (machine and "control_plane:read" in scopes):
        return True, set()
    if machine:
        _require_scope(user, "instances:read")
    return False, _job_access_owner_ids(user)


def _hpc_job(row) -> dict:
    from slurm_adapter import success_verdict

    (
        xid,
        cluster,
        sid,
        observed,
        slurm_state,
        reason,
        exit_code,
        desired,
        submitted,
        started,
        ended,
        terminal,
        observed_at,
        changed_at,
        missing,
        age,
        interval,
        derived_exit_code,
        time_limit,
        left_controller,
        ran,
    ) = row
    settled = terminal is not None
    age_s = None if age is None else round(float(age), 1)
    elapsed = None if ran is None or started is None else max(0, int(float(ran)))
    return {
        "xcelsior_job_id": xid,
        "cluster_id": cluster,
        "slurm_job_id": sid,
        "observed_state": observed,
        "slurm_state": slurm_state,
        "state_reason": reason,
        "exit_code": exit_code,
        "derived_exit_code": derived_exit_code,
        # succeeded only for COMPLETED with ExitCode and DerivedExitCode 0:0;
        # None until the job is settled.
        "verdict": success_verdict(observed, exit_code, derived_exit_code, settled),
        "desired_state": desired,
        "settled": settled,
        # Elapsed time against the time limit, from the scheduler. A job's
        # own progress report, when it writes one, is reported_progress.
        "scheduler_progress": {
            "elapsed_seconds": elapsed,
            "time_limit_seconds": time_limit,
            "fraction": (
                round(min(1.0, elapsed / time_limit), 4)
                if elapsed is not None and time_limit
                else None
            ),
        },
        # Gone from the controller but not yet settled by accounting.
        "awaiting_accounting_since": None if settled else _iso(left_controller),
        "submitted_at": _iso(submitted),
        "started_at": _iso(started),
        "ended_at": _iso(ended),
        "settled_at": _iso(terminal),
        "last_observed_at": _iso(observed_at),
        "last_changed_at": _iso(changed_at),
        "missing_since": _iso(missing),
        "observed_age_seconds": age_s,
        # A settled job is final; an open one goes stale when the reconciler
        # hasn't confirmed it for several of its cluster's poll intervals.
        "stale": (not settled) and (age_s is None or age_s > HPC_STALE_INTERVALS * int(interval)),
    }


@router.get("/api/v1/hpc/clusters")
def api_v1_hpc_clusters(request: Request):
    """Registered Slurm clusters with the reconciler's view of their health.

    Operator surface: platform admin, or a machine principal with
    `control_plane:read`. Read-only; never contacts a cluster.
    """
    from db import _get_pg_pool

    global_access, _owners = _hpc_reader(request)
    if not global_access:
        raise ProblemException(
            status=403,
            code="forbidden",
            detail="platform admin or a scoped machine principal is required",
        )
    with _get_pg_pool().connection() as conn:
        cur = conn.execute(
            """
            SELECT cluster_id, display_name, profile_key, enabled, research_only,
                   marketplace, agent_access, poll_interval_seconds, last_poll_at,
                   transport_ok_at, controller_ok_at, accounting_ok_at,
                   consecutive_failures, breaker_open_until, last_error_code,
                   COALESCE(facts->'orphans', '[]'::jsonb),
                   ssh_host, ssh_port, remote_user, known_hosts_line, credential_ref,
                   facts->'config', facts->>'config_captured_at'
              FROM slurm_clusters ORDER BY cluster_id
            """
        )
        rows = cur.fetchall()
    clusters = []
    for r in rows:
        clusters.append(
            {
                "cluster_id": r[0],
                "display_name": r[1],
                "profile": r[2],
                "enabled": r[3],
                "research_only": r[4],
                "marketplace": r[5],
                "agent_access": r[6],
                "poll_interval_seconds": r[7],
                "health": {
                    "last_poll_at": _iso(r[8]),
                    "transport_ok_at": _iso(r[9]),
                    "controller_ok_at": _iso(r[10]),
                    "accounting_ok_at": _iso(r[11]),
                    "consecutive_failures": r[12],
                    "backing_off_until": _iso(r[13]),
                    "last_error_code": r[14],
                },
                "orphan_slurm_job_ids": list(r[15] or []),
                "transport": _hpc_transport(r[16], r[17], r[18], r[19], r[20]),
                "config_facts": r[21],
                "config_captured_at": r[22],
            }
        )
    return {"ok": True, "clusters": clusters}


def _hpc_transport(host, port, user, known_hosts_line, credential_ref) -> dict:
    """How the reconciler reaches the cluster. The key line is shown as its
    fingerprint, and the credential only as the reference it is."""
    from slurm_transport import fingerprint

    if not host:
        return {"kind": "profile"}
    try:
        print_ = fingerprint(known_hosts_line or "")
    except Exception:
        print_ = None
    return {
        "kind": "registry",
        "ssh_host": host,
        "ssh_port": port or 22,
        "remote_user": user,
        "host_key_fingerprint": print_,
        "credential_ref": credential_ref,
    }


def _require_hpc_admin(request: Request) -> dict:
    """A human platform admin. Registering a cluster names a host and a key
    the control plane will use, so no machine principal may do it."""
    user = _require_auth(request)
    if str(user.get("grant_type", "")) == "client_credentials" or not _is_platform_admin(user):
        raise ProblemException(
            status=403, code="forbidden", detail="a platform admin is required"
        )
    return user


class _HpcClusterIn(BaseModel):
    profile_key: str = Field(min_length=1, max_length=64)
    cluster_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    # Off until a connection check has proven the transport.
    enabled: bool = False
    ssh_host: str | None = Field(default=None, max_length=253)
    ssh_port: int | None = Field(default=None, ge=1, le=65535)
    remote_user: str | None = Field(default=None, max_length=32)
    known_hosts_line: str | None = Field(default=None, max_length=8192)
    credential_ref: str | None = Field(default=None, max_length=4096)


@router.post("/api/v1/hpc/clusters", status_code=201)
def api_v1_register_hpc_cluster(request: Request, body: _HpcClusterIn):
    """Register (or update) a Slurm cluster the reconciler polls.

    Platform admin only, and audited. With `ssh_host`, give the host key to
    pin (a known_hosts or ssh-keyscan line verified out of band) and a
    `credential_ref`: `env:XCELSIOR_SLURM_KEY_<NAME>`, or `file:` under the
    API's key directory. Registered disabled unless `enabled` is set; run
    the connection check, then register again with `enabled: true`. The
    research-only, marketplace and agent flags come from the profile.
    """
    import psycopg

    import slurm_adapter as sa
    import slurm_reconciler as rc
    from slurm_transport import api_credential_allowed

    user = _require_hpc_admin(request)
    if body.ssh_host and not api_credential_allowed(body.credential_ref or ""):
        raise ProblemException(
            status=422,
            code="credential_ref_not_allowed",
            detail="credential_ref must be env:XCELSIOR_SLURM_KEY_<NAME> or a file in the "
            "API's Slurm key directory",
        )
    try:
        registered = rc.register_cluster(
            body.profile_key,
            enabled=body.enabled,
            cluster_id=body.cluster_id,
            ssh_host=body.ssh_host,
            ssh_port=body.ssh_port,
            remote_user=body.remote_user,
            known_hosts_line=body.known_hosts_line,
            credential_ref=body.credential_ref,
        )
    except sa.SlurmAdapterError as e:
        raise ProblemException(status=422, code=e.code, detail=str(e))
    except psycopg.errors.CheckViolation as e:
        raise ProblemException(
            status=422, code="invalid_cluster", detail=str(e).splitlines()[0]
        )
    append_user_audit_event(
        "hpc.cluster.registered",
        "slurm_cluster",
        registered["cluster_id"],
        user,
        data={
            "profile_key": body.profile_key,
            "enabled": body.enabled,
            "transport": registered["transport"],
            "ssh_host": body.ssh_host,
            "credential_ref": body.credential_ref,
        },
    )
    return {"ok": True, "cluster": registered}


@router.post("/api/v1/hpc/clusters/{cluster_id}/connection-check")
def api_v1_check_hpc_cluster(cluster_id: str, request: Request):
    """Connect to a registered cluster the way the reconciler will.

    Runs `scontrol ping` and `scontrol show config`, and stores the config
    facts on the cluster. Platform admin only, and audited. A failure is the
    result (`ok: false` with an `error_code` such as `host_key_mismatch`),
    not an error status; an unknown cluster is 404.
    """
    import slurm_reconciler as rc

    user = _require_hpc_admin(request)
    result = rc.check_cluster_connection(cluster_id)
    if result.get("error_code") == "not_found":
        raise ProblemException(
            status=404, code="hpc_cluster_not_found", detail=f"cluster {cluster_id} not found"
        )
    append_user_audit_event(
        "hpc.cluster.connection_checked",
        "slurm_cluster",
        cluster_id,
        user,
        data={"ok": result.get("ok"), "error_code": result.get("error_code")},
    )
    return {"ok": True, "check": result}


@router.get("/api/v1/hpc/jobs")
def api_v1_hpc_jobs(
    request: Request,
    cluster_id: str | None = None,
    state: str = "open",
    cursor: str | None = None,
    limit: int = Query(100, ge=1, le=200),
):
    """Slurm jobs as the reconciler last observed them, newest change first.

    `state`: `open` (not yet settled), `settled`, or `all`. Operators see
    every job; anyone else sees their own. Read-only; never contacts a cluster.
    """
    from db import _get_pg_pool

    global_access, owners = _hpc_reader(request)
    where, params = [], []
    if state == "open":
        where.append("m.terminal_at IS NULL")
    elif state == "settled":
        where.append("m.terminal_at IS NOT NULL")
    elif state != "all":
        raise ProblemException(
            status=422, code="invalid_state", detail="state must be open|settled|all"
        )
    if cluster_id:
        where.append("m.cluster_id = %s")
        params.append(cluster_id)
    if not global_access:
        where.append("m.tenant_id = ANY(%s)")
        params.append(sorted(owners))
    try:
        raw_cursor = cursor or "MA"
        offset = int(
            base64.urlsafe_b64decode(raw_cursor + "=" * ((4 - len(raw_cursor) % 4) % 4)).decode()
        )
    except (ValueError, TypeError):
        raise ProblemException(status=422, code="invalid_cursor", detail="cursor is invalid")
    query = (
        _HPC_JOB_SELECT
        + (" WHERE " + " AND ".join(where) if where else "")
        + " ORDER BY COALESCE(m.last_changed_at, m.submitted_at, to_timestamp(m.created_at)) DESC,"
        " m.xcelsior_job_id LIMIT %s OFFSET %s"
    )
    with _get_pg_pool().connection() as conn:
        rows = conn.execute(query, (*params, limit + 1, offset)).fetchall()
    more = len(rows) > limit
    jobs = [_hpc_job(r) for r in rows[:limit]]
    next_cursor = (
        base64.urlsafe_b64encode(str(offset + limit).encode()).decode().rstrip("=")
        if more
        else None
    )
    return {"ok": True, "jobs": jobs, "next_cursor": next_cursor}


@router.get("/api/v1/hpc/jobs/{xcelsior_job_id}")
def api_v1_hpc_job(xcelsior_job_id: str, request: Request):
    """One Slurm job as the reconciler last observed it.

    Adds `reported_progress`, the job's own progress report if it writes one
    (cached like output: a stale copy queues one fetch). Someone else's job
    is not-found, not forbidden: ids can't be probed. Read-only; never
    contacts a cluster.
    """
    from db import _get_pg_pool
    from slurm_adapter import read_reported_progress

    global_access, owners = _hpc_reader(request)
    query = _HPC_JOB_SELECT + " WHERE m.xcelsior_job_id = %s"
    params: list = [xcelsior_job_id]
    if not global_access:
        query += " AND m.tenant_id = ANY(%s)"
        params.append(sorted(owners))
    with _get_pg_pool().connection() as conn:
        row = conn.execute(query, params).fetchone()
    if row is None:
        raise ProblemException(
            status=404, code="hpc_job_not_found", detail=f"HPC job {xcelsior_job_id} not found"
        )
    job = _hpc_job(row)
    job["reported_progress"] = read_reported_progress(xcelsior_job_id)
    return {"ok": True, "job": job}


@router.get("/api/v1/hpc/jobs/{xcelsior_job_id}/output")
def api_v1_hpc_job_output(xcelsior_job_id: str, request: Request, stream: str = "out"):
    """The tail of a Slurm job's stdout (`out`) or stderr (`err`), from cache.

    Never contacts a cluster. A copy older than the cluster's poll interval
    queues one fetch for the reconciler's next pass (`refresh_pending`); read
    again after `poll_interval_seconds`. However many readers ask, the
    cluster is read at most once per interval per job and stream. A settled
    job's copy is `final`. Someone else's job is not-found.
    """
    from datetime import datetime, timezone

    from db import _get_pg_pool
    from slurm_adapter import read_cached_output

    if stream not in ("out", "err"):
        raise ProblemException(status=422, code="invalid_stream", detail="stream must be out|err")
    global_access, owners = _hpc_reader(request)
    query = "SELECT 1 FROM slurm_job_mappings m WHERE m.xcelsior_job_id = %s"
    params: list = [xcelsior_job_id]
    if not global_access:
        query += " AND m.tenant_id = ANY(%s)"
        params.append(sorted(owners))
    with _get_pg_pool().connection() as conn:
        visible = conn.execute(query, params).fetchone()
    output = read_cached_output(xcelsior_job_id, stream) if visible else None
    if output is None or "error" in output:
        raise ProblemException(
            status=404, code="hpc_job_not_found", detail=f"HPC job {xcelsior_job_id} not found"
        )
    fetched = output["fetched_at"]
    return {
        "ok": True,
        "output": {
            "xcelsior_job_id": output["xcelsior_job_id"],
            "cluster_id": output["cluster_id"],
            "slurm_job_id": output["slurm_job_id"],
            "stream": stream,
            "content": output["content"],
            "truncated": output["truncated"],
            "final": output["final"],
            "settled": output["settled"],
            "fetched_at": _iso(datetime.fromtimestamp(fetched, timezone.utc)) if fetched else None,
            "fetched_age_seconds": output["fetched_age_seconds"],
            "refresh_pending": output["refresh_pending"],
            "poll_interval_seconds": output["poll_interval_seconds"],
            "fetch_error_code": output["fetch_error_code"],
        },
    }
