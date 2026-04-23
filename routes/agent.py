"""Routes: agent."""

import gzip
import os
import re
import time
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from routes._deps import (
    broadcast_sse,
    _get_current_user,
    _require_user_grant,
)
from scheduler import (
    list_hosts,
    list_jobs,
    log,
    register_compute_score,
    update_job_status,
)
from events import Event, get_event_store, get_state_machine
from security import admit_node
from db import _get_pg_pool
import threading

router = APIRouter()


# ── Auth helper ───────────────────────────────────────────────────────
# Every /agent/* endpoint except the admission path (/agent/versions) and
# the public popular-images metadata endpoint requires a Bearer token or
# API key. The worker_agent.py already sends
# `Authorization: Bearer $XCELSIOR_API_TOKEN` on every request (see
# worker_agent.py:600-607), so enforcing this does not break deployed
# agents. Previously these endpoints were completely unauthenticated —
# anyone reachable could spoof telemetry, claim leases, inject logs,
# exfiltrate user SSH public keys via /agent/ssh-keys, etc.
def _require_agent_auth(request: Request, *, host_id: str | None = None) -> dict:
    """Require auth on /agent/* endpoints. Optionally gate by host ownership.

    Bypass rules (in order, first match wins):
      1. If XCELSIOR_ENV == 'production': NEVER bypass. A production host
         accidentally tagged with XCELSIOR_ENV=test (the original audit
         concern) is impossible because deploy.sh pins XCELSIOR_ENV=production
         at the unit level; this rule hardens it belt-and-braces.
      2. If XCELSIOR_ENV == 'test': accept unauth (test suite pattern).
      3. If unauth and XCELSIOR_ALLOW_UNAUTH_AGENT=1: accept with WARNING
         (emergency escape hatch during token rotation).
      4. Otherwise: require an authenticated user.

    B1: when XCELSIOR_AGENT_STRICT_HOST_BINDING=1 AND a bypass rule (2 or 3)
    would otherwise grant access, any supplied `host_id` must still resolve
    to a registered row in the hosts table. Fail-open if the DB lookup
    raises (avoids locking out the whole fleet during a DB incident).
    """
    env = os.environ.get("XCELSIOR_ENV", "").lower()
    # 1. Hard refuse in production — escape hatches do NOT apply.
    if env == "production":
        user = _get_current_user(request)
        if not user:
            raise HTTPException(401, "Authentication required")
        return user

    # B1 defense-in-depth: when bypass is active AND strict binding is enabled,
    # still require any supplied host_id to be registered. Prevents a rogue/
    # misconfigured caller in dev/test/staging from claiming an arbitrary
    # host_id. Default OFF for backward compat; production already hard-fails
    # above so this flag is a no-op there.
    strict = os.environ.get("XCELSIOR_AGENT_STRICT_HOST_BINDING", "").lower() in (
        "1", "true", "yes",
    )

    def _enforce_host_registered() -> None:
        if not strict or not host_id:
            return
        try:
            all_hosts = list_hosts(active_only=False)
        except Exception:
            return  # DB unavailable — fail-open rather than lock out fleet
        if not any(h.get("host_id") == host_id for h in all_hosts):
            raise HTTPException(403, "Unknown host_id")

    # 2. Test-mode bypass — tests drive /agent/* without bearer tokens.
    if env == "test":
        _enforce_host_registered()
        return {"unauth": True, "test": True}

    user = _get_current_user(request)
    if not user:
        # 3. Explicit escape hatch (non-prod only) — WARN loudly.
        if os.environ.get("XCELSIOR_ALLOW_UNAUTH_AGENT", "").lower() in ("1", "true", "yes"):
            log.warning(
                "unauthenticated /agent/* request accepted (XCELSIOR_ALLOW_UNAUTH_AGENT=true)",
            )
            _enforce_host_registered()
            return {"unauth": True}
        raise HTTPException(401, "Authentication required")

    # Admins bypass host-ownership checks.
    if user.get("is_admin"):
        return user

    if host_id:
        try:
            all_hosts = list_hosts(active_only=False)
            host = next((h for h in all_hosts if h.get("host_id") == host_id), None)
        except Exception:
            host = None
        if host:
            owner = host.get("owner") or ""
            if owner and owner != user.get("user_id") and owner != user.get("email"):
                raise HTTPException(403, "Host is not owned by the authenticated caller")
    return user


# Critical container log patterns to detect image pull / launch errors
_IMAGE_PULL_PATTERNS = (
    "manifest unknown",
    "pull access denied",
    "not found: manifest unknown",
    "Error response from daemon",
    "image not found",
    "ErrImagePull",
    "ImagePullBackOff",
)

# Agent work/preemption state
_agent_work: dict[str, list[dict]] = defaultdict(list)  # host_id -> [job, ...]
_agent_preempt: dict[str, list[str]] = defaultdict(list)  # host_id -> [job_id, ...]
_agent_lock = threading.Lock()
_host_telemetry: dict[str, dict] = {}


# ── Model: VersionReport ──


class VersionReport(BaseModel):
    host_id: str = Field(min_length=1, max_length=64)
    versions: dict


# ── Model: MiningAlert ──


class MiningAlert(BaseModel):
    host_id: str = Field(min_length=1, max_length=64)
    gpu_index: int = Field(ge=0, le=64)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=500)
    timestamp: float | None = None


# ── Model: BenchmarkReport ──


class BenchmarkReport(BaseModel):
    host_id: str = Field(min_length=1, max_length=64)
    gpu_model: str = Field(max_length=64)
    score: float = Field(ge=0)
    tflops: float = Field(ge=0)
    details: dict | None = None


@router.get("/agent/work/{host_id}", tags=["Agent"])
def api_agent_work(host_id: str, request: Request):
    """Pull pending work for an agent. Returns assigned jobs."""
    _require_agent_auth(request, host_id=host_id)
    all_jobs = list_jobs()
    pending = [
        j for j in all_jobs if j.get("host_id") == host_id and j.get("status") in ("assigned",)
    ]
    with _agent_lock:
        queued_work = _agent_work.pop(host_id, [])

    jobs = pending + queued_work
    if not jobs:
        return JSONResponse(status_code=204, content=None)

    # Enrich each job with volume mount paths from the volume_attachments table
    try:
        from volumes import get_volume_engine

        ve = get_volume_engine()
        for j in jobs:
            vol_ids = j.get("volume_ids", [])
            if vol_ids:
                mounts: dict[str, str] = {}
                try:
                    for att in ve.get_instance_volumes(j["job_id"]):
                        mounts[att["volume_id"]] = att.get("mount_path", "/workspace")
                except Exception:
                    pass
                # Auto-assign paths for volumes without attachments yet
                _idx = 0
                for vid in vol_ids:
                    if vid not in mounts:
                        mounts[vid] = "/workspace" if _idx == 0 else f"/workspace/vol-{_idx}"
                    _idx += 1
                j["volume_mounts"] = mounts
    except Exception:
        pass  # Best-effort: worker falls back to /workspace

    return {"ok": True, "instances": jobs}


@router.get("/agent/preempt/{host_id}", tags=["Agent"])
def api_agent_preempt(host_id: str, request: Request):
    """Check if any jobs on this host should be preempted."""
    _require_agent_auth(request, host_id=host_id)
    with _agent_lock:
        preempt_list = _agent_preempt.pop(host_id, [])
    return {"ok": True, "preempt_jobs": preempt_list}


@router.post("/agent/preempt/{host_id}/{job_id}", tags=["Agent"])
def api_schedule_preemption(host_id: str, job_id: str, request: Request):
    """Schedule a job for preemption on a host. Admin-only."""
    user = _require_user_grant(request, allow_api_key=True)
    if not user.get("is_admin"):
        raise HTTPException(403, "Admin role required")
    with _agent_lock:
        _agent_preempt[host_id].append(job_id)
    broadcast_sse("preemption_scheduled", {"host_id": host_id, "job_id": job_id})
    return {"ok": True, "host_id": host_id, "job_id": job_id}


# ── Agent command queue (admin control plane → worker) ────────────────
# Admins enqueue commands via /admin/* endpoints; worker agents drain
# them via this endpoint in their poll loop. Rows are deleted on fetch —
# delivery is at-most-once by design (re-inject is idempotent so a lost
# command is harmless; admins can re-issue).

_AGENT_COMMAND_ALLOWED = {
    "reinject_shell",
    "upgrade_agent",
    "rollback_agent",   # P1.2 — auto-rollback driver
    "stop_container",  # P3.2 — billing/admin-initiated container kill (+ rm)
    "pause_container",  # internal: state-preserving stop primitive used by /stop
    "start_container",  # P3.2 — (re)start a stopped container
    "reset_container",  # P2 — restart with fresh /workspace, preserve volumes
    "snapshot_container",  # P3.1 — docker commit → user_images
}


@router.get("/agent/commands/{host_id}", tags=["Agent"])
def api_agent_commands_drain(host_id: str, request: Request):
    """Drain pending agent commands for this host.

    Returns up to 50 pending, non-expired commands and deletes them in the
    same transaction. Also GCs rows whose expires_at has passed.
    """
    _require_agent_auth(request, host_id=host_id)
    if not host_id or len(host_id) > 128:
        raise HTTPException(400, "Invalid host_id")
    pool = _get_pg_pool()
    with pool.connection() as conn, conn.cursor() as cur:
        # Drop expired rows for any host (cheap housekeeping).
        cur.execute("DELETE FROM agent_commands WHERE expires_at < EXTRACT(EPOCH FROM NOW())")
        # Atomically claim pending rows for this host.
        cur.execute(
            """
            DELETE FROM agent_commands
            WHERE id IN (
                SELECT id FROM agent_commands
                WHERE host_id = %s AND status = 'pending'
                ORDER BY created_at ASC
                LIMIT 50
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, command, args, created_at, created_by
            """,
            (host_id,),
        )
        rows = cur.fetchall()
    commands = [
        {
            "id": r[0],
            "command": r[1],
            "args": r[2] or {},
            "created_at": float(r[3]),
            "created_by": r[4],
        }
        for r in rows
    ]
    return {"ok": True, "commands": commands}


def enqueue_agent_command(
    host_id: str,
    command: str,
    args: dict | None = None,
    created_by: str | None = None,
    ttl_sec: int = 900,
) -> int:
    """Insert an agent command row; returns the new command id.

    Rejects unknown command names at the API boundary so a typo in an
    admin endpoint can't ship a bogus instruction to workers.
    """
    if command not in _AGENT_COMMAND_ALLOWED:
        raise HTTPException(400, f"Unknown agent command: {command}")
    import json as _json

    # P3/C4 — args size cap. Serialised JSON > 16 KB almost certainly
    # indicates a bug or an abuse attempt (the payload has to cross a
    # JSONB column and later be rehydrated in the worker). Fail fast
    # with 413 so callers get a clear signal.
    _AGENT_ARGS_MAX_BYTES = int(os.environ.get("XCELSIOR_AGENT_ARGS_MAX_BYTES", "16384"))
    args_json = _json.dumps(args or {})
    if len(args_json) > _AGENT_ARGS_MAX_BYTES:
        raise HTTPException(
            413,
            f"Agent command args too large: {len(args_json)} bytes "
            f"(max {_AGENT_ARGS_MAX_BYTES}).",
        )

    pool = _get_pg_pool()
    with pool.connection() as conn, conn.cursor() as cur:
        # P3/C5 — per-host queue cap. Prevents a buggy caller or
        # compromised admin token from flooding a single host with more
        # pending commands than it can reasonably drain (default 50/drain,
        # so 1000 = ~20 drain cycles of backlog).
        _AGENT_QUEUE_MAX = int(os.environ.get("XCELSIOR_AGENT_QUEUE_MAX", "1000"))
        cur.execute(
            """
            SELECT COUNT(*) FROM agent_commands
             WHERE host_id=%s
               AND status='pending'
               AND expires_at > EXTRACT(EPOCH FROM NOW())
            """,
            (host_id,),
        )
        pending_count = int((cur.fetchone() or [0])[0])
        if pending_count >= _AGENT_QUEUE_MAX:
            raise HTTPException(
                503,
                f"Agent queue full for host {host_id}: "
                f"{pending_count} pending (max {_AGENT_QUEUE_MAX}). "
                f"Wait for worker to drain.",
            )
        cur.execute(
            """
            INSERT INTO agent_commands (host_id, command, args, created_by, expires_at)
            VALUES (%s, %s, %s::jsonb, %s, EXTRACT(EPOCH FROM NOW()) + %s)
            RETURNING id
            """,
            (host_id, command, args_json, created_by, int(ttl_sec)),
        )
        row = cur.fetchone()
    return int(row[0])


# ── Admission failure notification (throttled: once per hour per host) ─────
_admission_notified: dict[str, float] = {}


def _notify_provider_admission_failure(host: dict, details: dict):
    """Send in-app + push notification to host owner about admission failure."""
    host_id = host.get("host_id", "?")
    now = time.time()
    # Evict stale throttle entries (>1 hour) to prevent memory leak
    stale = [k for k, v in _admission_notified.items() if now - v > 3600]
    for k in stale:
        del _admission_notified[k]
    last = _admission_notified.get(host_id, 0)
    if now - last < 3600:
        return  # throttle: at most once per hour per host
    _admission_notified[host_id] = now

    owner_id = host.get("owner", "")
    if not owner_id:
        return  # no owner — agent-only host, can't notify

    try:
        from db import UserStore, NotificationStore

        user = UserStore.get_user_by_id(owner_id)
        if not user:
            return
        email = user.get("email", "")
        if not email:
            return

        reasons = details.get("rejection_reasons", [])
        reason_text = "; ".join(reasons) if reasons else "Unknown version requirements not met"
        gpu_label = host.get("gpu_model") or host.get("hostname") or host_id

        NotificationStore.create(
            user_email=email,
            notif_type="host_admission_failed",
            title=f"Host {gpu_label} failed admission",
            body=(
                f"Your host could not be admitted to the compute pool. "
                f"Issues: {reason_text}. "
                f"Please update the flagged components and restart the worker agent."
            ),
            data={
                "host_id": host_id,
                "rejection_reasons": reasons,
                "recommended_runtime": details.get("recommended_runtime", "runc"),
            },
            action_url="/dashboard/hosts",
            entity_type="host",
            entity_id=host_id,
            priority=2,  # critical
        )
        log.info("Notified provider %s about admission failure for host %s", email, host_id)
    except Exception:
        log.exception("Failed to send admission failure notification for host %s", host_id)


@router.post("/agent/versions", tags=["Agent"])
def api_agent_versions(report: VersionReport):
    """Receive and validate node component versions for admission control.

    When a host passes admission, updates the host record to admitted=True
    and status='active'. This is the gate that allows hosts to receive work.
    Per REPORT_FEATURE_FINAL.md §62 and REPORT_FEATURE_1.md:
    - CUDA >= 12.0, runc patched, NVIDIA Container Toolkit patched
    - Hosts that fail stay in 'pending' status
    """
    from security import admit_node

    admitted_result, details = admit_node(report.host_id, report.versions)

    # Update host record with admission status
    from scheduler import _atomic_mutation, _upsert_host_row, _migrate_hosts_if_needed

    hosts = list_hosts(active_only=False)
    host_found = False
    for h in hosts:
        if h.get("host_id") == report.host_id:
            host_found = True
            h["admitted"] = details["admitted"]
            h["recommended_runtime"] = details.get("recommended_runtime", "runc")
            h["admission_details"] = details
            if details["admitted"]:
                h["status"] = "draining" if h.get("status") == "draining" else "active"
                log.info(
                    "HOST %s ADMITTED — status set to active, runtime=%s",
                    report.host_id,
                    details.get("recommended_runtime", "runc"),
                )
            else:
                h["status"] = "pending"
                log.warning(
                    "HOST %s NOT ADMITTED — status remains pending: %s",
                    report.host_id,
                    details.get("rejection_reasons", []),
                )
                # Notify the provider about what needs fixing
                _notify_provider_admission_failure(h, details)
            # Persist
            with _atomic_mutation() as conn:
                _migrate_hosts_if_needed(conn)
                _upsert_host_row(conn, h)
            break

    if not host_found:
        # Host hasn't heartbeated yet — create a minimal record so the
        # admission state survives until the first heartbeat arrives.
        entry = {
            "host_id": report.host_id,
            "ip": "",
            "gpu_model": "",
            "total_vram_gb": 0,
            "free_vram_gb": 0,
            "cost_per_hour": 0,
            "admitted": details["admitted"],
            "admission_details": details,
            "recommended_runtime": details.get("recommended_runtime", "runc"),
            "status": "active" if details["admitted"] else "pending",
            "registered_at": time.time(),
            "last_seen": time.time(),
        }
        with _atomic_mutation() as conn:
            _migrate_hosts_if_needed(conn)
            _upsert_host_row(conn, entry)
        log.info(
            "HOST %s pre-registered via /agent/versions (admitted=%s)",
            report.host_id,
            details["admitted"],
        )

    broadcast_sse(
        "node_admission",
        {
            "host_id": report.host_id,
            "admitted": details["admitted"],
            "versions": report.versions,
            "runtime": details.get("recommended_runtime", "runc"),
        },
    )
    return {
        "ok": True,
        "admitted": details["admitted"],
        "details": details,
    }


@router.post("/agent/mining-alert", tags=["Agent"])
def api_mining_alert(alert: MiningAlert, request: Request):
    """Receive mining detection alert from an agent."""
    _require_agent_auth(request, host_id=alert.host_id)
    log.warning(
        "MINING ALERT host=%s gpu=%d confidence=%.0f%% — %s",
        alert.host_id,
        alert.gpu_index,
        alert.confidence * 100,
        alert.reason,
    )
    broadcast_sse(
        "mining_alert",
        {
            "host_id": alert.host_id,
            "gpu_index": alert.gpu_index,
            "confidence": alert.confidence,
            "reason": alert.reason,
        },
    )
    return {"ok": True, "received": True}


@router.post("/agent/benchmark", tags=["Agent"])
def api_agent_benchmark(report: BenchmarkReport, request: Request):
    """Receive compute benchmark results from an agent."""
    _require_agent_auth(request, host_id=report.host_id)
    register_compute_score(
        report.host_id,
        report.gpu_model,
        report.score,
        report.details,
    )
    broadcast_sse(
        "benchmark_result",
        {
            "host_id": report.host_id,
            "gpu_model": report.gpu_model,
            "xcu": report.score,
            "tflops": report.tflops,
        },
    )
    return {"ok": True, "xcu": report.score}


# ── Model: LeaseClaimRequest ──


class LeaseClaimRequest(BaseModel):
    host_id: str
    job_id: str


# ── Model: LeaseRenewRequest ──


class LeaseRenewRequest(BaseModel):
    host_id: str
    job_id: str


# ── Model: LeaseReleaseRequest ──


class LeaseReleaseRequest(BaseModel):
    job_id: str
    reason: str = "completed"

    @classmethod
    def _valid_reasons(cls):
        return {"completed", "failed", "preempted"}


@router.post("/agent/lease/claim", tags=["Agent"])
def api_agent_lease_claim(req: LeaseClaimRequest, request: Request):
    """Agent claims a lease for an assigned job.

    This transitions the job from ASSIGNED → LEASED and starts
    the lease clock. The agent must renew before expiry.
    """
    _require_agent_auth(request, host_id=req.host_id)
    store = get_event_store()
    sm = get_state_machine()

    # Validate the job is in a claimable state.
    # Normal flow: job must be "assigned".
    # Restart-recovery: allow re-claiming if job is already leased/starting/running
    # on this same host (worker restart without server-side requeue).
    _RECLAIM_STATUSES = {"leased", "starting", "running"}
    jobs = list_jobs()
    job = next((j for j in jobs if j.get("job_id") == req.job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {req.job_id} not found")

    current_status = job.get("status", "queued")
    is_reclaim = current_status in _RECLAIM_STATUSES

    if current_status != "assigned" and not is_reclaim:
        raise HTTPException(
            status_code=409,
            detail=f"Job {req.job_id} is '{current_status}', expected 'assigned'",
        )

    # Verify the requesting host is the one assigned to this job
    assigned_host = job.get("host_id", "")
    if assigned_host and assigned_host != req.host_id:
        raise HTTPException(
            status_code=403,
            detail=f"Host {req.host_id} is not assigned to job {req.job_id}",
        )

    # Grant lease (or re-grant after worker restart)
    lease = store.grant_lease(req.job_id, req.host_id)

    if not is_reclaim:
        # Normal first-claim: transition assigned → leased
        try:
            sm.transition(
                req.job_id,
                "assigned",
                "leased",
                actor=f"agent:{req.host_id}",
                data={"lease_id": lease.lease_id},
            )
        except ValueError:
            pass  # Event already recorded by grant_lease

        # Update scheduler's job status to leased
        update_job_status(req.job_id, "leased", host_id=req.host_id)
    else:
        # Restart-recovery reclaim: job already past "leased", don't regress its status.
        # Just refresh the lease clock so the scheduler doesn't requeue a live container.
        log.info(
            "LEASE RECLAIM job=%s host=%s (status was '%s') — refreshing lease without status change",
            req.job_id,
            req.host_id,
            current_status,
        )

    broadcast_sse(
        "lease_granted",
        {
            "job_id": req.job_id,
            "host_id": req.host_id,
            "lease_id": lease.lease_id,
            "expires_at": lease.expires_at,
        },
    )

    return {
        "ok": True,
        "lease_id": lease.lease_id,
        "expires_at": lease.expires_at,
        "duration_sec": lease.duration_sec,
    }


@router.post("/agent/lease/renew", tags=["Agent"])
def api_agent_lease_renew(req: LeaseRenewRequest, request: Request):
    """Agent renews its lease on a job. Must be called before expiry."""
    _require_agent_auth(request, host_id=req.host_id)
    store = get_event_store()
    lease = store.renew_lease(req.job_id, req.host_id)
    if not lease:
        raise HTTPException(
            status_code=404,
            detail=f"No active lease for job {req.job_id} on host {req.host_id}",
        )
    return {
        "ok": True,
        "lease_id": lease.lease_id,
        "expires_at": lease.expires_at,
    }


@router.post("/agent/lease/release", tags=["Agent"])
def api_agent_lease_release(req: LeaseReleaseRequest, request: Request):
    """Agent releases its lease (job completed/failed/preempted)."""
    _require_agent_auth(request)
    store = get_event_store()
    released = store.release_lease(req.job_id)
    if not released:
        return {"ok": True, "released": False, "detail": "No active lease"}
    return {"ok": True, "released": True}


@router.get("/agent/popular-images", tags=["Agent"])
def api_agent_popular_images(request: Request):
    """Return popular container images for agent pre-pulling.

    Agents call this during idle time to pre-cache frequently-used images,
    reducing cold-start latency for future jobs.
    """
    _require_agent_auth(request)
    # Aggregate image usage from completed/running jobs
    jobs = list_jobs()
    image_counts: dict[str, int] = defaultdict(int)
    for j in jobs:
        img = j.get("image") or j.get("docker_image", "")
        if img:
            image_counts[img] += 1

    # Sort by frequency, return top 10
    popular = sorted(image_counts.items(), key=lambda x: -x[1])
    return {"images": [img for img, _ in popular[:10]]}


# ── Model: TelemetryPayload ──


class TelemetryPayload(BaseModel):
    host_id: str
    timestamp: float = 0
    metrics: dict = {}


@router.post("/agent/telemetry", tags=["Telemetry"])
def api_agent_telemetry(payload: TelemetryPayload, request: Request):
    """Receive periodic GPU telemetry from agent (every 5s)."""
    _require_agent_auth(request, host_id=payload.host_id)
    _host_telemetry[payload.host_id] = {
        "timestamp": payload.timestamp or time.time(),
        "metrics": payload.metrics,
        "received_at": time.time(),
    }
    return {"ok": True}


@router.get("/agent/telemetry/{host_id}", tags=["Telemetry"])
def api_get_telemetry(host_id: str, request: Request):
    """Get latest telemetry for a host (dashboard live gauges)."""
    _require_user_grant(request, allow_api_key=True)
    if host_id not in _host_telemetry:
        raise HTTPException(404, f"No telemetry for host {host_id}")

    data = _host_telemetry[host_id]
    stale = (time.time() - data.get("received_at", 0)) > 30  # >30s = stale
    return {"ok": True, "host_id": host_id, "stale": stale, **data}


@router.get("/api/telemetry/all", tags=["Telemetry"])
def api_all_telemetry(request: Request):
    """Get latest telemetry for all hosts (dashboard overview)."""
    user = _require_user_grant(request, allow_api_key=True)
    if not user.get("is_admin") and user.get("role") != "provider":
        raise HTTPException(403, "Admin or provider role required")
    now = time.time()
    result = {}
    for host_id, data in _host_telemetry.items():
        result[host_id] = {
            **data,
            "stale": (now - data.get("received_at", 0)) > 30,
        }
    return {"ok": True, "hosts": result, "count": len(result)}


# ── Log Forwarding (worker → API → SSE/WS → frontend) ──


@router.post("/agent/logs/{job_id}", tags=["Agent"])
async def api_agent_logs(job_id: str, request: Request):
    """Ingest container log lines from the worker agent.

    Accepts JSON body (LogBatch) or gzip-compressed JSON.
    Persists to job_logs table, pushes to in-memory buffer + SSE.
    """
    _require_agent_auth(request)
    # Basic shape validation on job_id (prevents injection of rows with weird
    # keys). Accept hex/slug forms used by our job-id generator.
    if not job_id or len(job_id) > 128 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]*", job_id):
        raise HTTPException(400, "Invalid job_id")

    # Cap the raw body at 2 MiB (uncompressed or compressed) to prevent gzip
    # bombs and abuse. 2 MiB is generous for 500 log lines × ~4 KiB each.
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > 2 * 1024 * 1024:
        raise HTTPException(413, "Payload too large")

    # Handle gzip-compressed body
    content_encoding = request.headers.get("content-encoding", "")
    raw = await request.body()
    if len(raw) > 2 * 1024 * 1024:
        raise HTTPException(413, "Payload too large")
    if content_encoding == "gzip":
        try:
            # Bound decompression too — refuse anything that would expand past 8 MiB
            decomp = gzip.decompress(raw)
            if len(decomp) > 8 * 1024 * 1024:
                raise HTTPException(413, "Decompressed payload too large")
            raw = decomp
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(400, "Invalid gzip payload")

    import json as _json

    try:
        data = _json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "Invalid JSON")

    lines = data.get("lines", [])
    if not lines:
        return {"ok": True, "accepted": 0}

    # Cap batch size to prevent abuse
    if len(lines) > 500:
        lines = lines[:500]

    # Silently drop logs for jobs that don't exist OR that have been in a
    # terminal state for >1h. Prevents zombie log-line floods from a
    # forgotten/rogue worker agent and keeps the job_logs table bounded.
    try:
        from db import load_jobs_snapshot

        snap = {j["job_id"]: j for j in load_jobs_snapshot()}
        job = snap.get(job_id)
        if not job:
            return {"ok": True, "accepted": 0, "dropped": "unknown_job"}
        if job.get("status") in {"completed", "failed", "cancelled", "terminated", "preempted"}:
            last_ts = job.get("completed_at") or job.get("updated_at") or 0
            try:
                last_ts_f = float(last_ts) if last_ts else 0.0
            except (TypeError, ValueError):
                last_ts_f = 0.0
            if last_ts_f and (time.time() - last_ts_f) > 3600:
                return {"ok": True, "accepted": 0, "dropped": "job_terminal"}
    except HTTPException:
        raise
    except Exception:
        # Snapshot failure — fall through and accept the log (fail-open so
        # a transient DB blip doesn't lose diagnostics).
        pass

    now = time.time()
    accepted = 0

    # Detect critical errors in log lines (image pull failures, OOM, etc.)
    critical_error = None
    for entry in lines:
        msg = entry.get("message", "")
        if msg and any(pat.lower() in msg.lower() for pat in _IMAGE_PULL_PATTERNS):
            critical_error = msg
            break

    if critical_error:
        from db import emit_event

        emit_event(
            "job_error",
            {
                "job_id": job_id,
                "error": "image_pull_failed",
                "message": f"Container image error: {critical_error[:200]}",
            },
        )

    # Persist to PG and broadcast to SSE in one pass
    from routes.instances import push_job_log

    try:
        pool = _get_pg_pool()
        with pool.connection() as conn:
            for entry in lines:
                msg = entry.get("message", "")
                # Cap individual line length to 8 KiB — protects the log
                # viewer, SSE wire, and PG column from pathological lines.
                if isinstance(msg, str) and len(msg) > 8192:
                    msg = msg[:8192] + "…[truncated]"
                level = entry.get("level", "info")
                if level not in {
                    "debug",
                    "info",
                    "warn",
                    "warning",
                    "error",
                    "fatal",
                    "stdout",
                    "stderr",
                }:
                    level = "info"
                ts = entry.get("timestamp") or now
                if not msg:
                    continue

                # Persist
                conn.execute(
                    "INSERT INTO job_logs (job_id, ts, level, line) VALUES (%s, %s, %s, %s)",
                    (job_id, ts, level, msg),
                )

                # Push to in-memory buffer + SSE (feeds LogViewer in real-time)
                # persist=False because we already inserted to PG above
                push_job_log(job_id, msg, level, ts, persist=False)
                accepted += 1

            conn.commit()
    except Exception as e:
        log.error("Failed to persist logs for job %s: %s", job_id, e)
        # PG failed — push to SSE only (persist=True to retry PG for these)
        for entry in lines:
            msg = entry.get("message", "")
            if isinstance(msg, str) and len(msg) > 8192:
                msg = msg[:8192] + "…[truncated]"
            if msg:
                push_job_log(
                    job_id,
                    msg,
                    entry.get("level", "info"),
                    entry.get("timestamp") or now,
                    persist=True,
                )
                accepted += 1

    return {"ok": True, "accepted": accepted}


@router.get("/agent/ssh-keys/{job_id}", tags=["Agent"])
def api_agent_ssh_keys(job_id: str, request: Request):
    """Return the job owner's SSH public keys so the agent can inject them into containers.

    Requires authentication — this endpoint previously leaked every user's
    authorized public keys to any unauthenticated caller who guessed a job id.
    """
    _require_agent_auth(request)
    from db import UserStore

    # Look up the job to find its owner
    all_jobs = list_jobs()
    job = None
    for j in all_jobs:
        if j.get("job_id") == job_id:
            job = j
            break
    if not job:
        raise HTTPException(404, "Job not found")

    owner = job.get("owner", "")
    if not owner:
        return {"ok": True, "keys": []}

    # owner is customer_id or user_id — find the user
    user = UserStore.get_user_by_id(owner)
    if not user:
        # Try looking up by customer_id
        try:
            pool = _get_pg_pool()
            with pool.connection() as conn:
                row = conn.execute(
                    "SELECT email FROM users WHERE customer_id = %s LIMIT 1", (owner,)
                ).fetchone()
                if row:
                    user = {"email": row[0] if isinstance(row, tuple) else dict(row)["email"]}
        except Exception:
            pass

    if not user or not user.get("email"):
        return {"ok": True, "keys": []}

    keys = UserStore.list_ssh_keys(user["email"])
    return {
        "ok": True,
        "keys": [k["public_key"] for k in keys],
    }
