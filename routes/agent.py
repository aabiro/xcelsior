"""Routes: agent."""

import gzip
import time
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from routes._deps import (
    broadcast_sse,
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
    host_id: str
    versions: dict


# ── Model: MiningAlert ──

class MiningAlert(BaseModel):
    host_id: str
    gpu_index: int
    confidence: float
    reason: str
    timestamp: float | None = None


# ── Model: BenchmarkReport ──

class BenchmarkReport(BaseModel):
    host_id: str
    gpu_model: str
    score: float
    tflops: float
    details: dict | None = None

@router.get("/agent/work/{host_id}", tags=["Agent"])
def api_agent_work(host_id: str):
    """Pull pending work for an agent. Returns assigned jobs."""
    all_jobs = list_jobs()
    pending = [
        j for j in all_jobs if j.get("host_id") == host_id and j.get("status") in ("assigned",)
    ]
    with _agent_lock:
        queued_work = _agent_work.pop(host_id, [])

    jobs = pending + queued_work
    if not jobs:
        return JSONResponse(status_code=204, content=None)
    return {"ok": True, "instances": jobs}

@router.get("/agent/preempt/{host_id}", tags=["Agent"])
def api_agent_preempt(host_id: str):
    """Check if any jobs on this host should be preempted."""
    with _agent_lock:
        preempt_list = _agent_preempt.pop(host_id, [])
    return {"ok": True, "preempt_jobs": preempt_list}

@router.post("/agent/preempt/{host_id}/{job_id}", tags=["Agent"])
def api_schedule_preemption(host_id: str, job_id: str):
    """Schedule a job for preemption on a host."""
    with _agent_lock:
        _agent_preempt[host_id].append(job_id)
    broadcast_sse("preemption_scheduled", {"host_id": host_id, "job_id": job_id})
    return {"ok": True, "host_id": host_id, "job_id": job_id}

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
def api_mining_alert(alert: MiningAlert):
    """Receive mining detection alert from an agent."""
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
def api_agent_benchmark(report: BenchmarkReport):
    """Receive compute benchmark results from an agent."""
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
def api_agent_lease_claim(req: LeaseClaimRequest):
    """Agent claims a lease for an assigned job.

    This transitions the job from ASSIGNED → LEASED and starts
    the lease clock. The agent must renew before expiry.
    """
    store = get_event_store()
    sm = get_state_machine()

    # Validate the job is in ASSIGNED state
    jobs = list_jobs()
    job = next((j for j in jobs if j.get("job_id") == req.job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {req.job_id} not found")

    current_status = job.get("status", "queued")
    if current_status != "assigned":
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

    # Grant lease
    lease = store.grant_lease(req.job_id, req.host_id)

    # Transition state: assigned → leased
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
def api_agent_lease_renew(req: LeaseRenewRequest):
    """Agent renews its lease on a job. Must be called before expiry."""
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
def api_agent_lease_release(req: LeaseReleaseRequest):
    """Agent releases its lease (job completed/failed/preempted)."""
    store = get_event_store()
    released = store.release_lease(req.job_id)
    if not released:
        return {"ok": True, "released": False, "detail": "No active lease"}
    return {"ok": True, "released": True}

@router.get("/agent/popular-images", tags=["Agent"])
def api_agent_popular_images():
    """Return popular container images for agent pre-pulling.

    Agents call this during idle time to pre-cache frequently-used images,
    reducing cold-start latency for future jobs.
    """
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
def api_agent_telemetry(payload: TelemetryPayload):
    """Receive periodic GPU telemetry from agent (every 5s)."""
    _host_telemetry[payload.host_id] = {
        "timestamp": payload.timestamp or time.time(),
        "metrics": payload.metrics,
        "received_at": time.time(),
    }
    return {"ok": True}

@router.get("/agent/telemetry/{host_id}", tags=["Telemetry"])
def api_get_telemetry(host_id: str):
    """Get latest telemetry for a host (dashboard live gauges)."""
    if host_id not in _host_telemetry:
        raise HTTPException(404, f"No telemetry for host {host_id}")

    data = _host_telemetry[host_id]
    stale = (time.time() - data.get("received_at", 0)) > 30  # >30s = stale
    return {"ok": True, "host_id": host_id, "stale": stale, **data}

@router.get("/api/telemetry/all", tags=["Telemetry"])
def api_all_telemetry():
    """Get latest telemetry for all hosts (dashboard overview)."""
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
    # Handle gzip-compressed body
    content_encoding = request.headers.get("content-encoding", "")
    raw = await request.body()
    if content_encoding == "gzip":
        try:
            raw = gzip.decompress(raw)
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
        emit_event("job_error", {
            "job_id": job_id,
            "error": "image_pull_failed",
            "message": f"Container image error: {critical_error[:200]}",
        })

    # Persist to PG and broadcast to SSE in one pass
    from routes.instances import push_job_log

    try:
        pool = _get_pg_pool()
        with pool.connection() as conn:
            for entry in lines:
                msg = entry.get("message", "")
                level = entry.get("level", "info")
                ts = entry.get("timestamp") or now
                if not msg:
                    continue

                # Persist
                conn.execute(
                    "INSERT INTO job_logs (job_id, ts, level, line) VALUES (%s, %s, %s, %s)",
                    (job_id, ts, level, msg),
                )

                # Push to in-memory buffer + SSE (feeds LogViewer in real-time)
                push_job_log(job_id, msg, level, ts)
                accepted += 1

            conn.commit()
    except Exception as e:
        log.error("Failed to persist logs for job %s: %s", job_id, e)
        # Still try to push to SSE even if PG fails
        for entry in lines:
            msg = entry.get("message", "")
            if msg:
                push_job_log(job_id, msg, entry.get("level", "info"), entry.get("timestamp") or now)
                accepted += 1

    return {"ok": True, "accepted": accepted}


@router.get("/agent/ssh-keys/{job_id}", tags=["Agent"])
def api_agent_ssh_keys(job_id: str):
    """Return the job owner's SSH public keys so the agent can inject them into containers."""
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
