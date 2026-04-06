"""Routes: instances."""

import asyncio
import hmac
import json
import os
import time

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field, field_validator

from routes._deps import (
    AUTH_REQUIRED,
    _AUTH_COOKIE_NAME,
    _USE_PERSISTENT_AUTH,
    _api_keys,
    _require_auth,
    _sessions,
    _sse_lock,
    _sse_subscribers,
    _user_lock,
    broadcast_sse,
    log,
    otel_span,
)
from scheduler import (
    API_TOKEN,
    failover_and_reassign,
    kill_job,
    list_hosts,
    list_jobs,
    list_tiers,
    log,
    process_queue,
    process_queue_binpack,
    requeue_job,
    run_job,
    submit_job,
    update_job_status,
)
from db import UserStore
from collections import defaultdict
from routes.agent import _agent_lock, _agent_preempt

router = APIRouter()

# Job log and WebSocket state
_JOB_LOG_MAX = 500
_job_log_buffers: dict[str, list[dict]] = defaultdict(list)
_ws_connections: dict[str, set] = defaultdict(set)


def _check_job_access(user: dict, job_id: str):
    """Verify user owns the job or is admin."""
    if user.get("role") == "admin" or user.get("is_admin"):
        return
    from scheduler import get_job
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Instance {job_id} not found")
    job_owner = job.get("owner", "")
    customer_id = user.get("customer_id", user.get("user_id", ""))
    if job_owner and job_owner != customer_id:
        raise HTTPException(403, "Not authorized to access this instance")


class JobIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    vram_needed_gb: float = Field(default=0, ge=0)
    priority: int = Field(default=0, ge=0, le=10)
    tier: str | None = None
    num_gpus: int = Field(default=1, ge=1, le=64)
    host_id: str | None = None
    gpu_model: str | None = None
    nfs_server: str | None = None
    nfs_path: str | None = None
    nfs_mount_point: str | None = None
    image: str | None = None
    interactive: bool = True
    command: str | None = None
    ssh_port: int = Field(default=22, ge=1, le=65535)

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return v
        from security import validate_docker_image
        return validate_docker_image(v)


class StatusUpdate(BaseModel):
    status: str
    host_id: str | None = None
    container_id: str | None = None
    container_name: str | None = None
    ssh_port: int | None = None
    interactive: bool | None = None


# ── Image Templates ──

class ImageTemplate(BaseModel):
    id: str
    label: str
    image: str
    default_vram_gb: int
    icon: str
    category: str
    description: str


@router.get("/api/images/templates", tags=["Images"])
def api_image_templates():
    """Return the authoritative list of validated container image templates.

    Single source of truth consumed by frontend, wizard, and API callers.
    """
    from security import get_image_templates
    return {"templates": [ImageTemplate(**t).model_dump() for t in get_image_templates()]}


# ── Helper: _refresh_job ──

def _refresh_job(job_id: str):
    """Re-read a job from the DB to get updated status/host/container fields."""
    for j in list_jobs():
        if j["job_id"] == job_id:
            return j
    return None

@router.post("/instance", tags=["Instances"])
def api_submit_instance(j: JobIn, request: Request):
    """Submit a job to the queue or directly assign to a host.

    If host_id is provided (marketplace launch), the job is assigned directly
    to that host and container start is attempted immediately. Otherwise,
    the job is queued and process_queue runs to find a host.
    """
    user = _require_auth(request)

    # If host_id provided but no vram, look it up from the host record
    vram_needed = j.vram_needed_gb
    target_host_id = j.host_id
    if target_host_id and vram_needed <= 0:
        hosts = list_hosts()
        hmap = {h["host_id"]: h for h in hosts}
        target = hmap.get(target_host_id)
        if target:
            vram_needed = float(target.get("total_vram_gb", 24) or 24)
        else:
            vram_needed = 24.0
    elif vram_needed <= 0:
        # Auto GPU: use the smallest available host's free VRAM so the job
        # can actually be scheduled rather than defaulting to 24 GB.
        hosts = list_hosts()
        if hosts:
            vram_needed = float(min(h.get("free_vram_gb", 0) for h in hosts) or 4.0)
        else:
            vram_needed = 4.0  # minimal default when no hosts available

    with otel_span("job.submit", {"job.name": j.name, "job.tier": j.tier or "", "job.num_gpus": j.num_gpus}):
        customer_id = user.get("customer_id", user.get("user_id", ""))

        # ── Wallet pre-flight: block launch if wallet is broke ────────
        from billing import get_billing_engine
        be = get_billing_engine()
        wallet = be.get_wallet(customer_id)
        if wallet.get("status") == "suspended":
            raise HTTPException(402, detail="Wallet suspended — please add funds to resume service")
        if wallet["balance_cad"] <= 0 and wallet.get("grace_until", 0) < time.time():
            raise HTTPException(402, detail="Insufficient wallet balance — please deposit credits")

        job = submit_job(
            j.name,
            vram_needed,
            j.priority,
            tier=j.tier,
            num_gpus=j.num_gpus,
            nfs_server=j.nfs_server,
            nfs_path=j.nfs_path,
            nfs_mount_point=j.nfs_mount_point,
            image=j.image,
            interactive=j.interactive,
            command=j.command,
            ssh_port=j.ssh_port,
            owner=customer_id,
        )
        # Track job ownership (response-only, already persisted via owner param)
        job["submitted_by"] = user.get("email", "")
        job["customer_id"] = customer_id
        broadcast_sse("job_submitted", {"job_id": job["job_id"], "name": job["name"]})

        # Direct host assignment: assign + start immediately
        if target_host_id:
            try:
                updated = update_job_status(job["job_id"], "assigned", host_id=target_host_id)
                if updated and updated.get("status") == "assigned":
                    hosts = list_hosts()
                    hmap = {h["host_id"]: h for h in hosts}
                    host = hmap.get(target_host_id)
                    if host:
                        container_id = run_job(updated, host, docker_image=j.image or None)
                        if container_id:
                            job = _refresh_job(job["job_id"]) or job
                            log.info("Direct launch: job %s running on host %s", job["job_id"], target_host_id)
                        else:
                            job = _refresh_job(job["job_id"]) or job
                            log.warning("Direct launch: container start failed for job %s on host %s",
                                        job["job_id"], target_host_id)
                    else:
                        log.warning("Direct launch: host %s not found, job %s stays queued", target_host_id, job["job_id"])
                        update_job_status(job["job_id"], "queued")
                        job = _refresh_job(job["job_id"]) or job
            except Exception as e:
                log.error("Direct launch failed for job %s: %s", job["job_id"], e)
                job = _refresh_job(job["job_id"]) or job
        else:
            # Auto-process queue to try to assign immediately
            try:
                process_queue()
                # Refresh job status after queue processing
                job = _refresh_job(job["job_id"]) or job
            except Exception as e:
                log.warning("Queue processing after submit failed: %s", e)

        return {"ok": True, "instance": job}

@router.get("/instances", tags=["Instances"])
def api_list_instances(status: str | None = None):
    """List jobs. Optional filter by status."""
    jobs = list_jobs(status=status)
    # Enrich with host GPU info where available
    hosts = list_hosts()
    host_map = {h["host_id"]: h for h in hosts}
    for j in jobs:
        # Map 'image' → 'docker_image'
        if j.get("image") and not j.get("docker_image"):
            j["docker_image"] = j["image"]
        hid = j.get("host_id")
        if hid and hid in host_map:
            j.setdefault("gpu_type", host_map[hid].get("gpu_model", ""))
            j.setdefault("host_gpu", host_map[hid].get("gpu_model", ""))
        # Compute elapsed / duration
        started = float(j.get("started_at") or 0)
        completed = float(j.get("completed_at") or 0)
        if started > 0:
            if completed > started:
                j.setdefault("duration_sec", round(completed - started, 2))
            elif j.get("status") == "running":
                j.setdefault("elapsed_sec", round(time.time() - started, 2))
    return {"instances": jobs}

@router.get("/instance/{job_id}", tags=["Instances"])
def api_get_instance(job_id: str):
    """Get a specific instance by ID, enriched with connection info."""
    jobs = list_jobs()
    for j in jobs:
        if j["job_id"] == job_id:
            # Map 'image' → 'docker_image' for frontend
            if j.get("image") and not j.get("docker_image"):
                j["docker_image"] = j["image"]

            # Compute elapsed / duration from timestamps
            started = float(j.get("started_at") or 0)
            completed = float(j.get("completed_at") or 0)
            if started > 0:
                if completed > started:
                    j["duration_sec"] = round(completed - started, 2)
                    j["elapsed_sec"] = j["duration_sec"]
                elif j.get("status") == "running":
                    j["elapsed_sec"] = round(time.time() - started, 2)

            # Enrich with host connection details when running
            host = None
            if j.get("host_id") and j.get("status") in ("running", "completed", "failed"):
                hosts = list_hosts()
                host = next((h for h in hosts if h["host_id"] == j["host_id"]), None)
                if host:
                    j["host_ip"] = host.get("ip", "")
                    j["host_gpu"] = host.get("gpu_model", "")
                    j["host_vram_gb"] = host.get("total_vram_gb", 0)
                    j.setdefault("gpu_type", host.get("gpu_model", ""))

            # Compute cost_cad for running/completed jobs
            elapsed = j.get("elapsed_sec") or j.get("duration_sec") or 0
            if elapsed > 0 and j.get("cost_cad") is None:
                rate = 0.20  # default CAD/hr
                if host and host.get("cost_per_hour"):
                    rate = float(host["cost_per_hour"])
                j["cost_cad"] = round((elapsed / 3600) * rate, 4)

            return {"instance": j}
    raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")

@router.patch("/instance/{job_id}", tags=["Instances"])
def api_update_instance(job_id: str, update: StatusUpdate):
    """Update a job's status."""
    with otel_span("job.status_update", {"job.id": job_id, "job.status": update.status}):
        try:
            update_job_status(job_id, update.status, host_id=update.host_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        # Store container info if provided by worker agent
        extras = {}
        if update.container_id:
            extras["container_id"] = update.container_id
        if update.container_name:
            extras["container_name"] = update.container_name
        if update.ssh_port is not None:
            extras["ssh_port"] = update.ssh_port
        if update.interactive is not None:
            extras["interactive"] = update.interactive
        if extras:
            from scheduler import _set_job_fields
            _set_job_fields(job_id, **extras)
        broadcast_sse("job_status", {"job_id": job_id, "status": update.status})
        return {"ok": True, "job_id": job_id, "status": update.status}

@router.post("/queue/process", tags=["Instances"])
def api_process_queue():
    """Process the job queue — assign jobs to hosts."""
    assigned = process_queue()
    result = [{"job": j["name"], "job_id": j["job_id"], "host": h["host_id"]} for j, h in assigned]
    if result:
        broadcast_sse("queue_processed", {"assigned_count": len(result)})
    return {"assigned": result}

@router.post("/failover", tags=["Instances"])
def api_failover():
    """Run a full failover cycle: check hosts, requeue orphaned jobs, reassign."""
    requeued, assigned = failover_and_reassign()
    return {
        "requeued": [
            {"job_id": j["job_id"], "name": j["name"], "retries": j.get("retries", 0)}
            for j in requeued
        ],
        "assigned": [
            {"job": j["name"], "job_id": j["job_id"], "host": h["host_id"]} for j, h in assigned
        ],
    }

@router.post("/instances/{job_id}/cancel", tags=["Instances"])
def api_cancel_instance(job_id: str, request: Request):
    """Cancel a running or queued instance. For interactive instances, stops the container."""
    _require_auth(request)
    jobs = list_jobs()
    job = next((j for j in jobs if j["job_id"] == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")

    if job.get("status") in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail=f"Instance already {job['status']}")

    # If running on a host, kill the container directly and also schedule via agent
    if job.get("host_id") and job.get("status") in ("running", "assigned", "leased"):
        hosts = list_hosts()
        hmap = {h["host_id"]: h for h in hosts}
        host = hmap.get(job["host_id"])
        if host:
            try:
                kill_job(job, host)
            except Exception as e:
                log.warning("Container kill failed for %s: %s", job_id, e)
        with _agent_lock:
            _agent_preempt[job["host_id"]].append(job_id)

    update_job_status(job_id, "cancelled")
    broadcast_sse("job_cancelled", {"job_id": job_id})
    return {"ok": True, "job_id": job_id, "status": "cancelled"}

@router.post("/instance/{job_id}/requeue", tags=["Instances"])
def api_requeue_instance(job_id: str, request: Request):
    """Manually requeue a failed or stuck job."""
    _require_auth(request)
    result = requeue_job(job_id)
    if not result:
        raise HTTPException(
            status_code=400,
            detail=f"Could not requeue job {job_id} (max retries exceeded or not found)",
        )
    return {"ok": True, "instance": result}


# ── Pause / Resume ───────────────────────────────────────────────────

@router.post("/instances/{job_id}/pause", tags=["Instances"])
def api_pause_instance(job_id: str, request: Request):
    """Pause a running instance. Stops the container but preserves volumes.

    Requires authentication. Only the instance owner or an admin can pause.
    """
    user = _require_auth(request)
    customer_id = user.get("customer_id", user.get("user_id", ""))
    role = user.get("role", "")

    from scheduler import get_job
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")

    # Owner check (admins can pause any instance)
    job_owner = job.get("owner", "")
    if role != "admin" and job_owner != customer_id:
        raise HTTPException(status_code=403, detail="Not authorized to pause this instance")

    if job.get("status") != "running":
        raise HTTPException(status_code=400, detail=f"Instance is {job.get('status')}, not running")

    from billing import get_billing_engine
    be = get_billing_engine()
    result = be.pause_instance(job_id, reason="user_paused")
    if not result.get("paused"):
        raise HTTPException(status_code=400, detail=result.get("reason", "pause failed"))

    broadcast_sse("instance_paused", {"job_id": job_id})
    return {"ok": True, "instance": result}


@router.post("/instances/{job_id}/resume", tags=["Instances"])
def api_resume_instance(job_id: str, request: Request):
    """Resume a paused instance. Restarts the container from preserved state.

    Requires authentication. Only the instance owner or an admin can resume.
    Returns 402 if wallet has insufficient funds.
    """
    user = _require_auth(request)
    customer_id = user.get("customer_id", user.get("user_id", ""))
    role = user.get("role", "")

    from scheduler import get_job
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")

    # Owner check (admins can resume any instance)
    job_owner = job.get("owner", "")
    if role != "admin" and job_owner != customer_id:
        raise HTTPException(status_code=403, detail="Not authorized to resume this instance")

    status = job.get("status", "")
    if status not in ("paused_low_balance", "user_paused"):
        raise HTTPException(status_code=400, detail=f"Instance is {status}, not paused")

    # Wallet pre-flight — must have funds to resume
    from billing import get_billing_engine
    be = get_billing_engine()
    wallet = be.get_wallet(job_owner or customer_id)
    if wallet.get("status") == "suspended":
        raise HTTPException(status_code=402, detail="Wallet suspended — please add funds")
    if wallet["balance_cad"] <= 0:
        raise HTTPException(status_code=402, detail="Insufficient wallet balance to resume")

    result = be.resume_instance(job_id)
    if not result.get("resumed"):
        detail = result.get("reason", "resume failed")
        code = 402 if detail == "insufficient_balance" else 400
        raise HTTPException(status_code=code, detail=detail)

    broadcast_sse("instance_resumed", {"job_id": job_id})
    return {"ok": True, "instance": result}


# ── Helper: push_job_log ──

def push_job_log(job_id: str, line: str, level: str = "info", timestamp: float | None = None):
    """Push a log line into the per-job log buffer (called from scheduler/worker)."""
    entry = {"timestamp": timestamp or time.time(), "line": line, "message": line, "level": level}
    buf = _job_log_buffers[job_id]
    buf.append(entry)
    if len(buf) > _JOB_LOG_MAX:
        _job_log_buffers[job_id] = buf[-_JOB_LOG_MAX:]
    # Also broadcast to general SSE stream
    broadcast_sse("job_log", {"job_id": job_id, **entry})


# ── Helper: _load_pg_logs ──

def _load_pg_logs(job_id: str, limit: int = 200) -> list[dict]:
    """Load recent log lines from PG job_logs table (fallback when in-memory buffer is empty)."""
    try:
        from db import _get_pg_pool
        pool = _get_pg_pool()
        with pool.connection() as conn:
            rows = conn.execute(
                "SELECT ts AS timestamp, level, line, line AS message FROM job_logs "
                "WHERE job_id = %s ORDER BY ts DESC LIMIT %s",
                (job_id, limit),
            ).fetchall()
        if not rows:
            return []
        cols = ["timestamp", "level", "line", "message"]
        return [dict(zip(cols, r)) for r in reversed(rows)]  # chronological order
    except Exception:
        return []


# ── Helper: _job_log_generator ──

async def _job_log_generator(request: Request, job_id: str):
    """Async generator that yields SSE events for a specific job.

    Replays buffered log lines, then live-tails new events
    from the broadcast SSE bus filtered to this job_id.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    with _sse_lock:
        _sse_subscribers.append(queue)

    try:
        # SSE retry hint for automatic browser reconnection
        yield "retry: 3000\n\n"

        # Replay buffered log lines (PG fallback if buffer is empty)
        replay = list(_job_log_buffers.get(job_id, []))
        if not replay:
            replay = _load_pg_logs(job_id, limit=200)
        for entry in replay:
            data = json.dumps({"job_id": job_id, **entry})
            yield f"event: job_log\ndata: {data}\n\n"

        yield f"event: connected\ndata: {json.dumps({'job_id': job_id, 'status': 'streaming'})}\n\n"

        while True:
            if await request.is_disconnected():
                break
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30)
                event_type = msg.get("event", "message")
                event_data = msg.get("data", {})
                # Filter: only pass through events for this job
                if event_data.get("job_id") == job_id or event_type in (
                    "job_status",
                    "job_log",
                    "lease_claimed",
                    "lease_released",
                    "job_completed",
                    "job_failed",
                ):
                    if event_data.get("job_id", "") == job_id:
                        yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
    finally:
        with _sse_lock:
            if queue in _sse_subscribers:
                _sse_subscribers.remove(queue)

@router.get("/instances/{job_id}/logs/stream", tags=["Instances"])
async def api_instance_log_stream(request: Request, job_id: str):
    """Stream real-time logs for a specific job via Server-Sent Events.

    Connect with `EventSource('/jobs/{job_id}/logs/stream')` in the browser
    or `curl -N` from the CLI. Replays buffered log lines on connect, then
    live-tails new log entries until the client disconnects or the job completes.

    Events emitted:
    - `job_log` — individual log line (data: {job_id, timestamp, line, level})
    - `job_status` — status change (data: {job_id, status})
    - `connected` — initial handshake (data: {job_id, status: "streaming"})
    """
    _require_auth(request)
    # Verify job exists
    jobs = list_jobs()
    if not any(j["job_id"] == job_id for j in jobs):
        raise HTTPException(404, f"Job {job_id} not found")

    return StreamingResponse(
        _job_log_generator(request, job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@router.get("/instances/{job_id}/logs", tags=["Instances"])
def api_instance_logs(job_id: str, request: Request, limit: int = 100):
    """Get buffered log lines for a job (non-streaming).

    Returns the last `limit` log lines. Tries in-memory buffer first,
    falls back to persistent job_logs table in PG.
    For real-time streaming, use `/jobs/{job_id}/logs/stream` (SSE).
    """
    user = _require_auth(request)
    _check_job_access(user, job_id)
    limit = min(limit, 10_000)
    buf = _job_log_buffers.get(job_id, [])

    # If in-memory buffer is empty, load from PG
    if not buf:
        buf = _load_pg_logs(job_id, limit=limit)

    return {"ok": True, "job_id": job_id, "logs": buf[-limit:], "total": len(buf)}


@router.get("/instances/{job_id}/logs/download", tags=["Instances"])
def api_instance_logs_download(job_id: str, request: Request):
    """Download all available logs for a job as a plain text file."""
    user = _require_auth(request)
    _check_job_access(user, job_id)
    buf = _job_log_buffers.get(job_id, [])
    if not buf:
        buf = _load_pg_logs(job_id, limit=10_000)
    if not buf:
        raise HTTPException(404, "No logs available for this job")

    lines = []
    for entry in buf:
        ts = entry.get("timestamp", "")
        level = entry.get("level", "")
        msg = entry.get("message", "") or entry.get("line", "")
        lines.append(f"{ts} [{level.upper()}] {msg}")
    text = "\n".join(lines)

    import re
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', job_id)[:128]
    return Response(
        content=text,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{safe_id}-logs.txt"'},
    )


# ── Helper: _validate_ws_auth ──

def _validate_ws_auth(websocket: WebSocket) -> dict | None:
    """Validate auth for WebSocket connections (mirrors TokenAuthMiddleware).
    Returns user dict on success, None on failure."""
    if not AUTH_REQUIRED:
        return {"email": "anonymous", "user_id": "anonymous", "role": "admin", "is_admin": True}
    api_token = os.environ.get("XCELSIOR_API_TOKEN", API_TOKEN)
    token = websocket.cookies.get(_AUTH_COOKIE_NAME, "")
    if not token:
        token = websocket.query_params.get("token", "")
    if not token:
        return None
    if api_token and hmac.compare_digest(token, api_token):
        return {"email": "api-token", "user_id": "api-token", "role": "admin", "is_admin": True}
    if _USE_PERSISTENT_AUTH:
        session = UserStore.get_session(token)
        if session:
            return dict(session)
        api_key = UserStore.get_api_key(token)
        if api_key:
            return {
                "email": api_key["email"],
                "user_id": api_key["user_id"],
                "role": api_key.get("role", "submitter"),
            }
    else:
        with _user_lock:
            if token in _sessions and _sessions[token]["expires_at"] > time.time():
                return _sessions[token]
            if token in _api_keys:
                return _api_keys[token]
    return None

@router.get("/tiers", tags=["Instances"])
def api_list_tiers():
    """List all priority tiers with their multipliers."""
    return {"tiers": list_tiers()}

@router.post("/api/v2/scheduler/process-binpack", tags=["Jobs"])
def api_process_queue_binpack(canada_only: bool = False, province: str = ""):
    """Process job queue using best-fit-decreasing bin packing."""
    assigned = process_queue_binpack(
        canada_only=canada_only or None,
        province=province or None,
    )
    return {"ok": True, "assigned": assigned, "count": len(assigned)}



# ── WebSocket Instance Streaming ──────────────────────────────────

@router.websocket("/ws/instances/{job_id}")
async def ws_instance_stream(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time instance updates.

    Sends:
    - ``instance`` — full instance snapshot on connect and on status changes
    - ``job_log`` — individual log lines
    - ``job_status`` — status change notifications
    - ``ping`` — keepalive every 30 s

    Client can send ``{"event": "pong"}`` or ``{"event": "refresh"}``
    to request a fresh instance snapshot.
    """
    user = _validate_ws_auth(websocket)
    if not user:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()

    # Verify job exists and send initial snapshot
    jobs = list_jobs()
    instance = next((j for j in jobs if j["job_id"] == job_id), None)
    if not instance:
        await websocket.send_json({"event": "error", "data": {"message": "Instance not found"}})
        await websocket.close(code=4004)
        return

    # Ownership check — only job owner or admin may stream
    if not (user.get("role") == "admin" or user.get("is_admin")):
        job_owner = instance.get("owner", "")
        customer_id = user.get("customer_id", user.get("user_id", ""))
        if job_owner and job_owner != customer_id:
            await websocket.send_json({"event": "error", "data": {"message": "Not authorized"}})
            await websocket.close(code=4003)
            return

    _ws_connections[job_id].add(websocket)
    await websocket.send_json({"event": "instance", "data": instance})

    # Replay buffered logs (PG fallback if buffer is empty)
    replay = list(_job_log_buffers.get(job_id, []))[-50:]
    if not replay:
        replay = _load_pg_logs(job_id, limit=50)
    for entry in replay:
        await websocket.send_json({"event": "job_log", "data": {"job_id": job_id, **entry}})

    # Subscribe to the broadcast SSE bus
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    with _sse_lock:
        _sse_subscribers.append(queue)

    closed = False

    async def _send_loop():
        nonlocal closed
        while not closed:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_json({"event": "ping", "data": {"ts": time.time()}})
                continue
            event_type = msg.get("event", "message")
            event_data = msg.get("data", {})
            if event_data.get("job_id") != job_id:
                continue
            await websocket.send_json({"event": event_type, "data": event_data})
            # On status change, send full instance snapshot
            if event_type == "job_status":
                fresh = next((j for j in list_jobs() if j["job_id"] == job_id), None)
                if fresh:
                    await websocket.send_json({"event": "instance", "data": fresh})

    async def _recv_loop():
        nonlocal closed
        while not closed:
            try:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                if data.get("event") == "refresh":
                    fresh = next((j for j in list_jobs() if j["job_id"] == job_id), None)
                    if fresh:
                        await websocket.send_json({"event": "instance", "data": fresh})
            except (WebSocketDisconnect, RuntimeError):
                closed = True
                break
            except (json.JSONDecodeError, KeyError):
                pass

    try:
        done, pending = await asyncio.wait(
            [asyncio.ensure_future(_send_loop()), asyncio.ensure_future(_recv_loop())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    finally:
        closed = True
        _ws_connections[job_id].discard(websocket)
        with _sse_lock:
            if queue in _sse_subscribers:
                _sse_subscribers.remove(queue)
        try:
            await websocket.close()
        except Exception as e:
            log.debug("WS close error: %s", e)


# ── WebSocket Terminal Proxy ──────────────────────────────────────


# ── WebSocket Terminal Proxy ──────────────────────────────────────────
# xterm.js <-> WebSocket <-> docker exec via Tailscale mesh
# Security: JWT auth, 30-min session timeout, 10 KB/s rate limit

_TERMINAL_SESSION_TIMEOUT = 1800  # 30 minutes
_TERMINAL_MAX_SCROLLBACK = 50_000  # 50 KB
_TERMINAL_RATE_LIMIT_BYTES = 10_240  # 10 KB/s


@router.websocket("/ws/terminal/{instance_id}")
async def ws_terminal(websocket: WebSocket, instance_id: str):
    """Interactive terminal session for a running instance.

    Proxies stdin/stdout between the browser (xterm.js) and
    ``docker exec -it <container_id> /bin/bash`` on the worker host
    reached through the Tailscale mesh.

    Protocol:
    - Client sends: ``{"type": "input", "data": "<chars>"}``
    - Client sends: ``{"type": "resize", "cols": N, "rows": N}``
    - Server sends: ``{"type": "output", "data": "<chars>"}``
    - Server sends: ``{"type": "error", "message": "..."}``
    - Server sends: ``{"type": "exit", "code": N}``
    """
    user = _validate_ws_auth(websocket)
    if not user:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()

    # Resolve the instance
    jobs = list_jobs()
    instance = next((j for j in jobs if j["job_id"] == instance_id), None)
    if not instance:
        await websocket.send_json({"type": "error", "message": "Instance not found"})
        await websocket.close(code=4004)
        return

    # Ownership check
    if not (user.get("role") == "admin" or user.get("is_admin")):
        job_owner = instance.get("owner", "")
        customer_id = user.get("customer_id", user.get("user_id", ""))
        if job_owner and job_owner != customer_id:
            await websocket.send_json({"type": "error", "message": "Not authorized"})
            await websocket.close(code=4003)
            return

    if instance.get("status") != "running":
        await websocket.send_json({"type": "error", "message": f"Instance is {instance.get('status', 'unknown')}, not running"})
        await websocket.close(code=4003)
        return

    host_id = instance.get("host_id", "")
    # Prefer container_name (xcl-{job_id}) over container_id for reliability
    container_ref = instance.get("container_name") or instance.get("container_id") or f"xcl-{instance_id}"
    if not host_id:
        await websocket.send_json({"type": "error", "message": "No host assigned"})
        await websocket.close(code=4003)
        return

    # Look up host IP for remote SSH proxy
    host_ip = instance.get("host_ip", "")
    if not host_ip:
        hmap = {h["host_id"]: h for h in list_hosts(active_only=False)}
        host_record = hmap.get(host_id)
        host_ip = host_record.get("ip", "") if host_record else ""

    # Validate host_ip format before SSH
    import re as _re
    if host_ip and not _re.match(r'^[a-zA-Z0-9._-]+$', host_ip):
        await websocket.send_json({"type": "error", "message": "Invalid host address"})
        await websocket.close(code=4003)
        return

    shell = instance.get("shell", "/bin/bash")
    is_remote = bool(host_ip) and host_ip not in ("127.0.0.1", "localhost", "0.0.0.0")

    if is_remote:
        # SSH into remote host and run docker exec there
        from scheduler import SSH_KEY_PATH, SSH_USER
        import shlex
        docker_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
            "-i", SSH_KEY_PATH,
            "-tt",
            f"{SSH_USER}@{host_ip}",
            "docker", "exec", "-e", "TERM=xterm-256color", "-it",
            shlex.quote(container_ref), shell,
        ]
    else:
        # Local dev: direct docker exec
        docker_cmd = [
            "docker", "exec", "-e", "TERM=xterm-256color", "-it",
            container_ref, shell,
        ]

    session_start = time.time()
    process = None
    closed = False
    master_fd = None

    try:
        import pty as _pty, fcntl, struct, termios
        master_fd, slave_fd = _pty.openpty()
        process = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
        )
        os.close(slave_fd)
        os.set_blocking(master_fd, False)
    except (FileNotFoundError, ImportError, OSError):
        # Fallback: no PTY support (resize won't work)
        if master_fd is not None:
            try:
                os.close(master_fd)
            except OSError:
                pass
            master_fd = None
        try:
            docker_cmd[2] = "-i"  # downgrade to -i (no tty)
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            await websocket.send_json({"type": "error", "message": "Docker not available on this host"})
            await websocket.close(code=4003)
            return
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close(code=4003)
            return

    await websocket.send_json({"type": "output", "data": "Connected to " + instance.get('name', instance_id) + "\r\n"})

    bytes_this_second = 0
    last_rate_reset = time.time()

    async def _stdout_relay():
        """Read container stdout and relay to browser."""
        nonlocal closed, bytes_this_second, last_rate_reset
        loop = asyncio.get_event_loop()
        try:
            while not closed:
                if master_fd is not None:
                    # PTY mode: read from master fd
                    try:
                        chunk = await asyncio.wait_for(
                            loop.run_in_executor(None, lambda: os.read(master_fd, 4096)),
                            timeout=5.0,
                        )
                    except (OSError, asyncio.TimeoutError):
                        if closed:
                            break
                        continue
                elif process and process.stdout:
                    # Pipe mode fallback
                    try:
                        chunk = await asyncio.wait_for(
                            process.stdout.read(4096), timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        continue
                else:
                    break
                if not chunk:
                    break
                # Rate limiting
                now = time.time()
                if now - last_rate_reset >= 1.0:
                    bytes_this_second = 0
                    last_rate_reset = now
                bytes_this_second += len(chunk)
                if bytes_this_second > _TERMINAL_RATE_LIMIT_BYTES:
                    await asyncio.sleep(0.1)  # Throttle
                text = chunk.decode("utf-8", errors="replace")
                # Enforce scrollback limit
                if len(text) > _TERMINAL_MAX_SCROLLBACK:
                    text = text[-_TERMINAL_MAX_SCROLLBACK:]
                await websocket.send_json({"type": "output", "data": text})
        except asyncio.TimeoutError:
            pass
        except (WebSocketDisconnect, RuntimeError):
            closed = True
        except Exception as e:
            log.debug("Terminal stdout relay error: %s", e)
            closed = True

    async def _stdin_relay():
        """Read browser input and relay to container stdin."""
        nonlocal closed
        try:
            while not closed:
                # Session timeout check
                if time.time() - session_start > _TERMINAL_SESSION_TIMEOUT:
                    await websocket.send_json({"type": "error", "message": "Session timed out (30 min)"})
                    closed = True
                    break
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(), timeout=10.0
                    )
                except asyncio.TimeoutError:
                    continue
                msg = json.loads(raw)
                if msg.get("type") == "input":
                    data = msg.get("data", "")
                    if master_fd is not None:
                        os.write(master_fd, data.encode("utf-8"))
                    elif process and process.stdin:
                        process.stdin.write(data.encode("utf-8"))
                        await process.stdin.drain()
                elif msg.get("type") == "resize":
                    cols = int(msg.get("cols", 80))
                    rows = int(msg.get("rows", 24))
                    if master_fd is not None:
                        try:
                            import fcntl, struct, termios
                            winsize = struct.pack("HHHH", rows, cols, 0, 0)
                            fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
                        except (ImportError, OSError):
                            pass
                elif msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "ts": time.time()})
        except (WebSocketDisconnect, RuntimeError):
            closed = True
        except (json.JSONDecodeError, KeyError):
            pass

    try:
        done, pending = await asyncio.wait(
            [asyncio.ensure_future(_stdout_relay()), asyncio.ensure_future(_stdin_relay())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    finally:
        closed = True
        if master_fd is not None:
            try:
                os.close(master_fd)
            except OSError:
                pass
        if process:
            try:
                process.kill()
                await process.wait()
            except Exception as e:
                log.debug("Terminal process cleanup error: %s", e)
        # Send exit notification
        try:
            exit_code = process.returncode if process else -1
            await websocket.send_json({"type": "exit", "code": exit_code or 0})
            await websocket.close()
        except Exception as e:
            log.debug("Terminal exit notification error: %s", e)
