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
    _check_ws_connect_rate_limit,
    _consume_ws_ticket,
    _issue_ws_ticket,
    _require_auth,
    _require_scope,
    _sessions,
    _sse_lock,
    _sse_subscribers,
    _user_lock,
    _validate_ws_origin,
    _validate_ws_auth,  # re-exported for backward compat with tests
    broadcast_sse,
    log,
    otel_span,
)
from scheduler import (
    API_TOKEN,
    failover_and_reassign,
    get_job,
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
    max_bid: float | None = Field(default=None, gt=0)
    volume_ids: list[str] | None = Field(default=None, max_length=16)

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


def _wallet_preflight(customer_id: str):
    """Shared wallet check — raises 402 if wallet is suspended or empty."""
    from billing import get_billing_engine
    be = get_billing_engine()
    wallet = be.get_wallet(customer_id)
    if wallet.get("status") == "suspended":
        raise HTTPException(402, detail="Wallet suspended — please add funds to resume service")
    if wallet["balance_cad"] <= 0 and wallet.get("grace_until", 0) < time.time():
        raise HTTPException(402, detail="Insufficient wallet balance — please deposit credits")


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
    _require_scope(user, "instances:write")

    vram_needed = 0.0
    target_host_id = j.host_id
    target_host = None
    if target_host_id:
        all_hosts = list_hosts(active_only=False)
        hmap_all = {h["host_id"]: h for h in all_hosts}
        target_host = hmap_all.get(target_host_id)
        if not target_host:
            raise HTTPException(status_code=404, detail=f"Host {target_host_id} not found")
        if target_host.get("status") == "draining":
            raise HTTPException(
                status_code=409,
                detail=f"Host {target_host_id} is draining for maintenance and not accepting new instances",
            )
        if target_host.get("status") != "active":
            raise HTTPException(
                status_code=409,
                detail=f"Host {target_host_id} is {target_host.get('status', 'unavailable')} and not accepting new instances",
            )

    # Respect the requested VRAM for scheduler matching and host selection.
    # A zero value remains valid and means "no minimum VRAM preference".
    vram_needed = max(float(j.vram_needed_gb or 0.0), 0.0)

    # ── Marketplace flow requires a Docker image ──────────────────────
    if target_host_id and not j.image:
        raise HTTPException(
            status_code=422,
            detail="Docker image is required for marketplace launches — select a template or enter a custom image",
        )

    with otel_span("job.submit", {"job.name": j.name, "job.tier": j.tier or "", "job.num_gpus": j.num_gpus}):
        customer_id = user.get("customer_id", user.get("user_id", ""))
        _wallet_preflight(customer_id)

        # ── Validate volume_ids ownership and status ─────────────
        validated_volume_ids = None
        if j.volume_ids:
            from volumes import get_volume_engine
            ve = get_volume_engine()
            validated_volume_ids = []
            seen_vids: set[str] = set()
            for vid in j.volume_ids:
                if vid in seen_vids:
                    continue  # deduplicate
                seen_vids.add(vid)
                try:
                    vol = ve.get_volume(vid)
                except Exception:
                    raise HTTPException(status_code=404, detail=f"Volume {vid} not found")
                if not vol:
                    raise HTTPException(status_code=404, detail=f"Volume {vid} not found")
                if vol.get("owner_id") != customer_id:
                    raise HTTPException(status_code=404, detail=f"Volume {vid} not found")
                if vol.get("status") not in ("available", "attached"):
                    raise HTTPException(
                        status_code=409,
                        detail=f"Volume {vid} is {vol.get('status', 'unknown')} and cannot be attached",
                    )
                validated_volume_ids.append(vid)

        # Spot path: max_bid present → delegate to spot submission
        if j.max_bid is not None:
            from scheduler import submit_spot_job
            job = submit_spot_job(
                j.name, vram_needed, j.max_bid, j.priority,
                tier=j.tier, owner=customer_id, image=j.image,
            )
            event_name = "spot_job_submitted"
        else:
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
                volume_ids=validated_volume_ids,
            )
            event_name = "job_submitted"

        # Track job ownership (response-only, already persisted via owner param)
        job["submitted_by"] = user.get("email", "")
        job["customer_id"] = customer_id
        broadcast_sse(event_name, {"job_id": job["job_id"], "name": job["name"]})

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

def _enrich_instance(j: dict, host_map: dict[str, dict]) -> dict:
    """Shared enrichment for every instance response — list and detail.

    Adds docker_image mapping, host GPU info, elapsed/duration, connection
    details, and cost computation.  Single source of truth — no drift.
    """
    # Map internal 'image' → frontend 'docker_image'
    if j.get("image") is not None and not j.get("docker_image"):
        j["docker_image"] = j["image"]

    # Host GPU info
    hid = j.get("host_id")
    host = host_map.get(hid) if hid else None
    if host:
        j.setdefault("gpu_type", host.get("gpu_model", ""))
        j.setdefault("host_gpu", host.get("gpu_model", ""))

    # Elapsed / duration
    started = float(j.get("started_at") or 0)
    completed = float(j.get("completed_at") or 0)
    if started > 0:
        if completed > started:
            j.setdefault("duration_sec", round(completed - started, 2))
            j.setdefault("elapsed_sec", j["duration_sec"])
        elif j.get("status") == "running":
            j.setdefault("elapsed_sec", round(time.time() - started, 2))

    # Connection details (host IP, VRAM) — available once job has run
    if host and j.get("status") in ("running", "starting", "completed", "failed"):
        j.setdefault("host_ip", host.get("ip", ""))
        j.setdefault("host_gpu", host.get("gpu_model", ""))
        j.setdefault("host_vram_gb", host.get("total_vram_gb", 0))
        j.setdefault("gpu_type", host.get("gpu_model", ""))

    # Cost
    elapsed = j.get("elapsed_sec") or j.get("duration_sec") or 0
    if elapsed > 0 and j.get("cost_cad") is None:
        rate = 0.20  # default CAD/hr
        if host and host.get("cost_per_hour"):
            rate = float(host["cost_per_hour"])
        j["cost_cad"] = round((elapsed / 3600) * rate, 4)

    # Attached volumes
    try:
        from volumes import get_volume_engine
        ve = get_volume_engine()
        vols = ve.get_instance_volumes(j.get("job_id", ""))
        j["attached_volumes"] = [
            {
                "volume_id": v["volume_id"],
                "name": v.get("name", ""),
                "size_gb": v.get("size_gb", 0),
                "mount_path": v.get("mount_path", "/workspace"),
                "mode": v.get("mode", "rw"),
                "storage_type": v.get("storage_type", "nfs"),
                "encrypted": v.get("encrypted", False),
            }
            for v in vols
        ]
    except Exception:
        j.setdefault("attached_volumes", [])

    # Storage cost — sum billing_cycles for attached volumes
    volume_ids = j.get("volume_ids", [])
    if volume_ids:
        try:
            from db import _get_pg_pool
            from psycopg.rows import dict_row
            pool = _get_pg_pool()
            with pool.connection() as conn:
                conn.row_factory = dict_row
                row = conn.execute(
                    """SELECT COALESCE(SUM(amount_cad), 0) AS total
                       FROM billing_cycles
                       WHERE job_id = ANY(%s) AND gpu_model = 'storage' AND tier = 'volume'""",
                    (volume_ids,),
                ).fetchone()
            j["storage_cost_cad"] = round(float(row["total"]), 4) if row else 0
        except Exception:
            j.setdefault("storage_cost_cad", 0)
    else:
        j.setdefault("storage_cost_cad", 0)

    return j


@router.get("/instances", tags=["Instances"])
def api_list_instances(status: str | None = None):
    """List jobs. Optional filter by status."""
    jobs = list_jobs(status=status)
    hosts = list_hosts()
    host_map = {h["host_id"]: h for h in hosts}
    for j in jobs:
        _enrich_instance(j, host_map)
    return {"instances": jobs}

@router.get("/instance/{job_id}", tags=["Instances"])
def api_get_instance(job_id: str):
    """Get a specific instance by ID, enriched with connection info."""
    j = get_job(job_id)
    if not j:
        raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")
    hosts = list_hosts()
    host_map = {h["host_id"]: h for h in hosts}
    _enrich_instance(j, host_map)
    return {"instance": j}

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
    user = _require_auth(request)
    _require_scope(user, "instances:write")
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

    # Detach any volumes still attached to this instance
    try:
        from volumes import get_volume_engine
        get_volume_engine().detach_all_for_instance(job_id)
    except Exception as e:
        log.warning("Volume detach on cancel failed for %s: %s", job_id, e)

    return {"ok": True, "job_id": job_id, "status": "cancelled"}

@router.post("/instance/{job_id}/requeue", tags=["Instances"])
def api_requeue_instance(job_id: str, request: Request):
    """Manually requeue a failed or stuck job."""
    user = _require_auth(request)
    _require_scope(user, "instances:write")
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
    _require_scope(user, "instances:write")
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
    _require_scope(user, "instances:write")
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


# ── Stop / Start / Restart / Terminate ──────────────────────────────

@router.post("/instances/{job_id}/stop", tags=["Instances"])
def api_stop_instance(job_id: str, request: Request):
    """Gracefully stop a running instance.

    Sends SIGTERM to the container (docker stop -t 10). The container is
    preserved — data, volumes and configuration are intact. Storage billing
    continues at the per-GB rate. The instance can be started again at any time.
    """
    user = _require_auth(request)
    _require_scope(user, "instances:write")
    customer_id = user.get("customer_id", user.get("user_id", ""))
    role = user.get("role", "")

    from scheduler import get_job
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")

    job_owner = job.get("owner", "")
    if role != "admin" and job_owner != customer_id:
        raise HTTPException(status_code=403, detail="Not authorized to stop this instance")

    if job.get("status") != "running":
        raise HTTPException(status_code=400, detail=f"Instance is '{job.get('status')}', must be running to stop")

    from billing import get_billing_engine
    be = get_billing_engine()
    result = be.stop_instance(job_id, reason="user_stopped")
    if not result.get("stopped"):
        raise HTTPException(status_code=500, detail=result.get("reason", "stop failed"))

    broadcast_sse("instance_stopped", {"job_id": job_id})
    return {"ok": True, "instance": result}


@router.post("/instances/{job_id}/start", tags=["Instances"])
def api_start_instance(job_id: str, request: Request):
    """Start a stopped instance.

    Restores the exited container via docker start. Container data is
    preserved. Requires a positive wallet balance. Compute billing resumes
    immediately.
    """
    user = _require_auth(request)
    _require_scope(user, "instances:write")
    customer_id = user.get("customer_id", user.get("user_id", ""))
    role = user.get("role", "")

    from scheduler import get_job
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")

    job_owner = job.get("owner", "")
    if role != "admin" and job_owner != customer_id:
        raise HTTPException(status_code=403, detail="Not authorized to start this instance")

    allowed_statuses = {"stopped", "user_paused", "paused_low_balance"}
    if job.get("status") not in allowed_statuses:
        raise HTTPException(status_code=400, detail=f"Instance is '{job.get('status')}', must be stopped to start")

    from billing import get_billing_engine
    be = get_billing_engine()
    wallet = be.get_wallet(job_owner or customer_id)
    if wallet.get("status") == "suspended":
        raise HTTPException(status_code=402, detail="Wallet suspended — please add funds")
    if wallet["balance_cad"] <= 0:
        raise HTTPException(status_code=402, detail="Insufficient wallet balance to start instance")

    result = be.start_instance(job_id)
    if not result.get("started"):
        detail = result.get("reason", "start failed")
        code = 402 if detail == "insufficient_balance" else 500
        raise HTTPException(status_code=code, detail=detail)

    broadcast_sse("instance_started", {"job_id": job_id})
    return {"ok": True, "instance": result}


@router.post("/instances/{job_id}/restart", tags=["Instances"])
def api_restart_instance(job_id: str, request: Request):
    """Restart an instance. Container data is fully preserved.

    Works from both running and stopped states:
    - Running → graceful stop → start (docker stop + docker start)
    - Stopped → start (docker start only)

    Billing is continuous — no gap. Requires a positive wallet balance.
    """
    user = _require_auth(request)
    _require_scope(user, "instances:write")
    customer_id = user.get("customer_id", user.get("user_id", ""))
    role = user.get("role", "")

    from scheduler import get_job
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")

    job_owner = job.get("owner", "")
    if role != "admin" and job_owner != customer_id:
        raise HTTPException(status_code=403, detail="Not authorized to restart this instance")

    if job.get("status") not in ("running", "stopped"):
        raise HTTPException(status_code=400, detail=f"Instance is '{job.get('status')}', must be running or stopped to restart")

    from billing import get_billing_engine
    be = get_billing_engine()
    wallet = be.get_wallet(job_owner or customer_id)
    if wallet.get("status") == "suspended":
        raise HTTPException(status_code=402, detail="Wallet suspended — please add funds")
    if wallet["balance_cad"] <= 0:
        raise HTTPException(status_code=402, detail="Insufficient wallet balance to restart instance")

    result = be.restart_instance(job_id)
    if not result.get("restarted"):
        detail = result.get("reason", "restart failed")
        code = 402 if detail == "insufficient_balance" else 500
        raise HTTPException(status_code=code, detail=detail)

    broadcast_sse("instance_restarted", {"job_id": job_id})
    return {"ok": True, "instance": result}


@router.post("/instances/{job_id}/terminate", tags=["Instances"])
def api_terminate_instance(job_id: str, request: Request):
    """Permanently terminate an instance. This action is irreversible.

    The container and its anonymous volumes are hard-killed and removed.
    Named/NFS volumes are preserved. The instance cannot be restarted
    after termination.
    """
    user = _require_auth(request)
    _require_scope(user, "instances:write")
    customer_id = user.get("customer_id", user.get("user_id", ""))
    role = user.get("role", "")

    from scheduler import get_job
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")

    job_owner = job.get("owner", "")
    if role != "admin" and job_owner != customer_id:
        raise HTTPException(status_code=403, detail="Not authorized to terminate this instance")

    terminal_statuses = {"terminated", "completed", "failed", "preempted", "cancelled"}
    if job.get("status") in terminal_statuses:
        raise HTTPException(status_code=400, detail=f"Instance already {job.get('status')}")

    from billing import get_billing_engine
    be = get_billing_engine()
    result = be.terminate_instance(job_id)
    if not result.get("terminated"):
        raise HTTPException(status_code=400, detail=result.get("reason", "terminate failed"))

    broadcast_sse("instance_terminated", {"job_id": job_id})
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
        from psycopg.rows import dict_row
        pool = _get_pg_pool()
        with pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    "SELECT ts AS timestamp, level, line, line AS message FROM job_logs "
                    "WHERE job_id = %s ORDER BY ts DESC LIMIT %s",
                    (job_id, limit),
                ).fetchall()
        if not rows:
            return []
        return list(reversed(rows))  # chronological order, already dicts
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
    user = _require_auth(request)
    _require_scope(user, "instances:read")
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
    _require_scope(user, "instances:read")
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


@router.post("/api/instances/{job_id}/stream-ticket")
def api_instance_stream_ticket(job_id: str, request: Request) -> dict:
    """Issue a short-lived one-time WebSocket ticket for instance streaming."""
    user = _require_auth(request)
    _check_job_access(user, job_id)
    ticket = _issue_ws_ticket(
        user,
        request=request,
        purpose="instance_stream",
        target=job_id,
    )
    return {
        "ok": True,
        "ticket": ticket["ticket"],
        "expires_in": int(max(0, ticket["expires_at"] - time.time())),
    }



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
    if not _validate_ws_origin(
        websocket,
        require_for_cookie_auth=True,
        allow_query_token=False,
    ):
        await websocket.close(code=1008, reason="Invalid origin")
        return

    if not _check_ws_connect_rate_limit(websocket, bucket="instances"):
        await websocket.close(code=4429, reason="Connection rate limit exceeded")
        return

    ticket = websocket.query_params.get("ticket", "").strip()
    if ticket:
        user = _consume_ws_ticket(
            ticket,
            websocket,
            purpose="instance_stream",
            target=job_id,
        )
    else:
        user = _validate_ws_auth(websocket, allow_query_token=False)
    if not user:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()

    # Verify job exists and send initial snapshot
    jobs = list_jobs()
    hosts = list_hosts()
    host_map = {h["host_id"]: h for h in hosts}
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

    _enrich_instance(instance, host_map)
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
            # On status change, send full enriched instance snapshot
            if event_type == "job_status":
                fresh = next((j for j in list_jobs() if j["job_id"] == job_id), None)
                if fresh:
                    _enrich_instance(fresh, {h["host_id"]: h for h in list_hosts()})
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
                        _enrich_instance(fresh, {h["host_id"]: h for h in list_hosts()})
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
