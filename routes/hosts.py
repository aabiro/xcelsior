"""Routes: hosts."""

import asyncio
import os
import re
import time
import uuid
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from routes._deps import (
    _AUTH_RATE_BUCKETS,
    _RATE_BUCKETS,
    _USE_PERSISTENT_AUTH,
    _check_auth_rate_limit,
    _get_real_client_ip,
    _otel_tracer,
    _sse_lock,
    _sse_subscribers,
    broadcast_sse,
    log,
)
from scheduler import (
    check_hosts,
    get_compute_score,
    list_hosts,
    list_jobs,
    log,
    register_host,
    remove_host,
)
from db import NotificationStore, UserStore
from verification import get_verification_engine
from security import admit_node
from reputation import VerificationType, get_reputation_engine
import threading as _threading

router = APIRouter()

# Rate limits and notification events
_AUTH_RATE_LIMIT_REQUESTS = int(os.environ.get("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "10"))
_AUTH_RATE_LIMIT_WINDOW_SEC = 300
_NOTIF_EVENT_MAP = {
    "user_registered": ("system", "New User Registered", "{email} has joined the platform."),
    "job_submitted": ("instance", "Instance Submitted", "Your instance {name} has been submitted."),
    "job_status": ("instance", "Instance {status}", "Instance {job_id} is now {status}."),
    "host_registered": ("host", "Host Registered", "A new host has been registered."),
    "host_removed": ("host", "Host Removed", "Host {host_id} has been removed."),
    "job_completed": ("instance", "Instance Completed", "Instance {job_id} completed successfully."),
    "job_failed": ("instance", "Instance Failed", "Instance {job_id} has failed."),
    "preemption_scheduled": ("instance", "Preemption Scheduled", "Instance {job_id} is being preempted."),
}


# ── Helper: otel_span ──

def otel_span(name: str, attributes: dict | None = None):
    """Create a custom OpenTelemetry span (context manager).

    Usage:
        with otel_span("job.submit", {"job.id": job_id}):
            ...
    """
    if _otel_tracer is None:
        from contextlib import nullcontext
        return nullcontext()
    span = _otel_tracer.start_as_current_span(name, attributes=attributes or {})
    return span


# ── Helper: _sse_message_text ──

def _sse_message_text(event_type: str, data: dict) -> str:
    """Generate a human-readable message for an SSE event."""
    _templates = {
        "host_update": "Host {host_id} registered with {gpu_model}",
        "host_removed": "Host {host_id} removed",
        "job_submitted": "Instance {name} submitted (ID: {job_id})",
        "job_status": "Instance {job_id} is now {status}",
        "job_cancelled": "Instance {job_id} cancelled",
        "job_log": "Log entry for instance {job_id}",
        "queue_processed": "{assigned_count} instance(s) assigned to hosts",
        "user_registered": "New user registered: {email}",
        "team_created": "Team {name} created",
        "team_member_added": "Member {email} added to team {team_id}",
        "team_deleted": "Team {team_id} deleted",
        "preemption_scheduled": "Preemption scheduled on host {host_id} for instance {job_id}",
        "spot_prices_updated": "Spot prices updated",
    }
    template = _templates.get(event_type)
    if template:
        try:
            return template.format(**data)
        except (KeyError, IndexError):
            pass
    return event_type.replace("_", " ").title()


# ── Helper: broadcast_sse ──

def broadcast_sse(event_type: str, data: dict):
    """Push an event to all connected SSE clients."""
    message = {
        "event": event_type,
        "data": data,
        "timestamp": time.time(),
        "message": _sse_message_text(event_type, data),
    }
    with _sse_lock:
        dead = []
        for q in _sse_subscribers:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            _sse_subscribers.remove(q)
    # Deliver in-app notifications for user-facing events
    _threading.Thread(target=_deliver_notifications, args=(event_type, data), daemon=True).start()


# ── Helper: _deliver_notifications ──

def _deliver_notifications(event_type: str, data: dict):
    """Create per-user in-app notifications for relevant events."""
    template = _NOTIF_EVENT_MAP.get(event_type)
    if not template:
        return
    try:
        notif_type, title_tmpl, body_tmpl = template
        title = title_tmpl.format_map(defaultdict(str, **data))
        body = body_tmpl.format_map(defaultdict(str, **data))

        # Determine which users to notify based on event type
        if _USE_PERSISTENT_AUTH:
            # For job events, notify the submitter; for host/admin events, notify admins
            if event_type in ("job_submitted", "job_status", "job_completed", "job_failed",
                              "preemption_scheduled"):
                # Find the job owner from the jobs list
                job_id = data.get("job_id", "")
                jobs = list_jobs()
                job = next((j for j in jobs if j.get("job_id") == job_id), None)
                owner_email = job.get("owner_email", job.get("user_email", "")) if job else ""
                if owner_email:
                    user = UserStore.get_user(owner_email)
                    if user and user.get("notifications_enabled", 1):
                        NotificationStore.create(owner_email, notif_type, title, body, data)
                # Also notify admins for failures
                if event_type == "job_failed":
                    for u in UserStore.list_users():
                        if u.get("role") == "admin" and u["email"] != owner_email:
                            if u.get("notifications_enabled", 1):
                                NotificationStore.create(u["email"], notif_type, title, body, data)
            else:
                # Host/system events → notify admins
                for u in UserStore.list_users():
                    if u.get("role") == "admin" and u.get("notifications_enabled", 1):
                        NotificationStore.create(u["email"], notif_type, title, body, data)
    except Exception as e:
        log.debug("Notification delivery error: %s", e)


# ── Helper: _get_real_client_ip ──

def _get_real_client_ip(request: Request) -> str:
    """Extract real client IP, respecting X-Real-IP / X-Forwarded-For from trusted proxy."""
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ── Helper: _check_auth_rate_limit ──

def _check_auth_rate_limit(request: Request) -> None:
    """Enforce stricter rate limiting on auth endpoints. Raises 429 if exceeded."""
    now = time.time()
    client_ip = _get_real_client_ip(request)
    bucket = _AUTH_RATE_BUCKETS[client_ip]
    while bucket and bucket[0] <= now - _AUTH_RATE_LIMIT_WINDOW_SEC:
        bucket.popleft()
    if len(bucket) >= _AUTH_RATE_LIMIT_REQUESTS:
        raise HTTPException(429, "Too many attempts. Please try again later.")
    bucket.append(now)


# ── Model: HostIn ──

class HostIn(BaseModel):
    host_id: str
    ip: str
    gpu_model: str
    total_vram_gb: float
    free_vram_gb: float
    cost_per_hour: float = 0.20
    country: str = "CA"  # ISO 3166-1 alpha-2
    province: str = ""  # CA province code (ON, QC, BC, etc.)
    # Optional: agent-reported versions for inline admission
    versions: dict | None = None  # {"runc": "1.2.4", "nvidia_ctk": "1.17.8", ...}
    # Canadian company fields (Report #1.B — Provider Onboarding)
    corporation_name: str = ""  # Legal corporation name
    business_number: str = ""  # CRA Business Number (BN), e.g. 123456789RC0001
    gst_hst_number: str = ""  # GST/HST registration number
    legal_name: str = ""  # Legal name of individual or company


# ── Model: JobIn ──

class JobIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    vram_needed_gb: float = Field(default=0, ge=0)
    priority: int = Field(default=0, ge=0, le=10)
    tier: str | None = None
    num_gpus: int = Field(default=1, ge=1, le=64)
    host_id: str | None = None  # Direct host assignment (marketplace)
    gpu_model: str | None = None  # Hint for VRAM lookup
    nfs_server: str | None = None
    nfs_path: str | None = None
    nfs_mount_point: str | None = None
    image: str | None = None
    interactive: bool = True
    command: str | None = None
    ssh_port: int = Field(default=22, ge=1, le=65535)


# ── Model: StatusUpdate ──

class StatusUpdate(BaseModel):
    status: str
    host_id: str | None = None
    container_id: str | None = None
    container_name: str | None = None

@router.put("/host", tags=["Hosts"])
def api_register_host(h: HostIn):
    """Register or update a host with strict admission gating.

    Per REPORT_FEATURE_FINAL.md §62 and REPORT_FEATURE_2.md §37:
    - Hosts must pass node admission (version gating) before accepting work
    - Country/province recorded for jurisdiction-aware scheduling
    - Hosts register as 'pending' until agent completes benchmark + admission
    - If versions are provided inline, admission is checked immediately
    """
    from security import admit_node

    # Register the host with country/province metadata
    entry = register_host(
        h.host_id, h.ip, h.gpu_model, h.total_vram_gb, h.free_vram_gb, h.cost_per_hour
    )
    entry["country"] = h.country.upper()
    entry["province"] = h.province.upper() if h.province else ""
    # Persist Canadian company info if provided
    if h.corporation_name:
        entry["corporation_name"] = h.corporation_name
    if h.business_number:
        entry["business_number"] = h.business_number
    if h.gst_hst_number:
        entry["gst_hst_number"] = h.gst_hst_number
    if h.legal_name:
        entry["legal_name"] = h.legal_name

    # Inline admission check if versions provided
    if h.versions:
        admitted, details = admit_node(h.host_id, h.versions, h.gpu_model)
        entry["admitted"] = admitted
        entry["admission_details"] = details
        entry["recommended_runtime"] = details.get("recommended_runtime", "runc")
        if not admitted:
            # Host is registered but marked as not-admitted — won't receive work
            entry["status"] = "pending"
            log.warning(
                "HOST %s registered but NOT ADMITTED: %s",
                h.host_id,
                details.get("rejection_reasons", []),
            )
    else:
        # No versions provided — only set pending for NEW hosts.
        # Existing hosts preserve their admission status from /agent/versions.
        if not entry.get("admitted"):
            entry.setdefault("admitted", False)
            entry["status"] = "pending"

    # Persist the updated entry (country, province, admitted status)
    from scheduler import _atomic_mutation, _upsert_host_row, _migrate_hosts_if_needed

    with _atomic_mutation() as conn:
        _migrate_hosts_if_needed(conn)
        _upsert_host_row(conn, entry)

    # Auto-compute score and auto-list on marketplace
    from scheduler import estimate_compute_score, register_compute_score, list_rig

    score = estimate_compute_score(h.gpu_model)
    register_compute_score(h.host_id, h.gpu_model, score)
    entry["compute_score"] = score

    list_rig(
        h.host_id,
        h.gpu_model,
        h.total_vram_gb,
        h.cost_per_hour,
        description=f"{h.gpu_model} ({h.total_vram_gb}GB) in {h.country.upper()}",
        owner=h.host_id,
    )

    # ── Auto-create verification + reputation records ────────────────────
    # Ensures the host appears on the trust page immediately (as "unverified")
    # and gets a baseline reputation score.  Hardware verification scoring
    # happens later when the agent sends a full benchmark report.
    try:
        ve = get_verification_engine()
        if not ve.store.get_verification(h.host_id):
            from verification import HostVerification, HostVerificationState

            ve.store.save_verification(
                HostVerification(
                    verification_id=str(uuid.uuid4())[:12],
                    host_id=h.host_id,
                    state=HostVerificationState.UNVERIFIED,
                )
            )
            log.info("VERIFY RECORD created for new host %s", h.host_id)
        # Bootstrap reputation — email verification is implicit for registered users
        re = get_reputation_engine()
        re.add_verification(h.host_id, VerificationType.EMAIL)
    except Exception as e:
        log.exception("Non-fatal: could not bootstrap verification/reputation for %s", h.host_id)

    broadcast_sse(
        "host_update",
        {
            "host_id": h.host_id,
            "gpu_model": h.gpu_model,
            "admitted": entry.get("admitted", False),
            "country": entry.get("country", ""),
        },
    )
    return {"ok": True, "host": entry}

@router.get("/host/{host_id}", tags=["Hosts"])
def api_get_host(host_id: str):
    """Get a single host by ID."""
    hosts = list_hosts(active_only=False)
    host = next((h for h in hosts if h["host_id"] == host_id), None)
    if not host:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    return {"ok": True, "host": host}

@router.get("/hosts", tags=["Hosts"])
def api_list_hosts(active_only: bool = True):
    """List all hosts."""
    return {"hosts": list_hosts(active_only=active_only)}

@router.delete("/host/{host_id}", tags=["Hosts"])
def api_remove_host(host_id: str):
    """Remove a host."""
    hosts = list_hosts(active_only=False)
    if not any(h["host_id"] == host_id for h in hosts):
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    remove_host(host_id)
    broadcast_sse("host_removed", {"host_id": host_id})
    return {"ok": True, "removed": host_id}

@router.post("/hosts/check", tags=["Hosts"])
def api_check_hosts():
    """Ping all hosts and update status."""
    results = check_hosts()
    return {"results": results}

@router.get("/compute-score/{host_id}", tags=["Hosts"])
def api_get_compute_score(host_id: str):
    """Get the compute score (XCU) for a host."""
    score = get_compute_score(host_id)
    if score is None:
        raise HTTPException(status_code=404, detail=f"No compute score for host {host_id}")
    return {"ok": True, "host_id": host_id, "score": score}

@router.get("/compute-scores", tags=["Hosts"])
def api_list_compute_scores():
    """List compute scores for all hosts."""
    hosts = list_hosts(active_only=False)
    scores = {}
    for h in hosts:
        score = get_compute_score(h["host_id"])
        if score is not None:
            scores[h["host_id"]] = {
                "score": score,
                "gpu_model": h.get("gpu_model", "unknown"),
            }
    return {"ok": True, "scores": scores}

