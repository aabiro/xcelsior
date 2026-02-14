# Xcelsior API v2.0.0
# FastAPI. Every endpoint. Dashboard. Marketplace. Autoscale. SSE. Spot pricing. No fluff.

import asyncio
import hmac
import json
import os
import secrets
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

TEMPLATES_DIR = Path(os.path.dirname(__file__)) / "templates"

from db import start_pg_listen, UserStore

from scheduler import (
    register_host,
    remove_host,
    list_hosts,
    check_hosts,
    submit_job,
    list_jobs,
    update_job_status,
    process_queue,
    bill_job,
    bill_all_completed,
    get_total_revenue,
    load_billing,
    configure_alerts,
    ALERT_CONFIG,
    generate_ssh_keypair,
    get_public_key,
    API_TOKEN,
    failover_and_reassign,
    requeue_job,
    list_tiers,
    PRIORITY_TIERS,
    build_and_push,
    list_builds,
    generate_dockerfile,
    list_rig,
    unlist_rig,
    get_marketplace,
    marketplace_bill,
    marketplace_stats,
    register_host_ca,
    list_hosts_filtered,
    process_queue_filtered,
    set_canada_only,
    add_to_pool,
    remove_from_pool,
    load_autoscale_pool,
    autoscale_cycle,
    autoscale_up,
    autoscale_down,
    get_metrics_snapshot,
    storage_healthcheck,
    log,
    # v2.0.0 additions
    get_current_spot_prices,
    update_spot_prices,
    submit_spot_job,
    preemption_cycle,
    estimate_compute_score,
    register_compute_score,
    get_compute_score,
    allocate_compute_aware,
    # v2.1 additions
    allocate_jurisdiction_aware,
    process_queue_sovereign,
)

from security import admit_node, check_node_versions

# v2.1 module imports
from events import get_event_store, get_state_machine, JobState, EventType, Event
from verification import get_verification_engine
from jurisdiction import (
    TrustTier,
    JurisdictionConstraint,
    generate_residency_trace,
    compute_fund_eligible_amount,
    PROVINCE_COMPLIANCE,
    TRUST_TIER_REQUIREMENTS,
)
from billing import get_billing_engine, get_tax_rate_for_province, PROVINCE_TAX_RATES
from reputation import (
    get_reputation_engine,
    ReputationTier,
    VerificationType,
    estimate_job_cost,
    GPU_REFERENCE_PRICING_CAD,
)
from artifacts import get_artifact_manager
from privacy import (
    get_lifecycle_manager,
    PrivacyConfig,
    RETENTION_POLICIES,
    redact_job_record,
    requires_quebec_pia,
    DataCategory,
)
from sla import get_sla_engine, SLATier, SLA_TARGETS
from stripe_connect import get_stripe_manager

# ── OpenAPI Tag Definitions ───────────────────────────────────────────
# Per REPORT_FEATURE_1.md (Report #1.B): Interactive Documentation
# Groups all 70+ endpoints into logical sections for Swagger UI / Fern / Mintlify

OPENAPI_TAGS = [
    {"name": "Hosts", "description": "GPU host registration, admission gating, and management."},
    {"name": "Jobs", "description": "Job submission, scheduling, and lifecycle management."},
    {"name": "Billing", "description": "Wallet management, invoicing, CAF exports, refunds. Credit-first CAD billing."},
    {"name": "Marketplace", "description": "Rig listings, browsing, and marketplace billing."},
    {"name": "Spot Pricing", "description": "Dynamic spot pricing, interruptible jobs, preemption cycles."},
    {"name": "Reputation", "description": "Trust scoring, verification tiers (Bronze→Platinum), leaderboards."},
    {"name": "Verification", "description": "Automated hardware attestation: GPU identity, CUDA, thermals, network."},
    {"name": "SLA", "description": "Service Level Agreement enforcement, uptime tracking, credit calculation."},
    {"name": "Providers", "description": "Stripe Connect onboarding, Canadian company registration, payouts."},
    {"name": "Artifacts", "description": "Presigned upload/download URLs for model weights, checkpoints, outputs."},
    {"name": "Jurisdiction", "description": "Canada-first scheduling, province filtering, data residency traces."},
    {"name": "Compliance", "description": "Province compliance matrix, tax rates, Quebec PIA checks."},
    {"name": "Privacy", "description": "PIPEDA consent management, retention policies, privacy officer config."},
    {"name": "Transparency", "description": "Legal request handling, CLOUD Act canary, transparency reports."},
    {"name": "Telemetry", "description": "Real-time GPU metrics: utilization, temperature, memory, power."},
    {"name": "Agent", "description": "Worker agent endpoints: work assignment, leases, benchmarks, mining alerts."},
    {"name": "Autoscale", "description": "Auto-scaling pool management and provisioning cycles."},
    {"name": "Events", "description": "Event sourcing, state machine transitions, audit trail."},
    {"name": "Infrastructure", "description": "Health checks, readiness probes, metrics, SSE streaming, dashboard."},
]

app = FastAPI(
    title="Xcelsior",
    version="2.2.0",
    description=(
        "Distributed GPU orchestration platform for AI/ML workloads. "
        "Canadian-first data sovereignty (PIPEDA/Law 25), automated trust scoring, "
        "Stripe Connect marketplace billing, and real-time GPU telemetry.\n\n"
        "**Authentication**: Bearer token via `Authorization: Bearer <token>` header.\n\n"
        "**Regions**: Canada-first with province-level filtering (ON, QC, BC, AB, etc.).\n\n"
        "**SDK**: Generate idiomatic Python/TypeScript SDKs with [Fern](https://buildwithfern.com).\n\n"
        "**LLM Integration**: See `/llms.txt` for AI agent–optimized documentation."
    ),
    contact={"name": "Xcelsior", "url": "https://xcelsior.ca", "email": "admin@xcelsior.ca"},
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    openapi_tags=OPENAPI_TAGS,
    servers=[
        {"url": "https://xcelsior.ca", "description": "Production"},
        {"url": "http://localhost:8000", "description": "Development"},
    ],
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── SSE Infrastructure ───────────────────────────────────────────────
# In-memory event bus for server-sent events.
# Listeners register an asyncio.Queue per connection.

import threading as _threading

_sse_subscribers: list[asyncio.Queue] = []
_sse_lock = _threading.Lock()

# Track pending work and preemption commands for agents
_agent_work: dict[str, list[dict]] = defaultdict(list)  # host_id -> [job, ...]
_agent_preempt: dict[str, list[str]] = defaultdict(list)  # host_id -> [job_id, ...]
_agent_lock = _threading.Lock()


def broadcast_sse(event_type: str, data: dict):
    """Push an event to all connected SSE clients."""
    message = {"event": event_type, "data": data, "timestamp": time.time()}
    with _sse_lock:
        dead = []
        for q in _sse_subscribers:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            _sse_subscribers.remove(q)


# ── Bridge PgEventBus LISTEN/NOTIFY → SSE ────────────────────────────
# Per REPORT_XCELSIOR_TECHNICAL_FINAL.md Step 9:
# "start with Postgres LISTEN/NOTIFY for notifications"
# This bridges scheduler events (db.emit_event) into SSE delivery.

start_pg_listen(broadcast_sse)

XCELSIOR_ENV = os.environ.get("XCELSIOR_ENV", "dev").lower()
AUTH_REQUIRED = XCELSIOR_ENV not in {"dev", "development", "test"}
RATE_LIMIT_REQUESTS = int(os.environ.get("XCELSIOR_RATE_LIMIT_REQUESTS", "120"))
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("XCELSIOR_RATE_LIMIT_WINDOW_SEC", "60"))
_RATE_BUCKETS = defaultdict(deque)


# ── Phase 13: API Token Auth ─────────────────────────────────────────

# Public routes — no token required
PUBLIC_PATHS = {
    "/", "/docs", "/redoc", "/openapi.json", "/llms.txt", "/dashboard",
    "/healthz", "/readyz", "/metrics", "/api/stream",
    "/api/transparency/report",
}


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """
    Bearer token auth. If XCELSIOR_API_TOKEN is set, every request
    (except public routes) must include it. No token set = open access.
    """

    async def dispatch(self, request: Request, call_next):
        if not AUTH_REQUIRED:
            return await call_next(request)

        api_token = os.environ.get("XCELSIOR_API_TOKEN", API_TOKEN)
        if not api_token:
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": {
                        "code": "auth_config_error",
                        "message": "XCELSIOR_API_TOKEN must be set in non-dev environments",
                    },
                },
            )

        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
        else:
            token = request.query_params.get("token", "")

        if not token or not hmac.compare_digest(token, api_token):
            return JSONResponse(
                status_code=401,
                content={"ok": False, "error": {"code": "unauthorized", "message": "Unauthorized"}},
            )

        return await call_next(request)


app.add_middleware(TokenAuthMiddleware)


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Emit structured access logs for observability."""

    async def dispatch(self, request: Request, call_next):
        started = time.time()
        response = await call_next(request)
        duration_ms = round((time.time() - started) * 1000, 2)
        entry = {
            "event": "api_request",
            "path": request.url.path,
            "method": request.method,
            "status": response.status_code,
            "duration_ms": duration_ms,
            "client_ip": request.client.host if request.client else "unknown",
        }
        log.info(json.dumps(entry, sort_keys=True))
        return response


app.add_middleware(RequestLogMiddleware)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory IP rate limiting for API safety."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        now = time.time()
        client_ip = request.client.host if request.client else "unknown"
        bucket = _RATE_BUCKETS[client_ip]
        while bucket and bucket[0] <= now - RATE_LIMIT_WINDOW_SEC:
            bucket.popleft()

        if len(bucket) >= RATE_LIMIT_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={
                    "ok": False,
                    "error": {"code": "rate_limited", "message": "Too many requests"},
                },
            )

        bucket.append(now)
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "error": {"code": "http_error", "message": str(exc.detail)}},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "ok": False,
            "error": {
                "code": "validation_error",
                "message": "Request validation failed",
                "details": exc.errors(),
            },
        },
    )


# ── Request models ────────────────────────────────────────────────────


class HostIn(BaseModel):
    host_id: str
    ip: str
    gpu_model: str
    total_vram_gb: float
    free_vram_gb: float
    cost_per_hour: float = 0.20
    country: str = "CA"           # ISO 3166-1 alpha-2
    province: str = ""            # CA province code (ON, QC, BC, etc.)
    # Optional: agent-reported versions for inline admission
    versions: dict | None = None  # {"runc": "1.2.4", "nvidia_ctk": "1.17.8", ...}
    # Canadian company fields (Report #1.B — Provider Onboarding)
    corporation_name: str = ""    # Legal corporation name
    business_number: str = ""     # CRA Business Number (BN), e.g. 123456789RC0001
    gst_hst_number: str = ""      # GST/HST registration number
    legal_name: str = ""          # Legal name of individual or company


class JobIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    vram_needed_gb: float = Field(gt=0)
    priority: int = Field(default=0, ge=0, le=10)
    tier: str | None = None
    num_gpus: int = Field(default=1, ge=1, le=64)
    nfs_server: str | None = None
    nfs_path: str | None = None
    nfs_mount_point: str | None = None
    image: str | None = None


class StatusUpdate(BaseModel):
    status: str
    host_id: str | None = None


# ── Host endpoints ────────────────────────────────────────────────────


@app.put("/host", tags=["Hosts"])
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
            log.warning("HOST %s registered but NOT ADMITTED: %s",
                        h.host_id, details.get("rejection_reasons", []))
    else:
        # No versions provided — host starts as pending until agent reports
        entry["admitted"] = False
        entry["status"] = "pending"

    # Persist the updated entry (country, province, admitted status)
    from scheduler import _atomic_mutation, _upsert_host_row, _migrate_hosts_if_needed
    with _atomic_mutation() as conn:
        _migrate_hosts_if_needed(conn)
        _upsert_host_row(conn, entry)

    broadcast_sse("host_update", {
        "host_id": h.host_id, "gpu_model": h.gpu_model,
        "admitted": entry.get("admitted", False),
        "country": entry.get("country", ""),
    })
    return {"ok": True, "host": entry}


@app.get("/hosts", tags=["Hosts"])
def api_list_hosts(active_only: bool = True):
    """List all hosts."""
    return {"hosts": list_hosts(active_only=active_only)}


@app.delete("/host/{host_id}", tags=["Hosts"])
def api_remove_host(host_id: str):
    """Remove a host."""
    hosts = list_hosts(active_only=False)
    if not any(h["host_id"] == host_id for h in hosts):
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    remove_host(host_id)
    broadcast_sse("host_removed", {"host_id": host_id})
    return {"ok": True, "removed": host_id}


@app.post("/hosts/check", tags=["Hosts"])
def api_check_hosts():
    """Ping all hosts and update status."""
    results = check_hosts()
    return {"results": results}


# ── Job endpoints ─────────────────────────────────────────────────────


@app.post("/job", tags=["Jobs"])
def api_submit_job(j: JobIn):
    """Submit a job to the queue. Tier overrides priority.

    Multi-GPU: Set num_gpus > 1 for multi-GPU jobs.
    NFS: Optionally specify nfs_server + nfs_path for shared storage.
    """
    job = submit_job(j.name, j.vram_needed_gb, j.priority, tier=j.tier,
                     num_gpus=j.num_gpus, nfs_server=j.nfs_server,
                     nfs_path=j.nfs_path, nfs_mount_point=j.nfs_mount_point,
                     image=j.image)
    broadcast_sse("job_submitted", {"job_id": job["job_id"], "name": job["name"]})
    return {"ok": True, "job": job}


@app.get("/jobs", tags=["Jobs"])
def api_list_jobs(status: str | None = None):
    """List jobs. Optional filter by status."""
    return {"jobs": list_jobs(status=status)}


@app.get("/job/{job_id}", tags=["Jobs"])
def api_get_job(job_id: str):
    """Get a specific job by ID."""
    jobs = list_jobs()
    for j in jobs:
        if j["job_id"] == job_id:
            return {"job": j}
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@app.patch("/job/{job_id}", tags=["Jobs"])
def api_update_job(job_id: str, update: StatusUpdate):
    """Update a job's status."""
    try:
        update_job_status(job_id, update.status, host_id=update.host_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    broadcast_sse("job_status", {"job_id": job_id, "status": update.status})
    return {"ok": True, "job_id": job_id, "status": update.status}


@app.post("/queue/process", tags=["Jobs"])
def api_process_queue():
    """Process the job queue — assign jobs to hosts."""
    assigned = process_queue()
    result = [
        {"job": j["name"], "job_id": j["job_id"], "host": h["host_id"]} for j, h in assigned
    ]
    if result:
        broadcast_sse("queue_processed", {"assigned_count": len(result)})
    return {"assigned": result}


# ── Phase 14: Failover endpoints ──────────────────────────────────────


@app.post("/failover", tags=["Jobs"])
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


@app.post("/job/{job_id}/requeue", tags=["Jobs"])
def api_requeue_job(job_id: str):
    """Manually requeue a failed or stuck job."""
    result = requeue_job(job_id)
    if not result:
        raise HTTPException(
            status_code=400,
            detail=f"Could not requeue job {job_id} (max retries exceeded or not found)",
        )
    return {"ok": True, "job": result}


# ── Per-Job SSE Log Streaming ─────────────────────────────────────────
# Per Report #1.B: "SSE Log Streaming Design" — the /jobs/{job_id}/logs/stream
# endpoint uses an async generator that filters the broadcast SSE
# stream for events matching a specific job_id — tail-style, EventSource-compatible.

_job_log_buffers: dict[str, list[dict]] = defaultdict(list)  # job_id -> [log_entry, ...]
_JOB_LOG_MAX = 500  # max buffered lines per job


def push_job_log(job_id: str, line: str, level: str = "info"):
    """Push a log line into the per-job log buffer (called from scheduler/worker)."""
    entry = {"timestamp": time.time(), "line": line, "level": level}
    buf = _job_log_buffers[job_id]
    buf.append(entry)
    if len(buf) > _JOB_LOG_MAX:
        _job_log_buffers[job_id] = buf[-_JOB_LOG_MAX:]
    # Also broadcast to general SSE stream
    broadcast_sse("job_log", {"job_id": job_id, **entry})


async def _job_log_generator(request: Request, job_id: str):
    """Async generator that yields SSE events for a specific job.

    Replays buffered log lines, then live-tails new events
    from the broadcast SSE bus filtered to this job_id.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    with _sse_lock:
        _sse_subscribers.append(queue)

    try:
        # Replay buffered log lines
        for entry in list(_job_log_buffers.get(job_id, [])):
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
                    "job_status", "job_log", "lease_claimed", "lease_released",
                    "job_completed", "job_failed",
                ):
                    if event_data.get("job_id", "") == job_id:
                        yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
    finally:
        with _sse_lock:
            if queue in _sse_subscribers:
                _sse_subscribers.remove(queue)


@app.get("/jobs/{job_id}/logs/stream", tags=["Jobs"])
async def api_job_log_stream(request: Request, job_id: str):
    """Stream real-time logs for a specific job via Server-Sent Events.

    Connect with `EventSource('/jobs/{job_id}/logs/stream')` in the browser
    or `curl -N` from the CLI. Replays buffered log lines on connect, then
    live-tails new log entries until the client disconnects or the job completes.

    Events emitted:
    - `job_log` — individual log line (data: {job_id, timestamp, line, level})
    - `job_status` — status change (data: {job_id, status})
    - `connected` — initial handshake (data: {job_id, status: "streaming"})
    """
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


@app.get("/jobs/{job_id}/logs", tags=["Jobs"])
def api_job_logs(job_id: str, limit: int = 100):
    """Get buffered log lines for a job (non-streaming).

    Returns the last `limit` log lines from the in-memory buffer.
    For real-time streaming, use `/jobs/{job_id}/logs/stream` (SSE).
    """
    buf = _job_log_buffers.get(job_id, [])
    return {"ok": True, "job_id": job_id, "logs": buf[-limit:], "total": len(buf)}


# ── Billing endpoints ────────────────────────────────────────────────


@app.post("/billing/bill/{job_id}", tags=["Billing"])
def api_bill_job(job_id: str):
    """Bill a specific completed job."""
    record = bill_job(job_id)
    if not record:
        raise HTTPException(status_code=400, detail=f"Could not bill job {job_id}")
    return {"ok": True, "bill": record}


@app.post("/billing/bill-all", tags=["Billing"])
def api_bill_all():
    """Bill all unbilled completed jobs."""
    bills = bill_all_completed()
    return {"billed": len(bills), "bills": bills}


@app.get("/billing", tags=["Billing"])
def api_billing():
    """Get all billing records and total revenue."""
    records = load_billing()
    return {
        "records": records,
        "total_revenue": get_total_revenue(),
    }


# ── Phase 11: Dashboard ───────────────────────────────────────────────


@app.get("/dashboard", response_class=HTMLResponse, tags=["Infrastructure"])
def dashboard():
    """The dashboard. HTML + JS. No React. No npm. No build step."""
    html = (TEMPLATES_DIR / "dashboard.html").read_text()
    return HTMLResponse(content=html)


# ── Phase 12: Alerts config ───────────────────────────────────────────


class AlertConfig(BaseModel):
    email_enabled: bool | None = None
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_user: str | None = None
    smtp_pass: str | None = None
    email_from: str | None = None
    email_to: str | None = None
    telegram_enabled: bool | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None


@app.get("/alerts/config", tags=["Infrastructure"])
def api_get_alert_config():
    """Get current alert config (passwords redacted)."""
    safe = {k: ("***" if "pass" in k or "token" in k else v) for k, v in ALERT_CONFIG.items()}
    return {"config": safe}


@app.put("/alerts/config", tags=["Infrastructure"])
def api_set_alert_config(cfg: AlertConfig):
    """Update alert config at runtime."""
    updates = {k: v for k, v in cfg.model_dump().items() if v is not None}
    configure_alerts(**updates)
    return {"ok": True, "updated": list(updates.keys())}


# ── Phase 13: SSH key management ──────────────────────────────────────


@app.post("/ssh/keygen", tags=["Infrastructure"])
def api_generate_ssh_key():
    """Generate an Ed25519 SSH keypair for host access."""
    path = generate_ssh_keypair()
    pub = get_public_key(path)
    return {"ok": True, "key_path": path, "public_key": pub}


@app.get("/ssh/pubkey", tags=["Infrastructure"])
def api_get_pubkey():
    """Get the public key to add to hosts' authorized_keys."""
    pub = get_public_key()
    if not pub:
        raise HTTPException(status_code=404, detail="No SSH key found. POST /ssh/keygen first.")
    return {"public_key": pub}


@app.post("/token/generate", tags=["Infrastructure"])
def api_generate_token():
    """Generate a secure random API token. User must set it in .env themselves."""
    token = secrets.token_urlsafe(32)
    return {"token": token, "note": "Set XCELSIOR_API_TOKEN in your .env to enable auth."}


# ── OAuth2 Device Authorization Flow ─────────────────────────────────
# Implements RFC 8628 for CLI-to-web authentication.
# Flow: CLI calls /api/auth/device → gets user_code + verification_url
#        → user opens browser → enters code → CLI polls /api/auth/token

_device_codes: dict[str, dict] = {}  # device_code -> {user_code, expires, status, token}
_device_lock = _threading.Lock()

DEVICE_CODE_EXPIRY = 600  # 10 minutes
DEVICE_CODE_INTERVAL = 5  # poll interval seconds


class DeviceCodeResponse(BaseModel):
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int = DEVICE_CODE_EXPIRY
    interval: int = DEVICE_CODE_INTERVAL


class DeviceTokenRequest(BaseModel):
    device_code: str
    grant_type: str = "urn:ietf:params:oauth:grant-type:device_code"


@app.post("/api/auth/device", tags=["Infrastructure"])
def api_auth_device_code(request: Request):
    """Initiate OAuth2 device authorization flow (RFC 8628).

    Returns a device_code (for polling) and a user_code (for the user to enter
    in the browser at the verification_uri).
    """
    device_code = secrets.token_urlsafe(32)
    user_code = "-".join([
        secrets.token_hex(2).upper(),
        secrets.token_hex(2).upper(),
    ])  # e.g. "A1B2-C3D4"

    base_url = str(request.base_url).rstrip("/")
    verification_uri = f"{base_url}/api/auth/verify"

    entry = {
        "user_code": user_code,
        "device_code": device_code,
        "status": "pending",  # pending | authorized | expired
        "token": None,
        "created_at": time.time(),
        "expires_at": time.time() + DEVICE_CODE_EXPIRY,
    }

    with _device_lock:
        # Cleanup expired entries
        now = time.time()
        expired = [k for k, v in _device_codes.items() if v["expires_at"] < now]
        for k in expired:
            del _device_codes[k]

        _device_codes[device_code] = entry

    return DeviceCodeResponse(
        device_code=device_code,
        user_code=user_code,
        verification_uri=verification_uri,
    )


@app.post("/api/auth/token", tags=["Infrastructure"])
def api_auth_device_token(body: DeviceTokenRequest):
    """Poll for device authorization result (RFC 8628 §3.4).

    Returns:
    - 200 + access_token when authorized
    - 428 "authorization_pending" while waiting
    - 410 "expired_token" if timed out
    """
    with _device_lock:
        entry = _device_codes.get(body.device_code)

    if not entry:
        raise HTTPException(status_code=404, detail="invalid_device_code")

    now = time.time()
    if now > entry["expires_at"]:
        entry["status"] = "expired"
        raise HTTPException(status_code=410, detail="expired_token")

    if entry["status"] == "pending":
        raise HTTPException(
            status_code=428,
            detail="authorization_pending",
            headers={"Retry-After": str(DEVICE_CODE_INTERVAL)},
        )

    if entry["status"] == "authorized":
        return {
            "access_token": entry["token"],
            "token_type": "Bearer",
            "expires_in": 86400 * 30,  # 30 days
        }

    raise HTTPException(status_code=400, detail="unknown_status")


class DeviceVerifyRequest(BaseModel):
    user_code: str


@app.post("/api/auth/verify", tags=["Infrastructure"])
def api_auth_verify_device(body: DeviceVerifyRequest):
    """Verify a device code by entering the user_code shown in the CLI.

    Called from the web dashboard after the user logs in and enters their code.
    Generates a bearer token and marks the device flow as authorized.
    """
    with _device_lock:
        for dc, entry in _device_codes.items():
            if entry["user_code"] == body.user_code and entry["status"] == "pending":
                now = time.time()
                if now > entry["expires_at"]:
                    entry["status"] = "expired"
                    raise HTTPException(status_code=410, detail="Code expired")

                # Generate bearer token
                token = secrets.token_urlsafe(32)
                entry["status"] = "authorized"
                entry["token"] = token
                return {"message": "Device authorized", "user_code": body.user_code}

    raise HTTPException(status_code=404, detail="Invalid or expired user code")


@app.get("/api/auth/verify", response_class=HTMLResponse, tags=["Infrastructure"])
def api_auth_verify_page():
    """Browser-facing page where users enter their device code."""
    return HTMLResponse("""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Xcelsior — Device Authorization</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0a;color:#e5e7eb;
       display:flex;align-items:center;justify-content:center;min-height:100vh}
  .card{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:2rem;
        max-width:420px;width:100%;text-align:center}
  h1{font-size:1.5rem;margin-bottom:.5rem;color:#60a5fa}
  p{color:#9ca3af;margin-bottom:1.5rem;font-size:.9rem}
  input{width:100%;padding:.75rem;border:1px solid #374151;border-radius:8px;
        background:#1f2937;color:#f9fafb;font-size:1.2rem;text-align:center;
        letter-spacing:.2em;text-transform:uppercase;margin-bottom:1rem}
  input:focus{outline:none;border-color:#3b82f6}
  button{width:100%;padding:.75rem;border:none;border-radius:8px;
         background:#3b82f6;color:white;font-size:1rem;cursor:pointer;font-weight:600}
  button:hover{background:#2563eb}
  .msg{margin-top:1rem;padding:.75rem;border-radius:8px;font-size:.9rem}
  .ok{background:#064e3b;color:#6ee7b7;border:1px solid #065f46}
  .err{background:#7f1d1d;color:#fca5a5;border:1px solid #991b1b}
</style></head><body>
<div class="card">
  <h1>Xcelsior</h1>
  <p>Enter the code shown in your CLI to authorize this device.</p>
  <form id="f">
    <input id="code" placeholder="XXXX-XXXX" maxlength="9" autocomplete="off" autofocus>
    <button type="submit">Authorize Device</button>
  </form>
  <div id="msg"></div>
</div>
<script>
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault();
  const code=document.getElementById('code').value.trim();
  if(!code)return;
  const msg=document.getElementById('msg');
  try{
    const r=await fetch('/api/auth/verify',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({user_code:code})});
    const d=await r.json();
    if(r.ok){msg.className='msg ok';msg.textContent='✓ Device authorized! You can close this tab.';}
    else{msg.className='msg err';msg.textContent=d.detail||'Authorization failed.';}
  }catch(x){msg.className='msg err';msg.textContent='Network error.';}
};
</script></body></html>""")


# ── User Authentication (Email/Password + OAuth) ─────────────────────
# Per UI_ROADMAP Phase UI-1: Login/Signup with email + OAuth providers
# Password hashing uses PBKDF2-HMAC-SHA256 + salt
# Storage: persistent SQLite via db.UserStore (survives restarts)

import hashlib as _hashlib

# Legacy in-memory stores kept for backward compat in tests that mock them
_users_db: dict[str, dict] = {}  # DEPRECATED — use UserStore
_sessions: dict[str, dict] = {}  # DEPRECATED — use UserStore
_api_keys: dict[str, dict] = {}  # DEPRECATED — use UserStore
_user_lock = _threading.Lock()

# Feature flag: use persistent storage (default True, can disable for tests)
_USE_PERSISTENT_AUTH = os.environ.get("XCELSIOR_PERSISTENT_AUTH", "true").lower() != "false"

SESSION_EXPIRY = 86400 * 30  # 30 days


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Hash a password with PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = _hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return hashed, salt


def _create_session(email: str, user: dict) -> dict:
    """Create a session token for a user."""
    token = secrets.token_urlsafe(48)
    session = {
        "token": token,
        "email": email,
        "user_id": user.get("user_id", email),
        "role": user.get("role", "submitter"),
        "name": user.get("name", ""),
        "created_at": time.time(),
        "expires_at": time.time() + SESSION_EXPIRY,
    }
    if _USE_PERSISTENT_AUTH:
        UserStore.create_session(session)
    else:
        with _user_lock:
            _sessions[token] = session
    return session


def _get_current_user(request: Request) -> dict | None:
    """Extract user from Authorization header."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        if _USE_PERSISTENT_AUTH:
            session = UserStore.get_session(token)
            if session:
                return dict(session)
            api_key = UserStore.get_api_key(token)
            if api_key:
                return {"email": api_key["email"], "user_id": api_key["user_id"],
                        "role": api_key.get("role", "submitter"), "name": api_key.get("name", "")}
        else:
            with _user_lock:
                session = _sessions.get(token)
            if session and session["expires_at"] > time.time():
                return session
            with _user_lock:
                api_key = _api_keys.get(token)
            if api_key:
                api_key["last_used"] = time.time()
                return {"email": api_key["email"], "user_id": api_key["user_id"],
                        "role": api_key.get("role", "submitter"), "name": api_key.get("name", "")}
    return None


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str = ""
    role: str = "submitter"  # submitter | provider | admin


class LoginRequest(BaseModel):
    email: str
    password: str


class ProfileUpdateRequest(BaseModel):
    name: str | None = None
    role: str | None = None
    country: str | None = None
    province: str | None = None


@app.post("/api/auth/register", tags=["Auth"])
def api_auth_register(body: RegisterRequest):
    """Register a new user with email and password.

    Creates an account and returns a session token for immediate use.
    Password is hashed with PBKDF2-HMAC-SHA256 + random 16-byte salt.
    """
    email = body.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(400, "Invalid email address")
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")

    if _USE_PERSISTENT_AUTH:
        if UserStore.user_exists(email):
            raise HTTPException(409, "Email already registered")
    else:
        with _user_lock:
            if email in _users_db:
                raise HTTPException(409, "Email already registered")

    password_hash, salt = _hash_password(body.password)
    user_id = f"user-{uuid.uuid4().hex[:12]}"
    customer_id = f"cust-{uuid.uuid4().hex[:8]}"

    user = {
        "user_id": user_id,
        "email": email,
        "name": body.name or email.split("@")[0],
        "password_hash": password_hash,
        "salt": salt,
        "role": body.role,
        "customer_id": customer_id,
        "provider_id": None,
        "country": "CA",
        "province": "ON",
        "created_at": time.time(),
    }

    if _USE_PERSISTENT_AUTH:
        UserStore.create_user(user)
    else:
        with _user_lock:
            _users_db[email] = user

    # Create initial wallet
    try:
        be = get_billing_engine()
        be.deposit(customer_id, 0.0, "Account created")
    except Exception:
        pass

    session = _create_session(email, user)
    broadcast_sse("user_registered", {"email": email, "user_id": user_id})

    return {
        "ok": True,
        "access_token": session["token"],
        "token_type": "Bearer",
        "expires_in": SESSION_EXPIRY,
        "user": {
            "user_id": user_id,
            "email": email,
            "name": user["name"],
            "role": user["role"],
            "customer_id": customer_id,
        },
    }


@app.post("/api/auth/login", tags=["Auth"])
def api_auth_login(body: LoginRequest):
    """Authenticate with email and password.

    Returns a Bearer token valid for 30 days.
    """
    email = body.email.strip().lower()

    if _USE_PERSISTENT_AUTH:
        user = UserStore.get_user(email)
    else:
        with _user_lock:
            user = _users_db.get(email)

    if not user:
        raise HTTPException(401, "Invalid email or password")

    password_hash, _ = _hash_password(body.password, user["salt"])
    if not hmac.compare_digest(password_hash, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")

    session = _create_session(email, user)

    return {
        "ok": True,
        "access_token": session["token"],
        "token_type": "Bearer",
        "expires_in": SESSION_EXPIRY,
        "user": {
            "user_id": user["user_id"],
            "email": email,
            "name": user["name"],
            "role": user["role"],
            "customer_id": user["customer_id"],
            "provider_id": user.get("provider_id"),
        },
    }


@app.post("/api/auth/oauth/{provider}", tags=["Auth"])
def api_auth_oauth(provider: str, request: Request):
    """Handle OAuth callback from Google, GitHub, or HuggingFace.

    In production, this would exchange the OAuth authorization code
    for user profile info. Currently creates/returns a session for
    the OAuth email.
    """
    if provider not in ("google", "github", "huggingface"):
        raise HTTPException(400, f"Unsupported OAuth provider: {provider}")

    # In production: exchange code from query params / body for user profile
    # For now: create a dev user for this provider
    email = f"{provider}-user@xcelsior.ca"
    if _USE_PERSISTENT_AUTH:
        user = UserStore.get_user(email)
    else:
        with _user_lock:
            user = _users_db.get(email)

    if not user:
        user_id = f"user-{uuid.uuid4().hex[:12]}"
        customer_id = f"cust-{uuid.uuid4().hex[:8]}"
        user = {
            "user_id": user_id,
            "email": email,
            "name": f"{provider.title()} User",
            "password_hash": "",
            "salt": "",
            "role": "submitter",
            "customer_id": customer_id,
            "provider_id": None,
            "country": "CA",
            "province": "ON",
            "oauth_provider": provider,
            "created_at": time.time(),
        }
        if _USE_PERSISTENT_AUTH:
            UserStore.create_user(user)
        else:
            with _user_lock:
                _users_db[email] = user

    session = _create_session(email, user)

    return {
        "ok": True,
        "access_token": session["token"],
        "token_type": "Bearer",
        "expires_in": SESSION_EXPIRY,
        "user": {
            "user_id": user["user_id"],
            "email": email,
            "name": user["name"],
            "role": user["role"],
            "customer_id": user["customer_id"],
            "oauth_provider": provider,
        },
    }


@app.get("/api/auth/me", tags=["Auth"])
def api_auth_me(request: Request):
    """Get the currently authenticated user's profile.

    Requires Authorization: Bearer <token> header.
    """
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    email = user["email"]
    if _USE_PERSISTENT_AUTH:
        full_user = UserStore.get_user(email) or {}
    else:
        with _user_lock:
            full_user = _users_db.get(email, {})

    return {
        "ok": True,
        "user": {
            "user_id": user["user_id"],
            "email": email,
            "name": full_user.get("name", user.get("name", "")),
            "role": full_user.get("role", user.get("role", "submitter")),
            "customer_id": full_user.get("customer_id", ""),
            "provider_id": full_user.get("provider_id"),
            "country": full_user.get("country", "CA"),
            "province": full_user.get("province", "ON"),
            "team_id": full_user.get("team_id"),
            "created_at": full_user.get("created_at", 0),
        },
    }


@app.patch("/api/auth/me", tags=["Auth"])
def api_auth_update_profile(body: ProfileUpdateRequest, request: Request):
    """Update the current user's profile fields."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    if _USE_PERSISTENT_AUTH:
        updates = {}
        if body.name is not None:
            updates["name"] = body.name
        if body.role is not None:
            updates["role"] = body.role
        if body.country is not None:
            updates["country"] = body.country
        if body.province is not None:
            updates["province"] = body.province
        if not updates:
            return {"ok": True, "message": "No changes"}
        UserStore.update_user(user["email"], updates)
    else:
        with _user_lock:
            full_user = _users_db.get(user["email"])
            if not full_user:
                raise HTTPException(404, "User not found")
            if body.name is not None:
                full_user["name"] = body.name
            if body.role is not None:
                full_user["role"] = body.role
            if body.country is not None:
                full_user["country"] = body.country
            if body.province is not None:
                full_user["province"] = body.province

    return {"ok": True, "message": "Profile updated"}


@app.post("/api/auth/refresh", tags=["Auth"])
def api_auth_refresh(request: Request):
    """Refresh an existing session token.

    Returns a new token with a fresh 30-day expiry.
    """
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Session expired or invalid")

    if _USE_PERSISTENT_AUTH:
        full_user = UserStore.get_user(user["email"]) or user
    else:
        with _user_lock:
            full_user = _users_db.get(user["email"], user)

    # Invalidate old token
    old_token = request.headers.get("authorization", "")[7:]
    if _USE_PERSISTENT_AUTH:
        UserStore.delete_session(old_token)
    else:
        with _user_lock:
            _sessions.pop(old_token, None)

    session = _create_session(user["email"], full_user)
    return {
        "ok": True,
        "access_token": session["token"],
        "token_type": "Bearer",
        "expires_in": SESSION_EXPIRY,
    }


@app.delete("/api/auth/me", tags=["Auth"])
def api_auth_delete_account(request: Request):
    """Delete the current user's account."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    if _USE_PERSISTENT_AUTH:
        UserStore.delete_user(user["email"])
    else:
        with _user_lock:
            _users_db.pop(user["email"], None)
            to_remove = [k for k, v in _sessions.items() if v.get("email") == user["email"]]
            for k in to_remove:
                del _sessions[k]
            to_remove = [k for k, v in _api_keys.items() if v.get("email") == user["email"]]
            for k in to_remove:
                del _api_keys[k]

    return {"ok": True, "message": "Account deleted"}


@app.post("/api/keys/generate", tags=["Auth"])
def api_generate_api_key(request: Request, name: str = "default"):
    """Generate a named API key for the authenticated user.

    API keys can be used as Bearer tokens for programmatic access.
    """
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    key = f"xc-{secrets.token_urlsafe(32)}"
    key_data = {
        "key": key,
        "name": name,
        "email": user["email"],
        "user_id": user["user_id"],
        "role": user.get("role", "submitter"),
        "created_at": time.time(),
        "last_used": None,
    }
    with _user_lock:
        _api_keys[key] = key_data
    if _USE_PERSISTENT_AUTH:
        UserStore.create_api_key(key_data)

    return {
        "ok": True,
        "key": key,
        "name": name,
        "preview": key[:12] + "..." + key[-4:],
        "note": "Save this key — it will not be shown again.",
    }


@app.get("/api/keys", tags=["Auth"])
def api_list_keys(request: Request):
    """List all API keys for the authenticated user (keys are redacted)."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    if _USE_PERSISTENT_AUTH:
        all_keys = UserStore.list_api_keys(user["email"])
        keys = [
            {
                "name": v["name"],
                "preview": v["key"][:12] + "..." + v["key"][-4:],
                "created_at": v["created_at"],
                "last_used": v["last_used"],
            }
            for v in all_keys
        ]
    else:
        with _user_lock:
            keys = [
                {
                    "name": v["name"],
                    "preview": v["key"][:12] + "..." + v["key"][-4:],
                    "created_at": v["created_at"],
                    "last_used": v["last_used"],
                }
                for v in _api_keys.values()
                if v["email"] == user["email"]
            ]

    return {"ok": True, "keys": keys}


@app.delete("/api/keys/{key_preview}", tags=["Auth"])
def api_revoke_key(key_preview: str, request: Request):
    """Revoke an API key by its preview string."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    if _USE_PERSISTENT_AUTH:
        found = UserStore.delete_api_key_by_preview(user["email"], key_preview)
        if not found:
            raise HTTPException(404, "API key not found")
    else:
        with _user_lock:
            to_remove = [
                k for k, v in _api_keys.items()
                if v["email"] == user["email"] and (v["key"][:12] + "..." + v["key"][-4:]) == key_preview
            ]
            for k in to_remove:
                del _api_keys[k]
        if not to_remove:
            raise HTTPException(404, "API key not found")
    return {"ok": True, "message": "API key revoked"}


# ── Password Reset & Change ──────────────────────────────────────────


class PasswordResetRequest(BaseModel):
    email: str


@app.post("/api/auth/password-reset", tags=["Auth"])
def api_auth_password_reset(req: PasswordResetRequest):
    """Initiate a password reset. Returns a one-time reset token (dev mode)."""
    reset_token = None
    if _USE_PERSISTENT_AUTH:
        user = UserStore.get_user(req.email)
        if not user:
            return {"ok": True, "message": "If the email exists, a reset link has been sent."}
        reset_token = secrets.token_urlsafe(32)
        UserStore.update_user(req.email, {
            "reset_token": reset_token,
            "reset_token_expires": time.time() + 3600,
        })
    else:
        with _user_lock:
            user = _users_db.get(req.email)
            if not user:
                return {"ok": True, "message": "If the email exists, a reset link has been sent."}
            reset_token = secrets.token_urlsafe(32)
            user["reset_token"] = reset_token
            user["reset_token_expires"] = time.time() + 3600

    return {
        "ok": True,
        "message": "If the email exists, a reset link has been sent.",
        "reset_token": reset_token if os.environ.get("XCELSIOR_ENV") == "test" else None,
    }


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str


@app.post("/api/auth/password-reset/confirm", tags=["Auth"])
def api_auth_password_reset_confirm(req: PasswordResetConfirm):
    """Confirm password reset with token and set new password."""
    if len(req.new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")

    if _USE_PERSISTENT_AUTH:
        from db import auth_connection
        with auth_connection() as conn:
            row = conn.execute("SELECT email, reset_token_expires FROM users WHERE reset_token = ?", (req.token,)).fetchone()
            if not row:
                raise HTTPException(400, "Invalid or expired reset token")
            if time.time() > (row["reset_token_expires"] or 0):
                raise HTTPException(400, "Reset token has expired")
            salt = secrets.token_hex(16)
            new_hash, _ = _hash_password(req.new_password, salt)
            conn.execute(
                "UPDATE users SET password_hash=?, salt=?, reset_token=NULL, reset_token_expires=NULL WHERE email=?",
                (new_hash, salt, row["email"]),
            )
            UserStore.delete_user_sessions(row["email"])
        return {"ok": True, "message": "Password updated. Please log in again."}
    else:
        with _user_lock:
            for email, user in _users_db.items():
                if user.get("reset_token") == req.token:
                    if time.time() > user.get("reset_token_expires", 0):
                        raise HTTPException(400, "Reset token has expired")
                    salt = secrets.token_hex(16)
                    user["password_hash"], _ = _hash_password(req.new_password, salt)
                    user["salt"] = salt
                    user.pop("reset_token", None)
                    user.pop("reset_token_expires", None)
                    to_remove_s = [k for k, v in _sessions.items() if v.get("email") == email]
                    for k in to_remove_s:
                        del _sessions[k]
                    return {"ok": True, "message": "Password updated. Please log in again."}
        raise HTTPException(400, "Invalid or expired reset token")


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@app.post("/api/auth/change-password", tags=["Auth"])
def api_auth_change_password(request: Request, req: ChangePasswordRequest):
    """Change password for the authenticated user."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    if len(req.new_password) < 8:
        raise HTTPException(400, "New password must be at least 8 characters")

    if _USE_PERSISTENT_AUTH:
        stored = UserStore.get_user(user["email"])
        if not stored:
            raise HTTPException(404, "User not found")
        expected, _ = _hash_password(req.current_password, stored.get("salt", ""))
        if not hmac.compare_digest(expected, stored.get("password_hash", "")):
            raise HTTPException(400, "Current password is incorrect")
        salt = secrets.token_hex(16)
        new_hash, _ = _hash_password(req.new_password, salt)
        UserStore.update_user(user["email"], {"password_hash": new_hash, "salt": salt})
    else:
        with _user_lock:
            stored = _users_db.get(user["email"])
            if not stored:
                raise HTTPException(404, "User not found")
            expected, _ = _hash_password(req.current_password, stored.get("salt", ""))
            if not hmac.compare_digest(expected, stored.get("password_hash", "")):
                raise HTTPException(400, "Current password is incorrect")
            salt = secrets.token_hex(16)
            stored["password_hash"], _ = _hash_password(req.new_password, salt)
            stored["salt"] = salt

    return {"ok": True, "message": "Password changed successfully"}


# ── Team / Organization Management ───────────────────────────────────
# Per UI_ROADMAP competitor gap: team/org management
# Teams share billing, hosts, and job visibility


class CreateTeamRequest(BaseModel):
    name: str
    plan: str = "free"  # free | pro | enterprise


class AddTeamMemberRequest(BaseModel):
    email: str
    role: str = "member"  # admin | member | viewer


@app.post("/api/teams", tags=["Teams"])
def api_create_team(body: CreateTeamRequest, request: Request):
    """Create a new team/organization. Creator becomes team admin."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    team_id = f"team-{uuid.uuid4().hex[:8]}"
    max_members = {"free": 5, "pro": 25, "enterprise": 100}.get(body.plan, 5)

    team = {
        "team_id": team_id,
        "name": body.name,
        "owner_email": user["email"],
        "created_at": time.time(),
        "plan": body.plan,
        "max_members": max_members,
    }

    UserStore.create_team(team)
    UserStore.update_user(user["email"], {"team_id": team_id})
    broadcast_sse("team_created", {"team_id": team_id, "name": body.name})

    return {"ok": True, "team_id": team_id, "name": body.name, "plan": body.plan}


@app.get("/api/teams/me", tags=["Teams"])
def api_my_teams(request: Request):
    """Get teams the current user belongs to."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    teams = UserStore.get_user_teams(user["email"])
    return {"ok": True, "teams": teams}


@app.get("/api/teams/{team_id}", tags=["Teams"])
def api_get_team(team_id: str, request: Request):
    """Get team details including members."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    team = UserStore.get_team(team_id)
    if not team:
        raise HTTPException(404, "Team not found")

    members = UserStore.list_team_members(team_id)
    # Verify the requester is a member
    if not any(m["email"] == user["email"] for m in members):
        raise HTTPException(403, "Not a member of this team")

    return {"ok": True, "team": team, "members": members}


@app.post("/api/teams/{team_id}/members", tags=["Teams"])
def api_add_team_member(team_id: str, body: AddTeamMemberRequest, request: Request):
    """Add a member to a team. Only team admins can add members."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    team = UserStore.get_team(team_id)
    if not team:
        raise HTTPException(404, "Team not found")

    # Check requester is admin
    members = UserStore.list_team_members(team_id)
    requester = next((m for m in members if m["email"] == user["email"]), None)
    if not requester or requester["role"] != "admin":
        raise HTTPException(403, "Only team admins can add members")

    # Verify target user exists
    target = UserStore.get_user(body.email)
    if not target:
        raise HTTPException(404, f"User {body.email} not found")

    ok = UserStore.add_team_member(team_id, body.email, body.role)
    if not ok:
        raise HTTPException(400, "Team is at member capacity")

    broadcast_sse("team_member_added", {"team_id": team_id, "email": body.email})
    return {"ok": True, "message": f"{body.email} added to team as {body.role}"}


@app.delete("/api/teams/{team_id}/members/{email}", tags=["Teams"])
def api_remove_team_member(team_id: str, email: str, request: Request):
    """Remove a member from a team. Admins can remove anyone; members can leave."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    team = UserStore.get_team(team_id)
    if not team:
        raise HTTPException(404, "Team not found")

    members = UserStore.list_team_members(team_id)
    requester = next((m for m in members if m["email"] == user["email"]), None)
    if not requester:
        raise HTTPException(403, "Not a member of this team")

    # Non-admins can only remove themselves
    if requester["role"] != "admin" and email != user["email"]:
        raise HTTPException(403, "Only admins can remove other members")

    # Prevent removing the owner
    if email == team["owner_email"]:
        raise HTTPException(400, "Cannot remove team owner")

    UserStore.remove_team_member(team_id, email)
    return {"ok": True, "message": f"{email} removed from team"}


@app.delete("/api/teams/{team_id}", tags=["Teams"])
def api_delete_team(team_id: str, request: Request):
    """Delete a team. Only the team owner can delete it."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    team = UserStore.get_team(team_id)
    if not team:
        raise HTTPException(404, "Team not found")

    if team["owner_email"] != user["email"]:
        raise HTTPException(403, "Only the team owner can delete this team")

    UserStore.delete_team(team_id)
    return {"ok": True, "message": "Team deleted"}


# ── Marketplace Search ───────────────────────────────────────────────


@app.get("/marketplace/search", tags=["Marketplace"])
def api_marketplace_search(
    gpu_model: str | None = None,
    min_vram: float | None = None,
    max_price: float | None = None,
    province: str | None = None,
    country: str | None = None,
    min_reputation: int | None = None,
    sort_by: str = "price",
    limit: int = 50,
):
    """Search marketplace listings with filters and sorting."""
    listings = get_marketplace(active_only=True)

    if gpu_model:
        listings = [l for l in listings if gpu_model.lower() in (l.get("gpu_model", "")).lower()]
    if min_vram is not None:
        listings = [l for l in listings if (l.get("vram_gb", 0) or 0) >= min_vram]
    if max_price is not None:
        listings = [l for l in listings if (l.get("price_per_hour", 999) or 999) <= max_price]
    if province:
        listings = [l for l in listings if l.get("province", "").upper() == province.upper()]
    if country:
        listings = [l for l in listings if (l.get("country", "").upper()) == country.upper()]
    if min_reputation is not None:
        listings = [l for l in listings if (l.get("reputation_score", 0) or 0) >= min_reputation]

    # Sort
    sort_keys = {
        "price": lambda x: x.get("price_per_hour", 999),
        "vram": lambda x: -(x.get("vram_gb", 0) or 0),
        "reputation": lambda x: -(x.get("reputation_score", 0) or 0),
        "score": lambda x: -(x.get("compute_score", 0) or 0),
    }
    if sort_by in sort_keys:
        listings.sort(key=sort_keys[sort_by])

    return {
        "ok": True,
        "total": len(listings),
        "listings": listings[:limit],
        "filters_applied": {
            "gpu_model": gpu_model,
            "min_vram": min_vram,
            "max_price": max_price,
            "province": province,
            "country": country,
            "min_reputation": min_reputation,
            "sort_by": sort_by,
        },
    }


# ── Slurm Cluster Adapter ────────────────────────────────────────────


class SlurmSubmitIn(BaseModel):
    name: str
    vram_needed_gb: float = 0
    priority: int = 0
    tier: str | None = None
    num_gpus: int = 1
    image: str = ""
    profile: str | None = None
    dry_run: bool = False


@app.post("/api/slurm/submit", tags=["Infrastructure"])
def api_slurm_submit(body: SlurmSubmitIn):
    """Submit an Xcelsior job to a Slurm cluster (HPC bridge).

    Translates the job to an sbatch script and submits. Set dry_run=true
    to see the generated script without submitting.
    """
    from slurm_adapter import submit_to_slurm, register_slurm_job

    job_dict = {
        "job_id": secrets.token_hex(4),
        "name": body.name,
        "vram_needed_gb": body.vram_needed_gb,
        "priority": body.priority,
        "tier": body.tier or "free",
        "num_gpus": body.num_gpus,
        "image": body.image,
    }

    result = submit_to_slurm(job_dict, profile_name=body.profile, dry_run=body.dry_run)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    if not body.dry_run and "slurm_job_id" in result:
        register_slurm_job(job_dict["job_id"], result["slurm_job_id"])

    return result


@app.get("/api/slurm/status/{slurm_job_id}", tags=["Infrastructure"])
def api_slurm_status(slurm_job_id: str):
    """Check the status of a Slurm job."""
    from slurm_adapter import get_slurm_job_status

    status = get_slurm_job_status(slurm_job_id)
    if "error" in status:
        raise HTTPException(status_code=400, detail=status["error"])
    return status


@app.delete("/api/slurm/{slurm_job_id}", tags=["Infrastructure"])
def api_slurm_cancel(slurm_job_id: str):
    """Cancel a Slurm job."""
    from slurm_adapter import cancel_slurm_job

    result = cancel_slurm_job(slurm_job_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/slurm/profiles", tags=["Infrastructure"])
def api_slurm_profiles():
    """List available Slurm cluster profiles (Nibi, Graham, Narval, generic)."""
    from slurm_adapter import CLUSTER_PROFILES
    return {"profiles": {k: v["name"] for k, v in CLUSTER_PROFILES.items()}}


# ── NFS Configuration ────────────────────────────────────────────────


@app.get("/api/nfs/config", tags=["Infrastructure"])
def api_nfs_config():
    """Get current NFS configuration from environment."""
    return {
        "nfs_server": os.environ.get("XCELSIOR_NFS_SERVER", ""),
        "nfs_path": os.environ.get("XCELSIOR_NFS_PATH", ""),
        "nfs_mount_point": os.environ.get("XCELSIOR_NFS_MOUNT", "/mnt/xcelsior-nfs"),
        "configured": bool(os.environ.get("XCELSIOR_NFS_SERVER")),
    }


@app.get("/tiers", tags=["Jobs"])
def api_list_tiers():
    """List all priority tiers with their multipliers."""
    return {"tiers": list_tiers()}


# ── Phase 16: Docker Image Builder ───────────────────────────────────


class BuildIn(BaseModel):
    model: str
    base_image: str = "python:3.11-slim"
    quantize: str | None = None
    push: bool = False


@app.post("/build", tags=["Infrastructure"])
def api_build_image(b: BuildIn):
    """Build a Docker image for a model. Optionally quantize and push."""
    result = build_and_push(b.model, quantize=b.quantize, base_image=b.base_image, push=b.push)
    if not result["built"]:
        raise HTTPException(status_code=500, detail=f"Build failed for {b.model}")
    return {"ok": True, "build": result}


@app.get("/builds", tags=["Infrastructure"])
def api_list_builds():
    """List all local build directories."""
    return {"builds": list_builds()}


@app.post("/build/{model}/dockerfile", tags=["Infrastructure"])
def api_generate_dockerfile(
    model: str, base_image: str = "python:3.11-slim", quantize: str | None = None
):
    """Preview the generated Dockerfile without building."""
    content = generate_dockerfile(model, base_image=base_image, quantize=quantize)
    return {"model": model, "dockerfile": content}


# ── Phase 17: Marketplace ────────────────────────────────────────────


class RigListing(BaseModel):
    host_id: str
    gpu_model: str
    vram_gb: float
    price_per_hour: float
    description: str = ""
    owner: str = "anonymous"


@app.post("/marketplace/list", tags=["Marketplace"])
def api_list_rig(rig: RigListing):
    """List a rig on the marketplace."""
    listing = list_rig(
        rig.host_id, rig.gpu_model, rig.vram_gb, rig.price_per_hour, rig.description, rig.owner
    )
    return {"ok": True, "listing": listing}


@app.delete("/marketplace/{host_id}", tags=["Marketplace"])
def api_unlist_rig(host_id: str):
    """Remove a rig from the marketplace."""
    if not unlist_rig(host_id):
        raise HTTPException(status_code=404, detail=f"Listing {host_id} not found")
    return {"ok": True, "unlisted": host_id}


@app.get("/marketplace", tags=["Marketplace"])
def api_get_marketplace(active_only: bool = True):
    """Browse marketplace listings."""
    return {"listings": get_marketplace(active_only=active_only)}


@app.post("/marketplace/bill/{job_id}", tags=["Marketplace"])
def api_marketplace_bill(job_id: str):
    """Bill a marketplace job — split between host and platform."""
    result = marketplace_bill(job_id)
    if not result:
        raise HTTPException(status_code=400, detail=f"Could not bill marketplace job {job_id}")
    return {"ok": True, "bill": result}


@app.get("/marketplace/stats", tags=["Marketplace"])
def api_marketplace_stats():
    """Marketplace aggregate stats."""
    return {"stats": marketplace_stats()}


# ── Phase 18: Canada-Only Toggle ─────────────────────────────────────


class CanadaToggle(BaseModel):
    enabled: bool


@app.get("/canada", tags=["Jurisdiction"])
def api_canada_status():
    """Check if Canada-only mode is active."""
    import scheduler

    return {"canada_only": scheduler.CANADA_ONLY}


@app.put("/canada", tags=["Jurisdiction"])
def api_set_canada(toggle: CanadaToggle):
    """Toggle Canada-only mode."""
    set_canada_only(toggle.enabled)
    return {"ok": True, "canada_only": toggle.enabled}


@app.get("/hosts/ca", tags=["Jurisdiction"])
def api_list_canadian_hosts():
    """List only Canadian hosts."""
    return {"hosts": list_hosts_filtered(canada_only=True)}


@app.post("/queue/process/ca", tags=["Jurisdiction"])
def api_process_queue_ca():
    """Process queue with Canada-only hosts."""
    assigned = process_queue_filtered(canada_only=True)
    return {
        "canada_only": True,
        "assigned": [
            {
                "job": j["name"],
                "job_id": j["job_id"],
                "host": h["host_id"],
                "country": h.get("country", "?"),
            }
            for j, h in assigned
        ],
    }


# ── Phase 19: Auto-Scaling ───────────────────────────────────────────


class PoolHost(BaseModel):
    host_id: str
    ip: str
    gpu_model: str
    vram_gb: float
    cost_per_hour: float = 0.20
    country: str = "CA"


@app.post("/autoscale/pool", tags=["Autoscale"])
def api_add_to_pool(h: PoolHost):
    """Add a host to the autoscale pool."""
    entry = add_to_pool(h.host_id, h.ip, h.gpu_model, h.vram_gb, h.cost_per_hour, h.country)
    return {"ok": True, "pool_entry": entry}


@app.delete("/autoscale/pool/{host_id}", tags=["Autoscale"])
def api_remove_from_pool(host_id: str):
    """Remove a host from the autoscale pool."""
    remove_from_pool(host_id)
    return {"ok": True, "removed": host_id}


@app.get("/autoscale/pool", tags=["Autoscale"])
def api_get_pool():
    """List the autoscale pool."""
    return {"pool": load_autoscale_pool()}


@app.post("/autoscale/cycle", tags=["Autoscale"])
def api_autoscale_cycle():
    """Run a full autoscale cycle: scale up, process queue, scale down."""
    provisioned, assigned, deprovisioned = autoscale_cycle()
    return {
        "provisioned": [{"host_id": h["host_id"], "gpu": h["gpu_model"]} for h in provisioned],
        "assigned": [
            {"job": j["name"], "job_id": j["job_id"], "host": h["host_id"]} for j, h in assigned
        ],
        "deprovisioned": deprovisioned,
    }


@app.post("/autoscale/up", tags=["Autoscale"])
def api_autoscale_up():
    """Scale up: provision hosts for queued jobs."""
    provisioned = autoscale_up()
    return {"provisioned": [{"host_id": h["host_id"], "gpu": h["gpu_model"]} for h in provisioned]}


@app.post("/autoscale/down", tags=["Autoscale"])
def api_autoscale_down():
    """Scale down: deprovision idle autoscaled hosts."""
    deprovisioned = autoscale_down()
    return {"deprovisioned": deprovisioned}


# ── SSE Streaming ─────────────────────────────────────────────────────


async def _sse_generator(request: Request):
    """Async generator that yields SSE events until the client disconnects."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    with _sse_lock:
        _sse_subscribers.append(queue)
    try:
        yield f"event: connected\ndata: {json.dumps({'status': 'connected'})}\n\n"

        while True:
            if await request.is_disconnected():
                break
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30)
                event_type = msg.get("event", "message")
                data = json.dumps(msg.get("data", {}))
                yield f"event: {event_type}\ndata: {data}\n\n"
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
    finally:
        with _sse_lock:
            if queue in _sse_subscribers:
                _sse_subscribers.remove(queue)


@app.get("/api/stream", tags=["Infrastructure"])
async def sse_stream(request: Request):
    """Server-Sent Events stream for real-time dashboard updates."""
    return StreamingResponse(
        _sse_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Agent Endpoints (Pull-Based Architecture) ────────────────────────


class VersionReport(BaseModel):
    host_id: str
    versions: dict


class MiningAlert(BaseModel):
    host_id: str
    gpu_index: int
    confidence: float
    reason: str
    timestamp: float | None = None


class BenchmarkReport(BaseModel):
    host_id: str
    gpu_model: str
    score: float
    tflops: float
    details: dict | None = None


@app.get("/agent/work/{host_id}", tags=["Agent"])
def api_agent_work(host_id: str):
    """Pull pending work for an agent. Returns assigned jobs."""
    all_jobs = list_jobs()
    pending = [
        j for j in all_jobs
        if j.get("host_id") == host_id and j.get("status") in ("assigned",)
    ]
    with _agent_lock:
        queued_work = _agent_work.pop(host_id, [])

    jobs = pending + queued_work
    if not jobs:
        return JSONResponse(status_code=204, content=None)
    return {"ok": True, "jobs": jobs}


@app.get("/agent/preempt/{host_id}", tags=["Agent"])
def api_agent_preempt(host_id: str):
    """Check if any jobs on this host should be preempted."""
    with _agent_lock:
        preempt_list = _agent_preempt.pop(host_id, [])
    return {"ok": True, "preempt_jobs": preempt_list}


@app.post("/agent/preempt/{host_id}/{job_id}", tags=["Agent"])
def api_schedule_preemption(host_id: str, job_id: str):
    """Schedule a job for preemption on a host."""
    with _agent_lock:
        _agent_preempt[host_id].append(job_id)
    broadcast_sse("preemption_scheduled", {"host_id": host_id, "job_id": job_id})
    return {"ok": True, "host_id": host_id, "job_id": job_id}


@app.post("/agent/versions", tags=["Agent"])
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
    hosts = list_hosts(active_only=False)
    for h in hosts:
        if h.get("host_id") == report.host_id:
            h["admitted"] = details["admitted"]
            h["recommended_runtime"] = details.get("recommended_runtime", "runc")
            h["admission_details"] = details
            if details["admitted"]:
                h["status"] = "active"
                log.info("HOST %s ADMITTED — status set to active, runtime=%s",
                         report.host_id, details.get("recommended_runtime", "runc"))
            else:
                h["status"] = "pending"
                log.warning("HOST %s NOT ADMITTED — status remains pending: %s",
                            report.host_id, details.get("rejection_reasons", []))
            # Persist
            from scheduler import _atomic_mutation, _upsert_host_row, _migrate_hosts_if_needed
            with _atomic_mutation() as conn:
                _migrate_hosts_if_needed(conn)
                _upsert_host_row(conn, h)
            break

    broadcast_sse("node_admission", {
        "host_id": report.host_id,
        "admitted": details["admitted"],
        "versions": report.versions,
        "runtime": details.get("recommended_runtime", "runc"),
    })
    return {
        "ok": True,
        "admitted": details["admitted"],
        "details": details,
    }


@app.post("/agent/mining-alert", tags=["Agent"])
def api_mining_alert(alert: MiningAlert):
    """Receive mining detection alert from an agent."""
    log.warning(
        "MINING ALERT host=%s gpu=%d confidence=%.0f%% — %s",
        alert.host_id, alert.gpu_index, alert.confidence * 100, alert.reason,
    )
    broadcast_sse("mining_alert", {
        "host_id": alert.host_id,
        "gpu_index": alert.gpu_index,
        "confidence": alert.confidence,
        "reason": alert.reason,
    })
    return {"ok": True, "received": True}


@app.post("/agent/benchmark", tags=["Agent"])
def api_agent_benchmark(report: BenchmarkReport):
    """Receive compute benchmark results from an agent."""
    register_compute_score(
        report.host_id, report.gpu_model, report.score, report.details,
    )
    broadcast_sse("benchmark_result", {
        "host_id": report.host_id,
        "gpu_model": report.gpu_model,
        "xcu": report.score,
        "tflops": report.tflops,
    })
    return {"ok": True, "xcu": report.score}


# ── Agent Lease Protocol ──────────────────────────────────────────────
# Per REPORT_FEATURE_FINAL.md: "clean lease/claim protocol"
# (assign → lease renewal → completion) — not conflating assigned/running.

class LeaseClaimRequest(BaseModel):
    host_id: str
    job_id: str


class LeaseRenewRequest(BaseModel):
    host_id: str
    job_id: str


class LeaseReleaseRequest(BaseModel):
    job_id: str
    reason: str = "completed"  # completed, failed, preempted


@app.post("/agent/lease/claim", tags=["Agent"])
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

    # Grant lease
    lease = store.grant_lease(req.job_id, req.host_id)

    # Transition state: assigned → leased
    try:
        sm.transition(req.job_id, "assigned", "leased",
                      actor=f"agent:{req.host_id}",
                      data={"lease_id": lease.lease_id})
    except ValueError:
        pass  # Event already recorded by grant_lease

    # Update scheduler's job status to leased
    update_job_status(req.job_id, "leased", host_id=req.host_id)

    broadcast_sse("lease_granted", {
        "job_id": req.job_id,
        "host_id": req.host_id,
        "lease_id": lease.lease_id,
        "expires_at": lease.expires_at,
    })

    return {
        "ok": True,
        "lease_id": lease.lease_id,
        "expires_at": lease.expires_at,
        "duration_sec": lease.duration_sec,
    }


@app.post("/agent/lease/renew", tags=["Agent"])
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


@app.post("/agent/lease/release", tags=["Agent"])
def api_agent_lease_release(req: LeaseReleaseRequest):
    """Agent releases its lease (job completed/failed/preempted)."""
    store = get_event_store()
    released = store.release_lease(req.job_id)
    if not released:
        return {"ok": True, "released": False, "detail": "No active lease"}
    return {"ok": True, "released": True}


@app.get("/agent/popular-images", tags=["Agent"])
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


# ── Spot Pricing Endpoints ───────────────────────────────────────────


class SpotJobIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    vram_needed_gb: float = Field(gt=0)
    max_bid: float = Field(gt=0)
    priority: int = Field(default=0, ge=0, le=10)
    tier: str | None = None


@app.get("/spot-prices", tags=["Spot Pricing"])
def api_spot_prices():
    """Get current spot prices for all GPU models."""
    return {"ok": True, "prices": get_current_spot_prices()}


@app.post("/spot-prices/update", tags=["Spot Pricing"])
def api_update_spot_prices():
    """Trigger spot price recalculation."""
    prices = update_spot_prices()
    broadcast_sse("spot_prices_updated", {"prices": prices})
    return {"ok": True, "prices": prices}


@app.post("/spot/job", tags=["Spot Pricing"])
def api_submit_spot_job(j: SpotJobIn):
    """Submit a spot job with a maximum bid price."""
    job = submit_spot_job(j.name, j.vram_needed_gb, j.max_bid, j.priority, tier=j.tier)
    broadcast_sse("spot_job_submitted", {
        "job_id": job["job_id"], "name": job["name"], "max_bid": j.max_bid,
    })
    return {"ok": True, "job": job}


@app.post("/spot/preemption-cycle", tags=["Spot Pricing"])
def api_preemption_cycle():
    """Run a preemption cycle — reclaim resources from underbidding spot jobs."""
    preempted = preemption_cycle()
    return {"ok": True, "preempted": preempted}


# ── Compute Score Endpoints ──────────────────────────────────────────


@app.get("/compute-score/{host_id}", tags=["Hosts"])
def api_get_compute_score(host_id: str):
    """Get the compute score (XCU) for a host."""
    score = get_compute_score(host_id)
    if score is None:
        raise HTTPException(status_code=404, detail=f"No compute score for host {host_id}")
    return {"ok": True, "host_id": host_id, "score": score}


@app.get("/compute-scores", tags=["Hosts"])
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


# ── Health ────────────────────────────────────────────────────────────


@app.get("/healthz", tags=["Infrastructure"])
def healthz():
    return {"ok": True, "status": "healthy", "env": XCELSIOR_ENV}


@app.get("/readyz", tags=["Infrastructure"])
def readyz():
    token = os.environ.get("XCELSIOR_API_TOKEN", API_TOKEN)
    if AUTH_REQUIRED and not token:
        raise HTTPException(
            status_code=503, detail="API token not configured for non-dev environment"
        )

    storage = storage_healthcheck()
    if not storage.get("ok"):
        raise HTTPException(
            status_code=503, detail=f"Storage not ready: {storage.get('error', 'unknown')}"
        )

    return {"ok": True, "status": "ready", "storage": storage}


@app.get("/metrics", tags=["Infrastructure"])
def metrics():
    return {"ok": True, "metrics": get_metrics_snapshot()}


@app.get("/", tags=["Infrastructure"])
def root():
    return {"name": "Xcelsior", "status": "running"}


# ═══════════════════════════════════════════════════════════════════════
# v2.1 API — Events, Verification, Jurisdiction, Billing, Reputation
# ═══════════════════════════════════════════════════════════════════════


# ── Events ────────────────────────────────────────────────────────────

@app.get("/api/events/{entity_type}/{entity_id}", tags=["Events"])
def api_get_events(entity_type: str, entity_id: str, limit: int = 50):
    """Get event history for a job or host."""
    store = get_event_store()
    events = store.get_events(entity_type, entity_id, limit=limit)
    return {"ok": True, "entity_type": entity_type, "entity_id": entity_id, "events": events}


@app.get("/api/events/leases/{job_id}", tags=["Events"])
def api_get_lease(job_id: str):
    """Get active lease for a job."""
    store = get_event_store()
    lease = store.get_lease(job_id)
    if not lease:
        raise HTTPException(status_code=404, detail=f"No active lease for job {job_id}")
    return {"ok": True, "lease": lease}


# ── Verification ──────────────────────────────────────────────────────

class VerifyHostRequest(BaseModel):
    host_id: str
    gpu_info: dict = Field(default_factory=dict)
    network_info: dict = Field(default_factory=dict)


@app.post("/api/verify/{host_id}", tags=["Verification"])
def api_verify_host(host_id: str, req: VerifyHostRequest):
    """Run verification checks on a host."""
    ve = get_verification_engine()
    result = ve.run_verification(host_id, req.gpu_info, req.network_info)
    return {"ok": True, "host_id": host_id, "verification": result}


@app.get("/api/verify/{host_id}/status", tags=["Verification"])
def api_verification_status(host_id: str):
    """Get current verification status for a host."""
    store = get_verification_engine().store
    status = store.get_status(host_id)
    if not status:
        return {"ok": True, "host_id": host_id, "status": "unverified"}
    return {"ok": True, "host_id": host_id, "verification": dict(status)}


@app.get("/api/verified-hosts", tags=["Verification"])
def api_verified_hosts():
    """List all verified hosts with full verification details.

    Returns host_id, state, gpu_model, country, last_check, overall_score
    for every host that has any verification record (not just 'verified').
    """
    ve = get_verification_engine()
    store = ve.store
    # Return all hosts with verification records (any state)
    with store._conn() as conn:
        rows = conn.execute(
            "SELECT host_id, state, overall_score, last_check_at, gpu_fingerprint, deverify_reason FROM host_verifications ORDER BY state, host_id"
        ).fetchall()
    # Enrich with host data
    all_hosts = list_hosts(active_only=False)
    host_map = {h["host_id"]: h for h in all_hosts}
    result = []
    for r in rows:
        h = host_map.get(r["host_id"], {})
        result.append({
            "host_id": r["host_id"],
            "status": r["state"],
            "overall_score": r["overall_score"],
            "last_check": r["last_check_at"],
            "gpu_fingerprint": r["gpu_fingerprint"],
            "deverify_reason": r["deverify_reason"] or "",
            "gpu_model": h.get("gpu_model", "—"),
            "country": h.get("country", ""),
            "province": h.get("province", ""),
        })
    return {"ok": True, "count": len(result), "hosts": result}


@app.post("/api/verify/{host_id}/approve", tags=["Verification"])
def api_admin_approve_host(host_id: str, notes: str = ""):
    """Admin manually approves a host, overriding automated checks.

    Sets host verification state to 'verified' regardless of check results.
    Useful when an admin has physically inspected hardware or reviewed logs.
    """
    ve = get_verification_engine()
    store = ve.store
    existing = store.get_verification(host_id)
    if not existing:
        # Create a new verification record for this host
        from verification import HostVerification, HostVerificationState
        existing = HostVerification(
            verification_id=str(uuid.uuid4())[:12],
            host_id=host_id,
            state=HostVerificationState.UNVERIFIED,
        )
    existing.state = "verified"
    existing.verified_at = time.time()
    existing.deverified_at = None
    existing.deverify_reason = ""
    existing.overall_score = 100.0
    existing.last_check_at = time.time()
    existing.next_check_at = time.time() + 86400
    store.save_verification(existing)
    log.info("ADMIN APPROVED host=%s notes=%s", host_id, notes or "(none)")
    emit_event("verification_override", {"host_id": host_id, "action": "approve", "notes": notes})
    return {"ok": True, "host_id": host_id, "status": "verified", "approved_by": "admin"}


@app.post("/api/verify/{host_id}/reject", tags=["Verification"])
def api_admin_reject_host(host_id: str, reason: str = "Admin rejection"):
    """Admin manually rejects/deverifies a host.

    Sets host verification state to 'deverified' so it cannot receive jobs.
    """
    ve = get_verification_engine()
    store = ve.store
    existing = store.get_verification(host_id)
    if not existing:
        from verification import HostVerification, HostVerificationState
        existing = HostVerification(
            verification_id=str(uuid.uuid4())[:12],
            host_id=host_id,
            state=HostVerificationState.UNVERIFIED,
        )
    existing.state = "deverified"
    existing.deverified_at = time.time()
    existing.deverify_reason = f"Admin: {reason}"
    existing.last_check_at = time.time()
    store.save_verification(existing)
    log.warning("ADMIN REJECTED host=%s reason=%s", host_id, reason)
    emit_event("verification_override", {"host_id": host_id, "action": "reject", "reason": reason})
    return {"ok": True, "host_id": host_id, "status": "deverified", "reason": reason}


# ── Jurisdiction ──────────────────────────────────────────────────────

class JurisdictionFilterRequest(BaseModel):
    canada_only: bool = True
    province: str = None
    trust_tier: str = None


@app.post("/api/jurisdiction/hosts", tags=["Jurisdiction"])
def api_jurisdiction_hosts(req: JurisdictionFilterRequest):
    """Filter hosts by jurisdiction constraints."""
    hosts = list_hosts(active_only=True)
    constraint = JurisdictionConstraint(
        canada_only=req.canada_only,
        province=req.province,
        trust_tier=TrustTier(req.trust_tier) if req.trust_tier else None,
    )
    from jurisdiction import filter_hosts_by_jurisdiction
    filtered = filter_hosts_by_jurisdiction(hosts, constraint)
    return {"ok": True, "count": len(filtered), "hosts": filtered}


@app.get("/api/jurisdiction/residency-trace/{job_id}", tags=["Jurisdiction"])
def api_residency_trace(job_id: str):
    """Generate a residency trace for a job (compliance artifact)."""
    jobs = list_jobs()
    job = next((j for j in jobs if j["job_id"] == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    host_data = {}
    if job.get("host_id"):
        hosts = list_hosts(active_only=False)
        host_data = next((h for h in hosts if h["host_id"] == job["host_id"]), {})

    # Build jurisdiction object from host data
    from jurisdiction import HostJurisdiction
    jurisdiction = None
    if host_data:
        jurisdiction = HostJurisdiction(
            country=host_data.get("country", "CA"),
            province=host_data.get("province", ""),
            city=host_data.get("city", ""),
        )
    trace = generate_residency_trace(
        job_id=job_id,
        host_id=job.get("host_id", ""),
        jurisdiction=jurisdiction,
        started_at=job.get("started_at", job.get("submitted_at", 0)) or 0,
        completed_at=job.get("completed_at") or time.time(),
    )
    return {"ok": True, "job_id": job_id, "trace": trace}


@app.get("/api/trust-tiers", tags=["Jurisdiction"])
def api_trust_tiers():
    """List available trust tiers and their requirements."""
    from jurisdiction import TRUST_TIER_REQUIREMENTS
    return {
        "ok": True,
        "tiers": {t.value: v for t, v in TRUST_TIER_REQUIREMENTS.items()},
    }


# ── Billing ───────────────────────────────────────────────────────────

@app.get("/api/billing/wallet/{customer_id}", tags=["Billing"])
def api_get_wallet(customer_id: str):
    """Get credit wallet balance and status."""
    be = get_billing_engine()
    wallet = be.get_wallet(customer_id)
    return {"ok": True, "wallet": wallet}


class DepositRequest(BaseModel):
    amount_cad: float
    description: str = "Credit deposit"


@app.post("/api/billing/wallet/{customer_id}/deposit", tags=["Billing"])
def api_deposit(customer_id: str, req: DepositRequest):
    """Deposit credits into a customer wallet."""
    be = get_billing_engine()
    result = be.deposit(customer_id, req.amount_cad, req.description)
    return {"ok": True, **result}


@app.get("/api/billing/wallet/{customer_id}/history", tags=["Billing"])
def api_wallet_history(customer_id: str, limit: int = 50):
    """Get transaction history for a wallet."""
    be = get_billing_engine()
    history = be.get_wallet_history(customer_id, limit)
    return {"ok": True, "customer_id": customer_id, "transactions": history}


@app.get("/api/billing/usage/{customer_id}", tags=["Billing"])
def api_usage_summary(customer_id: str, period_start: float = 0, period_end: float = 0):
    """Get usage summary for a customer."""
    if period_end == 0:
        period_end = time.time()
    if period_start == 0:
        period_start = period_end - 30 * 86400  # Last 30 days
    be = get_billing_engine()
    summary = be.get_usage_summary(customer_id, period_start, period_end)
    return {"ok": True, **summary}


@app.get("/api/billing/invoice/{customer_id}", tags=["Billing"])
def api_generate_invoice(customer_id: str, customer_name: str = "",
                         period_start: float = 0, period_end: float = 0,
                         tax_rate: float = 0.13):
    """Generate an AI Compute Access Fund–aligned invoice."""
    if period_end == 0:
        period_end = time.time()
    if period_start == 0:
        period_start = period_end - 30 * 86400
    be = get_billing_engine()
    invoice = be.generate_invoice(customer_id, customer_name, period_start, period_end, tax_rate)
    return {"ok": True, "invoice": invoice.to_dict()}


@app.get("/api/billing/export/caf/{customer_id}", tags=["Billing"])
def api_export_caf(customer_id: str, period_start: float = 0, period_end: float = 0,
                   format: str = "json"):
    """Export AI Compute Access Fund rebate documentation.

    From REPORT_FEATURE_2.md: /billing/export?format=caf
    Supports json and csv formats.
    """
    if period_end == 0:
        period_end = time.time()
    if period_start == 0:
        period_start = period_end - 30 * 86400
    be = get_billing_engine()

    if format == "csv":
        csv_data = be.export_caf_csv(customer_id, period_start, period_end)
        return StreamingResponse(
            iter([csv_data]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=xcelsior-caf-{customer_id}.csv"},
        )

    report = be.export_caf_report(customer_id, period_start, period_end)
    return {"ok": True, **report}


@app.get("/api/billing/invoices/{customer_id}", tags=["Billing"])
def api_list_invoices(customer_id: str, limit: int = 12):
    """List past invoices for a customer (monthly summaries).

    Generates monthly invoice stubs for the last N months showing
    total spend, tax, job count, and top GPUs used.
    """
    be = get_billing_engine()
    now = time.time()
    invoices = []
    for i in range(limit):
        period_end = now - (i * 30 * 86400)
        period_start = period_end - 30 * 86400
        try:
            inv = be.generate_invoice(customer_id, "", period_start, period_end, 0.13)
            inv_dict = inv.to_dict()
            # Only include months with actual usage
            if inv_dict.get("total_compute_cad", 0) > 0 or inv_dict.get("line_items"):
                invoices.append({
                    "invoice_id": f"INV-{customer_id[:8]}-{i+1:03d}",
                    "period_start": period_start,
                    "period_end": period_end,
                    "total_cad": inv_dict.get("total_with_tax_cad", inv_dict.get("total_compute_cad", 0)),
                    "subtotal_cad": inv_dict.get("total_compute_cad", 0),
                    "tax_cad": inv_dict.get("tax_cad", 0),
                    "tax_rate": inv_dict.get("tax_rate", 0.13),
                    "line_items": len(inv_dict.get("line_items", [])),
                    "caf_eligible_cad": inv_dict.get("caf_eligible_cad", 0),
                    "status": "paid",
                })
        except Exception:
            pass
    return {"ok": True, "invoices": invoices, "count": len(invoices)}


@app.get("/api/billing/attestation", tags=["Billing"])
def api_provider_attestation():
    """Get Xcelsior supplier attestation bundle for Fund claims."""
    be = get_billing_engine()
    attestation = be.generate_attestation()
    return {"ok": True, "attestation": attestation.to_dict()}


class RefundRequest(BaseModel):
    job_id: str
    exit_code: int
    failure_reason: str = ""


@app.post("/api/billing/refund", tags=["Billing"])
def api_process_refund(req: RefundRequest):
    """Process a refund for a failed job.

    From REPORT_FEATURE_1.md:
    - Hardware error → full refund
    - User OOM (exit 137) → zero refund
    """
    be = get_billing_engine()
    result = be.process_refund(req.job_id, req.exit_code, req.failure_reason)
    return {"ok": True, **result}


# ── Reputation ────────────────────────────────────────────────────────

@app.get("/api/reputation/{entity_id}", tags=["Reputation"])
def api_get_reputation(entity_id: str):
    """Get reputation score and tier for a host or user."""
    re = get_reputation_engine()
    score = re.compute_score(entity_id)
    return {"ok": True, "reputation": score.to_dict()}


@app.get("/api/reputation/leaderboard", tags=["Reputation"])
def api_reputation_leaderboard(entity_type: str = "host", limit: int = 20):
    """Top hosts/users by reputation score."""
    re = get_reputation_engine()
    board = re.get_leaderboard(entity_type, limit)
    return {"ok": True, "entity_type": entity_type, "leaderboard": board}


@app.get("/api/reputation/{entity_id}/history", tags=["Reputation"])
def api_reputation_history(entity_id: str, limit: int = 50):
    """Get reputation event history."""
    re = get_reputation_engine()
    history = re.store.get_event_history(entity_id, limit)
    return {"ok": True, "entity_id": entity_id, "events": history}


class VerificationGrant(BaseModel):
    entity_id: str
    verification_type: str  # email, phone, gov_id, hardware_audit, incorporation, data_center


@app.post("/api/reputation/verify", tags=["Reputation"])
def api_grant_verification(req: VerificationGrant):
    """Grant a verification badge to a host/user."""
    try:
        vtype = VerificationType(req.verification_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid verification type: {req.verification_type}")
    re = get_reputation_engine()
    score = re.add_verification(req.entity_id, vtype)
    return {"ok": True, "reputation": score.to_dict()}


# ── Pricing & Estimation ─────────────────────────────────────────────

class EstimateRequest(BaseModel):
    gpu_model: str = "RTX 4090"
    duration_hours: float = 1.0
    spot: bool = False
    sovereignty: bool = False
    is_canadian: bool = True


@app.post("/api/pricing/estimate", tags=["Billing"])
def api_estimate_cost(req: EstimateRequest):
    """Estimate job cost with AI Compute Access Fund rebate preview.

    From REPORT_FEATURE_2.md: --estimate-rebate / simulate=true
    """
    estimate = estimate_job_cost(
        req.gpu_model, req.duration_hours,
        spot=req.spot, sovereignty=req.sovereignty,
        is_canadian=req.is_canadian,
    )
    return {"ok": True, **estimate}


@app.get("/api/pricing/reference", tags=["Billing"])
def api_reference_pricing():
    """Get reference GPU pricing table in CAD."""
    return {"ok": True, "currency": "CAD", "pricing": GPU_REFERENCE_PRICING_CAD}


# ── Reserved Pricing ─────────────────────────────────────────────────
# Per Report #1.B: "Reserved: Discounts for 1-month or 1-year terms."
# Complements on-demand (standard POST /job) and spot (POST /spot/job).

RESERVED_PRICING_TIERS = {
    "1_month": {
        "commitment": "1 month",
        "discount_pct": 20,
        "description": "20% off on-demand rates for 1-month commitment",
        "min_hours_per_day": 4,
    },
    "3_month": {
        "commitment": "3 months",
        "discount_pct": 30,
        "description": "30% off on-demand rates for 3-month commitment",
        "min_hours_per_day": 4,
    },
    "1_year": {
        "commitment": "1 year",
        "discount_pct": 45,
        "description": "45% off on-demand rates for 1-year commitment",
        "min_hours_per_day": 0,
    },
}


class ReservedCommitmentRequest(BaseModel):
    customer_id: str
    gpu_model: str = "RTX 4090"
    commitment_type: str = "1_month"  # 1_month | 3_month | 1_year
    quantity: int = 1  # number of GPU slots reserved
    province: str = "ON"


@app.get("/api/pricing/reserved-plans", tags=["Billing"])
def api_reserved_plans():
    """List available reserved pricing tiers with discount percentages.

    Three commitment levels:
    - **1_month**: 20% discount, minimum 4 hrs/day usage
    - **3_month**: 30% discount, minimum 4 hrs/day usage
    - **1_year**: 45% discount, no minimum daily usage

    Compare with on-demand (`POST /job`) and spot/interruptible (`POST /spot/job`).
    """
    # Enrich each tier with sample pricing based on reference GPU pricing
    enriched = {}
    for tier_key, tier in RESERVED_PRICING_TIERS.items():
        samples = {}
        for gpu, ref in GPU_REFERENCE_PRICING_CAD.items():
            rate = ref.get("base_rate_cad", ref.get("cad_per_hour", 0)) if isinstance(ref, dict) else ref
            samples[gpu] = round(rate * (1 - tier["discount_pct"] / 100), 4)
        enriched[tier_key] = {**tier, "sample_hourly_rates_cad": samples}
    return {"ok": True, "currency": "CAD", "reserved_tiers": enriched}


@app.post("/api/pricing/reserve", tags=["Billing"])
def api_reserve_commitment(req: ReservedCommitmentRequest):
    """Create a reserved pricing commitment for a customer.

    Reserved instances are 20-45% cheaper than on-demand, depending on
    commitment length. The customer pre-commits to a term and receives
    a guaranteed discount on all GPU hours consumed during that period.
    """
    tier = RESERVED_PRICING_TIERS.get(req.commitment_type)
    if not tier:
        raise HTTPException(400, f"Invalid commitment_type: {req.commitment_type}. "
                           f"Valid: {list(RESERVED_PRICING_TIERS.keys())}")

    # Calculate pricing
    ref_pricing = GPU_REFERENCE_PRICING_CAD.get(req.gpu_model, {})
    base_rate = ref_pricing.get("base_rate_cad", ref_pricing.get("cad_per_hour", 0)) if isinstance(ref_pricing, dict) else (
        ref_pricing if isinstance(ref_pricing, (int, float)) else 0
    )
    if base_rate <= 0:
        raise HTTPException(400, f"Unknown GPU model: {req.gpu_model}")

    discounted_rate = round(base_rate * (1 - tier["discount_pct"] / 100), 4)
    tax_rate, tax_desc = get_tax_rate_for_province(req.province)

    commitment = {
        "commitment_id": str(uuid.uuid4()),
        "customer_id": req.customer_id,
        "commitment_type": req.commitment_type,
        "gpu_model": req.gpu_model,
        "quantity": req.quantity,
        "base_rate_cad": base_rate,
        "discounted_rate_cad": discounted_rate,
        "discount_pct": tier["discount_pct"],
        "province": req.province,
        "tax_rate": tax_rate,
        "tax_description": tax_desc,
        "commitment_description": tier["description"],
        "min_hours_per_day": tier["min_hours_per_day"],
        "created_at": time.time(),
        "status": "active",
    }

    # Deposit placeholder — in production, charge upfront or set up recurring billing
    billing = get_billing_engine()
    monthly_estimate = discounted_rate * req.quantity * 24 * 30
    commitment["monthly_estimate_cad"] = round(monthly_estimate, 2)
    commitment["monthly_estimate_with_tax_cad"] = round(monthly_estimate * (1 + tax_rate), 2)

    broadcast_sse("reservation_created", {
        "commitment_id": commitment["commitment_id"],
        "customer_id": req.customer_id,
        "type": req.commitment_type,
    })
    return {"ok": True, **commitment}


# ── GST/HST Small-Supplier Threshold ─────────────────────────────────
# Per Report #1.B: "$30,000 Threshold — Xcelsior must register for GST/HST
# once total revenue exceeds $30k over four consecutive quarters."

GST_SMALL_SUPPLIER_THRESHOLD_CAD = 30_000.00


@app.get("/api/billing/gst-threshold", tags=["Compliance"])
def api_gst_threshold_status():
    """Check platform-wide GST/HST small-supplier threshold status.

    Under the Excise Tax Act, a distribution platform operator **must**
    register for GST/HST once total taxable revenue exceeds $30,000 CAD
    over any four consecutive calendar quarters.

    Returns:
    - `exceeded`: whether the $30k threshold is passed
    - `total_revenue_cad`: estimated revenue from all billing
    - `threshold_cad`: the $30,000 statutory limit
    - `quarters_assessed`: number of quarters with data
    """
    billing = get_billing_engine()
    now = time.time()
    # Look back 4 quarters (~365 days)
    one_year_ago = now - (365.25 * 86400)

    try:
        with billing._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(total_cost_cad), 0) AS total "
                "FROM usage_meters WHERE started_at >= ?",
                (one_year_ago,),
            ).fetchone()
            total_rev = row["total"] if row else 0.0

            # Count distinct quarters
            qrow = conn.execute(
                "SELECT COUNT(DISTINCT (CAST(strftime('%%Y', started_at, 'unixepoch') AS INT) * 4 "
                "+ CAST(strftime('%%m', started_at, 'unixepoch') AS INT) / 4)) AS q_count "
                "FROM usage_meters WHERE started_at >= ?",
                (one_year_ago,),
            ).fetchone()
            quarters = qrow["q_count"] if qrow else 0
    except Exception:
        total_rev = 0.0
        quarters = 0

    exceeded = total_rev >= GST_SMALL_SUPPLIER_THRESHOLD_CAD
    return {
        "ok": True,
        "exceeded": exceeded,
        "total_revenue_cad": round(total_rev, 2),
        "threshold_cad": GST_SMALL_SUPPLIER_THRESHOLD_CAD,
        "quarters_assessed": quarters,
        "must_register": exceeded,
        "message": (
            "GST/HST registration REQUIRED — revenue exceeds $30,000 threshold."
            if exceeded
            else f"Below threshold (${total_rev:,.2f} / $30,000). "
                 "Registration not yet required but recommended."
        ),
    }


@app.get("/api/billing/gst-threshold/{provider_id}", tags=["Compliance"])
def api_provider_gst_threshold(provider_id: str):
    """Check whether a specific provider has exceeded the $30,000 GST/HST
    small-supplier threshold based on their historical payouts.

    Used by providers to determine if they need to independently register
    for GST/HST. The simplified regime is recommended for non-resident
    providers serving Canadians.
    """
    billing = get_billing_engine()
    now = time.time()
    one_year_ago = now - (365.25 * 86400)

    try:
        with billing._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(provider_payout_cad), 0) AS total "
                "FROM payout_ledger WHERE provider_id = ? AND created_at >= ?",
                (provider_id, one_year_ago),
            ).fetchone()
            total_payouts = row["total"] if row else 0.0
    except Exception:
        total_payouts = 0.0

    exceeded = total_payouts >= GST_SMALL_SUPPLIER_THRESHOLD_CAD
    return {
        "ok": True,
        "provider_id": provider_id,
        "exceeded": exceeded,
        "total_payouts_cad": round(total_payouts, 2),
        "threshold_cad": GST_SMALL_SUPPLIER_THRESHOLD_CAD,
        "must_register_gst": exceeded,
        "message": (
            "Provider should register for GST/HST — payouts exceed $30,000."
            if exceeded
            else f"Below threshold (${total_payouts:,.2f} / $30,000)."
        ),
        "simplified_regime_eligible": True,
    }


# ── Usage Analytics ──────────────────────────────────────────────────
# Per Report #1.B Phase 3: "Usage Analytics Dashboard — Providing both
# providers and submitters with deep insights into cost, performance,
# and hardware health over time."

@app.get("/api/analytics/usage", tags=["Billing"])
def api_usage_analytics(
    customer_id: str = "",
    provider_id: str = "",
    days: int = 30,
    group_by: str = "day",  # day | week | gpu_model | province
):
    """Usage analytics for both providers and submitters.

    Provides cost breakdowns, GPU utilization trends, and hardware health
    aggregates over time. Supports grouping by day, week, GPU model,
    or province for detailed reporting.

    Query params:
    - `customer_id` — filter to one customer (submitter view)
    - `provider_id` — filter to one provider (earnings view)
    - `days` — lookback window (default 30)
    - `group_by` — aggregation: `day`, `week`, `gpu_model`, `province`
    """
    billing = get_billing_engine()
    now = time.time()
    since = now - (days * 86400)

    group_sql = {
        "day": "date(started_at, 'unixepoch') AS period",
        "week": "strftime('%%Y-W%%W', started_at, 'unixepoch') AS period",
        "gpu_model": "gpu_model AS period",
        "province": "province AS period",
    }.get(group_by, "date(started_at, 'unixepoch') AS period")

    where_clauses = ["started_at >= ?"]
    params: list = [since]
    if customer_id:
        where_clauses.append("owner = ?")
        params.append(customer_id)
    # Provider filter: match host_id (providers are hosts)
    if provider_id:
        where_clauses.append("host_id = ?")
        params.append(provider_id)

    where_sql = " AND ".join(where_clauses)

    try:
        with billing._conn() as conn:
            rows = conn.execute(
                f"SELECT {group_sql}, "
                "COUNT(*) AS job_count, "
                "ROUND(SUM(total_cost_cad), 2) AS total_cost_cad, "
                "ROUND(SUM(gpu_seconds), 0) AS total_gpu_seconds, "
                "ROUND(AVG(gpu_utilization_pct), 1) AS avg_gpu_util_pct, "
                "SUM(is_canadian_compute) AS canadian_jobs, "
                "COUNT(*) - SUM(is_canadian_compute) AS international_jobs "
                f"FROM usage_meters WHERE {where_sql} "
                "GROUP BY period ORDER BY period",
                params,
            ).fetchall()

            analytics = [
                {
                    "period": r["period"],
                    "job_count": r["job_count"],
                    "total_cost_cad": r["total_cost_cad"],
                    "total_gpu_hours": round(r["total_gpu_seconds"] / 3600, 2) if r["total_gpu_seconds"] else 0,
                    "avg_gpu_utilization_pct": r["avg_gpu_util_pct"],
                    "canadian_jobs": r["canadian_jobs"],
                    "international_jobs": r["international_jobs"],
                }
                for r in rows
            ]

            # Summary
            summary_row = conn.execute(
                "SELECT COUNT(*) AS total_jobs, "
                "ROUND(SUM(total_cost_cad), 2) AS total_spend, "
                "ROUND(SUM(gpu_seconds) / 3600.0, 2) AS total_gpu_hours, "
                "ROUND(AVG(gpu_utilization_pct), 1) AS avg_util "
                f"FROM usage_meters WHERE {where_sql}",
                params,
            ).fetchone()
    except Exception as e:
        return {"ok": False, "error": str(e), "analytics": [], "summary": {}}

    return {
        "ok": True,
        "days": days,
        "group_by": group_by,
        "analytics": analytics,
        "summary": {
            "total_jobs": summary_row["total_jobs"] if summary_row else 0,
            "total_spend_cad": summary_row["total_spend"] if summary_row else 0,
            "total_gpu_hours": summary_row["total_gpu_hours"] if summary_row else 0,
            "avg_gpu_utilization_pct": summary_row["avg_util"] if summary_row else 0,
        },
    }


# ── Artifacts ─────────────────────────────────────────────────────────

class UploadRequest(BaseModel):
    job_id: str
    filename: str
    artifact_type: str = "job_output"
    residency_policy: str = "canada_only"


@app.post("/api/artifacts/upload", tags=["Artifacts"])
def api_request_upload(req: UploadRequest):
    """Get a presigned upload URL for an artifact."""
    from artifacts import ArtifactType, ResidencyPolicy
    try:
        atype = ArtifactType(req.artifact_type)
        rpolicy = ResidencyPolicy(req.residency_policy)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    mgr = get_artifact_manager()
    result = mgr.request_upload(req.job_id, req.filename, atype, rpolicy)
    return {"ok": True, **result}


class DownloadRequest(BaseModel):
    job_id: str
    filename: str
    artifact_type: str = "job_output"


@app.post("/api/artifacts/download", tags=["Artifacts"])
def api_request_download(req: DownloadRequest):
    """Get a presigned download URL for an artifact."""
    from artifacts import ArtifactType
    try:
        atype = ArtifactType(req.artifact_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    mgr = get_artifact_manager()
    result = mgr.request_download(req.job_id, req.filename, atype)
    return {"ok": True, **result}


@app.get("/api/artifacts/{job_id}", tags=["Artifacts"])
def api_list_artifacts(job_id: str):
    """List all artifacts for a job."""
    mgr = get_artifact_manager()
    artifacts = mgr.get_job_artifacts(job_id)
    return {"ok": True, "job_id": job_id, "artifacts": artifacts}


# ── Sovereign Queue Processing ───────────────────────────────────────

class SovereignQueueRequest(BaseModel):
    canada_only: bool = True
    province: str = None
    trust_tier: str = None


@app.post("/api/queue/process-sovereign", tags=["Jurisdiction"])
def api_process_queue_sovereign(req: SovereignQueueRequest):
    """Process queue with jurisdiction + verification + reputation awareness."""
    assigned = process_queue_sovereign(
        canada_only=req.canada_only,
        province=req.province,
        trust_tier=req.trust_tier,
    )
    results = []
    for job, host in assigned:
        results.append({
            "job_id": job["job_id"],
            "job_name": job.get("name"),
            "host_id": host["host_id"],
            "gpu_model": host.get("gpu_model"),
            "country": host.get("country", ""),
        })
        broadcast_sse("job_assigned", {
            "job_id": job["job_id"],
            "host_id": host["host_id"],
        })
    return {"ok": True, "assigned": len(results), "jobs": results}


# ═══ Province Compliance Matrix ══════════════════════════════════════
# REPORT_MARKETING_FINAL.md: "maintaining a small policy matrix embedded
# in the scheduler product and documentation"

@app.get("/api/compliance/provinces", tags=["Compliance"])
def api_compliance_provinces():
    """Province-specific compliance matrix for scheduling guidance."""
    matrix = {}
    for prov, info in PROVINCE_COMPLIANCE.items():
        prov_code = prov.value if hasattr(prov, "value") else str(prov)
        tax_rate, tax_desc = get_tax_rate_for_province(prov_code)
        matrix[prov_code] = {
            **info,
            "tax_rate": tax_rate,
            "tax_description": tax_desc,
        }
    return {"provinces": matrix}


@app.get("/api/compliance/tax-rates", tags=["Compliance"])
def api_tax_rates():
    """Canadian GST/HST/PST rates by province for billing."""
    return {
        "rates": {
            code: {"rate": rate, "description": desc}
            for code, (rate, desc) in PROVINCE_TAX_RATES.items()
        }
    }


@app.get("/api/compliance/trust-tier-requirements", tags=["Compliance"])
def api_trust_tier_requirements():
    """Full trust tier requirements matrix."""
    return {"tiers": [
        {"tier": tier.value, **reqs}
        for tier, reqs in TRUST_TIER_REQUIREMENTS.items()
    ]}


# ═══ Québec Law 25 PIA Check ════════════════════════════════════════

class PIACheckRequest(BaseModel):
    data_origin_province: str = "QC"
    processing_province: str = "ON"
    data_contains_pi: bool = False


@app.post("/api/compliance/quebec-pia-check", tags=["Compliance"])
def api_quebec_pia_check(req: PIACheckRequest):
    """Check if Québec Law 25 PIA is required for cross-border transfer."""
    return requires_quebec_pia(
        req.data_origin_province,
        req.processing_province,
        req.data_contains_pi,
    )


# ═══ Privacy Controls ════════════════════════════════════════════════
# REPORT_FEATURE_FINAL.md § "Privacy-by-default and governance hooks"

@app.get("/api/privacy/retention-policies", tags=["Privacy"])
def api_retention_policies():
    """Data retention policies per PIPEDA fair information principles."""
    policies = {}
    for cat, policy in RETENTION_POLICIES.items():
        cat_key = cat.value if hasattr(cat, "value") else str(cat)
        policies[cat_key] = {
            "retention_days": policy["retention_sec"] // 86400,
            "description": policy["description"],
            "redact_on_completion": policy.get("redact_on_completion", False),
        }
    return {"policies": policies}


@app.get("/api/privacy/retention-summary", tags=["Privacy"])
def api_retention_summary():
    """Current retention status across all data categories."""
    lm = get_lifecycle_manager()
    return lm.get_retention_summary()


@app.post("/api/privacy/purge-expired", tags=["Privacy"])
def api_purge_expired():
    """Purge all expired retention records (daily maintenance)."""
    lm = get_lifecycle_manager()
    count = lm.purge_expired()
    return {"ok": True, "purged": count}


class PrivacyConfigRequest(BaseModel):
    org_id: str
    privacy_level: str = "strict"
    privacy_officer_name: str = ""
    privacy_officer_email: str = ""
    enable_identification: bool = False
    enable_location_tracking: bool = False
    enable_profiling: bool = False
    redact_pii_in_logs: bool = True
    redact_env_vars: bool = True
    redact_ip_addresses: bool = True
    log_retention_days: int = None
    telemetry_retention_days: int = None


@app.post("/api/privacy/config", tags=["Privacy"])
def api_save_privacy_config(req: PrivacyConfigRequest):
    """Save privacy configuration for an organization."""
    lm = get_lifecycle_manager()
    config = PrivacyConfig(
        privacy_level=req.privacy_level,
        privacy_officer_name=req.privacy_officer_name,
        privacy_officer_email=req.privacy_officer_email,
        privacy_officer_designated=bool(req.privacy_officer_name),
        enable_identification=req.enable_identification,
        enable_location_tracking=req.enable_location_tracking,
        enable_profiling=req.enable_profiling,
        redact_pii_in_logs=req.redact_pii_in_logs,
        redact_env_vars=req.redact_env_vars,
        redact_ip_addresses=req.redact_ip_addresses,
        log_retention_days=req.log_retention_days,
        telemetry_retention_days=req.telemetry_retention_days,
    )
    lm.save_config(req.org_id, config)
    return {"ok": True, "org_id": req.org_id, "privacy_level": req.privacy_level}


@app.get("/api/privacy/config/{org_id}", tags=["Privacy"])
def api_get_privacy_config(org_id: str):
    """Get privacy configuration for an organization (defaults to STRICT)."""
    lm = get_lifecycle_manager()
    config = lm.get_config(org_id)
    return config.to_dict()


class ConsentRequest(BaseModel):
    entity_id: str
    consent_type: str  # "cross_border", "data_collection", "telemetry", "profiling"
    details: dict = None


@app.post("/api/privacy/consent", tags=["Privacy"])
def api_record_consent(req: ConsentRequest):
    """Record explicit consent (PIPEDA principle: Consent)."""
    lm = get_lifecycle_manager()
    consent_id = lm.record_consent(req.entity_id, req.consent_type, req.details)
    return {"ok": True, "consent_id": consent_id}


@app.delete("/api/privacy/consent/{entity_id}/{consent_type}", tags=["Privacy"])
def api_revoke_consent(entity_id: str, consent_type: str):
    """Revoke consent (PIPEDA: individuals can withdraw consent)."""
    lm = get_lifecycle_manager()
    lm.revoke_consent(entity_id, consent_type)
    return {"ok": True, "revoked": consent_type}


@app.get("/api/privacy/consent/{entity_id}", tags=["Privacy"])
def api_get_consents(entity_id: str):
    """Get all consent records for an entity (PIPEDA: Individual Access)."""
    lm = get_lifecycle_manager()
    consents = lm.get_consents(entity_id)
    return {"consents": consents}


# ── Transparency Report (REPORT_FEATURE_2.md Phase B §3) ─────────────
# "Log all access/subpoenas in DB; API /transparency/report"
# Tracks legal requests, data disclosures, and CLOUD Act diligence.

_transparency_db_path = os.path.join(os.path.dirname(__file__), "xcelsior_transparency.db")


def _get_transparency_db():
    """Lazy-init transparency SQLite DB."""
    import sqlite3
    conn = sqlite3.connect(_transparency_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS legal_requests (
            request_id TEXT PRIMARY KEY,
            received_at REAL NOT NULL,
            request_type TEXT NOT NULL,
            jurisdiction TEXT DEFAULT '',
            authority TEXT DEFAULT '',
            scope TEXT DEFAULT '',
            status TEXT DEFAULT 'received',
            responded_at REAL DEFAULT 0,
            complied INTEGER DEFAULT 0,
            challenged INTEGER DEFAULT 0,
            notes TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS data_disclosures (
            disclosure_id TEXT PRIMARY KEY,
            request_id TEXT,
            disclosed_at REAL NOT NULL,
            data_category TEXT DEFAULT '',
            record_count INTEGER DEFAULT 0,
            entities_affected INTEGER DEFAULT 0,
            was_mandatory INTEGER DEFAULT 0,
            notes TEXT DEFAULT ''
        )
    """)
    conn.commit()
    return conn


class LegalRequestRecord(BaseModel):
    request_type: str = "subpoena"   # subpoena, warrant, mlat, production_order, informal
    jurisdiction: str = "CA"
    authority: str = ""
    scope: str = ""
    notes: str = ""


@app.post("/api/transparency/legal-request", tags=["Transparency"])
def api_record_legal_request(req: LegalRequestRecord):
    """Record a legal request (subpoena, warrant, MLAT, etc.)."""
    import uuid
    conn = _get_transparency_db()
    request_id = str(uuid.uuid4())[:12]
    conn.execute(
        """INSERT INTO legal_requests
           (request_id, received_at, request_type, jurisdiction, authority, scope, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (request_id, time.time(), req.request_type, req.jurisdiction,
         req.authority, req.scope, req.notes),
    )
    conn.commit()
    conn.close()

    # Also record as an auditable event in the hash chain
    store = get_event_store()
    store.append(Event(
        event_type="transparency.legal_request",
        entity_type="legal",
        entity_id=request_id,
        actor="admin",
        data={"request_type": req.request_type, "jurisdiction": req.jurisdiction},
    ))

    return {"ok": True, "request_id": request_id}


@app.post("/api/transparency/legal-request/{request_id}/respond", tags=["Transparency"])
def api_respond_legal_request(request_id: str, complied: bool = False,
                               challenged: bool = False, notes: str = ""):
    """Record response to a legal request."""
    conn = _get_transparency_db()
    conn.execute(
        """UPDATE legal_requests
           SET status = 'responded', responded_at = ?, complied = ?, challenged = ?, notes = ?
           WHERE request_id = ?""",
        (time.time(), int(complied), int(challenged), notes, request_id),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "request_id": request_id}


@app.get("/api/transparency/report", tags=["Transparency"])
def api_transparency_report(months: int = 12):
    """Generate transparency report — CLOUD Act diligence artifact.

    Returns summary of all legal requests and data disclosures.
    Monthly JSON per REPORT_FEATURE_2.md Phase B §3.
    """
    conn = _get_transparency_db()
    since = time.time() - (months * 30 * 86400)

    requests_rows = conn.execute(
        "SELECT * FROM legal_requests WHERE received_at >= ? ORDER BY received_at DESC",
        (since,),
    ).fetchall()

    disclosures_rows = conn.execute(
        "SELECT * FROM data_disclosures WHERE disclosed_at >= ? ORDER BY disclosed_at DESC",
        (since,),
    ).fetchall()
    conn.close()

    requests_list = [dict(r) for r in requests_rows]
    disclosures_list = [dict(r) for r in disclosures_rows]

    # Summary statistics
    total = len(requests_list)
    complied = sum(1 for r in requests_list if r.get("complied"))
    challenged = sum(1 for r in requests_list if r.get("challenged"))
    by_type = {}
    for r in requests_list:
        t = r.get("request_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
    by_jurisdiction = {}
    for r in requests_list:
        j = r.get("jurisdiction", "unknown")
        by_jurisdiction[j] = by_jurisdiction.get(j, 0) + 1

    return {
        "ok": True,
        "period_months": months,
        "generated_at": time.time(),
        "summary": {
            "requests_received": total,
            "complied": complied,
            "challenged": challenged,
            "pending": total - complied - challenged,
            "by_type": by_type,
            "by_jurisdiction": by_jurisdiction,
            "data_disclosures": len(disclosures_list),
        },
        "cloud_act_note": (
            "Xcelsior is a Canadian-controlled entity. All data resides in Canadian "
            "jurisdiction. Foreign legal process requires MLAT through Canadian courts. "
            "No US CLOUD Act compelled disclosure has been made."
        ),
        "requests": requests_list,
        "disclosures": disclosures_list,
    }


# ── Tamper-Evident Audit Verification (REPORT_FEATURE_2.md Phase C §1) ──

@app.get("/api/audit/verify-chain", tags=["Events"])
def api_verify_event_chain():
    """Verify the tamper-evident hash chain on all events.

    Returns chain integrity status. If any event was modified after
    being written, the chain will report the break point.
    """
    store = get_event_store()
    result = store.verify_chain()
    return {"ok": True, "chain_integrity": result}


@app.get("/api/audit/job/{job_id}", tags=["Events"])
def api_job_audit_trail(job_id: str):
    """Full auditable trail for a job — every event with hash chain.

    This is the dispute-resolution artifact: every state change,
    lease renewal, billing event, ordered by time with tamper-evident hashes.
    """
    sm = get_state_machine()
    timeline = sm.get_job_timeline(job_id)
    if not timeline:
        raise HTTPException(404, f"No events for job {job_id}")
    return {"ok": True, "job_id": job_id, "events": timeline, "count": len(timeline)}


# ── Agent Telemetry Endpoint (REPORT_FEATURE_2.md Phase A §3) ─────────
# "Agent endpoint /agent/telemetry pushes JSON: utilization, temp, memory_errors"

_host_telemetry: dict[str, dict] = {}  # host_id -> latest metrics


class TelemetryPayload(BaseModel):
    host_id: str
    timestamp: float = 0
    metrics: dict = {}


@app.post("/agent/telemetry", tags=["Telemetry"])
def api_agent_telemetry(payload: TelemetryPayload):
    """Receive periodic GPU telemetry from agent (every 5s)."""
    _host_telemetry[payload.host_id] = {
        "timestamp": payload.timestamp or time.time(),
        "metrics": payload.metrics,
        "received_at": time.time(),
    }
    return {"ok": True}


@app.get("/agent/telemetry/{host_id}", tags=["Telemetry"])
def api_get_telemetry(host_id: str):
    """Get latest telemetry for a host (dashboard live gauges)."""
    if host_id not in _host_telemetry:
        raise HTTPException(404, f"No telemetry for host {host_id}")

    data = _host_telemetry[host_id]
    stale = (time.time() - data.get("received_at", 0)) > 30  # >30s = stale
    return {"ok": True, "host_id": host_id, "stale": stale, **data}


@app.get("/api/telemetry/all", tags=["Telemetry"])
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


# ── Agent Verification Endpoint ───────────────────────────────────────
# Full verification report from agent benchmark → verification.py checks

class VerificationReportPayload(BaseModel):
    host_id: str
    report: dict


@app.post("/agent/verify", tags=["Verification"])
def api_agent_verify(payload: VerificationReportPayload):
    """Receive comprehensive benchmark report and run verification checks."""
    ve = get_verification_engine()
    result = ve.run_verification(payload.host_id, payload.report)
    return {
        "ok": True,
        "host_id": payload.host_id,
        "state": result.state.value if hasattr(result.state, 'value') else str(result.state),
        "score": result.overall_score,
        "checks": result.checks,
        "gpu_fingerprint": result.gpu_fingerprint,
    }


# ═══════════════════════════════════════════════════════════════════════
# v2.2 API — SLA Enforcement, Stripe Connect, Provider Onboarding
# Per REPORT_FEATURE_1.md (Report #1.B)
# ═══════════════════════════════════════════════════════════════════════


# ── SLA Enforcement (Report #1.B: "SLA Enforcement" section) ─────────

class SLAEnforceRequest(BaseModel):
    host_id: str
    month: str            # YYYY-MM
    tier: str = "community"
    monthly_spend_cad: float = 0.0


@app.post("/api/sla/enforce", tags=["SLA"])
def api_sla_enforce(req: SLAEnforceRequest):
    """Run monthly SLA enforcement for a host.

    Calculates uptime percentage, downtime incidents, and credits owed
    based on the SLA tier. Credits follow the Google Cloud / Azure model:
    - 95–99% uptime → 10% credit
    - 90–95% uptime → 25% credit
    - <90% uptime   → 100% credit
    """
    engine = get_sla_engine()
    record = engine.enforce_monthly(
        req.host_id, req.tier, req.month, req.monthly_spend_cad,
    )
    return {
        "ok": True,
        "host_id": record.host_id,
        "month": record.month,
        "tier": record.tier,
        "uptime_pct": round(record.uptime_pct, 4),
        "downtime_seconds": record.downtime_seconds,
        "incidents": record.incidents,
        "credit_pct": record.credit_pct,
        "credit_cad": record.credit_cad,
    }


@app.get("/api/sla/hosts-summary", tags=["SLA"])
def api_sla_hosts_summary():
    """Get SLA status summary for all known hosts.

    Returns per-host cards with uptime %, violation count, and SLA tier.
    Used by dashboard UI-8.1 SLA Dashboard.
    """
    engine = get_sla_engine()
    import scheduler as _sched
    hosts = _sched.list_hosts(active_only=False)
    summaries = []
    for h in hosts:
        hid = h.get("host_id", "")
        if not hid:
            continue
        uptime = engine.get_host_uptime_pct(hid)
        violations = engine.get_violations(hid)
        tier = h.get("sla_tier", "community")
        summaries.append({
            "host_id": hid,
            "gpu_model": h.get("gpu_model", "Unknown"),
            "status": h.get("status", "unknown"),
            "sla_tier": tier,
            "uptime_30d_pct": round(uptime, 4),
            "violation_count": len(violations),
            "last_violation": violations[-1] if violations else None,
            "country": h.get("country", ""),
            "province": h.get("province", ""),
        })
    return {"ok": True, "hosts": summaries, "count": len(summaries)}


@app.get("/api/sla/{host_id}", tags=["SLA"])
def api_sla_status(host_id: str, month: str = ""):
    """Get SLA record and rolling uptime for a host."""
    engine = get_sla_engine()
    uptime_30d = engine.get_host_uptime_pct(host_id)
    record = None
    if month:
        rec = engine.get_host_sla(host_id, month)
        record = {
            "month": rec.month, "tier": rec.tier,
            "uptime_pct": round(rec.uptime_pct, 4),
            "downtime_seconds": rec.downtime_seconds,
            "incidents": rec.incidents,
            "credit_pct": rec.credit_pct,
            "credit_cad": rec.credit_cad,
        } if rec else None
    return {
        "ok": True,
        "host_id": host_id,
        "uptime_30d_pct": round(uptime_30d, 4),
        "monthly_record": record,
    }


@app.get("/api/sla/violations/{host_id}", tags=["SLA"])
def api_sla_violations(host_id: str, since: float = 0):
    """Get SLA violation history for a host."""
    engine = get_sla_engine()
    violations = engine.get_violations(host_id, since)
    return {"ok": True, "host_id": host_id, "violations": violations, "count": len(violations)}


@app.get("/api/sla/downtimes", tags=["SLA"])
def api_sla_active_downtimes():
    """Get all currently-open downtime periods across all hosts."""
    engine = get_sla_engine()
    downtimes = engine.get_active_downtimes()
    return {"ok": True, "downtimes": downtimes, "count": len(downtimes)}


@app.get("/api/sla/targets", tags=["SLA"])
def api_sla_targets():
    """Get SLA target definitions for all tiers."""
    from dataclasses import asdict
    targets = {t.value: asdict(v) for t, v in SLA_TARGETS.items()}
    return {"ok": True, "targets": targets}


# ── Provider Onboarding (Report #1.B: Stripe Connect + Canadian Co) ──

class ProviderRegisterRequest(BaseModel):
    provider_id: str
    email: str
    provider_type: str = "individual"  # "individual" or "company"
    corporation_name: str = ""         # Required for company type
    business_number: str = ""          # CRA Business Number (BN)
    gst_hst_number: str = ""           # GST/HST registration number
    province: str = ""                 # ON, QC, BC, AB, etc.
    legal_name: str = ""               # Legal name of individual or entity


class IncorporationUploadRequest(BaseModel):
    file_id: str  # Reference to file uploaded via /api/artifacts/upload


@app.post("/api/providers/register", tags=["Providers"])
def api_register_provider(req: ProviderRegisterRequest):
    """Register a GPU provider with Stripe Connect onboarding.

    For Canadian companies, include corporation_name, business_number,
    and gst_hst_number. Returns a Stripe onboarding URL for KYC completion.

    Per Report #1.B "Five Pillars of Compliance":
    1. Identity Verification (Stripe Identity)
    2. Financial Enrollment (bank details via Stripe Express)
    3. Credentialing (GPU/bandwidth checked at admission)
    4. Tax Compliance (GST/HST auto-collected per province)
    """
    if req.provider_type == "company" and not req.corporation_name:
        raise HTTPException(400, "corporation_name required for company providers")

    mgr = get_stripe_manager()
    result = mgr.create_provider_account(
        provider_id=req.provider_id,
        email=req.email,
        provider_type=req.provider_type,
        corporation_name=req.corporation_name,
        business_number=req.business_number,
        gst_hst_number=req.gst_hst_number,
        province=req.province,
        legal_name=req.legal_name,
    )
    broadcast_sse("provider_registered", {
        "provider_id": req.provider_id,
        "type": req.provider_type,
        "corporation_name": req.corporation_name,
    })
    return {"ok": True, **result}


@app.get("/api/providers/{provider_id}", tags=["Providers"])
def api_get_provider(provider_id: str):
    """Get provider account details including company info and payout status."""
    mgr = get_stripe_manager()
    provider = mgr.get_provider(provider_id)
    if not provider:
        raise HTTPException(404, f"Provider {provider_id} not found")
    # Redact sensitive fields
    provider.pop("stripe_account_id", None)
    return {"ok": True, "provider": provider}


@app.get("/api/providers", tags=["Providers"])
def api_list_providers(status: str = ""):
    """List all provider accounts, optionally filtered by status."""
    mgr = get_stripe_manager()
    providers = mgr.list_providers(status)
    # Redact Stripe IDs
    for p in providers:
        p.pop("stripe_account_id", None)
    return {"ok": True, "providers": providers, "count": len(providers)}


@app.post("/api/providers/{provider_id}/incorporation", tags=["Providers"])
def api_upload_incorporation(provider_id: str, req: IncorporationUploadRequest):
    """Link an uploaded incorporation document to a provider account.

    The file itself should first be uploaded via POST /api/artifacts/upload
    with artifact_type='incorporation_doc'. Then pass the resulting file_id here.
    """
    mgr = get_stripe_manager()
    provider = mgr.get_provider(provider_id)
    if not provider:
        raise HTTPException(404, f"Provider {provider_id} not found")
    result = mgr.upload_incorporation_file(provider_id, req.file_id)

    # Also add 'incorporation' verification to reputation
    try:
        re = get_reputation_engine()
        re.add_verification(provider_id, VerificationType.INCORPORATION)
    except Exception:
        pass  # Non-critical

    return {"ok": True, **result}


@app.get("/api/providers/{provider_id}/earnings", tags=["Providers"])
def api_provider_earnings(provider_id: str):
    """Get aggregate earnings and payout history for a provider."""
    mgr = get_stripe_manager()
    earnings = mgr.get_provider_earnings(provider_id)
    payouts = mgr.get_provider_payouts(provider_id, limit=20)
    return {"ok": True, "earnings": earnings, "recent_payouts": payouts}


@app.post("/api/providers/{provider_id}/payout", tags=["Providers"])
def api_provider_payout(provider_id: str, job_id: str = "", total_cad: float = 0):
    """Split a job payment between provider (85%) and platform (15%).

    Applies province-specific GST/HST. If Stripe is configured,
    creates a real Transfer to the provider's connected account.
    """
    if not job_id or total_cad <= 0:
        raise HTTPException(400, "job_id and total_cad (>0) required")
    mgr = get_stripe_manager()
    provider = mgr.get_provider(provider_id)
    if not provider:
        raise HTTPException(404, f"Provider {provider_id} not found")
    result = mgr.split_payout(job_id, provider_id, total_cad, provider.get("province", "ON"))
    return {"ok": True, **result}


class StripeWebhookRaw(BaseModel):
    """Raw Stripe webhook — in production, read from request body directly."""
    payload: str = ""
    signature: str = ""


@app.post("/api/providers/webhook", tags=["Providers"])
def api_stripe_webhook(req: StripeWebhookRaw):
    """Handle Stripe Connect webhooks (account.updated, payment_intent.succeeded, etc.)."""
    mgr = get_stripe_manager()
    result = mgr.handle_webhook(req.payload.encode(), req.signature)
    return {"ok": True, **result}


# ── LLMs.txt Endpoint (Report #1.B: LLM Optimization) ────────────────

@app.get("/llms.txt", tags=["Infrastructure"])
def api_llms_txt():
    """Serve LLM-optimized documentation for AI agents.

    Per Report #1.B: "Standard llms.txt for AI agents".
    See https://llmstxt.org for specification.
    """
    llms_path = Path(os.path.dirname(__file__)) / "llms.txt"
    if llms_path.exists():
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(content=llms_path.read_text(), media_type="text/plain")
    raise HTTPException(404, "llms.txt not found")


# ── User Data Export (PIPEDA / Subject Access Request) ────────────────

@app.get("/api/auth/me/data-export", tags=["Auth"])
def api_data_export(request: Request):
    """Export all personal data for the current user (PIPEDA right).

    Returns a JSON bundle of all user data: profile, jobs, billing,
    reputation, artifacts, and consent records.
    """
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    email = user["email"]
    customer_id = ""

    # Gather profile
    if _USE_PERSISTENT_AUTH:
        profile = UserStore.get_user(email) or {}
    else:
        with _user_lock:
            profile = _users_db.get(email, {})
    customer_id = profile.get("customer_id", "")
    safe_profile = {
        k: v for k, v in profile.items()
        if k not in ("hashed_password", "password")
    }

    # Gather jobs
    all_jobs = list_jobs()
    user_jobs = [
        j for j in all_jobs
        if j.get("customer_id") == customer_id or j.get("submitted_by") == email
    ]

    # Gather billing
    billing_txns = []
    if customer_id:
        try:
            be = get_billing_engine()
            billing_txns = be.get_wallet_history(customer_id, limit=500)
        except Exception:
            pass

    # Gather reputation
    rep_data = {}
    try:
        re = get_reputation_engine()
        rep_data = re.store.get_score(customer_id or email) or {}
    except Exception:
        pass

    export = {
        "exported_at": time.time(),
        "profile": safe_profile,
        "jobs": user_jobs[:200],
        "billing_transactions": billing_txns[:200],
        "reputation": rep_data,
        "total_jobs": len(user_jobs),
        "total_transactions": len(billing_txns),
    }
    return {"ok": True, "data_export": export}


# ── Artifact TTL / Expiry Info ────────────────────────────────────────

@app.get("/api/artifacts/{job_id}/expiry", tags=["Artifacts"])
def api_artifact_expiry(job_id: str):
    """Get expiry/cleanup dates for artifacts of a given job.

    Returns each artifact with its created_at and estimated expiry date
    based on the configured retention policy.
    """
    try:
        am = get_artifact_manager()
        arts = am.get_job_artifacts(job_id)
    except Exception:
        arts = []

    # Default retention: 90 days for job_output, 180 for model_checkpoint, 30 for logs
    retention_days = {
        "job_output": 90,
        "model_checkpoint": 180,
        "dataset": 365,
        "log_bundle": 30,
    }
    result = []
    for a in arts:
        art_type = a.get("artifact_type", "job_output")
        created = a.get("created_at", time.time())
        ttl_days = retention_days.get(art_type, 90)
        expiry = created + ttl_days * 86400
        result.append({
            "artifact_id": a.get("artifact_id", ""),
            "artifact_type": art_type,
            "created_at": created,
            "ttl_days": ttl_days,
            "expires_at": expiry,
            "days_remaining": max(0, int((expiry - time.time()) / 86400)),
        })

    return {"ok": True, "job_id": job_id, "artifacts": result}


# ── Reputation Score Breakdown ────────────────────────────────────────

@app.get("/api/reputation/{entity_id}/breakdown", tags=["Reputation"])
def api_reputation_breakdown(entity_id: str):
    """Get a detailed breakdown of how a reputation score is calculated.

    Returns component scores: jobs completed, uptime bonus, penalties, decay.
    """
    re = get_reputation_engine()
    score_data = re.store.get_score(entity_id) or {}
    history = re.store.get_event_history(entity_id, limit=100)

    # Calculate component breakdown from history
    jobs_points = 0
    uptime_bonus = 0
    penalties = 0
    decay = 0

    for event in history:
        delta = event.get("score_delta", event.get("delta", 0))
        reason = (event.get("reason", "") or "").lower()
        if "job" in reason or "complete" in reason:
            jobs_points += max(0, delta)
        elif "uptime" in reason or "bonus" in reason:
            uptime_bonus += max(0, delta)
        elif "penalt" in reason or "violat" in reason or "fail" in reason:
            penalties += abs(min(0, delta))
        elif "decay" in reason:
            decay += abs(min(0, delta))
        else:
            if delta >= 0:
                jobs_points += delta
            else:
                penalties += abs(delta)

    total_score = score_data.get("final_score", score_data.get("score", 0))

    return {
        "ok": True,
        "entity_id": entity_id,
        "total_score": total_score,
        "tier": score_data.get("tier", "new_user"),
        "breakdown": {
            "jobs_completed": round(jobs_points, 1),
            "uptime_bonus": round(uptime_bonus, 1),
            "penalties": round(penalties, 1),
            "decay": round(decay, 1),
        },
        "events_analyzed": len(history),
    }


if __name__ == "__main__":
    import uvicorn

    log.info("API STARTING on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
