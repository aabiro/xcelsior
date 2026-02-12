# Xcelsior API v2.0.0
# FastAPI. Every endpoint. Dashboard. Marketplace. Autoscale. SSE. Spot pricing. No fluff.

import asyncio
import hmac
import json
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

TEMPLATES_DIR = Path(os.path.dirname(__file__)) / "templates"

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
)

from security import admit_node, check_node_versions

app = FastAPI(title="Xcelsior", version="2.0.0")

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

XCELSIOR_ENV = os.environ.get("XCELSIOR_ENV", "dev").lower()
AUTH_REQUIRED = XCELSIOR_ENV not in {"dev", "development", "test"}
RATE_LIMIT_REQUESTS = int(os.environ.get("XCELSIOR_RATE_LIMIT_REQUESTS", "120"))
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("XCELSIOR_RATE_LIMIT_WINDOW_SEC", "60"))
_RATE_BUCKETS = defaultdict(deque)


# ── Phase 13: API Token Auth ─────────────────────────────────────────

# Public routes — no token required
PUBLIC_PATHS = {
    "/", "/docs", "/openapi.json", "/dashboard", "/healthz", "/readyz",
    "/metrics", "/api/stream",
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


class JobIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    vram_needed_gb: float = Field(gt=0)
    priority: int = Field(default=0, ge=0, le=10)
    tier: str | None = None


class StatusUpdate(BaseModel):
    status: str
    host_id: str | None = None


# ── Host endpoints ────────────────────────────────────────────────────


@app.put("/host")
def api_register_host(h: HostIn):
    """Register or update a host."""
    entry = register_host(
        h.host_id, h.ip, h.gpu_model, h.total_vram_gb, h.free_vram_gb, h.cost_per_hour
    )
    broadcast_sse("host_update", {"host_id": h.host_id, "gpu_model": h.gpu_model})
    return {"ok": True, "host": entry}


@app.get("/hosts")
def api_list_hosts(active_only: bool = True):
    """List all hosts."""
    return {"hosts": list_hosts(active_only=active_only)}


@app.delete("/host/{host_id}")
def api_remove_host(host_id: str):
    """Remove a host."""
    hosts = list_hosts(active_only=False)
    if not any(h["host_id"] == host_id for h in hosts):
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    remove_host(host_id)
    broadcast_sse("host_removed", {"host_id": host_id})
    return {"ok": True, "removed": host_id}


@app.post("/hosts/check")
def api_check_hosts():
    """Ping all hosts and update status."""
    results = check_hosts()
    return {"results": results}


# ── Job endpoints ─────────────────────────────────────────────────────


@app.post("/job")
def api_submit_job(j: JobIn):
    """Submit a job to the queue. Tier overrides priority."""
    job = submit_job(j.name, j.vram_needed_gb, j.priority, tier=j.tier)
    broadcast_sse("job_submitted", {"job_id": job["job_id"], "name": job["name"]})
    return {"ok": True, "job": job}


@app.get("/jobs")
def api_list_jobs(status: str | None = None):
    """List jobs. Optional filter by status."""
    return {"jobs": list_jobs(status=status)}


@app.get("/job/{job_id}")
def api_get_job(job_id: str):
    """Get a specific job by ID."""
    jobs = list_jobs()
    for j in jobs:
        if j["job_id"] == job_id:
            return {"job": j}
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@app.patch("/job/{job_id}")
def api_update_job(job_id: str, update: StatusUpdate):
    """Update a job's status."""
    try:
        update_job_status(job_id, update.status, host_id=update.host_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    broadcast_sse("job_status", {"job_id": job_id, "status": update.status})
    return {"ok": True, "job_id": job_id, "status": update.status}


@app.post("/queue/process")
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


@app.post("/failover")
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


@app.post("/job/{job_id}/requeue")
def api_requeue_job(job_id: str):
    """Manually requeue a failed or stuck job."""
    result = requeue_job(job_id)
    if not result:
        raise HTTPException(
            status_code=400,
            detail=f"Could not requeue job {job_id} (max retries exceeded or not found)",
        )
    return {"ok": True, "job": result}


# ── Billing endpoints ────────────────────────────────────────────────


@app.post("/billing/bill/{job_id}")
def api_bill_job(job_id: str):
    """Bill a specific completed job."""
    record = bill_job(job_id)
    if not record:
        raise HTTPException(status_code=400, detail=f"Could not bill job {job_id}")
    return {"ok": True, "bill": record}


@app.post("/billing/bill-all")
def api_bill_all():
    """Bill all unbilled completed jobs."""
    bills = bill_all_completed()
    return {"billed": len(bills), "bills": bills}


@app.get("/billing")
def api_billing():
    """Get all billing records and total revenue."""
    records = load_billing()
    return {
        "records": records,
        "total_revenue": get_total_revenue(),
    }


# ── Phase 11: Dashboard ───────────────────────────────────────────────


@app.get("/dashboard", response_class=HTMLResponse)
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


@app.get("/alerts/config")
def api_get_alert_config():
    """Get current alert config (passwords redacted)."""
    safe = {k: ("***" if "pass" in k or "token" in k else v) for k, v in ALERT_CONFIG.items()}
    return {"config": safe}


@app.put("/alerts/config")
def api_set_alert_config(cfg: AlertConfig):
    """Update alert config at runtime."""
    updates = {k: v for k, v in cfg.model_dump().items() if v is not None}
    configure_alerts(**updates)
    return {"ok": True, "updated": list(updates.keys())}


# ── Phase 13: SSH key management ──────────────────────────────────────


@app.post("/ssh/keygen")
def api_generate_ssh_key():
    """Generate an Ed25519 SSH keypair for host access."""
    path = generate_ssh_keypair()
    pub = get_public_key(path)
    return {"ok": True, "key_path": path, "public_key": pub}


@app.get("/ssh/pubkey")
def api_get_pubkey():
    """Get the public key to add to hosts' authorized_keys."""
    pub = get_public_key()
    if not pub:
        raise HTTPException(status_code=404, detail="No SSH key found. POST /ssh/keygen first.")
    return {"public_key": pub}


@app.post("/token/generate")
def api_generate_token():
    """Generate a secure random API token. User must set it in .env themselves."""
    token = secrets.token_urlsafe(32)
    return {"token": token, "note": "Set XCELSIOR_API_TOKEN in your .env to enable auth."}


# ── Phase 15: Priority Tiers ─────────────────────────────────────────


@app.get("/tiers")
def api_list_tiers():
    """List all priority tiers with their multipliers."""
    return {"tiers": list_tiers()}


# ── Phase 16: Docker Image Builder ───────────────────────────────────


class BuildIn(BaseModel):
    model: str
    base_image: str = "python:3.11-slim"
    quantize: str | None = None
    push: bool = False


@app.post("/build")
def api_build_image(b: BuildIn):
    """Build a Docker image for a model. Optionally quantize and push."""
    result = build_and_push(b.model, quantize=b.quantize, base_image=b.base_image, push=b.push)
    if not result["built"]:
        raise HTTPException(status_code=500, detail=f"Build failed for {b.model}")
    return {"ok": True, "build": result}


@app.get("/builds")
def api_list_builds():
    """List all local build directories."""
    return {"builds": list_builds()}


@app.post("/build/{model}/dockerfile")
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


@app.post("/marketplace/list")
def api_list_rig(rig: RigListing):
    """List a rig on the marketplace."""
    listing = list_rig(
        rig.host_id, rig.gpu_model, rig.vram_gb, rig.price_per_hour, rig.description, rig.owner
    )
    return {"ok": True, "listing": listing}


@app.delete("/marketplace/{host_id}")
def api_unlist_rig(host_id: str):
    """Remove a rig from the marketplace."""
    if not unlist_rig(host_id):
        raise HTTPException(status_code=404, detail=f"Listing {host_id} not found")
    return {"ok": True, "unlisted": host_id}


@app.get("/marketplace")
def api_get_marketplace(active_only: bool = True):
    """Browse marketplace listings."""
    return {"listings": get_marketplace(active_only=active_only)}


@app.post("/marketplace/bill/{job_id}")
def api_marketplace_bill(job_id: str):
    """Bill a marketplace job — split between host and platform."""
    result = marketplace_bill(job_id)
    if not result:
        raise HTTPException(status_code=400, detail=f"Could not bill marketplace job {job_id}")
    return {"ok": True, "bill": result}


@app.get("/marketplace/stats")
def api_marketplace_stats():
    """Marketplace aggregate stats."""
    return {"stats": marketplace_stats()}


# ── Phase 18: Canada-Only Toggle ─────────────────────────────────────


class CanadaToggle(BaseModel):
    enabled: bool


@app.get("/canada")
def api_canada_status():
    """Check if Canada-only mode is active."""
    import scheduler

    return {"canada_only": scheduler.CANADA_ONLY}


@app.put("/canada")
def api_set_canada(toggle: CanadaToggle):
    """Toggle Canada-only mode."""
    set_canada_only(toggle.enabled)
    return {"ok": True, "canada_only": toggle.enabled}


@app.get("/hosts/ca")
def api_list_canadian_hosts():
    """List only Canadian hosts."""
    return {"hosts": list_hosts_filtered(canada_only=True)}


@app.post("/queue/process/ca")
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


@app.post("/autoscale/pool")
def api_add_to_pool(h: PoolHost):
    """Add a host to the autoscale pool."""
    entry = add_to_pool(h.host_id, h.ip, h.gpu_model, h.vram_gb, h.cost_per_hour, h.country)
    return {"ok": True, "pool_entry": entry}


@app.delete("/autoscale/pool/{host_id}")
def api_remove_from_pool(host_id: str):
    """Remove a host from the autoscale pool."""
    remove_from_pool(host_id)
    return {"ok": True, "removed": host_id}


@app.get("/autoscale/pool")
def api_get_pool():
    """List the autoscale pool."""
    return {"pool": load_autoscale_pool()}


@app.post("/autoscale/cycle")
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


@app.post("/autoscale/up")
def api_autoscale_up():
    """Scale up: provision hosts for queued jobs."""
    provisioned = autoscale_up()
    return {"provisioned": [{"host_id": h["host_id"], "gpu": h["gpu_model"]} for h in provisioned]}


@app.post("/autoscale/down")
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


@app.get("/api/stream")
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


@app.get("/agent/work/{host_id}")
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


@app.get("/agent/preempt/{host_id}")
def api_agent_preempt(host_id: str):
    """Check if any jobs on this host should be preempted."""
    with _agent_lock:
        preempt_list = _agent_preempt.pop(host_id, [])
    return {"ok": True, "preempt_jobs": preempt_list}


@app.post("/agent/preempt/{host_id}/{job_id}")
def api_schedule_preemption(host_id: str, job_id: str):
    """Schedule a job for preemption on a host."""
    with _agent_lock:
        _agent_preempt[host_id].append(job_id)
    broadcast_sse("preemption_scheduled", {"host_id": host_id, "job_id": job_id})
    return {"ok": True, "host_id": host_id, "job_id": job_id}


@app.post("/agent/versions")
def api_agent_versions(report: VersionReport):
    """Receive and validate node component versions for admission control."""
    admitted, reasons = check_node_versions(report.versions)
    broadcast_sse("node_admission", {
        "host_id": report.host_id,
        "admitted": admitted,
        "versions": report.versions,
    })
    return {
        "ok": True,
        "admitted": admitted,
        "details": {
            "host_id": report.host_id,
            "versions": report.versions,
            "rejection_reasons": reasons,
        },
    }


@app.post("/agent/mining-alert")
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


@app.post("/agent/benchmark")
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


# ── Spot Pricing Endpoints ───────────────────────────────────────────


class SpotJobIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    vram_needed_gb: float = Field(gt=0)
    max_bid: float = Field(gt=0)
    priority: int = Field(default=0, ge=0, le=10)
    tier: str | None = None


@app.get("/spot-prices")
def api_spot_prices():
    """Get current spot prices for all GPU models."""
    return {"ok": True, "prices": get_current_spot_prices()}


@app.post("/spot-prices/update")
def api_update_spot_prices():
    """Trigger spot price recalculation."""
    prices = update_spot_prices()
    broadcast_sse("spot_prices_updated", {"prices": prices})
    return {"ok": True, "prices": prices}


@app.post("/spot/job")
def api_submit_spot_job(j: SpotJobIn):
    """Submit a spot job with a maximum bid price."""
    job = submit_spot_job(j.name, j.vram_needed_gb, j.max_bid, j.priority, tier=j.tier)
    broadcast_sse("spot_job_submitted", {
        "job_id": job["job_id"], "name": job["name"], "max_bid": j.max_bid,
    })
    return {"ok": True, "job": job}


@app.post("/spot/preemption-cycle")
def api_preemption_cycle():
    """Run a preemption cycle — reclaim resources from underbidding spot jobs."""
    preempted = preemption_cycle()
    return {"ok": True, "preempted": preempted}


# ── Compute Score Endpoints ──────────────────────────────────────────


@app.get("/compute-score/{host_id}")
def api_get_compute_score(host_id: str):
    """Get the compute score (XCU) for a host."""
    score = get_compute_score(host_id)
    if score is None:
        raise HTTPException(status_code=404, detail=f"No compute score for host {host_id}")
    return {"ok": True, "host_id": host_id, "score": score}


@app.get("/compute-scores")
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


@app.get("/healthz")
def healthz():
    return {"ok": True, "status": "healthy", "env": XCELSIOR_ENV}


@app.get("/readyz")
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


@app.get("/metrics")
def metrics():
    return {"ok": True, "metrics": get_metrics_snapshot()}


@app.get("/")
def root():
    return {"name": "Xcelsior", "status": "running"}


if __name__ == "__main__":
    import uvicorn

    log.info("API STARTING on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
