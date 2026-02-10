# Xcelsior API v1.0.0
# FastAPI. Every endpoint. Dashboard. Marketplace. Autoscale. No fluff.

import hmac
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

TEMPLATES_DIR = Path(os.path.dirname(__file__)) / "templates"

from scheduler import (
    register_host, remove_host, list_hosts, check_hosts,
    submit_job, list_jobs, update_job_status, process_queue,
    bill_job, bill_all_completed, get_total_revenue, load_billing,
    configure_alerts, ALERT_CONFIG,
    generate_ssh_keypair, get_public_key, API_TOKEN,
    failover_and_reassign, requeue_job,
    list_tiers, PRIORITY_TIERS,
    build_and_push, list_builds, generate_dockerfile,
    list_rig, unlist_rig, get_marketplace, marketplace_bill, marketplace_stats,
    register_host_ca, list_hosts_filtered, process_queue_filtered,
    set_canada_only,
    add_to_pool, remove_from_pool, load_autoscale_pool,
    autoscale_cycle, autoscale_up, autoscale_down,
    log,
)

app = FastAPI(title="Xcelsior", version="1.0.0")


# ── Phase 13: API Token Auth ─────────────────────────────────────────

# Public routes — no token required
PUBLIC_PATHS = {"/", "/docs", "/openapi.json", "/dashboard"}


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """
    Bearer token auth. If XCELSIOR_API_TOKEN is set, every request
    (except public routes) must include it. No token set = open access.
    """
    async def dispatch(self, request: Request, call_next):
        if not API_TOKEN:
            return await call_next(request)

        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
        else:
            token = request.query_params.get("token", "")

        if not token or not hmac.compare_digest(token, API_TOKEN):
            return HTMLResponse(
                content='{"detail":"Unauthorized"}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)


app.add_middleware(TokenAuthMiddleware)


# ── Request models ────────────────────────────────────────────────────

class HostIn(BaseModel):
    host_id: str
    ip: str
    gpu_model: str
    total_vram_gb: float
    free_vram_gb: float
    cost_per_hour: float = 0.20


class JobIn(BaseModel):
    name: str
    vram_needed_gb: float
    priority: int = 0
    tier: str | None = None


class StatusUpdate(BaseModel):
    status: str
    host_id: str | None = None


# ── Host endpoints ────────────────────────────────────────────────────

@app.put("/host")
def api_register_host(h: HostIn):
    """Register or update a host."""
    entry = register_host(h.host_id, h.ip, h.gpu_model,
                          h.total_vram_gb, h.free_vram_gb, h.cost_per_hour)
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
    return {"ok": True, "job_id": job_id, "status": update.status}


@app.post("/queue/process")
def api_process_queue():
    """Process the job queue — assign jobs to hosts."""
    assigned = process_queue()
    return {
        "assigned": [
            {"job": j["name"], "job_id": j["job_id"], "host": h["host_id"]}
            for j, h in assigned
        ]
    }


# ── Phase 14: Failover endpoints ──────────────────────────────────────

@app.post("/failover")
def api_failover():
    """Run a full failover cycle: check hosts, requeue orphaned jobs, reassign."""
    requeued, assigned = failover_and_reassign()
    return {
        "requeued": [{"job_id": j["job_id"], "name": j["name"], "retries": j.get("retries", 0)} for j in requeued],
        "assigned": [{"job": j["name"], "job_id": j["job_id"], "host": h["host_id"]} for j, h in assigned],
    }


@app.post("/job/{job_id}/requeue")
def api_requeue_job(job_id: str):
    """Manually requeue a failed or stuck job."""
    result = requeue_job(job_id)
    if not result:
        raise HTTPException(status_code=400, detail=f"Could not requeue job {job_id} (max retries exceeded or not found)")
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
    safe = {k: ("***" if "pass" in k or "token" in k else v)
            for k, v in ALERT_CONFIG.items()}
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
    result = build_and_push(b.model, quantize=b.quantize,
                             base_image=b.base_image, push=b.push)
    if not result["built"]:
        raise HTTPException(status_code=500, detail=f"Build failed for {b.model}")
    return {"ok": True, "build": result}


@app.get("/builds")
def api_list_builds():
    """List all local build directories."""
    return {"builds": list_builds()}


@app.post("/build/{model}/dockerfile")
def api_generate_dockerfile(model: str, base_image: str = "python:3.11-slim",
                             quantize: str | None = None):
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
    listing = list_rig(rig.host_id, rig.gpu_model, rig.vram_gb,
                        rig.price_per_hour, rig.description, rig.owner)
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
            {"job": j["name"], "job_id": j["job_id"], "host": h["host_id"], "country": h.get("country", "?")}
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
    entry = add_to_pool(h.host_id, h.ip, h.gpu_model, h.vram_gb,
                         h.cost_per_hour, h.country)
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
        "assigned": [{"job": j["name"], "job_id": j["job_id"], "host": h["host_id"]} for j, h in assigned],
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


# ── Health ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"name": "Xcelsior", "status": "running"}


if __name__ == "__main__":
    import uvicorn
    log.info("API STARTING on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
