# Xcelsior API — Phase 9 + 11
# FastAPI. POST /job. GET /status. PUT /host. Dashboard. No fluff.

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

TEMPLATES_DIR = Path(os.path.dirname(__file__)) / "templates"

from scheduler import (
    register_host, remove_host, list_hosts, check_hosts,
    submit_job, list_jobs, update_job_status, process_queue,
    bill_job, bill_all_completed, get_total_revenue, load_billing,
    configure_alerts, ALERT_CONFIG,
    log,
)

app = FastAPI(title="Xcelsior", version="0.1.0")


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
    """Submit a job to the queue."""
    job = submit_job(j.name, j.vram_needed_gb, j.priority)
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
    update_job_status(job_id, update.status, host_id=update.host_id)
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


# ── Health ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"name": "Xcelsior", "status": "running"}


if __name__ == "__main__":
    import uvicorn
    log.info("API STARTING on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
