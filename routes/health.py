"""Routes: health."""

import asyncio
import json
import os
import secrets
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)
from pydantic import BaseModel, Field

from routes._deps import (
    AUTH_REQUIRED,
    XCELSIOR_ENV,
    _create_session,
    _get_current_user,
    _is_platform_admin,
    _require_admin,
    _require_auth,
    _require_provider_or_admin,
    _sse_lock,
    _sse_subscribers,
    log,
)
from scheduler import (
    ALERT_CONFIG,
    API_TOKEN,
    build_and_push,
    configure_alerts,
    generate_dockerfile,
    generate_ssh_keypair,
    get_metrics_snapshot,
    get_public_key,
    list_builds,
    log,
    storage_healthcheck,
)

router = APIRouter()

# Templates and device code constants
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
DEVICE_CODE_EXPIRY = 600  # 10 minutes
DEVICE_CODE_INTERVAL = 5  # poll interval seconds
_device_codes: dict[str, dict] = {}
_device_lock = threading.Lock()


@router.get("/healthz", tags=["Infrastructure"])
def healthz():
    """Lightweight liveness probe for frontend polling."""
    return {"ok": True}


@router.get("/dashboard", response_class=HTMLResponse, tags=["Infrastructure"])
def dashboard():
    """The dashboard. HTML + JS. No React. No npm. No build step."""
    html = (TEMPLATES_DIR / "dashboard.html").read_text()
    return HTMLResponse(content=html)


@router.get("/legacy", response_class=HTMLResponse, tags=["Infrastructure"])
@router.get("/legacy/{path:path}", response_class=HTMLResponse, tags=["Infrastructure"])
def legacy_dashboard(path: str = ""):
    """Legacy dashboard preserved at /legacy while Next.js serves /."""
    html = (TEMPLATES_DIR / "dashboard.html").read_text()
    return HTMLResponse(content=html)


# ── Model: AlertConfig ──


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


@router.get("/alerts/config", tags=["Infrastructure"])
def api_get_alert_config(request: Request):
    """Get current alert config (passwords redacted)."""
    _require_admin(request)
    safe = {k: ("***" if "pass" in k or "token" in k else v) for k, v in ALERT_CONFIG.items()}
    return {"config": safe}


@router.put("/alerts/config", tags=["Infrastructure"])
def api_set_alert_config(cfg: AlertConfig, request: Request):
    """Update alert config at runtime."""
    _require_admin(request)
    updates = {k: v for k, v in cfg.model_dump().items() if v is not None}
    configure_alerts(**updates)
    return {"ok": True, "updated": list(updates.keys())}


@router.post("/ssh/keygen", tags=["Infrastructure"])
def api_generate_ssh_key(request: Request):
    """Generate an Ed25519 SSH keypair for host access."""
    _require_auth(request)
    path = generate_ssh_keypair()
    pub = get_public_key(path)
    return {"ok": True, "key_path": path, "public_key": pub}


@router.get("/ssh/pubkey", tags=["Infrastructure"])
@router.get("/api/ssh/pubkey", tags=["Infrastructure"])
def api_get_pubkey():
    """Get the public key to add to hosts' authorized_keys."""
    pub = get_public_key()
    return {"public_key": pub or ""}


@router.post("/token/generate", tags=["Infrastructure"])
def api_generate_token():
    """Generate a secure random API token. User must set it in .env themselves."""
    token = secrets.token_urlsafe(32)
    return {"token": token, "note": "Set XCELSIOR_API_TOKEN in your .env to enable auth."}


# ── Model: DeviceCodeResponse ──


class DeviceCodeResponse(BaseModel):
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int = DEVICE_CODE_EXPIRY
    interval: int = DEVICE_CODE_INTERVAL


# ── Model: DeviceTokenRequest ──


class DeviceTokenRequest(BaseModel):
    device_code: str
    grant_type: str = "urn:ietf:params:oauth:grant-type:device_code"


@router.post("/_internal/legacy-auth/device", tags=["Infrastructure"])
def api_auth_device_code(request: Request):
    """Initiate OAuth2 device authorization flow (RFC 8628).

    Returns a device_code (for polling) and a user_code (for the user to enter
    in the browser at the verification_uri).
    """
    device_code = secrets.token_urlsafe(32)
    user_code = "-".join(
        [
            secrets.token_hex(2).upper(),
            secrets.token_hex(2).upper(),
        ]
    )  # e.g. "A1B2-C3D4"

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


@router.post("/_internal/legacy-auth/token", tags=["Infrastructure"])
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


# ── Model: DeviceVerifyRequest ──


class DeviceVerifyRequest(BaseModel):
    user_code: str


@router.post("/_internal/legacy-auth/verify", tags=["Infrastructure"])
def api_auth_verify_device(body: DeviceVerifyRequest, request: Request):
    """Verify a device code by entering the user_code shown in the CLI.

    Called from the web dashboard after the user logs in and enters their code.
    Requires a valid session (cookie or bearer token) so the CLI token can be
    tied to the authenticated user.  Falls back to creating an anonymous session
    when no user context is available (e.g. first-time sign-up via device flow).
    """
    # Resolve the logged-in user from cookie/bearer
    user = _get_current_user(request)

    with _device_lock:
        for dc, entry in _device_codes.items():
            if entry["user_code"] == body.user_code and entry["status"] == "pending":
                now = time.time()
                if now > entry["expires_at"]:
                    entry["status"] = "expired"
                    raise HTTPException(status_code=410, detail="Code expired")

                # Create a real session so the token works with TokenAuthMiddleware
                if user:
                    session = _create_session(user.get("email", "device-user"), user, request)
                else:
                    # Anonymous device auth — create a minimal session
                    anon_user = {
                        "user_id": f"device-{secrets.token_hex(8)}",
                        "email": "device-auth@xcelsior.ca",
                        "role": "submitter",
                        "name": "Device User",
                    }
                    session = _create_session(anon_user["email"], anon_user, request)

                token = session["token"]
                entry["status"] = "authorized"
                entry["token"] = token
                return {"message": "Device authorized", "user_code": body.user_code}

    raise HTTPException(status_code=404, detail="Invalid or expired user code")


@router.get("/_internal/legacy-auth/verify", response_class=HTMLResponse, tags=["Infrastructure"])
def api_auth_verify_page():
    """Browser-facing page where users enter their device code."""
    return HTMLResponse("""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="https://xcelsior.ca/favicon.svg" type="image/svg+xml">
<title>Xcelsior — Device Authorization</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,-apple-system,sans-serif;background:#060a14;color:#e5e7eb;
       display:flex;align-items:center;justify-content:center;min-height:100vh;
       background-image:radial-gradient(ellipse at 50% 0%,rgba(0,212,255,.06) 0%,transparent 60%)}
  .brand-line{height:2px;width:100%;background:linear-gradient(90deg,#00d4ff 0%,#8b5cf6 50%,#ef4444 100%);
              border-radius:1px}
  .card{background:rgba(13,19,32,.85);border:1px solid rgba(75,85,99,.4);border-radius:16px;
        padding:2.5rem 2rem 2rem;max-width:440px;width:100%;text-align:center;
        backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
        box-shadow:0 0 60px rgba(0,212,255,.05)}
  .logo-wrap{display:flex;justify-content:center;margin-bottom:1.5rem}
  .logo{width:180px;height:auto}
  .subtitle{color:#9ca3af;margin-bottom:2rem;font-size:.95rem;line-height:1.5}
  .subtitle strong{color:#00d4ff}
  input{width:100%;padding:.875rem;border:1px solid rgba(75,85,99,.5);border-radius:10px;
        background:rgba(31,41,55,.6);color:#f9fafb;font-size:1.3rem;text-align:center;
        letter-spacing:.25em;text-transform:uppercase;margin-bottom:1rem;
        transition:border-color .2s,box-shadow .2s}
  input:focus{outline:none;border-color:#00d4ff;box-shadow:0 0 0 3px rgba(0,212,255,.15)}
  input::placeholder{color:#4b5563;letter-spacing:.15em;text-transform:none;font-size:.95rem}
  button{width:100%;padding:.875rem;border:none;border-radius:10px;
         background:linear-gradient(135deg,#00d4ff 0%,#0ea5e9 100%);color:#0a0a0a;
         font-size:1rem;cursor:pointer;font-weight:700;letter-spacing:.02em;
         transition:opacity .2s,transform .1s}
  button:hover:not(:disabled){opacity:.9;transform:translateY(-1px)}
  button:active:not(:disabled){transform:translateY(0)}
  button:disabled{opacity:.4;cursor:not-allowed}
  .msg{margin-top:1.25rem;padding:.875rem;border-radius:10px;font-size:.9rem;font-weight:500}
  .ok{background:rgba(6,78,59,.6);color:#6ee7b7;border:1px solid rgba(6,95,70,.6)}
  .err{background:rgba(127,29,29,.5);color:#fca5a5;border:1px solid rgba(153,27,27,.5)}
  .footer{margin-top:1.5rem;font-size:.8rem;color:#4b5563}
  .footer a{color:#00d4ff;text-decoration:none}
  .footer a:hover{text-decoration:underline}
</style></head><body>
<div class="card">
  <div class="brand-line" style="margin-bottom:1.5rem"></div>
  <div class="logo-wrap">
    <img src="https://xcelsior.ca/xcelsior-logo-wordmark-iconbg.svg" alt="Xcelsior" class="logo"
         onerror="this.parentElement.style.display='none';document.getElementById('fallback-title').style.display='block'">
  </div>
  <h1 id="fallback-title" style="display:none;font-size:1.8rem;margin-bottom:1rem;
      background:linear-gradient(135deg,#00d4ff,#8b5cf6);-webkit-background-clip:text;
      -webkit-text-fill-color:transparent;font-weight:800">Xcelsior</h1>
  <p class="subtitle">Enter the code shown in your <strong>CLI</strong> to authorize this device.</p>
  <form id="f">
    <input id="code" placeholder="XXXX-XXXX" maxlength="9" autocomplete="off" autofocus>
    <button type="submit" id="btn">Authorize Device</button>
  </form>
  <div id="msg"></div>
  <div class="brand-line" style="margin-top:1.5rem"></div>
  <div class="footer">
    <a href="https://xcelsior.ca">xcelsior.ca</a> · GPU Cloud Platform
  </div>
</div>
<script>
const btn=document.getElementById('btn');
const codeInput=document.getElementById('code');
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault();
  const code=codeInput.value.trim();
  if(!code)return;
  const msg=document.getElementById('msg');
  btn.disabled=true;btn.textContent='Authorizing...';
  try{
    const r=await fetch('/api/auth/verify',{method:'POST',credentials:'include',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({user_code:code})});
    const d=await r.json();
    if(r.ok){
      msg.className='msg ok';
      msg.textContent='\\u2713 Device authorized! You can close this tab.';
      codeInput.disabled=true;
      btn.textContent='Authorized';
    }else{
      msg.className='msg err';
      msg.textContent=d.detail||'Authorization failed. Check your code and try again.';
      btn.disabled=false;btn.textContent='Authorize Device';
    }
  }catch(x){
    msg.className='msg err';msg.textContent='Network error — check your connection.';
    btn.disabled=false;btn.textContent='Authorize Device';
  }
};
</script></body></html>""")


# ── Helper: _require_provider_or_admin ──


def _require_provider_or_admin(request: Request) -> dict:
    """Return the current user or raise 403 if they lack provider/admin access."""
    user = _require_auth(request)
    if user.get("role") != "provider" and not _is_platform_admin(user):
        raise HTTPException(403, "Provider or admin access required")
    return user


# ── Model: SlurmSubmitIn ──


class SlurmSubmitIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    vram_needed_gb: float = Field(gt=0)
    priority: int = 0
    tier: str | None = None
    num_gpus: int = 1
    image: str = ""
    profile: str | None = None
    dry_run: bool = False


@router.post("/api/slurm/submit", tags=["Infrastructure"])
def api_slurm_submit(body: SlurmSubmitIn, request: Request):
    """Submit an Xcelsior job to a Slurm cluster (HPC bridge).

    Translates the job to an sbatch script and submits. Set dry_run=true
    to see the generated script without submitting.
    """
    _require_admin(request)
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


@router.get("/api/slurm/status/{slurm_job_id}", tags=["Infrastructure"])
def api_slurm_status(slurm_job_id: str, request: Request):
    """Check the status of a Slurm job."""
    _require_admin(request)
    from slurm_adapter import get_slurm_job_status

    status = get_slurm_job_status(slurm_job_id)
    if "error" in status:
        raise HTTPException(status_code=400, detail=status["error"])
    return status


@router.delete("/api/slurm/{slurm_job_id}", tags=["Infrastructure"])
def api_slurm_cancel(slurm_job_id: str, request: Request):
    """Cancel a Slurm job."""
    _require_admin(request)
    from slurm_adapter import cancel_slurm_job

    result = cancel_slurm_job(slurm_job_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/api/slurm/profiles", tags=["Infrastructure"])
def api_slurm_profiles(request: Request):
    """List available Slurm cluster profiles (Nibi, Graham, Narval, generic)."""
    _require_admin(request)
    from slurm_adapter import CLUSTER_PROFILES

    return {"profiles": {k: v["name"] for k, v in CLUSTER_PROFILES.items()}}


@router.get("/api/nfs/config", tags=["Infrastructure"])
def api_nfs_config():
    """Get current NFS configuration from environment."""
    return {
        "nfs_server": os.environ.get("XCELSIOR_NFS_SERVER", ""),
        "nfs_path": os.environ.get("XCELSIOR_NFS_PATH", ""),
        "nfs_mount_point": os.environ.get("XCELSIOR_NFS_MOUNT", "/mnt/xcelsior-nfs"),
        "configured": bool(os.environ.get("XCELSIOR_NFS_SERVER")),
    }


# ── Model: BuildIn ──


class BuildIn(BaseModel):
    model: str
    base_image: str = "python:3.11-slim"
    quantize: str | None = None
    push: bool = False


@router.post("/build", tags=["Infrastructure"])
def api_build_image(b: BuildIn):
    """Build a Docker image for a model. Optionally quantize and push."""
    result = build_and_push(b.model, quantize=b.quantize, base_image=b.base_image, push=b.push)
    if not result["built"]:
        raise HTTPException(status_code=500, detail=f"Build failed for {b.model}")
    return {"ok": True, "build": result}


@router.get("/builds", tags=["Infrastructure"])
def api_list_builds():
    """List all local build directories."""
    return {"builds": list_builds()}


@router.post("/build/{model}/dockerfile", tags=["Infrastructure"])
def api_generate_dockerfile(
    model: str, base_image: str = "python:3.11-slim", quantize: str | None = None
):
    """Preview the generated Dockerfile without building."""
    content = generate_dockerfile(model, base_image=base_image, quantize=quantize)
    return {"model": model, "dockerfile": content}


# ── Helper: _sse_generator ──


async def _sse_generator(request: Request):
    """Async generator that yields SSE events until the client disconnects."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    with _sse_lock:
        _sse_subscribers.append(queue)
    try:
        yield "retry: 3000\n\n"
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


@router.get("/api/stream", tags=["Infrastructure"])
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


# ── Model: VersionReport ──


class VersionReport(BaseModel):
    host_id: str
    versions: dict


@router.get("/healthz", tags=["Infrastructure"])
def healthz():
    """Health check — verifies database connectivity and returns real system status."""
    checks = {"env": XCELSIOR_ENV}
    try:
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            r = conn.execute("SELECT 1 AS ok").fetchone()
            checks["database"] = "connected" if r else "error"

            # Counts for observability
            hosts = conn.execute(
                "SELECT COUNT(*) as cnt FROM hosts WHERE status = 'active'"
            ).fetchone()
            jobs = conn.execute(
                "SELECT COUNT(*) as cnt FROM jobs WHERE status = 'running'"
            ).fetchone()
            queued = conn.execute(
                "SELECT COUNT(*) as cnt FROM jobs WHERE status = 'queued'"
            ).fetchone()
            checks["active_hosts"] = hosts["cnt"] if hosts else 0
            checks["running_jobs"] = jobs["cnt"] if jobs else 0
            checks["queued_jobs"] = queued["cnt"] if queued else 0
    except Exception as e:
        checks["database"] = f"error: {e}"
        return JSONResponse(
            status_code=503,
            content={"ok": False, "status": "unhealthy", **checks},
        )
    checks["status"] = "healthy"
    return {"ok": True, **checks}


@router.get("/readyz", tags=["Infrastructure"])
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


@router.get("/metrics", tags=["Infrastructure"])
def metrics():
    return {"ok": True, "metrics": get_metrics_snapshot()}


@router.get("/metrics/prometheus", tags=["Infrastructure"])
def metrics_prometheus():
    """Prometheus-compatible /metrics endpoint.

    Exports xcelsior_* gauges, counters, and histograms in Prometheus
    text exposition format for scraping by Prometheus/Grafana.
    """
    snap = get_metrics_snapshot()

    lines = [
        "# HELP xcelsior_queue_depth Number of queued jobs",
        "# TYPE xcelsior_queue_depth gauge",
        f'xcelsior_queue_depth {snap.get("queue_depth", 0)}',
        "",
        "# HELP xcelsior_active_hosts Number of active GPU hosts",
        "# TYPE xcelsior_active_hosts gauge",
        f'xcelsior_active_hosts {snap.get("active_hosts", 0)}',
        "",
        "# HELP xcelsior_running_jobs Number of running jobs",
        "# TYPE xcelsior_running_jobs gauge",
        f'xcelsior_running_jobs {snap.get("running_jobs", 0)}',
        "",
        "# HELP xcelsior_failed_jobs_total Total number of failed jobs",
        "# TYPE xcelsior_failed_jobs_total gauge",
        f'xcelsior_failed_jobs_total {snap.get("failed_jobs", 0)}',
        "",
        "# HELP xcelsior_billing_revenue_cad Total revenue in CAD",
        "# TYPE xcelsior_billing_revenue_cad gauge",
        f'xcelsior_billing_revenue_cad {snap.get("billing_totals", {}).get("total_revenue", 0)}',
        "",
        "# HELP xcelsior_billing_records_total Total billing records",
        "# TYPE xcelsior_billing_records_total gauge",
        f'xcelsior_billing_records_total {snap.get("billing_totals", {}).get("records", 0)}',
    ]

    notifications = snap.get("notifications", {})
    lines.extend(
        [
            "",
            "# HELP xcelsior_notifications_retained_total Total retained in-app notifications",
            "# TYPE xcelsior_notifications_retained_total gauge",
            f'xcelsior_notifications_retained_total {notifications.get("retained_total", 0)}',
        ]
    )

    web_push = snap.get("web_push", {})
    lines.extend(
        [
            "",
            "# HELP xcelsior_web_push_configured Whether web push is configured for this API instance",
            "# TYPE xcelsior_web_push_configured gauge",
            f'xcelsior_web_push_configured {web_push.get("configured", 0)}',
            "",
            "# HELP xcelsior_web_push_active_subscriptions Active web push subscriptions",
            "# TYPE xcelsior_web_push_active_subscriptions gauge",
            f'xcelsior_web_push_active_subscriptions {web_push.get("active_subscriptions", 0)}',
            "",
            "# HELP xcelsior_web_push_revoked_subscriptions Revoked web push subscriptions awaiting cleanup",
            "# TYPE xcelsior_web_push_revoked_subscriptions gauge",
            f'xcelsior_web_push_revoked_subscriptions {web_push.get("revoked_subscriptions", 0)}',
            "",
            "# HELP xcelsior_web_push_stale_subscriptions Active subscriptions not touched within the stale window",
            "# TYPE xcelsior_web_push_stale_subscriptions gauge",
            f'xcelsior_web_push_stale_subscriptions {web_push.get("stale_subscriptions", 0)}',
            "",
            "# HELP xcelsior_web_push_delivery_attempts_total Web push delivery attempts",
            "# TYPE xcelsior_web_push_delivery_attempts_total counter",
            f'xcelsior_web_push_delivery_attempts_total {web_push.get("delivery_attempts_total", 0)}',
            "",
            "# HELP xcelsior_web_push_delivery_success_total Successful web push deliveries",
            "# TYPE xcelsior_web_push_delivery_success_total counter",
            f'xcelsior_web_push_delivery_success_total {web_push.get("delivery_success_total", 0)}',
            "",
            "# HELP xcelsior_web_push_delivery_failure_total Failed web push deliveries",
            "# TYPE xcelsior_web_push_delivery_failure_total counter",
            f'xcelsior_web_push_delivery_failure_total {web_push.get("delivery_failure_total", 0)}',
            "",
            "# HELP xcelsior_web_push_delivery_revoked_total Deliveries that revoked stale endpoints",
            "# TYPE xcelsior_web_push_delivery_revoked_total counter",
            f'xcelsior_web_push_delivery_revoked_total {web_push.get("delivery_revoked_total", 0)}',
        ]
    )

    # GPU telemetry if available
    try:
        from nvml_telemetry import get_all_gpu_stats

        gpu_stats = get_all_gpu_stats()
        if gpu_stats:
            lines.extend(
                [
                    "",
                    "# HELP xcelsior_gpu_utilization_percent GPU utilization percentage",
                    "# TYPE xcelsior_gpu_utilization_percent gauge",
                ]
            )
            for gs in gpu_stats:
                idx = gs.get("index", 0)
                model = gs.get("name", "unknown").replace(" ", "_")
                lines.append(
                    f'xcelsior_gpu_utilization_percent{{gpu="{idx}",model="{model}"}} '
                    f'{gs.get("utilization", 0)}'
                )
            lines.extend(
                [
                    "",
                    "# HELP xcelsior_gpu_temperature_celsius GPU temperature",
                    "# TYPE xcelsior_gpu_temperature_celsius gauge",
                ]
            )
            for gs in gpu_stats:
                idx = gs.get("index", 0)
                lines.append(
                    f'xcelsior_gpu_temperature_celsius{{gpu="{idx}"}} {gs.get("temperature", 0)}'
                )
            lines.extend(
                [
                    "",
                    "# HELP xcelsior_gpu_memory_used_bytes GPU memory used",
                    "# TYPE xcelsior_gpu_memory_used_bytes gauge",
                ]
            )
            for gs in gpu_stats:
                idx = gs.get("index", 0)
                lines.append(
                    f'xcelsior_gpu_memory_used_bytes{{gpu="{idx}"}} '
                    f'{gs.get("memory_used_bytes", 0)}'
                )
    except Exception as e:
        log.debug("GPU metrics collection failed: %s", e)

    # Webhook backlog
    try:
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            row = conn.execute(
                "SELECT COUNT(*) as pending FROM stripe_event_inbox WHERE status = 'pending'"
            ).fetchone()
            backlog = row["pending"] if row else 0
        lines.extend(
            [
                "",
                "# HELP xcelsior_webhook_backlog Pending webhook events",
                "# TYPE xcelsior_webhook_backlog gauge",
                f"xcelsior_webhook_backlog {backlog}",
            ]
        )
    except Exception as e:
        log.debug("webhook backlog metric failed: %s", e)

    # Scheduling latency histogram (approximated from last scheduling cycle)
    try:
        _snap_running = snap.get("running_jobs", 0)
        _snap_queued = snap.get("queue_depth", 0)
        lines.extend(
            [
                "",
                "# HELP xcelsior_scheduling_latency_seconds Scheduling cycle latency",
                "# TYPE xcelsior_scheduling_latency_seconds histogram",
                f'xcelsior_scheduling_latency_seconds_bucket{{le="0.1"}} {_snap_running}',
                f'xcelsior_scheduling_latency_seconds_bucket{{le="1.0"}} {_snap_running}',
                f'xcelsior_scheduling_latency_seconds_bucket{{le="10.0"}} {_snap_running + _snap_queued}',
                f'xcelsior_scheduling_latency_seconds_bucket{{le="+Inf"}} {_snap_running + _snap_queued}',
                f"xcelsior_scheduling_latency_seconds_sum 0",
                f"xcelsior_scheduling_latency_seconds_count {_snap_running + _snap_queued}",
            ]
        )
    except Exception as e:
        log.debug("scheduling latency metric failed: %s", e)

    # Wallet depletion events counter
    try:
        from db import _get_pg_pool as _pgp2
        from psycopg.rows import dict_row as _dr2

        pool2 = _pgp2()
        with pool2.connection() as conn2:
            conn2.row_factory = _dr2
            dep_row = conn2.execute(
                "SELECT COUNT(*) as cnt FROM wallet_transactions WHERE description LIKE '%grace%' OR description LIKE '%suspend%'"
            ).fetchone()
            dep_cnt = dep_row["cnt"] if dep_row else 0
        lines.extend(
            [
                "",
                "# HELP xcelsior_wallet_depletion_events_total Total wallet depletion events",
                "# TYPE xcelsior_wallet_depletion_events_total counter",
                f"xcelsior_wallet_depletion_events_total {dep_cnt}",
            ]
        )
    except Exception as e:
        log.debug("wallet depletion metric failed: %s", e)

    # Inference cold start rate
    try:
        from db import _get_pg_pool as _pgp3
        from psycopg.rows import dict_row as _dr3

        pool3 = _pgp3()
        with pool3.connection() as conn3:
            conn3.row_factory = _dr3
            cache_row = conn3.execute(
                "SELECT COUNT(*) FILTER (WHERE state = 'loading') as cold, COUNT(*) FILTER (WHERE state = 'ready') as warm FROM worker_model_cache"
            ).fetchone()
            cold = cache_row["cold"] if cache_row else 0
            warm = cache_row["warm"] if cache_row else 0
            total_infer = cold + warm
            cold_rate = round(cold / total_infer, 4) if total_infer > 0 else 0
        lines.extend(
            [
                "",
                "# HELP xcelsior_inference_cold_start_rate Fraction of inference requests requiring cold start",
                "# TYPE xcelsior_inference_cold_start_rate gauge",
                f"xcelsior_inference_cold_start_rate {cold_rate}",
            ]
        )
    except Exception as e:
        log.debug("inference cold start metric failed: %s", e)

    # Inference tokens per second
    try:
        from db import _get_pg_pool as _pgp4
        from psycopg.rows import dict_row as _dr4

        pool4 = _pgp4()
        with pool4.connection() as conn4:
            conn4.row_factory = _dr4
            tps_row = conn4.execute(
                """SELECT COALESCE(SUM((result->>'tokens_generated')::float), 0) /
                          GREATEST(COALESCE(SUM((result->>'duration_sec')::float), 1), 1) as tps
                   FROM jobs WHERE status = 'completed'
                     AND result IS NOT NULL
                     AND result->>'tokens_generated' IS NOT NULL"""
            ).fetchone()
            tps = round(tps_row["tps"], 2) if tps_row else 0
        lines.extend(
            [
                "",
                "# HELP xcelsior_inference_tokens_per_second Aggregate inference throughput",
                "# TYPE xcelsior_inference_tokens_per_second gauge",
                f"xcelsior_inference_tokens_per_second {tps}",
            ]
        )
    except Exception as e:
        log.debug("inference tokens/sec metric failed: %s", e)

    # Append the live prometheus_client default registry (Counter/Gauge/Histogram
    # objects declared in routes/terminal.py and elsewhere). Without this the
    # hand-rolled lines above would miss xcelsior_terminal_* and any other
    # metric that uses the prometheus_client API directly.
    try:
        from prometheus_client import generate_latest, REGISTRY

        # Phase E/E7 — refresh registry-health gauges from the shared
        # DB cache before serialising. The bg_worker process runs the
        # actual probes; without this refresh the API process's gauges
        # would stay at 0 because each process has its own in-module
        # state.
        try:
            import registry_health

            registry_health.refresh_prometheus_gauges()
        except Exception as e:
            log.debug("registry_health gauge refresh failed: %s", e)

        prom_bytes = generate_latest(REGISTRY)
        if prom_bytes:
            lines.append("")
            lines.append(prom_bytes.decode("utf-8").rstrip("\n"))
    except Exception as e:
        log.debug("prometheus_client registry export failed: %s", e)

    from starlette.responses import Response

    return Response(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/", tags=["Infrastructure"], include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/dashboard")


@router.get("/llms.txt", tags=["Infrastructure"])
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


@router.get("/api/alerts/config", tags=["Infrastructure"])
def api_get_alert_config_alias(request: Request):
    """Alias for /alerts/config with /api/ prefix."""
    _require_admin(request)
    safe = {k: ("***" if "pass" in k or "token" in k else v) for k, v in ALERT_CONFIG.items()}
    return {"config": safe}


@router.put("/api/alerts/config", tags=["Infrastructure"])
def api_set_alert_config_alias(cfg: AlertConfig, request: Request):
    """Alias for PUT /alerts/config with /api/ prefix."""
    _require_admin(request)
    updates = {k: v for k, v in cfg.model_dump().items() if v is not None}
    configure_alerts(**updates)
    return {"ok": True, "updated": list(updates.keys())}


@router.get("/api/slurm/instances", tags=["Infrastructure"])
def api_slurm_list_instances(request: Request):
    """List all tracked Slurm jobs."""
    _require_admin(request)
    from slurm_adapter import _load_slurm_map, get_slurm_job_status

    job_map = _load_slurm_map()
    jobs = []
    for xcelsior_id, slurm_id in job_map.items():
        jobs.append({"job_id": xcelsior_id, "slurm_job_id": slurm_id})
    return {"ok": True, "jobs": jobs}
