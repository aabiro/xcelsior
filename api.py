# Xcelsior API v2.3.0
# FastAPI orchestrator. Middleware, lifespan, background tasks, router mounting.

import asyncio
import hmac
import json
import os
import time
import urllib.parse
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

TEMPLATES_DIR = Path(os.path.dirname(__file__)) / "templates"

from db import start_pg_listen, UserStore

from scheduler import (
    list_jobs,
    API_TOKEN,
    start_autoscale_monitor,
    AUTOSCALE_ENABLED,
    log,
)

from routes._deps import (
    AUTH_REQUIRED,
    XCELSIOR_ENV,
    _AUTH_COOKIE_NAME,
    _RATE_BUCKETS,
    _USE_PERSISTENT_AUTH,
    _api_keys,
    _get_real_client_ip,
    _sessions,
    _user_lock,
    broadcast_sse,
    AGENT_RATE_LIMIT_EXEMPT_PREFIXES,
    PUBLIC_PATHS,
    PUBLIC_PATH_PREFIXES,
)


# ── Rate limit constants (read from env, used by middleware) ──────────

RATE_LIMIT_REQUESTS = int(os.environ.get("XCELSIOR_RATE_LIMIT_REQUESTS", "300"))
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("XCELSIOR_RATE_LIMIT_WINDOW_SEC", "60"))


# ── OpenAPI Tag Definitions ───────────────────────────────────────────

OPENAPI_TAGS = [
    {"name": "Hosts", "description": "GPU host registration, admission gating, and management."},
    {"name": "Jobs", "description": "Job submission, scheduling, and lifecycle management."},
    {
        "name": "Billing",
        "description": "Wallet management, invoicing, CAF exports, refunds. Credit-first CAD billing.",
    },
    {"name": "Marketplace", "description": "Rig listings, browsing, and marketplace billing."},
    {
        "name": "Spot Pricing",
        "description": "Dynamic spot pricing, interruptible jobs, preemption cycles.",
    },
    {
        "name": "Reputation",
        "description": "Trust scoring, verification tiers (Bronze→Platinum), leaderboards.",
    },
    {
        "name": "Verification",
        "description": "Automated hardware attestation: GPU identity, CUDA, thermals, network.",
    },
    {
        "name": "SLA",
        "description": "Service Level Agreement enforcement, uptime tracking, credit calculation.",
    },
    {
        "name": "Providers",
        "description": "Stripe Connect onboarding, Canadian company registration, payouts.",
    },
    {
        "name": "Artifacts",
        "description": "Presigned upload/download URLs for model weights, checkpoints, outputs.",
    },
    {
        "name": "Jurisdiction",
        "description": "Canada-first scheduling, province filtering, data residency traces.",
    },
    {
        "name": "Compliance",
        "description": "Province compliance matrix, tax rates, Quebec PIA checks.",
    },
    {
        "name": "Privacy",
        "description": "PIPEDA consent management, retention policies, privacy officer config.",
    },
    {
        "name": "Transparency",
        "description": "Legal request handling, CLOUD Act canary, transparency reports.",
    },
    {
        "name": "Telemetry",
        "description": "Real-time GPU metrics: utilization, temperature, memory, power.",
    },
    {
        "name": "Agent",
        "description": "Worker agent endpoints: work assignment, leases, benchmarks, mining alerts.",
    },
    {"name": "Autoscale", "description": "Auto-scaling pool management and provisioning cycles."},
    {"name": "Events", "description": "Event sourcing, state machine transitions, audit trail."},
    {
        "name": "Infrastructure",
        "description": "Health checks, readiness probes, metrics, SSE streaming, dashboard.",
    },
]


# ── Background Task Lifespan Manager ─────────────────────────────────

import threading as _bg_threading

_bg_stop_event = _bg_threading.Event()
_bg_threads: list[_bg_threading.Thread] = []


def _bg_worker(name, func, interval_sec):
    """Generic background worker that calls func every interval_sec."""
    log.info("BG TASK started: %s (interval=%ds)", name, interval_sec)
    while not _bg_stop_event.is_set():
        try:
            func()
        except Exception as e:
            log.error("BG TASK %s error: %s", name, e)
        _bg_stop_event.wait(interval_sec)
    log.info("BG TASK stopped: %s", name)


def _start_background_tasks():
    """Start all periodic background tasks as daemon threads."""
    global _bg_threads

    tasks = []

    # 1. Auto-billing cycle (every 5 minutes)
    def _billing_cycle():
        try:
            from billing import get_billing_engine
            be = get_billing_engine()
            with otel_span("billing.cycle"):
                be.auto_billing_cycle()
                be.check_low_balance_and_topup()
                be.stop_jobs_for_suspended_wallets()
        except Exception as e:
            log.error("Billing cycle error: %s", e)
    tasks.append(("billing_cycle", _billing_cycle, 300))

    # 2. Stripe webhook event processor (every 30 seconds)
    def _webhook_processor():
        try:
            from stripe_connect import get_stripe_manager
            sm = get_stripe_manager()
            with otel_span("webhook.process_pending"):
                sm.process_pending_events()
        except Exception as e:
            log.error("Webhook processor error: %s", e)
    tasks.append(("webhook_processor", _webhook_processor, 30))

    # 3. Spot price updater (every 10 minutes)
    def _spot_updater():
        try:
            from marketplace import get_marketplace_engine
            me = get_marketplace_engine()
            me.update_spot_prices()
            me.expire_reservations()
        except Exception as e:
            log.error("Spot updater error: %s", e)
    tasks.append(("spot_updater", _spot_updater, 600))

    # 4. Inference scaledown (every 5 minutes)
    def _inference_scaledown():
        try:
            from inference import get_inference_engine
            ie = get_inference_engine()
            ie.scaledown_idle_workers()
        except Exception as e:
            log.error("Inference scaledown error: %s", e)
    tasks.append(("inference_scaledown", _inference_scaledown, 300))

    # 5. Cloud burst evaluator (every 2 minutes)
    def _burst_evaluator():
        try:
            from cloudburst import get_burst_engine
            cbe = get_burst_engine()
            cbe.evaluate_burst_need()
            cbe.drain_idle_instances()
            cbe.update_burst_spending()
        except Exception as e:
            log.error("Burst evaluator error: %s", e)
    tasks.append(("burst_evaluator", _burst_evaluator, 120))

    # 6. SLA credit issuance (every hour)
    def _sla_credits():
        try:
            from sla import get_sla_engine
            se = get_sla_engine()
            se.auto_issue_credits()
        except Exception as e:
            log.error("SLA credits error: %s", e)
    tasks.append(("sla_credits", _sla_credits, 3600))

    # 7. Event snapshotting (every 15 minutes)
    def _event_snapshots():
        try:
            from events import get_snapshot_manager, get_event_store
            sm = get_snapshot_manager()
            sm.snapshot_all_jobs(get_event_store())
        except Exception as e:
            log.error("Event snapshots error: %s", e)
    tasks.append(("event_snapshots", _event_snapshots, 900))

    # 8. Data retention / privacy purge (every 6 hours)
    def _privacy_purge():
        try:
            from privacy import get_lifecycle_manager, get_consent_manager
            lm = get_lifecycle_manager()
            lm.purge_expired()
            cm = get_consent_manager()
            cm.expire_implied_consents()
        except Exception as e:
            log.error("Privacy purge error: %s", e)
    tasks.append(("privacy_purge", _privacy_purge, 21600))

    # 9. Session cleanup (every hour)
    def _session_cleanup():
        try:
            count = UserStore.cleanup_expired_sessions()
            if count:
                log.info("Session cleanup: purged %d expired sessions", count)
        except Exception as e:
            log.error("Session cleanup error: %s", e)
    tasks.append(("session_cleanup", _session_cleanup, 3600))

    # 10. FINTRAC compliance check (every hour)
    def _fintrac_check():
        try:
            from billing import get_billing_engine
            be = get_billing_engine()
            be.fintrac_check_transaction(amount_cad=0, user_id="__periodic_scan__")
        except Exception as e:
            log.debug("fintrac periodic check error (expected for zero amount): %s", e)
    tasks.append(("fintrac_check", _fintrac_check, 3600))

    # 11. Job log cleanup — prune old entries from job_logs (daily)
    def _job_log_cleanup():
        try:
            from db import _get_pg_pool
            cutoff = time.time() - (7 * 86400)  # 7 days
            pool = _get_pg_pool()
            with pool.connection() as conn:
                cur = conn.execute("DELETE FROM job_logs WHERE ts < %s", (cutoff,))
                deleted = cur.rowcount
                conn.commit()
            if deleted:
                log.info("Job log cleanup: purged %d old log rows", deleted)
        except Exception as e:
            log.error("Job log cleanup error: %s", e)
    tasks.append(("job_log_cleanup", _job_log_cleanup, 86400))

    for name, func, interval in tasks:
        t = _bg_threading.Thread(target=_bg_worker, args=(name, func, interval), daemon=True)
        t.start()
        _bg_threads.append(t)

    log.info("LIFESPAN: started %d background tasks", len(tasks))


def _stop_background_tasks():
    """Signal all background tasks to stop."""
    _bg_stop_event.set()
    for t in _bg_threads:
        t.join(timeout=5)
    log.info("LIFESPAN: all background tasks stopped")


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan: start background tasks on startup, stop on shutdown."""
    # ── One-time backfill: ensure owner field exists on active jobs ────
    try:
        from db import get_pg_pool
        pool = get_pg_pool()
        with pool.connection() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET payload = jsonb_set(
                    payload,
                    '{owner}',
                    to_jsonb(COALESCE(payload->>'submitted_by', ''))
                )
                WHERE status IN ('running', 'queued', 'assigned')
                  AND (payload->>'owner' IS NULL OR payload->>'owner' = '')
                  AND payload->>'submitted_by' IS NOT NULL
                """
            )
            if cur.rowcount:
                log.info("BACKFILL: set owner on %d active jobs", cur.rowcount)
            conn.commit()
    except Exception as e:
        log.warning("Owner backfill skipped: %s", e)

    _start_background_tasks()
    yield
    _stop_background_tasks()


# ── FastAPI Application ───────────────────────────────────────────────

app = FastAPI(
    title="Xcelsior",
    version="2.3.0",
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
        {"url": "http://localhost:9500", "description": "Development"},
    ],
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── CORS ──────────────────────────────────────────────────────────────

_CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "XCELSIOR_CORS_ORIGINS",
        "https://xcelsior.ca,https://www.xcelsior.ca",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


# ── CSRF Origin Validation ────────────────────────────────────────────

_CSRF_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_CSRF_ALLOWED_ORIGINS = {urllib.parse.urlparse(o).netloc for o in _CORS_ORIGINS}


@app.middleware("http")
async def csrf_origin_check(request: Request, call_next):
    if request.method not in _CSRF_SAFE_METHODS:
        origin = request.headers.get("origin")
        if origin:
            origin_host = urllib.parse.urlparse(origin).netloc
            if origin_host and origin_host not in _CSRF_ALLOWED_ORIGINS:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-origin request rejected"},
                )
    return await call_next(request)


# ── OpenTelemetry Instrumentation ─────────────────────────────────────

_OTEL_ENABLED = False
_otel_tracer = None

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.trace.propagation import set_global_textmap
    from opentelemetry.propagators.composite import CompositePropagator
    from opentelemetry.trace import StatusCode as _OtelStatusCode

    _otel_resource = Resource.create({SERVICE_NAME: "xcelsior-api"})
    _otel_provider = TracerProvider(resource=_otel_resource)

    _otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if _otel_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        _otel_exporter = OTLPSpanExporter(endpoint=_otel_endpoint)
        _otel_provider.add_span_processor(BatchSpanProcessor(_otel_exporter))

    trace.set_tracer_provider(_otel_provider)

    from opentelemetry.propagators.textmap import DefaultGetter
    from opentelemetry.trace.propagation import TraceContextTextMapPropagator
    from opentelemetry.baggage.propagation import W3CBaggagePropagator
    _w3c_propagator = CompositePropagator([
        TraceContextTextMapPropagator(),
        W3CBaggagePropagator(),
    ])
    from opentelemetry.context.contextvars_context import ContextVarsRuntimeContext
    from opentelemetry import context as _otel_ctx
    set_global_textmap(_w3c_propagator)

    FastAPIInstrumentor.instrument_app(app)

    _OTEL_ENABLED = True
    _otel_tracer = trace.get_tracer("xcelsior.api", "2.3.0")
    # Inject tracer into _deps so route modules can use otel_span()
    from routes import _deps as _route_deps
    _route_deps._otel_tracer = _otel_tracer
    log.info("OTEL: OpenTelemetry instrumentation active (endpoint=%s)", _otel_endpoint or "none")
except ImportError:
    log.info("OTEL: opentelemetry not installed — tracing disabled")
except Exception as _otel_err:
    log.warning("OTEL: failed to initialize — %s", _otel_err)


from routes._deps import otel_span  # re-export for tests / bg tasks


# ── Bridge PgEventBus LISTEN/NOTIFY → SSE ────────────────────────────

start_pg_listen(broadcast_sse)


# ── Autoscale Monitor ─────────────────────────────────────────────────

if AUTOSCALE_ENABLED:
    start_autoscale_monitor(
        interval=15,
        callback=lambda p, a, d: broadcast_sse(
            "autoscale",
            {"provisioned": len(p), "assigned": len(a), "deprovisioned": len(d)},
        ),
    )
    log.info("AUTOSCALE MONITOR started (interval=15s)")


# ── Middleware ────────────────────────────────────────────────────────


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Bearer token auth."""

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

        if request.url.path in PUBLIC_PATHS or request.url.path.startswith(PUBLIC_PATH_PREFIXES):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
        else:
            token = request.query_params.get("token", "")
        if not token:
            token = request.cookies.get(_AUTH_COOKIE_NAME, "")

        if not token or not hmac.compare_digest(token, api_token):
            if token:
                if _USE_PERSISTENT_AUTH:
                    session = UserStore.get_session(token)
                    if session:
                        request.state.user_id = session.get("user_id", "")
                        request.state.customer_id = session.get("customer_id", session.get("user_id", ""))
                        return await call_next(request)
                    api_key = UserStore.get_api_key(token)
                    if api_key:
                        request.state.user_id = api_key.get("user_id", "")
                        request.state.customer_id = api_key.get("customer_id", api_key.get("user_id", ""))
                        return await call_next(request)
                else:
                    with _user_lock:
                        if token in _sessions and _sessions[token]["expires_at"] > time.time():
                            return await call_next(request)
                        if token in _api_keys:
                            return await call_next(request)
            return JSONResponse(
                status_code=401,
                content={"ok": False, "error": {"code": "unauthorized", "message": "Unauthorized"}},
            )

        return await call_next(request)


app.add_middleware(TokenAuthMiddleware)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory IP rate limiting for API safety."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in PUBLIC_PATHS:
            return await call_next(request)
        if path.startswith(AGENT_RATE_LIMIT_EXEMPT_PREFIXES):
            return await call_next(request)

        now = time.time()
        client_ip = _get_real_client_ip(request)
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


class ReadOnlyScopeMiddleware(BaseHTTPMiddleware):
    """Enforce read-only API key scopes — block mutating requests."""

    async def dispatch(self, request: Request, call_next):
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
            key_data = None
            if _USE_PERSISTENT_AUTH:
                key_data = UserStore.get_api_key(token)
            else:
                with _user_lock:
                    key_data = _api_keys.get(token)
            if key_data and key_data.get("scope") == "read-only":
                return JSONResponse(
                    status_code=403,
                    content={
                        "ok": False,
                        "error": {
                            "code": "read_only_key",
                            "message": "This API key has read-only scope",
                        },
                    },
                )
        return await call_next(request)


app.add_middleware(ReadOnlyScopeMiddleware)


class ComplianceGateMiddleware:
    """Enforce Canadian compliance at the API boundary.

    Pure ASGI implementation — avoids BaseHTTPMiddleware's response-buffering
    which causes "Response content longer than Content-Length" on streaming
    endpoints (SSE, large downloads).
    """

    CONSENT_REQUIRED_PATHS = {"/api/jobs", "/api/v2/marketplace/allocate"}
    RESIDENCY_PATHS = {"/api/jobs", "/api/v2/scheduler/process-binpack"}

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        path = scope.get("path", "")
        raw_headers = dict(scope.get("headers", []))

        # Consent check — only blocks early, never buffers response
        if method == "POST" and path in self.CONSENT_REQUIRED_PATHS:
            customer_id = raw_headers.get(b"x-customer-id", b"").decode()
            if customer_id:
                try:
                    from privacy import get_consent_manager
                    cm = get_consent_manager()
                    if not cm.has_consent(customer_id, "billing"):
                        response = JSONResponse(
                            status_code=403,
                            content={"error": "CASL consent required for billing purpose"},
                        )
                        await response(scope, receive, send)
                        return
                except Exception as e:
                    log.debug("Consent check skipped: %s", e)

        # Attach province + redact_pii to request state so route handlers can read them
        province = raw_headers.get(b"x-province", b"").decode().upper()
        if "state" not in scope:
            from starlette.datastructures import State
            scope["state"] = State()
        scope["state"].province = province if (province and path in self.RESIDENCY_PATHS) else ""
        scope["state"].redact_pii = True

        # Wrap send to inject compliance headers without buffering the body
        async def send_with_compliance(message):
            if message["type"] == "http.response.start":
                hdrs = list(message.get("headers", []))
                hdrs.append((b"x-data-residency", b"CA"))
                hdrs.append((b"x-compliance-version", b"PIPEDA-2024"))
                message = {**message, "headers": hdrs}
            await send(message)

        await self.app(scope, receive, send_with_compliance)


app.add_middleware(ComplianceGateMiddleware)


# ── Exception Handlers ────────────────────────────────────────────────


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    accept = request.headers.get("accept", "")
    if exc.status_code in (404, 403, 500) and "text/html" in accept:
        title = {404: "Page Not Found", 403: "Forbidden", 500: "Server Error"}.get(
            exc.status_code, "Error"
        )
        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{exc.status_code} — {title} | Xcelsior</title>
<style>
body{{margin:0;background:#0a0a0a;color:#e0e0e0;font-family:'Courier New',monospace;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100vh}}
.code{{font-size:120px;font-weight:bold;color:#ff3333;margin:0;line-height:1}}.title{{font-size:24px;margin:12px 0 8px;color:#fff}}
.msg{{color:#888;margin-bottom:24px;text-align:center;max-width:480px}}.home{{color:#2196f3;text-decoration:none;padding:8px 24px;border:1px solid #2196f3;border-radius:4px}}.home:hover{{background:#2196f3;color:#000}}
.maple{{font-size:64px;margin-bottom:16px}}</style></head>
<body><div class="maple">🍁</div><p class="code">{exc.status_code}</p><p class="title">{title}</p>
<p class="msg">{exc.detail}</p><a href="/dashboard" class="home">← Back to Dashboard</a></body></html>"""
        return HTMLResponse(content=html, status_code=exc.status_code)
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


# ── Mount all route modules ──────────────────────────────────────────

from routes import ALL_ROUTERS

for router in ALL_ROUTERS:
    app.include_router(router)
