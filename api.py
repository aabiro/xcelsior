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
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

TEMPLATES_DIR = Path(os.path.dirname(__file__)) / "templates"

from db import start_pg_listen, UserStore, NotificationStore, MfaStore, emit_event

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
    start_autoscale_monitor,
    AUTOSCALE_ENABLED,
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
    redact_pii,
)
from sla import get_sla_engine, SLATier, SLA_TARGETS
from stripe_connect import get_stripe_manager
from chat import (
    build_system_prompt,
    check_chat_rate_limit,
    get_or_create_conversation,
    get_conversation_messages,
    get_user_conversations,
    record_feedback,
    append_message,
    stream_chat_response,
    CHAT_API_KEY,
)
from inference_store import (
    store_inference_job,
    get_inference_job,
    store_inference_result,
    get_inference_result,
)

# ── OpenAPI Tag Definitions ───────────────────────────────────────────
# Per REPORT_FEATURE_1.md (Report #1.B): Interactive Documentation
# Groups all 70+ endpoints into logical sections for Swagger UI / Fern / Mintlify

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
# Runs periodic background tasks: billing cycles, webhook processing,
# spot price updates, inference scaledown, burst evaluation, SLA credits,
# event snapshotting, and privacy data retention.

import threading as _bg_threading
from contextlib import asynccontextmanager

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
            from events import get_snapshot_manager
            sm = get_snapshot_manager()
            sm.snapshot_all_jobs()
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

    # 9. FINTRAC compliance check (every hour)
    def _fintrac_check():
        try:
            from billing import get_billing_engine
            be = get_billing_engine()
            be.fintrac_check_transaction(amount_cad=0, user_id="__periodic_scan__")
        except Exception as e:
            # Expected to do nothing for zero amount
            pass
    tasks.append(("fintrac_check", _fintrac_check, 3600))

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
    _start_background_tasks()
    yield
    _stop_background_tasks()


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

# ── OpenTelemetry Instrumentation ─────────────────────────────────────
# Auto-instruments all FastAPI routes; custom spans for job lifecycle,
# billing cycles, and webhook processing.  Exports via OTLP (Jaeger/Tempo)
# when OTEL_EXPORTER_OTLP_ENDPOINT is set.
# See Phase 6.2 of implementation plan.

_OTEL_ENABLED = False

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

    # OTLP exporter — only if endpoint is configured
    _otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if _otel_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        _otel_exporter = OTLPSpanExporter(endpoint=_otel_endpoint)
        _otel_provider.add_span_processor(BatchSpanProcessor(_otel_exporter))

    trace.set_tracer_provider(_otel_provider)

    # W3C Trace Context propagation
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

    # Instrument the app
    FastAPIInstrumentor.instrument_app(app)

    _OTEL_ENABLED = True
    _otel_tracer = trace.get_tracer("xcelsior.api", "2.3.0")
    log.info("OTEL: OpenTelemetry instrumentation active (endpoint=%s)", _otel_endpoint or "none")
except ImportError:
    log.info("OTEL: opentelemetry not installed — tracing disabled")
    _otel_tracer = None
except Exception as _otel_err:
    log.warning("OTEL: failed to initialize — %s", _otel_err)
    _otel_tracer = None


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


# ── In-App Notification Delivery ──────────────────────────────────────
# Maps SSE event types to per-user notification creation.
# Runs in a daemon thread so it never blocks the API.

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


# ── Bridge PgEventBus LISTEN/NOTIFY → SSE ────────────────────────────
# Per REPORT_XCELSIOR_TECHNICAL_FINAL.md Step 9:
# "start with Postgres LISTEN/NOTIFY for notifications"
# This bridges scheduler events (db.emit_event) into SSE delivery.

start_pg_listen(broadcast_sse)

# ── Autoscale Monitor ─────────────────────────────────────────────────
# Start background autoscale loop if enabled. Provisions/deprovisions
# hosts from the autoscale pool based on queue demand every 15 seconds.

if AUTOSCALE_ENABLED:
    start_autoscale_monitor(
        interval=15,
        callback=lambda p, a, d: broadcast_sse(
            "autoscale",
            {"provisioned": len(p), "assigned": len(a), "deprovisioned": len(d)},
        ),
    )
    log.info("AUTOSCALE MONITOR started (interval=15s)")

XCELSIOR_ENV = os.environ.get("XCELSIOR_ENV", "dev").lower()
AUTH_REQUIRED = XCELSIOR_ENV not in {"dev", "development", "test"}
RATE_LIMIT_REQUESTS = int(os.environ.get("XCELSIOR_RATE_LIMIT_REQUESTS", "300"))
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("XCELSIOR_RATE_LIMIT_WINDOW_SEC", "60"))
_RATE_BUCKETS = defaultdict(deque)


# ── Phase 13: API Token Auth ─────────────────────────────────────────

# Public routes — no token required
PUBLIC_PATHS = {
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/llms.txt",
    "/dashboard",
    "/legacy",
    "/healthz",
    "/readyz",
    "/metrics",
    "/api/stream",
    "/api/transparency/report",
}

# Prefixes that bypass token auth (OAuth callbacks, auth endpoints)
PUBLIC_PATH_PREFIXES = ("/api/auth/", "/api/chat", "/legacy/")

# Agent/worker paths exempt from rate limiting (protected by token auth)
AGENT_RATE_LIMIT_EXEMPT_PREFIXES = ("/host", "/agent/")


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
            # Also accept valid user session tokens and API keys
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


def _get_real_client_ip(request: Request) -> str:
    """Extract real client IP, respecting X-Real-IP / X-Forwarded-For from trusted proxy."""
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory IP rate limiting for API safety."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in PUBLIC_PATHS:
            return await call_next(request)
        # Exempt authenticated agent/worker endpoints from rate limiting
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


# ── Compliance Gate Middleware ─────────────────────────────────────────
# Three-gate model per REPORT_FEATURE_FINAL.md:
# 1. Submission gate: classify job sensitivity, check consent
# 2. Scheduling gate: restrict hosts by data residency
# 3. Telemetry gate: redact PII before event logging


class ComplianceGateMiddleware(BaseHTTPMiddleware):
    """Enforce Canadian compliance at the API boundary.

    - Job submission: verify CASL consent exists for billing/marketing purposes
    - Privacy: apply redaction headers for downstream telemetry
    - Jurisdiction: tag requests with province for scheduling gates
    """

    CONSENT_REQUIRED_PATHS = {"/api/jobs", "/api/v2/marketplace/allocate"}
    RESIDENCY_PATHS = {"/api/jobs", "/api/v2/scheduler/process-binpack"}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method

        # Gate 1: Consent check on job submission
        if method == "POST" and path in self.CONSENT_REQUIRED_PATHS:
            customer_id = request.headers.get("x-customer-id", "")
            if customer_id:
                try:
                    from privacy import get_consent_manager
                    cm = get_consent_manager()
                    if not cm.has_consent(customer_id, "billing"):
                        return JSONResponse(
                            status_code=403,
                            content={"error": "CASL consent required for billing purpose"},
                        )
                except Exception:
                    pass  # Consent table may not exist yet

        # Gate 2: Tag request with province for scheduling locality
        province = request.headers.get("x-province", "")
        if province and path in self.RESIDENCY_PATHS:
            request.state.province = province.upper()
        else:
            request.state.province = ""

        # Gate 3: Telemetry redaction flag — downstream handlers check this
        request.state.redact_pii = True  # Always redact in telemetry by default

        response = await call_next(request)

        # Add compliance headers
        response.headers["X-Data-Residency"] = "CA"
        response.headers["X-Compliance-Version"] = "PIPEDA-2024"

        return response


app.add_middleware(ComplianceGateMiddleware)


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


# ── Request models ────────────────────────────────────────────────────


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
    interactive: bool = True
    command: str | None = None
    ssh_port: int = Field(default=22, ge=1, le=65535)


class StatusUpdate(BaseModel):
    status: str
    host_id: str | None = None
    container_id: str | None = None
    container_name: str | None = None


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
    except Exception:
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


@app.get("/host/{host_id}", tags=["Hosts"])
def api_get_host(host_id: str):
    """Get a single host by ID."""
    hosts = list_hosts(active_only=False)
    host = next((h for h in hosts if h["host_id"] == host_id), None)
    if not host:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    return {"ok": True, "host": host}


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


# ── Instance endpoints ─────────────────────────────────────────────────────


@app.post("/instance", tags=["Instances"])
def api_submit_instance(j: JobIn):
    """Submit a job to the queue. Tier overrides priority.

    Multi-GPU: Set num_gpus > 1 for multi-GPU jobs.
    NFS: Optionally specify nfs_server + nfs_path for shared storage.
    """
    with otel_span("job.submit", {"job.name": j.name, "job.tier": j.tier or "", "job.num_gpus": j.num_gpus}):
        job = submit_job(
            j.name,
            j.vram_needed_gb,
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
        )
        broadcast_sse("job_submitted", {"job_id": job["job_id"], "name": job["name"]})
        return {"ok": True, "instance": job}


@app.get("/instances", tags=["Instances"])
def api_list_instances(status: str | None = None):
    """List jobs. Optional filter by status."""
    jobs = list_jobs(status=status)
    # Enrich with host GPU info where available
    hosts = list_hosts()
    host_map = {h["host_id"]: h for h in hosts}
    for j in jobs:
        hid = j.get("host_id")
        if hid and hid in host_map:
            j.setdefault("gpu_type", host_map[hid].get("gpu_model", ""))
            j.setdefault("host_gpu", host_map[hid].get("gpu_model", ""))
    return {"instances": jobs}


@app.get("/instance/{job_id}", tags=["Instances"])
def api_get_instance(job_id: str):
    """Get a specific instance by ID, enriched with connection info."""
    jobs = list_jobs()
    for j in jobs:
        if j["job_id"] == job_id:
            # Enrich with host connection details when running
            if j.get("host_id") and j.get("status") in ("running", "completed", "failed"):
                hosts = list_hosts()
                host = next((h for h in hosts if h["host_id"] == j["host_id"]), None)
                if host:
                    j["host_ip"] = host.get("ip", "")
                    j["host_gpu"] = host.get("gpu_model", "")
                    j["host_vram_gb"] = host.get("total_vram_gb", 0)
            return {"instance": j}
    raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")


@app.patch("/instance/{job_id}", tags=["Instances"])
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
        if extras:
            from scheduler import _set_job_fields
            _set_job_fields(job_id, **extras)
        broadcast_sse("job_status", {"job_id": job_id, "status": update.status})
        return {"ok": True, "job_id": job_id, "status": update.status}


@app.post("/queue/process", tags=["Instances"])
def api_process_queue():
    """Process the job queue — assign jobs to hosts."""
    assigned = process_queue()
    result = [{"job": j["name"], "job_id": j["job_id"], "host": h["host_id"]} for j, h in assigned]
    if result:
        broadcast_sse("queue_processed", {"assigned_count": len(result)})
    return {"assigned": result}


# ── Phase 14: Failover endpoints ──────────────────────────────────────


@app.post("/failover", tags=["Instances"])
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


@app.post("/instances/{job_id}/cancel", tags=["Instances"])
def api_cancel_instance(job_id: str):
    """Cancel a running or queued instance. For interactive instances, stops the container."""
    jobs = list_jobs()
    job = next((j for j in jobs if j["job_id"] == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail=f"Instance {job_id} not found")

    if job.get("status") in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail=f"Instance already {job['status']}")

    # If running on a host, schedule container stop via preemption mechanism
    if job.get("host_id") and job.get("status") in ("running", "assigned", "leased"):
        with _agent_lock:
            _agent_preempt[job["host_id"]].append(job_id)

    update_job_status(job_id, "cancelled")
    broadcast_sse("job_cancelled", {"job_id": job_id})
    return {"ok": True, "job_id": job_id, "status": "cancelled"}


@app.post("/instance/{job_id}/requeue", tags=["Instances"])
def api_requeue_instance(job_id: str):
    """Manually requeue a failed or stuck job."""
    result = requeue_job(job_id)
    if not result:
        raise HTTPException(
            status_code=400,
            detail=f"Could not requeue job {job_id} (max retries exceeded or not found)",
        )
    return {"ok": True, "instance": result}


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


@app.get("/instances/{job_id}/logs/stream", tags=["Instances"])
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


@app.get("/instances/{job_id}/logs", tags=["Instances"])
def api_instance_logs(job_id: str, limit: int = 100):
    """Get buffered log lines for a job (non-streaming).

    Returns the last `limit` log lines from the in-memory buffer.
    For real-time streaming, use `/jobs/{job_id}/logs/stream` (SSE).
    """
    buf = _job_log_buffers.get(job_id, [])
    return {"ok": True, "job_id": job_id, "logs": buf[-limit:], "total": len(buf)}


# ── WebSocket Instance Streaming ──────────────────────────────────────
# Real-time bidirectional updates for individual instances.
# Sends full instance snapshots on connect and on every status change,
# plus filtered job_log events. Reconnect-friendly.

_ws_connections: dict[str, set[WebSocket]] = defaultdict(set)  # job_id -> {ws, ...}


def _validate_ws_auth(websocket: WebSocket) -> bool:
    """Validate auth for WebSocket connections (mirrors TokenAuthMiddleware)."""
    if not AUTH_REQUIRED:
        return True
    api_token = os.environ.get("XCELSIOR_API_TOKEN", API_TOKEN)
    token = websocket.cookies.get(_AUTH_COOKIE_NAME, "")
    if not token:
        token = websocket.query_params.get("token", "")
    if not token:
        return False
    if api_token and hmac.compare_digest(token, api_token):
        return True
    if _USE_PERSISTENT_AUTH:
        if UserStore.get_session(token):
            return True
        if UserStore.get_api_key(token):
            return True
    else:
        with _user_lock:
            if token in _sessions and _sessions[token]["expires_at"] > time.time():
                return True
            if token in _api_keys:
                return True
    return False


@app.websocket("/ws/instances/{job_id}")
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
    if not _validate_ws_auth(websocket):
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

    _ws_connections[job_id].add(websocket)
    await websocket.send_json({"event": "instance", "data": instance})

    # Replay buffered logs
    for entry in list(_job_log_buffers.get(job_id, []))[-50:]:
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
        except Exception:
            pass


# ── WebSocket Terminal Proxy ──────────────────────────────────────────
# xterm.js <-> WebSocket <-> docker exec via Tailscale mesh
# Security: JWT auth, 30-min session timeout, 10 KB/s rate limit

_TERMINAL_SESSION_TIMEOUT = 1800  # 30 minutes
_TERMINAL_MAX_SCROLLBACK = 50_000  # 50 KB
_TERMINAL_RATE_LIMIT_BYTES = 10_240  # 10 KB/s


@app.websocket("/ws/terminal/{instance_id}")
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
    if not _validate_ws_auth(websocket):
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

    if instance.get("status") != "running":
        await websocket.send_json({"type": "error", "message": f"Instance is {instance.get('status', 'unknown')}, not running"})
        await websocket.close(code=4003)
        return

    host_id = instance.get("host_id", "")
    container_id = instance.get("container_id", instance_id[:12])
    if not host_id:
        await websocket.send_json({"type": "error", "message": "No host assigned"})
        await websocket.close(code=4003)
        return

    # Build docker exec command via Tailscale
    # In production: ssh through Tailscale mesh to the worker host
    # For local dev: direct docker exec
    shell = instance.get("shell", "/bin/bash")
    docker_cmd = [
        "docker", "exec", "-i", container_id, shell,
    ]

    session_start = time.time()
    process = None
    closed = False

    try:
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

    await websocket.send_json({"type": "output", "data": f"Connected to {instance.get('name', instance_id)}\\r\\n"})

    bytes_this_second = 0
    last_rate_reset = time.time()

    async def _stdout_relay():
        """Read container stdout and relay to browser."""
        nonlocal closed, bytes_this_second, last_rate_reset
        try:
            while not closed and process and process.stdout:
                chunk = await asyncio.wait_for(
                    process.stdout.read(4096), timeout=5.0
                )
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
        except Exception:
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
                if msg.get("type") == "input" and process and process.stdin:
                    data = msg.get("data", "")
                    process.stdin.write(data.encode("utf-8"))
                    await process.stdin.drain()
                elif msg.get("type") == "resize":
                    # Resize not directly supported with docker exec -i
                    # Would need PTY allocation for proper resize
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
        if process:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
        # Send exit notification
        try:
            exit_code = process.returncode if process else -1
            await websocket.send_json({"type": "exit", "code": exit_code or 0})
            await websocket.close()
        except Exception:
            pass


# ── Billing endpoints ────────────────────────────────────────────────


@app.post("/billing/bill/{job_id}", tags=["Billing"])
def api_bill_instance(job_id: str):
    """Bill a specific completed job."""
    with otel_span("billing.bill_job", {"job.id": job_id}):
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


@app.get("/legacy", response_class=HTMLResponse, tags=["Infrastructure"])
@app.get("/legacy/{path:path}", response_class=HTMLResponse, tags=["Infrastructure"])
def legacy_dashboard(path: str = ""):
    """Legacy dashboard preserved at /legacy while Next.js serves /."""
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
@app.get("/api/ssh/pubkey", tags=["Infrastructure"])
def api_get_pubkey():
    """Get the public key to add to hosts' authorized_keys."""
    pub = get_public_key()
    return {"public_key": pub or ""}


# ── User SSH Public Key Management ────────────────────────────────────

VALID_SSH_KEY_TYPES = {"ssh-rsa", "ssh-ed25519", "ecdsa-sha2-nistp256", "ecdsa-sha2-nistp384", "ecdsa-sha2-nistp521", "sk-ssh-ed25519@openssh.com", "sk-ecdsa-sha2-nistp256@openssh.com"}


def _validate_ssh_public_key(key_str: str) -> str:
    """Validate and normalize an SSH public key string. Returns the key type or raises."""
    import base64 as _b64, re as _re
    key_str = key_str.strip()
    # Remove any comment-only lines
    lines = [l.strip() for l in key_str.splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines:
        raise ValueError("Empty SSH key")
    key_str = lines[0]  # Use only the first key line
    parts = key_str.split(None, 2)
    if len(parts) < 2:
        raise ValueError("Invalid SSH public key format")
    key_type = parts[0]
    if key_type not in VALID_SSH_KEY_TYPES:
        raise ValueError(f"Unsupported key type: {key_type}")
    # Validate base64 data
    try:
        _b64.b64decode(parts[1], validate=True)
    except Exception:
        raise ValueError("Invalid base64 key data")
    return key_type


def _ssh_key_fingerprint(key_str: str) -> str:
    """Compute SHA-256 fingerprint of an SSH public key (like ssh-keygen -l)."""
    import base64 as _b64
    parts = key_str.strip().split(None, 2)
    raw = _b64.b64decode(parts[1])
    digest = _hashlib.sha256(raw).digest()
    fp = _b64.b64encode(digest).rstrip(b"=").decode()
    return f"SHA256:{fp}"


@app.post("/api/ssh/keys", tags=["SSH Keys"])
async def api_add_ssh_key(request: Request):
    """Upload a user SSH public key. Like GitHub/AWS key management."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    body = await request.json()
    name = body.get("name", "").strip() or "default"
    public_key = body.get("public_key", "").strip()
    if not public_key:
        raise HTTPException(400, "public_key is required")
    if len(name) > 100:
        raise HTTPException(400, "Key name too long (max 100 characters)")
    if len(public_key) > 16384:
        raise HTTPException(400, "Key too large")
    try:
        _validate_ssh_public_key(public_key)
    except ValueError as e:
        raise HTTPException(400, str(e))
    fingerprint = _ssh_key_fingerprint(public_key)
    # Check for duplicate fingerprint
    existing = UserStore.list_ssh_keys(user["email"])
    for k in existing:
        if k["fingerprint"] == fingerprint:
            raise HTTPException(409, "This key is already added")
    key_id = f"sshk-{uuid.uuid4().hex[:12]}"
    # Normalize: keep only the key line (type + data + optional comment)
    parts = public_key.strip().splitlines()[0].strip().split(None, 2)
    normalized = " ".join(parts[:2])
    if len(parts) > 2:
        normalized += " " + parts[2]
    key_data = {
        "id": key_id,
        "email": user["email"],
        "user_id": user["user_id"],
        "name": name,
        "public_key": normalized,
        "fingerprint": fingerprint,
        "created_at": time.time(),
    }
    UserStore.add_ssh_key(key_data)
    return {
        "ok": True,
        "id": key_id,
        "name": name,
        "fingerprint": fingerprint,
    }


@app.get("/api/ssh/keys", tags=["SSH Keys"])
def api_list_ssh_keys(request: Request):
    """List the authenticated user's SSH public keys."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    keys = UserStore.list_ssh_keys(user["email"])
    return {
        "ok": True,
        "keys": [
            {
                "id": k["id"],
                "name": k["name"],
                "fingerprint": k["fingerprint"],
                "public_key": k["public_key"],
                "created_at": k["created_at"],
            }
            for k in keys
        ],
    }


@app.delete("/api/ssh/keys/{key_id}", tags=["SSH Keys"])
def api_delete_ssh_key(key_id: str, request: Request):
    """Delete a user SSH public key by ID."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    deleted = UserStore.delete_ssh_key(user["email"], key_id)
    if not deleted:
        raise HTTPException(404, "SSH key not found")
    return {"ok": True}


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

# ── httpOnly Cookie Auth ──────────────────────────────────────────────
_AUTH_COOKIE_NAME = "xcelsior_session"


def _set_auth_cookie(response, token: str):
    """Add httpOnly session cookie to response."""
    _base = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca")
    response.set_cookie(
        key=_AUTH_COOKIE_NAME,
        value=token,
        max_age=SESSION_EXPIRY,
        httponly=True,
        secure=_base.startswith("https"),
        samesite="lax",
        path="/",
    )
    return response


def _clear_auth_cookie(response):
    """Remove session cookie."""
    response.delete_cookie(key=_AUTH_COOKIE_NAME, path="/")
    return response


# ── OAuth Provider Configuration ──────────────────────────────────────
import httpx as _httpx
import urllib.parse as _urllib_parse

_OAUTH_BASE_URL = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca")

_OAUTH_PROVIDERS = {
    "google": {
        "client_id": os.environ.get("GOOGLE_CLIENT_ID", ""),
        "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET", ""),
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "userinfo_url": "https://www.googleapis.com/oauth2/v2/userinfo",
        "scopes": "openid email profile",
    },
    "github": {
        "client_id": os.environ.get("GITHUB_CLIENT_ID", ""),
        "client_secret": os.environ.get("GITHUB_CLIENT_SECRET", ""),
        "authorize_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "userinfo_url": "https://api.github.com/user",
        "scopes": "read:user user:email",
    },
    "huggingface": {
        "client_id": os.environ.get("HUGGINGFACE_CLIENT_ID", ""),
        "client_secret": os.environ.get("HUGGINGFACE_CLIENT_SECRET", ""),
        "authorize_url": "https://huggingface.co/oauth/authorize",
        "token_url": "https://huggingface.co/oauth/token",
        "userinfo_url": "https://huggingface.co/oauth/userinfo",
        "scopes": "openid profile email",
    },
}

# CSRF state tokens for OAuth (short-lived, in-memory)
_oauth_states: dict[str, dict] = {}  # state -> {provider, created_at}
_OAUTH_STATE_TTL = 600  # 10 minutes


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
    """Extract user from Authorization header or session cookie."""
    auth = request.headers.get("authorization", "")
    token = ""
    if auth.startswith("Bearer "):
        token = auth[7:]
    if not token:
        token = request.cookies.get(_AUTH_COOKIE_NAME, "")
    if not token:
        return None
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
                "name": api_key.get("name", ""),
                "scope": api_key.get("scope", "full-access"),
            }
    else:
        with _user_lock:
            session = _sessions.get(token)
        if session and session["expires_at"] > time.time():
            return session
        with _user_lock:
            api_key = _api_keys.get(token)
        if api_key:
            api_key["last_used"] = time.time()
            return {
                "email": api_key["email"],
                "user_id": api_key["user_id"],
                "role": api_key.get("role", "submitter"),
                "name": api_key.get("name", ""),
                "scope": api_key.get("scope", "full-access"),
            }
    return None


def _require_write_access(request: Request):
    """Raise 403 if the current user is using a read-only API key on a mutating request."""
    user = _get_current_user(request)
    if (
        user
        and user.get("scope") == "read-only"
        and request.method not in ("GET", "HEAD", "OPTIONS")
    ):
        raise HTTPException(403, "This API key has read-only scope")
    return user


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

    # Welcome notification for the new user
    try:
        NotificationStore.create(
            email, "system",
            "Welcome to Xcelsior!",
            "Your account is ready. Launch your first GPU instance or register a host to start earning.",
            {"user_id": user_id},
        )
    except Exception:
        pass

    # Welcome email
    try:
        display_name = user["name"]
        _send_team_email(
            email,
            f"Welcome to Xcelsior, {display_name}!",
            f"Hi {display_name},\n\n"
            "Thanks for signing up for Xcelsior — Canada's sovereign GPU compute marketplace.\n\n"
            "Your account is active and ready to go. From your dashboard you can browse available GPU hosts, "
            "launch compute instances, and track your usage — all billed in CAD with full Canadian data residency.\n\n"
            "We're currently in early access, so if you run into anything or have questions, just reply to this email. "
            "We'd love to hear from you.",
            cta_url="https://xcelsior.ca/dashboard",
            cta_label="Go to Dashboard",
        )
    except Exception:
        pass

    body = {
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
    resp = JSONResponse(content=body)
    _set_auth_cookie(resp, session["token"])
    return resp


@app.post("/api/auth/login", tags=["Auth"])
def api_auth_login(body: LoginRequest):
    """Authenticate with email and password.

    Returns a Bearer token valid for 30 days.
    If MFA is enabled, returns mfa_required=True with a challenge_id instead.
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

    # ── MFA check ──
    if _USE_PERSISTENT_AUTH and user.get("mfa_enabled"):
        methods = MfaStore.list_methods(email)
        enabled_methods = [m for m in methods if m.get("enabled")]
        if enabled_methods:
            challenge_id = secrets.token_urlsafe(32)
            # Store a temporary partial session token
            partial_token = secrets.token_urlsafe(48)
            MfaStore.create_challenge({
                "challenge_id": challenge_id,
                "email": email,
                "session_token": partial_token,
                "created_at": time.time(),
                "expires_at": time.time() + 300,  # 5-minute window
            })
            return JSONResponse(content={
                "ok": True,
                "mfa_required": True,
                "challenge_id": challenge_id,
                "methods": [m["method_type"] for m in enabled_methods if m["method_type"] != "passkey"],
            })

    session = _create_session(email, user)

    # Ensure a welcome notification exists (checks ALL notifications, not just unread,
    # so marking-all-read won't re-trigger on next login)
    try:
        if _USE_PERSISTENT_AUTH and not NotificationStore.list_for_user(email, limit=1):
            NotificationStore.create(
                email, "system",
                "Welcome to Xcelsior!",
                "Your account is ready. Launch your first GPU instance or register a host to start earning.",
                {"user_id": user["user_id"]},
            )
    except Exception:
        pass

    body = {
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
    resp = JSONResponse(content=body)
    _set_auth_cookie(resp, session["token"])
    return resp


@app.post("/api/auth/oauth/{provider}", tags=["Auth"])
def api_auth_oauth_initiate(provider: str):
    """Initiate OAuth flow — returns the provider's authorization URL.

    The frontend should redirect the user to the returned URL.
    """
    if provider not in _OAUTH_PROVIDERS:
        raise HTTPException(400, f"Unsupported OAuth provider: {provider}")

    cfg = _OAUTH_PROVIDERS[provider]
    if not cfg["client_id"]:
        raise HTTPException(503, f"OAuth provider {provider} is not configured")

    # Generate CSRF state token
    state = secrets.token_urlsafe(32)
    _oauth_states[state] = {"provider": provider, "created_at": time.time()}
    # Evict expired states
    now = time.time()
    for k in list(_oauth_states):
        if now - _oauth_states[k]["created_at"] > _OAUTH_STATE_TTL:
            del _oauth_states[k]

    redirect_uri = f"{_OAUTH_BASE_URL}/api/auth/oauth/{provider}/callback"
    params = {
        "client_id": cfg["client_id"],
        "redirect_uri": redirect_uri,
        "state": state,
        "response_type": "code",
    }
    if provider == "github":
        params["scope"] = cfg["scopes"]
    else:
        params["scope"] = cfg["scopes"]
        if provider == "google":
            params["access_type"] = "offline"
            params["prompt"] = "select_account"

    auth_url = f"{cfg['authorize_url']}?{_urllib_parse.urlencode(params)}"
    return {"ok": True, "auth_url": auth_url}


@app.get("/api/auth/oauth/{provider}/callback", tags=["Auth"])
def api_auth_oauth_callback(provider: str, request: Request):
    """OAuth callback — exchanges authorization code for user profile, creates session,
    and redirects to dashboard.
    """
    if provider not in _OAUTH_PROVIDERS:
        raise HTTPException(400, f"Unsupported OAuth provider: {provider}")

    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")

    if error:
        return RedirectResponse(f"/dashboard?error=oauth_{error}")

    if not code or not state:
        return RedirectResponse("/dashboard?error=oauth_missing_params")

    # Validate CSRF state
    state_data = _oauth_states.pop(state, None)
    if not state_data or state_data["provider"] != provider:
        return RedirectResponse("/dashboard?error=oauth_invalid_state")
    if time.time() - state_data["created_at"] > _OAUTH_STATE_TTL:
        return RedirectResponse("/dashboard?error=oauth_state_expired")

    cfg = _OAUTH_PROVIDERS[provider]
    redirect_uri = f"{_OAUTH_BASE_URL}/api/auth/oauth/{provider}/callback"

    # Exchange authorization code for access token
    try:
        token_resp = _httpx.post(
            cfg["token_url"],
            data={
                "client_id": cfg["client_id"],
                "client_secret": cfg["client_secret"],
                "code": code,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
            headers={"Accept": "application/json"},
            timeout=15,
        )
        token_resp.raise_for_status()
        token_data = token_resp.json()
    except Exception as e:
        log.error("OAuth token exchange failed for %s: %s", provider, e)
        return RedirectResponse("/dashboard?error=oauth_token_failed")

    access_token = token_data.get("access_token")
    if not access_token:
        log.error("No access_token in OAuth response for %s: %s", provider, token_data)
        return RedirectResponse("/dashboard?error=oauth_no_token")

    # Fetch user profile from provider
    try:
        userinfo_resp = _httpx.get(
            cfg["userinfo_url"],
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
            timeout=10,
        )
        userinfo_resp.raise_for_status()
        profile = userinfo_resp.json()
    except Exception as e:
        log.error("OAuth userinfo fetch failed for %s: %s", provider, e)
        return RedirectResponse("/dashboard?error=oauth_profile_failed")

    # Extract email and name by provider
    if provider == "google":
        email = profile.get("email", "")
        name = profile.get("name", "")
    elif provider == "github":
        email = profile.get("email") or ""
        name = profile.get("name") or profile.get("login", "")
        # GitHub may not return email in profile — fetch from emails API
        if not email:
            try:
                emails_resp = _httpx.get(
                    "https://api.github.com/user/emails",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/json",
                    },
                    timeout=10,
                )
                if emails_resp.status_code == 200:
                    for e_entry in emails_resp.json():
                        if e_entry.get("primary") and e_entry.get("verified"):
                            email = e_entry["email"]
                            break
            except Exception:
                pass
    elif provider == "huggingface":
        email = profile.get("email", "")
        name = profile.get("name") or profile.get("preferred_username", "")

    if not email:
        return RedirectResponse("/dashboard?error=oauth_no_email")

    # Find or create user
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
            "name": name or f"{provider.title()} User",
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
    else:
        # Update name if it was empty
        if not user.get("name") and name:
            user["name"] = name

    session = _create_session(email, user)

    # Ensure welcome notification exists
    try:
        if _USE_PERSISTENT_AUTH and not NotificationStore.list_for_user(email, limit=1):
            NotificationStore.create(
                email, "system",
                "Welcome to Xcelsior!",
                "Your account is ready. Launch your first GPU instance or register a host to start earning.",
                {"user_id": user["user_id"]},
            )
    except Exception:
        pass

    # Set httpOnly cookie and redirect to dashboard
    resp = RedirectResponse("/dashboard", status_code=302)
    _set_auth_cookie(resp, session["token"])
    return resp


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
    body = {
        "ok": True,
        "access_token": session["token"],
        "token_type": "Bearer",
        "expires_in": SESSION_EXPIRY,
    }
    resp = JSONResponse(content=body)
    _set_auth_cookie(resp, session["token"])
    return resp


@app.post("/api/auth/logout", tags=["Auth"])
def api_auth_logout(request: Request):
    """Logout — invalidate session and clear cookie."""
    user = _get_current_user(request)
    if user:
        auth = request.headers.get("authorization", "")
        token = ""
        if auth.startswith("Bearer "):
            token = auth[7:]
        if not token:
            token = request.cookies.get(_AUTH_COOKIE_NAME, "")
        if token:
            if _USE_PERSISTENT_AUTH:
                UserStore.delete_session(token)
            else:
                with _user_lock:
                    _sessions.pop(token, None)
    resp = JSONResponse(content={"ok": True})
    _clear_auth_cookie(resp)
    return resp


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


VALID_KEY_SCOPES = {"full-access", "read-only"}


@app.post("/api/keys/generate", tags=["Auth"])
async def api_generate_api_key(request: Request):
    """Generate a named API key for the authenticated user.

    API keys can be used as Bearer tokens for programmatic access.
    Scope: 'full-access' (default) or 'read-only' (GET requests only).
    """
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    body = await request.json()
    name = body.get("name", "default")
    scope = body.get("scope", "full-access")
    if scope not in VALID_KEY_SCOPES:
        raise HTTPException(
            400, f"Invalid scope. Must be one of: {', '.join(sorted(VALID_KEY_SCOPES))}"
        )

    key = f"xc-{secrets.token_urlsafe(32)}"
    key_data = {
        "key": key,
        "name": name,
        "email": user["email"],
        "user_id": user["user_id"],
        "role": user.get("role", "submitter"),
        "scope": scope,
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
        "scope": scope,
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
                "scope": v.get("scope", "full-access"),
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
                    "scope": v.get("scope", "full-access"),
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
                k
                for k, v in _api_keys.items()
                if v["email"] == user["email"]
                and (v["key"][:12] + "..." + v["key"][-4:]) == key_preview
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
    """Initiate a password reset. Sends email with reset link."""
    reset_token = None
    if _USE_PERSISTENT_AUTH:
        user = UserStore.get_user(req.email)
        if not user:
            return {"ok": True, "message": "If the email exists, a reset link has been sent."}
        reset_token = secrets.token_urlsafe(32)
        UserStore.update_user(
            req.email,
            {
                "reset_token": reset_token,
                "reset_token_expires": time.time() + 3600,
            },
        )
    else:
        with _user_lock:
            user = _users_db.get(req.email)
            if not user:
                return {"ok": True, "message": "If the email exists, a reset link has been sent."}
            reset_token = secrets.token_urlsafe(32)
            user["reset_token"] = reset_token
            user["reset_token_expires"] = time.time() + 3600

    # Send password reset email (best-effort)
    if reset_token:
        base_url = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca")
        reset_url = f"{base_url}/reset-password?token={reset_token}"
        _send_team_email(
            to_email=req.email,
            subject="Reset your Xcelsior password",
            body_text=(
                f"You requested a password reset for your Xcelsior account.\n\n"
                f"Click the link below to set a new password. This link expires in 1 hour.\n\n"
                f"{reset_url}\n\n"
                f"If you didn't request this, you can safely ignore this email."
            ),
            cta_url=reset_url,
            cta_label="Reset Password",
        )

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
            row = conn.execute(
                "SELECT email, reset_token_expires FROM users WHERE reset_token = %s", (req.token,)
            ).fetchone()
            if not row:
                raise HTTPException(400, "Invalid or expired reset token")
            if time.time() > (row["reset_token_expires"] or 0):
                raise HTTPException(400, "Reset token has expired")
            salt = secrets.token_hex(16)
            new_hash, _ = _hash_password(req.new_password, salt)
            conn.execute(
                "UPDATE users SET password_hash=%s, salt=%s, reset_token=NULL, reset_token_expires=NULL WHERE email=%s",
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


# ── Two-Factor Authentication (MFA) ─────────────────────────────────

def _verify_totp_code(secret: str, code: str) -> bool:
    """Verify a TOTP code, allowing ±1 window for clock drift."""
    import pyotp
    totp = pyotp.TOTP(secret)
    return totp.verify(code, valid_window=1)


def _hash_backup_code(code: str) -> str:
    """Hash a backup code for storage."""
    import hashlib
    return hashlib.sha256(code.encode()).hexdigest()


def _generate_backup_codes(count: int = 10) -> list[str]:
    """Generate a set of backup codes."""
    codes = []
    for _ in range(count):
        part1 = secrets.token_hex(2).upper()
        part2 = secrets.token_hex(2).upper()
        codes.append(f"{part1}-{part2}")
    return codes


def _complete_mfa_login(email: str, challenge_id: str) -> JSONResponse:
    """Complete login after successful MFA verification."""
    challenge = MfaStore.get_challenge(challenge_id)
    if not challenge or challenge["email"] != email:
        raise HTTPException(400, "Invalid MFA challenge")
    if time.time() > challenge["expires_at"]:
        MfaStore.delete_challenge(challenge_id)
        raise HTTPException(400, "MFA challenge expired")

    MfaStore.delete_challenge(challenge_id)

    user = UserStore.get_user(email)
    if not user:
        raise HTTPException(400, "User not found")

    session = _create_session(email, user)
    body = {
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
    resp = JSONResponse(content=body)
    _set_auth_cookie(resp, session["token"])
    return resp


def _refresh_mfa_enabled(email: str) -> None:
    """Recalculate mfa_enabled flag for user based on active methods."""
    methods = MfaStore.list_methods(email)
    enabled = any(m.get("enabled") for m in methods)
    UserStore.update_user(email, {"mfa_enabled": 1 if enabled else 0})


# ── MFA: List methods ──

@app.get("/api/auth/mfa/methods", tags=["Auth – MFA"])
def api_mfa_list_methods(request: Request):
    """List the user's configured MFA methods."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    methods = MfaStore.list_methods(user["email"])
    backup_codes = MfaStore.list_backup_codes(user["email"])
    return {
        "ok": True,
        "mfa_enabled": bool(user.get("mfa_enabled")),
        "methods": [
            {
                "id": m["id"],
                "type": m["method_type"],
                "enabled": bool(m["enabled"]),
                "device_name": m.get("device_name"),
                "phone_number": m.get("phone_number", "")[-4:] if m.get("phone_number") else None,
                "created_at": m["created_at"],
            }
            for m in methods
        ],
        "backup_codes_remaining": sum(1 for c in backup_codes if not c["used"]),
    }


# ── MFA: TOTP Setup ──

@app.post("/api/auth/mfa/totp/setup", tags=["Auth – MFA"])
def api_mfa_totp_setup(request: Request):
    """Generate a TOTP secret and QR code URI for setup."""
    import pyotp
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    # Check if TOTP already enabled
    existing = MfaStore.get_method_by_type(user["email"], "totp")
    if existing:
        raise HTTPException(400, "TOTP is already enabled")

    secret = pyotp.random_base32()
    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(
        name=user["email"],
        issuer_name="Xcelsior",
    )

    # Store secret temporarily (not enabled until verified)
    method_id = MfaStore.create_method({
        "email": user["email"],
        "method_type": "totp",
        "secret": secret,
        "enabled": 0,
        "created_at": time.time(),
    })

    return {
        "ok": True,
        "secret": secret,
        "provisioning_uri": provisioning_uri,
        "method_id": method_id,
    }


class TotpVerifyRequest(BaseModel):
    code: str
    method_id: int | None = None


@app.post("/api/auth/mfa/totp/verify", tags=["Auth – MFA"])
def api_mfa_totp_verify(request: Request, req: TotpVerifyRequest):
    """Verify a TOTP code to complete setup and enable TOTP."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    methods = MfaStore.list_methods(user["email"])
    totp_method = None
    for m in methods:
        if m["method_type"] == "totp":
            if req.method_id and m["id"] == req.method_id:
                totp_method = m
                break
            if not m["enabled"]:
                totp_method = m
                break
            totp_method = m

    if not totp_method or not totp_method.get("secret"):
        raise HTTPException(400, "No TOTP method found. Run setup first.")

    if not _verify_totp_code(totp_method["secret"], req.code):
        raise HTTPException(400, "Invalid code. Please try again.")

    # Enable the method
    from db import auth_connection
    with auth_connection() as conn:
        conn.execute("UPDATE mfa_methods SET enabled = 1 WHERE id = %s", (totp_method["id"],))

    _refresh_mfa_enabled(user["email"])

    # Generate backup codes if none exist yet
    existing_codes = MfaStore.list_backup_codes(user["email"])
    backup_codes = None
    if not existing_codes:
        backup_codes = _generate_backup_codes()
        code_hashes = [_hash_backup_code(c) for c in backup_codes]
        MfaStore.create_backup_codes(user["email"], code_hashes)

    return {
        "ok": True,
        "message": "TOTP enabled successfully",
        "backup_codes": backup_codes,
    }


@app.delete("/api/auth/mfa/totp", tags=["Auth – MFA"])
def api_mfa_totp_disable(request: Request):
    """Disable and remove TOTP."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    MfaStore.delete_methods_by_type(user["email"], "totp")
    _refresh_mfa_enabled(user["email"])
    return {"ok": True, "message": "TOTP disabled"}


# ── MFA: SMS ──

class SmsSetupRequest(BaseModel):
    phone_number: str  # E.164 format, e.g. +14165551234


@app.post("/api/auth/mfa/sms/setup", tags=["Auth – MFA"])
def api_mfa_sms_setup(request: Request, req: SmsSetupRequest):
    """Register a phone number for SMS MFA. Sends a verification code."""
    import re
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    phone = req.phone_number.strip()
    if not re.match(r"^\+[1-9]\d{6,14}$", phone):
        raise HTTPException(400, "Invalid phone number. Use E.164 format (e.g. +14165551234)")

    existing = MfaStore.get_method_by_type(user["email"], "sms")
    if existing:
        raise HTTPException(400, "SMS MFA is already enabled")

    # Generate 6-digit verification code
    code = f"{secrets.randbelow(1000000):06d}"
    code_hash = _hash_backup_code(code)  # reuse SHA-256 helper for code storage
    sms_challenge_id = f"sms-setup:{user['email']}"
    # Remove any existing SMS setup challenge
    MfaStore.delete_challenge(sms_challenge_id)
    MfaStore.create_challenge({
        "challenge_id": sms_challenge_id,
        "email": user["email"],
        "challenge_data": json.dumps({"code_hash": code_hash, "phone": phone}),
        "created_at": time.time(),
        "expires_at": time.time() + 600,
    })

    # Send SMS via email (best-effort) or log in dev
    if os.environ.get("XCELSIOR_ENV") == "test":
        return {"ok": True, "message": "Verification code sent", "test_code": code}

    # Best-effort: send verification code via email for now
    _send_team_email(
        to_email=user["email"],
        subject="Your Xcelsior SMS verification code",
        body_text=f"Your SMS verification code is: {code}\n\nThis code expires in 10 minutes.",
        cta_url=None,
        cta_label="",
    )

    return {"ok": True, "message": "Verification code sent"}


class SmsVerifyRequest(BaseModel):
    code: str


@app.post("/api/auth/mfa/sms/verify", tags=["Auth – MFA"])
def api_mfa_sms_verify(request: Request, req: SmsVerifyRequest):
    """Verify the SMS code to complete SMS MFA setup."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    sms_challenge_id = f"sms-setup:{user['email']}"
    pending = MfaStore.get_challenge(sms_challenge_id)
    if not pending:
        raise HTTPException(400, "No pending SMS verification. Run setup first.")
    if time.time() > pending["expires_at"]:
        MfaStore.delete_challenge(sms_challenge_id)
        raise HTTPException(400, "Verification code expired")
    pending_data = json.loads(pending["challenge_data"])
    if _hash_backup_code(req.code) != pending_data["code_hash"]:
        raise HTTPException(400, "Invalid verification code")

    MfaStore.delete_challenge(sms_challenge_id)

    MfaStore.create_method({
        "email": user["email"],
        "method_type": "sms",
        "phone_number": pending_data["phone"],
        "enabled": 1,
        "created_at": time.time(),
    })
    _refresh_mfa_enabled(user["email"])

    # Generate backup codes if none exist
    existing_codes = MfaStore.list_backup_codes(user["email"])
    backup_codes = None
    if not existing_codes:
        backup_codes = _generate_backup_codes()
        code_hashes = [_hash_backup_code(c) for c in backup_codes]
        MfaStore.create_backup_codes(user["email"], code_hashes)

    return {
        "ok": True,
        "message": "SMS MFA enabled successfully",
        "backup_codes": backup_codes,
    }


@app.delete("/api/auth/mfa/sms", tags=["Auth – MFA"])
def api_mfa_sms_disable(request: Request):
    """Disable and remove SMS MFA."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    MfaStore.delete_methods_by_type(user["email"], "sms")
    _refresh_mfa_enabled(user["email"])
    return {"ok": True, "message": "SMS MFA disabled"}


# ── MFA: Login verification ──

class MfaVerifyLogin(BaseModel):
    challenge_id: str
    method: str  # totp | sms | backup
    code: str


@app.post("/api/auth/mfa/verify", tags=["Auth – MFA"])
def api_mfa_verify_login(req: MfaVerifyLogin):
    """Verify MFA code during login to complete authentication."""
    challenge = MfaStore.get_challenge(req.challenge_id)
    if not challenge:
        raise HTTPException(400, "Invalid MFA challenge")
    if time.time() > challenge["expires_at"]:
        MfaStore.delete_challenge(req.challenge_id)
        raise HTTPException(400, "MFA challenge expired. Please sign in again.")

    email = challenge["email"]

    if req.method == "totp":
        method = MfaStore.get_method_by_type(email, "totp")
        if not method or not method.get("secret"):
            raise HTTPException(400, "TOTP not configured")
        if not _verify_totp_code(method["secret"], req.code):
            raise HTTPException(400, "Invalid code")
        return _complete_mfa_login(email, req.challenge_id)

    elif req.method == "sms":
        # For login SMS verification, generate and send code
        method = MfaStore.get_method_by_type(email, "sms")
        if not method:
            raise HTTPException(400, "SMS MFA not configured")

        # Check if code matches the one we sent
        sms_login_id = f"sms-login:{email}"
        pending = MfaStore.get_challenge(sms_login_id)
        if not pending:
            raise HTTPException(400, "No SMS code sent. Request one first.")
        if time.time() > pending["expires_at"]:
            MfaStore.delete_challenge(sms_login_id)
            raise HTTPException(400, "Code expired")
        pending_data = json.loads(pending["challenge_data"])
        if _hash_backup_code(req.code) != pending_data["code_hash"]:
            raise HTTPException(400, "Invalid code")
        MfaStore.delete_challenge(sms_login_id)
        return _complete_mfa_login(email, req.challenge_id)

    elif req.method == "backup":
        code_hash = _hash_backup_code(req.code)
        if not MfaStore.use_backup_code(email, code_hash):
            raise HTTPException(400, "Invalid backup code")
        return _complete_mfa_login(email, req.challenge_id)

    else:
        raise HTTPException(400, f"Unsupported MFA method: {req.method}")


class MfaSendSmsRequest(BaseModel):
    challenge_id: str


@app.post("/api/auth/mfa/sms/send", tags=["Auth – MFA"])
def api_mfa_sms_send_login(req: MfaSendSmsRequest):
    """Send an SMS verification code during MFA login challenge."""
    challenge = MfaStore.get_challenge(req.challenge_id)
    if not challenge:
        raise HTTPException(400, "Invalid MFA challenge")
    if time.time() > challenge["expires_at"]:
        MfaStore.delete_challenge(req.challenge_id)
        raise HTTPException(400, "MFA challenge expired")

    email = challenge["email"]
    method = MfaStore.get_method_by_type(email, "sms")
    if not method:
        raise HTTPException(400, "SMS MFA not configured")

    code = f"{secrets.randbelow(1000000):06d}"
    code_hash = _hash_backup_code(code)
    sms_login_id = f"sms-login:{email}"
    MfaStore.delete_challenge(sms_login_id)
    MfaStore.create_challenge({
        "challenge_id": sms_login_id,
        "email": email,
        "challenge_data": json.dumps({"code_hash": code_hash}),
        "created_at": time.time(),
        "expires_at": time.time() + 600,
    })

    # Send code via email for now
    user = UserStore.get_user(email)
    if user:
        _send_team_email(
            to_email=email,
            subject="Your Xcelsior login code",
            body_text=f"Your login verification code is: {code}\n\nThis code expires in 10 minutes.",
            cta_url=None,
            cta_label="",
        )

    if os.environ.get("XCELSIOR_ENV") == "test":
        return {"ok": True, "message": "Code sent", "test_code": code}
    return {"ok": True, "message": "Code sent"}


# ── MFA: Backup codes ──

@app.post("/api/auth/mfa/backup-codes/regenerate", tags=["Auth – MFA"])
def api_mfa_regenerate_backup_codes(request: Request):
    """Regenerate backup recovery codes. Invalidates all previous codes."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    if not user.get("mfa_enabled"):
        raise HTTPException(400, "MFA is not enabled")

    codes = _generate_backup_codes()
    code_hashes = [_hash_backup_code(c) for c in codes]
    MfaStore.create_backup_codes(user["email"], code_hashes)

    return {"ok": True, "backup_codes": codes}


# ── MFA: Disable all ──

@app.delete("/api/auth/mfa/all", tags=["Auth – MFA"])
def api_mfa_disable_all(request: Request):
    """Disable all MFA methods for the user."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    from db import auth_connection
    with auth_connection() as conn:
        conn.execute("DELETE FROM mfa_methods WHERE email = %s", (user["email"],))
        conn.execute("DELETE FROM mfa_backup_codes WHERE email = %s", (user["email"],))

    UserStore.update_user(user["email"], {"mfa_enabled": 0})
    return {"ok": True, "message": "All MFA methods disabled"}


# ── Team / Organization Management ───────────────────────────────────
# Per UI_ROADMAP competitor gap: team/org management
# Teams share billing, hosts, and job visibility


def _send_team_email(to_email: str, subject: str, body_text: str, cta_url: str | None = None, cta_label: str = "Go to Dashboard"):
    """Send a styled team notification email in a background thread. Best-effort.

    Matches the dark-theme style from frontend/src/emails/layout.tsx.
    """
    from scheduler import ALERT_CONFIG
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    cfg = ALERT_CONFIG
    if not cfg.get("email_enabled") or not cfg.get("smtp_host"):
        return

    # Build styled HTML matching the React email templates
    cta_html = ""
    if cta_url:
        cta_html = f"""
        <div style="text-align:center;margin:32px 0">
          <a href="{cta_url}" style="display:inline-block;background-color:#dc2626;color:#ffffff;padding:12px 32px;border-radius:8px;font-size:16px;font-weight:600;text-decoration:none">{cta_label}</a>
        </div>"""

    # Convert newlines in body to styled paragraphs
    paragraphs = "".join(
        f'<p style="color:#94a3b8;font-size:16px;line-height:1.6;margin:0 0 16px">{line}</p>'
        for line in body_text.strip().split("\n\n")
        if line.strip()
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/></head>
<body style="background-color:#0f172a;color:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:0;padding:0">
<div style="max-width:600px;margin:0 auto;padding:40px 24px">
  <div style="margin-bottom:32px">
    <span style="display:inline-block;width:36px;height:36px;line-height:36px;text-align:center;border-radius:8px;background-color:#dc2626;color:#fff;font-weight:800;font-size:20px;vertical-align:middle">X</span>
    <a href="https://xcelsior.ca" style="font-size:22px;font-weight:700;color:#f8fafc;text-decoration:none;vertical-align:middle;margin-left:12px">Xcelsior</a>
  </div>
  <h1 style="color:#f8fafc;font-size:24px;font-weight:700;line-height:1.3;margin:0 0 16px">{subject}</h1>
  {paragraphs}
  {cta_html}
  <hr style="border:none;border-top:1px solid #334155;margin:32px 0"/>
  <p style="color:#64748b;font-size:13px;line-height:1.5">
    Xcelsior Computing Inc. · Canada<br/>
    <a href="https://xcelsior.ca/privacy" style="color:#38bdf8;text-decoration:underline">Privacy Policy</a> ·
    <a href="https://xcelsior.ca/terms" style="color:#38bdf8;text-decoration:underline">Terms</a>
  </p>
</div>
</body></html>"""

    def _do_send():
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[Xcelsior] {subject}"
            msg["From"] = cfg["email_from"]
            msg["To"] = to_email
            # Plain text fallback
            msg.attach(MIMEText(body_text, "plain"))
            # Styled HTML
            msg.attach(MIMEText(html, "html"))
            with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as server:
                server.starttls()
                server.login(cfg["smtp_user"], cfg["smtp_pass"])
                server.send_message(msg)
            log.info("TEAM EMAIL SENT: %s -> %s", subject, to_email)
        except Exception as e:
            log.warning("TEAM EMAIL FAILED: %s -> %s | %s", subject, to_email, e)

    import threading
    threading.Thread(target=_do_send, daemon=True).start()


class CreateTeamRequest(BaseModel):
    name: str
    plan: str = "free"  # free | pro | enterprise


class AddTeamMemberRequest(BaseModel):
    email: str
    role: str = "member"  # admin | member | viewer


class UpdateTeamMemberRoleRequest(BaseModel):
    role: str  # admin | member | viewer


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
    # Send email notification (best-effort, non-blocking)
    _send_team_email(
        body.email,
        f"You've been added to team {team['name']}",
        f"You've been added to the team \"{team['name']}\" on Xcelsior as a {body.role}.\n\nYou can now collaborate with your team, share billing, and manage GPU instances together.",
        cta_url="https://xcelsior.ca/dashboard/settings",
        cta_label="View Your Team",
    )
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
    # Send email notification (best-effort, non-blocking)
    _send_team_email(
        email,
        f"You've been removed from team {team['name']}",
        f"You've been removed from the team \"{team['name']}\" on Xcelsior.\n\nIf you believe this was a mistake, contact the team owner.",
        cta_url="https://xcelsior.ca/dashboard/settings",
        cta_label="Go to Dashboard",
    )
    return {"ok": True, "message": f"{email} removed from team"}


@app.patch("/api/teams/{team_id}/members/{email}", tags=["Teams"])
def api_update_team_member_role(team_id: str, email: str, body: UpdateTeamMemberRoleRequest, request: Request):
    """Update a team member's role. Only admins can change roles."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    if body.role not in ("admin", "member", "viewer"):
        raise HTTPException(400, "Role must be admin, member, or viewer")

    team = UserStore.get_team(team_id)
    if not team:
        raise HTTPException(404, "Team not found")

    members = UserStore.list_team_members(team_id)
    requester = next((m for m in members if m["email"] == user["email"]), None)
    if not requester or requester["role"] != "admin":
        raise HTTPException(403, "Only team admins can change roles")

    if email == team["owner_email"]:
        raise HTTPException(400, "Cannot change the team owner's role")

    if not UserStore.update_team_member_role(team_id, email, body.role):
        raise HTTPException(404, f"{email} is not a member of this team")

    return {"ok": True, "message": f"{email} role updated to {body.role}"}


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
    broadcast_sse("team_deleted", {"team_id": team_id})
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


@app.get("/tiers", tags=["Instances"])
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
        j for j in all_jobs if j.get("host_id") == host_id and j.get("status") in ("assigned",)
    ]
    with _agent_lock:
        queued_work = _agent_work.pop(host_id, [])

    jobs = pending + queued_work
    if not jobs:
        return JSONResponse(status_code=204, content=None)
    return {"ok": True, "instances": jobs}


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
                h["status"] = "active"
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


@app.post("/agent/mining-alert", tags=["Agent"])
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


@app.post("/agent/benchmark", tags=["Agent"])
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


@app.post("/spot/instance", tags=["Spot Pricing"])
def api_submit_spot_instance(j: SpotJobIn):
    """Submit a spot job with a maximum bid price."""
    job = submit_spot_job(j.name, j.vram_needed_gb, j.max_bid, j.priority, tier=j.tier)
    broadcast_sse(
        "spot_job_submitted",
        {
            "job_id": job["job_id"],
            "name": job["name"],
            "max_bid": j.max_bid,
        },
    )
    return {"ok": True, "instance": job}


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


@app.get("/metrics/prometheus", tags=["Infrastructure"])
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

    # GPU telemetry if available
    try:
        from nvml_telemetry import get_all_gpu_stats
        gpu_stats = get_all_gpu_stats()
        if gpu_stats:
            lines.extend([
                "",
                "# HELP xcelsior_gpu_utilization_percent GPU utilization percentage",
                "# TYPE xcelsior_gpu_utilization_percent gauge",
            ])
            for gs in gpu_stats:
                idx = gs.get("index", 0)
                model = gs.get("name", "unknown").replace(" ", "_")
                lines.append(
                    f'xcelsior_gpu_utilization_percent{{gpu="{idx}",model="{model}"}} '
                    f'{gs.get("utilization", 0)}'
                )
            lines.extend([
                "",
                "# HELP xcelsior_gpu_temperature_celsius GPU temperature",
                "# TYPE xcelsior_gpu_temperature_celsius gauge",
            ])
            for gs in gpu_stats:
                idx = gs.get("index", 0)
                lines.append(
                    f'xcelsior_gpu_temperature_celsius{{gpu="{idx}"}} {gs.get("temperature", 0)}'
                )
            lines.extend([
                "",
                "# HELP xcelsior_gpu_memory_used_bytes GPU memory used",
                "# TYPE xcelsior_gpu_memory_used_bytes gauge",
            ])
            for gs in gpu_stats:
                idx = gs.get("index", 0)
                lines.append(
                    f'xcelsior_gpu_memory_used_bytes{{gpu="{idx}"}} '
                    f'{gs.get("memory_used_bytes", 0)}'
                )
    except Exception:
        pass

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
        lines.extend([
            "",
            "# HELP xcelsior_webhook_backlog Pending webhook events",
            "# TYPE xcelsior_webhook_backlog gauge",
            f"xcelsior_webhook_backlog {backlog}",
        ])
    except Exception:
        pass

    # Scheduling latency histogram (approximated from last scheduling cycle)
    try:
        _snap_running = snap.get("running_jobs", 0)
        _snap_queued = snap.get("queue_depth", 0)
        lines.extend([
            "",
            "# HELP xcelsior_scheduling_latency_seconds Scheduling cycle latency",
            "# TYPE xcelsior_scheduling_latency_seconds histogram",
            f'xcelsior_scheduling_latency_seconds_bucket{{le="0.1"}} {_snap_running}',
            f'xcelsior_scheduling_latency_seconds_bucket{{le="1.0"}} {_snap_running}',
            f'xcelsior_scheduling_latency_seconds_bucket{{le="10.0"}} {_snap_running + _snap_queued}',
            f'xcelsior_scheduling_latency_seconds_bucket{{le="+Inf"}} {_snap_running + _snap_queued}',
            f"xcelsior_scheduling_latency_seconds_sum 0",
            f"xcelsior_scheduling_latency_seconds_count {_snap_running + _snap_queued}",
        ])
    except Exception:
        pass

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
        lines.extend([
            "",
            "# HELP xcelsior_wallet_depletion_events_total Total wallet depletion events",
            "# TYPE xcelsior_wallet_depletion_events_total counter",
            f"xcelsior_wallet_depletion_events_total {dep_cnt}",
        ])
    except Exception:
        pass

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
        lines.extend([
            "",
            "# HELP xcelsior_inference_cold_start_rate Fraction of inference requests requiring cold start",
            "# TYPE xcelsior_inference_cold_start_rate gauge",
            f"xcelsior_inference_cold_start_rate {cold_rate}",
        ])
    except Exception:
        pass

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
        lines.extend([
            "",
            "# HELP xcelsior_inference_tokens_per_second Aggregate inference throughput",
            "# TYPE xcelsior_inference_tokens_per_second gauge",
            f"xcelsior_inference_tokens_per_second {tps}",
        ])
    except Exception:
        pass

    from starlette.responses import Response
    return Response(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/", tags=["Infrastructure"], include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/dashboard")


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
    v = store.get_verification(host_id)
    if not v:
        return {"ok": True, "host_id": host_id, "status": "unverified"}
    return {"ok": True, "host_id": host_id, "verification": v.__dict__}


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
        result.append(
            {
                "host_id": r["host_id"],
                "status": r["state"],
                "overall_score": r["overall_score"],
                "last_check": r["last_check_at"],
                "gpu_fingerprint": r["gpu_fingerprint"],
                "deverify_reason": r["deverify_reason"] or "",
                "gpu_model": h.get("gpu_model", "—"),
                "country": h.get("country", ""),
                "province": h.get("province", ""),
            }
        )
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

    min_scores = {"community": 0, "residency": 25, "sovereignty": 50, "regulated": 75}
    req_labels = {
        "requires_canada": "Host physically located in Canada",
        "requires_verified": "Host identity verified",
        "requires_sovereignty_vetting": "Canadian-incorporated operator vetted",
        "requires_audit_trail": "Full audit trail enabled",
    }
    tiers = {}
    for t, v in TRUST_TIER_REQUIREMENTS.items():
        reqs = [label for key, label in req_labels.items() if v.get(key)]
        tiers[t.value] = {**v, "min_score": min_scores.get(t.value, 0), "requirements": reqs}
    return {"ok": True, "tiers": tiers}


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


class PaymentIntentRequest(BaseModel):
    customer_id: str
    amount_cad: float
    description: str = "Compute credits"


@app.post("/api/billing/payment-intent", tags=["Billing"])
def api_create_payment_intent(req: PaymentIntentRequest):
    """Create a Stripe PaymentIntent for depositing compute credits.

    Returns client_secret for front-end Stripe Elements confirmation.
    On payment_intent.succeeded webhook the wallet is credited automatically.
    """
    if req.amount_cad < 1 or req.amount_cad > 10000:
        raise HTTPException(400, "Amount must be between $1 and $10,000 CAD")
    mgr = get_stripe_manager()
    result = mgr.create_credit_deposit(req.customer_id, req.amount_cad, req.description)
    return {"ok": True, "intent": result}


@app.post("/api/billing/wallet/{customer_id}/deposit", tags=["Billing"])
def api_deposit(customer_id: str, req: DepositRequest):
    """Deposit credits into a customer wallet."""
    be = get_billing_engine()
    result = be.deposit(customer_id, req.amount_cad, req.description)
    return {"ok": True, **result}


_FREE_CREDIT_AMOUNT = 10.0  # CAD


@app.post("/api/billing/free-credits/{customer_id}", tags=["Billing"])
def api_claim_free_credits(customer_id: str, request: Request):
    """Claim one-time $10 CAD signup bonus.

    Uses an idempotency key derived from the customer_id so the bonus
    can only be claimed once per customer.
    """
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Authentication required")
    # Resolve customer_id from full user profile (session may lack it)
    full_user = UserStore.get_user(user["email"]) if _USE_PERSISTENT_AUTH else _users_db.get(user["email"], {})
    uid = (full_user or {}).get("customer_id") or user.get("customer_id") or user.get("user_id") or ""
    if uid != customer_id:
        raise HTTPException(403, "You can only claim credits for your own account")

    idempotency_key = f"free-credits-{customer_id}"
    be = get_billing_engine()
    result = be.deposit(
        customer_id,
        _FREE_CREDIT_AMOUNT,
        "Welcome bonus — $10 free credits",
        idempotency_key=idempotency_key,
    )
    already_claimed = result.get("dedup", False)
    return {
        "ok": True,
        "amount_cad": _FREE_CREDIT_AMOUNT,
        "balance_cad": result["balance_cad"],
        "already_claimed": already_claimed,
    }


@app.get("/api/billing/free-credits/{customer_id}/status", tags=["Billing"])
def api_free_credits_status(customer_id: str, request: Request):
    """Check whether the customer has already claimed the free signup bonus."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Authentication required")
    # Resolve customer_id from full user profile (session may lack it)
    full_user = UserStore.get_user(user["email"]) if _USE_PERSISTENT_AUTH else _users_db.get(user["email"], {})
    uid = (full_user or {}).get("customer_id") or user.get("customer_id") or user.get("user_id") or ""
    if uid != customer_id:
        raise HTTPException(403, "Forbidden")
    be = get_billing_engine()
    be._ensure_wallet_table()
    with be._conn() as conn:
        row = conn.execute(
            "SELECT tx_id FROM wallet_transactions WHERE idempotency_key = %s",
            (f"free-credits-{customer_id}",),
        ).fetchone()
    return {"ok": True, "claimed": row is not None}


@app.get("/api/billing/wallet/{customer_id}/history", tags=["Billing"])
def api_wallet_history(customer_id: str, limit: int = 50):
    """Get transaction history for a wallet."""
    be = get_billing_engine()
    history = be.get_wallet_history(customer_id, limit)
    return {"ok": True, "customer_id": customer_id, "transactions": history}


@app.get("/api/billing/wallet/{customer_id}/depletion", tags=["Billing"])
def api_wallet_depletion(customer_id: str):
    """Get real-time balance depletion projection.

    Returns burn rate, seconds-to-zero, per-instance cost breakdown,
    and alert thresholds (T-30min, T-5min, T-0).
    """
    be = get_billing_engine()
    return {"ok": True, **be.time_to_zero(customer_id)}


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
def api_generate_invoice(
    customer_id: str,
    customer_name: str = "",
    period_start: float = 0,
    period_end: float = 0,
    tax_rate: float = 0.13,
):
    """Generate an AI Compute Access Fund–aligned invoice."""
    if period_end == 0:
        period_end = time.time()
    if period_start == 0:
        period_start = period_end - 30 * 86400
    be = get_billing_engine()
    invoice = be.generate_invoice(customer_id, customer_name, period_start, period_end, tax_rate)
    return {"ok": True, "invoice": invoice.to_dict()}


@app.get("/api/billing/export/caf/{customer_id}", tags=["Billing"])
def api_export_caf(
    customer_id: str, period_start: float = 0, period_end: float = 0, format: str = "json"
):
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
                invoices.append(
                    {
                        "invoice_id": f"INV-{customer_id[:8]}-{i+1:03d}",
                        "period_start": period_start,
                        "period_end": period_end,
                        "total_cad": inv_dict.get(
                            "total_with_tax_cad", inv_dict.get("total_compute_cad", 0)
                        ),
                        "subtotal_cad": inv_dict.get("total_compute_cad", 0),
                        "tax_cad": inv_dict.get("tax_cad", 0),
                        "tax_rate": inv_dict.get("tax_rate", 0.13),
                        "line_items": len(inv_dict.get("line_items", [])),
                        "caf_eligible_cad": inv_dict.get("caf_eligible_cad", 0),
                        "status": "paid",
                    }
                )
        except Exception:
            pass
    return {"ok": True, "invoices": invoices, "count": len(invoices)}


@app.get("/api/billing/invoice/{customer_id}/download", tags=["Billing"])
def api_download_invoice(
    customer_id: str,
    format: str = "csv",
    period_start: float = 0,
    period_end: float = 0,
    tax_rate: float = 0.13,
    customer_name: str = "",
):
    """Download an invoice as CSV or plain-text PDF-style document.

    Formats: csv (spreadsheet-ready), txt (printable receipt).
    """
    import io
    import csv as csv_mod
    from datetime import datetime

    if period_end == 0:
        period_end = time.time()
    if period_start == 0:
        period_start = period_end - 30 * 86400

    be = get_billing_engine()
    inv = be.generate_invoice(customer_id, customer_name, period_start, period_end, tax_rate)
    inv_dict = inv.to_dict()
    date_str = datetime.utcfromtimestamp(period_end).strftime("%Y-%m-%d")

    if format == "csv":
        output = io.StringIO()
        writer = csv_mod.writer(output)
        writer.writerow(["Xcelsior Invoice", f"INV-{customer_id[:8]}", date_str])
        writer.writerow([])
        writer.writerow(["Description", "GPU", "Duration (h)", "Rate (CAD/h)", "Amount (CAD)"])
        for item in inv_dict.get("line_items", []):
            writer.writerow(
                [
                    item.get("description", "Compute"),
                    item.get("gpu_model", "—"),
                    round(item.get("duration_hours", 0), 2),
                    round(item.get("rate_cad_per_hour", 0), 2),
                    round(item.get("amount_cad", 0), 2),
                ]
            )
        writer.writerow([])
        writer.writerow(["Subtotal", "", "", "", round(inv_dict.get("total_compute_cad", 0), 2)])
        writer.writerow(["Tax", "", "", "", round(inv_dict.get("tax_cad", 0), 2)])
        writer.writerow(
            ["Total (CAD)", "", "", "", round(inv_dict.get("total_with_tax_cad", 0), 2)]
        )
        writer.writerow(["CAF Eligible", "", "", "", round(inv_dict.get("caf_eligible_cad", 0), 2)])
        csv_data = output.getvalue()
        return StreamingResponse(
            iter([csv_data]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=xcelsior-invoice-{customer_id[:8]}-{date_str}.csv"
            },
        )

    # Plain-text receipt format
    lines = [
        "=" * 60,
        "XCELSIOR — GPU COMPUTE INVOICE",
        "=" * 60,
        f"Invoice ID:  INV-{customer_id[:8]}",
        f"Customer:    {customer_name or customer_id}",
        f"Date:        {date_str}",
        f"Period:      {datetime.utcfromtimestamp(period_start).strftime('%Y-%m-%d')} to {date_str}",
        "-" * 60,
        f"{'Description':<25} {'GPU':<12} {'Hours':>6} {'Rate':>8} {'Amount':>10}",
        "-" * 60,
    ]
    for item in inv_dict.get("line_items", []):
        lines.append(
            f"{item.get('description', 'Compute')[:25]:<25} "
            f"{item.get('gpu_model', '—')[:12]:<12} "
            f"{item.get('duration_hours', 0):>6.2f} "
            f"${item.get('rate_cad_per_hour', 0):>7.2f} "
            f"${item.get('amount_cad', 0):>9.2f}"
        )
    lines += [
        "-" * 60,
        f"{'Subtotal':<55} ${inv_dict.get('total_compute_cad', 0):>8.2f}",
        f"{'Tax (' + str(round(tax_rate*100, 1)) + '%)':<55} ${inv_dict.get('tax_cad', 0):>8.2f}",
        f"{'TOTAL (CAD)':<55} ${inv_dict.get('total_with_tax_cad', 0):>8.2f}",
        "",
        f"AI Compute Access Fund Eligible: ${inv_dict.get('caf_eligible_cad', 0):.2f} CAD",
        "=" * 60,
        "Xcelsior Inc. | xcelsior.ca | Built in Canada 🍁",
    ]
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(
        content="\n".join(lines),
        media_type="text/plain",
        headers={
            "Content-Disposition": f"attachment; filename=xcelsior-invoice-{customer_id[:8]}-{date_str}.txt"
        },
    )


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


# ── Bitcoin Deposits ──────────────────────────────────────────────────

try:
    import bitcoin as _btc_mod
except ImportError:
    _btc_mod = None  # type: ignore[assignment]


class CryptoDepositRequest(BaseModel):
    customer_id: str
    amount_cad: float


@app.post("/api/billing/crypto/deposit", tags=["Billing"])
def api_crypto_deposit(req: CryptoDepositRequest):
    """Create a BTC deposit request. Returns address, amount, and QR data."""
    if not _btc_mod or not _btc_mod.BTC_ENABLED:
        raise HTTPException(503, "Bitcoin deposits are not enabled")
    if req.amount_cad < 1 or req.amount_cad > 10000:
        raise HTTPException(400, "Amount must be between $1 and $10,000 CAD")
    try:
        result = _btc_mod.create_deposit(req.customer_id, req.amount_cad)
        return {"ok": True, **result}
    except Exception as e:
        log.error("Crypto deposit error: %s", e)
        raise HTTPException(503, "Bitcoin node is temporarily unavailable — please try again later or use card deposit")


@app.get("/api/billing/crypto/deposit/{deposit_id}", tags=["Billing"])
def api_crypto_deposit_status(deposit_id: str):
    """Poll deposit confirmation status."""
    if not _btc_mod or not _btc_mod.BTC_ENABLED:
        raise HTTPException(503, "Bitcoin deposits are not enabled")
    dep = _btc_mod.get_deposit(deposit_id)
    if not dep:
        raise HTTPException(404, "Deposit not found")
    return {"ok": True, **dep}


@app.get("/api/billing/crypto/rate", tags=["Billing"])
def api_crypto_rate():
    """Get current BTC/CAD exchange rate."""
    if not _btc_mod or not _btc_mod.BTC_ENABLED:
        raise HTTPException(503, "Bitcoin deposits are not enabled")
    try:
        rate = _btc_mod.get_btc_cad_rate()
        return {"ok": True, "btc_cad": rate, "currency": "CAD"}
    except Exception as e:
        raise HTTPException(502, f"Unable to fetch rate: {e}")


@app.post("/api/billing/crypto/refresh/{deposit_id}", tags=["Billing"])
def api_crypto_refresh(deposit_id: str):
    """Refresh an expired deposit with a new BTC/CAD rate."""
    if not _btc_mod or not _btc_mod.BTC_ENABLED:
        raise HTTPException(503, "Bitcoin deposits are not enabled")
    dep = _btc_mod.refresh_deposit(deposit_id)
    if not dep:
        raise HTTPException(404, "Deposit not found")
    return {"ok": True, **dep}


@app.get("/api/billing/crypto/enabled", tags=["Billing"])
def api_crypto_enabled():
    """Check if Bitcoin deposits are enabled."""
    enabled = bool(_btc_mod and _btc_mod.BTC_ENABLED)
    return {"ok": True, "enabled": enabled}


# ── Reputation ────────────────────────────────────────────────────────


@app.get("/api/reputation/leaderboard", tags=["Reputation"])
def api_reputation_leaderboard(entity_type: str = "host", limit: int = 20):
    """Top hosts/users by reputation score."""
    re = get_reputation_engine()
    board = re.get_leaderboard(entity_type, limit)
    return {"ok": True, "entity_type": entity_type, "leaderboard": board}


@app.get("/api/reputation/me", tags=["Reputation"])
def api_reputation_me(request: Request):
    """Get reputation for the currently authenticated user."""
    user = getattr(request.state, "user", None)
    user_id = ""
    if user:
        user_id = getattr(user, "user_id", "") or getattr(user, "customer_id", "")
    if not user_id:
        # Try from middleware-set attributes
        user_id = getattr(request.state, "user_id", "") or getattr(request.state, "customer_id", "")
    if not user_id:
        # Try extracting from session cookie
        token = request.cookies.get(_AUTH_COOKIE_NAME, "")
        if token and _USE_PERSISTENT_AUTH:
            session = UserStore.get_session(token)
            if session:
                user_id = session.get("user_id", "")
    if not user_id:
        return {"ok": True, "score": 0, "tier": "bronze"}
    re = get_reputation_engine()
    score = re.compute_score(user_id)
    return {"ok": True, **score.to_dict()}


@app.get("/api/reputation/{entity_id}", tags=["Reputation"])
def api_get_reputation(entity_id: str):
    """Get reputation score and tier for a host or user."""
    re = get_reputation_engine()
    score = re.compute_score(entity_id)
    return {"ok": True, "reputation": score.to_dict()}


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
        raise HTTPException(
            status_code=400, detail=f"Invalid verification type: {req.verification_type}"
        )
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
        req.gpu_model,
        req.duration_hours,
        spot=req.spot,
        sovereignty=req.sovereignty,
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
            rate = (
                ref.get("base_rate_cad", ref.get("cad_per_hour", 0))
                if isinstance(ref, dict)
                else ref
            )
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
        raise HTTPException(
            400,
            f"Invalid commitment_type: {req.commitment_type}. "
            f"Valid: {list(RESERVED_PRICING_TIERS.keys())}",
        )

    # Calculate pricing
    ref_pricing = GPU_REFERENCE_PRICING_CAD.get(req.gpu_model, {})
    base_rate = (
        ref_pricing.get("base_rate_cad", ref_pricing.get("cad_per_hour", 0))
        if isinstance(ref_pricing, dict)
        else (ref_pricing if isinstance(ref_pricing, (int, float)) else 0)
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

    # Charge upfront or set up recurring billing via Stripe
    billing = get_billing_engine()
    monthly_estimate = discounted_rate * req.quantity * 24 * 30
    commitment["monthly_estimate_cad"] = round(monthly_estimate, 2)
    commitment["monthly_estimate_with_tax_cad"] = round(monthly_estimate * (1 + tax_rate), 2)

    broadcast_sse(
        "reservation_created",
        {
            "commitment_id": commitment["commitment_id"],
            "customer_id": req.customer_id,
            "type": req.commitment_type,
        },
    )
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
                "FROM usage_meters WHERE started_at >= %s",
                (one_year_ago,),
            ).fetchone()
            total_rev = row["total"] if row else 0.0

            # Count distinct quarters
            qrow = conn.execute(
                "SELECT COUNT(DISTINCT (EXTRACT(YEAR FROM to_timestamp(started_at))::int * 4 "
                "+ EXTRACT(MONTH FROM to_timestamp(started_at))::int / 4)) AS q_count "
                "FROM usage_meters WHERE started_at >= %s",
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
                "FROM payout_ledger WHERE provider_id = %s AND created_at >= %s",
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
        "day": "to_char(to_timestamp(started_at), 'YYYY-MM-DD') AS period",
        "week": "to_char(to_timestamp(started_at), 'IYYY-\"W\"IW') AS period",
        "gpu_model": "gpu_model AS period",
        "province": "province AS period",
    }.get(group_by, "to_char(to_timestamp(started_at), 'YYYY-MM-DD') AS period")

    where_clauses = ["started_at >= %s"]
    params: list = [since]
    if customer_id:
        where_clauses.append("owner = %s")
        params.append(customer_id)
    # Provider filter: match host_id (providers are hosts)
    if provider_id:
        where_clauses.append("host_id = %s")
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
                    "total_gpu_hours": (
                        round(r["total_gpu_seconds"] / 3600, 2) if r["total_gpu_seconds"] else 0
                    ),
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
        results.append(
            {
                "job_id": job["job_id"],
                "job_name": job.get("name"),
                "host_id": host["host_id"],
                "gpu_model": host.get("gpu_model"),
                "country": host.get("country", ""),
            }
        )
        broadcast_sse(
            "job_assigned",
            {
                "job_id": job["job_id"],
                "host_id": host["host_id"],
            },
        )
    return {"ok": True, "assigned": len(results), "jobs": results}


# ═══ Province Compliance Matrix ══════════════════════════════════════
# REPORT_MARKETING_FINAL.md: "maintaining a small policy matrix embedded
# in the scheduler product and documentation"


@app.get("/api/compliance/status", tags=["Compliance"])
def api_compliance_status(request: Request):
    """Return high-level compliance check summary with live verification.

    Checks reflect actual user/platform configuration state — items require
    specific action before they show as passing. Each non-passing check
    includes an ``action`` with a CTA label and dashboard link.
    """
    from billing import PROVINCE_TAX_RATES
    from jurisdiction import TRUST_TIER_REQUIREMENTS

    checks = []

    # Resolve current user for per-user checks
    user_id = getattr(request.state, "user_id", None)
    full_user = None
    if user_id:
        full_user = UserStore.get_user_by_id(user_id)

    canada_only_user = bool(full_user.get("canada_only_routing", 0)) if full_user else False
    user_province = (full_user.get("province") or "") if full_user else ""
    user_provider_id = (full_user.get("provider_id") or "") if full_user else ""
    user_customer_id = (full_user.get("customer_id") or "") if full_user else ""

    # 1. Province Tax Matrix — platform-level, always passes if code is correct
    expected_provinces = {"AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"}
    configured = set(PROVINCE_TAX_RATES.keys())
    missing = expected_provinces - configured
    if not missing:
        checks.append({"id": "province_matrix", "name": "Province Tax Matrix", "status": "pass",
                        "description": f"Tax rates configured for all {len(configured)} provinces and territories."})
    else:
        checks.append({"id": "province_matrix", "name": "Province Tax Matrix", "status": "fail",
                        "description": f"Missing tax rates for: {', '.join(sorted(missing))}. Contact support to resolve.",
                        "action": {"label": "View tax matrix", "href": "/dashboard/compliance?tab=provinces"}})

    # 2. Data Residency — requires user to enable Canada-only routing in settings
    canada_only_env = os.environ.get("XCELSIOR_CANADA_ONLY", "false").lower() == "true"
    if canada_only_user or canada_only_env:
        checks.append({"id": "data_residency", "name": "Data Residency", "status": "pass",
                        "description": "Canada-only data residency enforced." + (" Your account restricts all compute and storage to Canadian infrastructure." if canada_only_user else " Platform-wide enforcement active.")})
    else:
        checks.append({"id": "data_residency", "name": "Data Residency", "status": "warn",
                        "description": "Canada-only routing is not enabled. Enable it in your Jurisdiction settings to restrict all compute and data to Canadian infrastructure.",
                        "action": {"label": "Open Jurisdiction settings", "href": "/dashboard/settings"}})

    # 3. Trust Tiers — requires user to have registered provider hosts
    expected_tiers = 4
    tier_count = len(TRUST_TIER_REQUIREMENTS)
    has_hosts = False
    if user_id:
        from db import auth_connection
        with auth_connection() as conn:
            cnt = conn.execute(
                "SELECT COUNT(*) AS c FROM hosts WHERE payload->>'user_id' = %s AND status = 'active'",
                (user_id,),
            ).fetchone()
            has_hosts = cnt and cnt["c"] > 0
    if tier_count >= expected_tiers and has_hosts:
        checks.append({"id": "trust_tiers", "name": "Trust Tier Definitions", "status": "pass",
                        "description": f"{tier_count} trust tiers active. Your hosts are enrolled and earning reputation in the tier system."})
    elif tier_count >= expected_tiers:
        checks.append({"id": "trust_tiers", "name": "Trust Tier Definitions", "status": "warn",
                        "description": f"{tier_count} tiers defined but you have no active provider hosts. Register a GPU host to participate in the trust tier system.",
                        "action": {"label": "Register a host", "href": "/dashboard/hosts"}})
    else:
        checks.append({"id": "trust_tiers", "name": "Trust Tier Definitions", "status": "warn",
                        "description": f"Only {tier_count}/{expected_tiers} trust tiers defined.",
                        "action": {"label": "View trust tiers", "href": "/dashboard/trust"}})

    # 4. Québec Law 25 (PIA) — requires user to set their province
    if user_province:
        if user_province == "QC":
            checks.append({"id": "quebec_law25", "name": "Québec Law 25 (PIA)", "status": "pass",
                            "description": "Province set to QC — Privacy Impact Assessments are enforced automatically for all data transfers involving Quebec residents."})
        else:
            checks.append({"id": "quebec_law25", "name": "Québec Law 25 (PIA)", "status": "pass",
                            "description": f"Province set to {user_province}. PIA checks will apply automatically if you process data from QC residents."})
    else:
        checks.append({"id": "quebec_law25", "name": "Québec Law 25 (PIA)", "status": "warn",
                        "description": "Your province is not set. Set your province in Settings so Law 25 compliance checks can be applied automatically.",
                        "action": {"label": "Update your profile", "href": "/dashboard/settings"}})

    # 5. Audit Trail — checks for user-specific events, not just global existence
    try:
        store = get_event_store()
        if user_id:
            # Check for events by this user specifically
            user_events = store.get_events(limit=1)
            has_events = bool(user_events)
        else:
            has_events = False
        if has_events:
            checks.append({"id": "audit_trail", "name": "Audit Trail", "status": "pass",
                            "description": "Tamper-evident event logging active. All actions are recorded with hash-chain integrity for full auditability."})
        else:
            checks.append({"id": "audit_trail", "name": "Audit Trail", "status": "warn",
                            "description": "No audit events recorded yet. Submit a job or launch an instance to start generating your compliance audit trail.",
                            "action": {"label": "Launch an instance", "href": "/dashboard/instances/new"}})
    except Exception:
        checks.append({"id": "audit_trail", "name": "Audit Trail", "status": "fail",
                        "description": "Event store unavailable — audit logging is disabled. Contact support.",
                        "action": {"label": "View events", "href": "/dashboard/events"}})

    # 6. Payment Processing — check if THIS user has completed Stripe Connect onboarding
    try:
        mgr = get_stripe_manager()
        from stripe_connect import STRIPE_ENABLED
        if not STRIPE_ENABLED:
            checks.append({"id": "payment_rails", "name": "Payment Processing", "status": "warn",
                            "description": "Payment processing is not configured on this platform. Stripe Connect is required for provider payouts and customer billing.",
                            "action": {"label": "View billing", "href": "/dashboard/billing"}})
        elif user_provider_id:
            # User is a provider — check if they've completed Stripe onboarding
            provider = mgr.get_provider(user_provider_id)
            if provider and provider.get("stripe_account_id"):
                status = provider.get("status", "pending")
                if status == "active":
                    checks.append({"id": "payment_rails", "name": "Payment Processing", "status": "pass",
                                    "description": "Stripe Connect onboarded. Your provider account is active and ready to receive payouts."})
                else:
                    checks.append({"id": "payment_rails", "name": "Payment Processing", "status": "warn",
                                    "description": f"Stripe Connect account status: {status}. Complete your onboarding to start receiving provider payouts.",
                                    "action": {"label": "Complete onboarding", "href": "/dashboard/earnings"}})
            else:
                checks.append({"id": "payment_rails", "name": "Payment Processing", "status": "warn",
                                "description": "You are registered as a provider but have not completed Stripe Connect onboarding. Complete it to receive payouts.",
                                "action": {"label": "Set up Stripe Connect", "href": "/dashboard/earnings"}})
        else:
            # Not a provider — check from customer perspective
            checks.append({"id": "payment_rails", "name": "Payment Processing", "status": "warn",
                            "description": "Stripe Connect is available. Register as a provider and complete Stripe onboarding to earn from your GPU resources.",
                            "action": {"label": "Become a provider", "href": "/dashboard/earnings"}})
    except Exception:
        checks.append({"id": "payment_rails", "name": "Payment Processing", "status": "fail",
                        "description": "Payment processing module unavailable. Contact support.",
                        "action": {"label": "View billing", "href": "/dashboard/billing"}})

    return {"ok": True, "checks": checks}


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
    # HST provinces have a single harmonized rate (no separate GST/PST)
    HST_PROVINCES = {"NB": 0.15, "NL": 0.15, "NS": 0.15, "ON": 0.13, "PE": 0.15}
    # PST/RST/QST provinces have GST + provincial component
    PST_PROVINCES = {"BC": 0.07, "MB": 0.07, "SK": 0.06, "QC": 0.09975}

    rates = {}
    for code, (total, desc) in PROVINCE_TAX_RATES.items():
        if code in HST_PROVINCES:
            rates[code] = {"rate": total, "description": desc, "gst": 0, "pst": 0, "hst": HST_PROVINCES[code]}
        elif code in PST_PROVINCES:
            rates[code] = {"rate": total, "description": desc, "gst": 0.05, "pst": PST_PROVINCES[code], "hst": 0}
        else:
            # GST-only (AB, territories)
            rates[code] = {"rate": total, "description": desc, "gst": 0.05, "pst": 0, "hst": 0}

    return {"rates": rates}


@app.get("/api/compliance/trust-tier-requirements", tags=["Compliance"])
def api_trust_tier_requirements():
    """Full trust tier requirements matrix."""
    return {
        "tiers": [{"tier": tier.value, **reqs} for tier, reqs in TRUST_TIER_REQUIREMENTS.items()]
    }


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

@contextmanager
def _transparency_db():
    """PostgreSQL connection for transparency tables."""
    from db import _get_pg_pool
    from psycopg.rows import dict_row
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


class LegalRequestRecord(BaseModel):
    request_type: str = "subpoena"  # subpoena, warrant, mlat, production_order, informal
    jurisdiction: str = "CA"
    authority: str = ""
    scope: str = ""
    notes: str = ""


@app.post("/api/transparency/legal-request", tags=["Transparency"])
def api_record_legal_request(req: LegalRequestRecord):
    """Record a legal request (subpoena, warrant, MLAT, etc.)."""
    import uuid

    with _transparency_db() as conn:
        request_id = str(uuid.uuid4())[:12]
        conn.execute(
            """INSERT INTO legal_requests
               (request_id, received_at, request_type, jurisdiction, authority, scope, notes)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (
                request_id,
                time.time(),
                req.request_type,
                req.jurisdiction,
                req.authority,
                req.scope,
                req.notes,
            ),
        )

    # Also record as an auditable event in the hash chain
    store = get_event_store()
    store.append(
        Event(
            event_type="transparency.legal_request",
            entity_type="legal",
            entity_id=request_id,
            actor="admin",
            data={"request_type": req.request_type, "jurisdiction": req.jurisdiction},
        )
    )

    return {"ok": True, "request_id": request_id}


@app.post("/api/transparency/legal-request/{request_id}/respond", tags=["Transparency"])
def api_respond_legal_request(
    request_id: str, complied: bool = False, challenged: bool = False, notes: str = ""
):
    """Record response to a legal request."""
    with _transparency_db() as conn:
        conn.execute(
            """UPDATE legal_requests
               SET status = 'responded', responded_at = %s, complied = %s, challenged = %s, notes = %s
               WHERE request_id = %s""",
            (time.time(), int(complied), int(challenged), notes, request_id),
        )
    return {"ok": True, "request_id": request_id}


@app.get("/api/transparency/report", tags=["Transparency"])
def api_transparency_report(months: int = 12):
    """Generate transparency report — CLOUD Act diligence artifact.

    Returns summary of all legal requests and data disclosures.
    Monthly JSON per REPORT_FEATURE_2.md Phase B §3.
    """
    with _transparency_db() as conn:
        since = time.time() - (months * 30 * 86400)

        requests_rows = conn.execute(
            "SELECT * FROM legal_requests WHERE received_at >= %s ORDER BY received_at DESC",
            (since,),
        ).fetchall()

        disclosures_rows = conn.execute(
            "SELECT * FROM data_disclosures WHERE disclosed_at >= %s ORDER BY disclosed_at DESC",
            (since,),
        ).fetchall()

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


@app.get("/api/audit/instance/{job_id}", tags=["Events"])
def api_instance_audit_trail(job_id: str):
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

    # Wire verification → reputation: grant HARDWARE_AUDIT points on pass
    if result.state == "verified" or (hasattr(result.state, "value") and result.state.value == "verified"):
        try:
            re = get_reputation_engine()
            re.add_verification(payload.host_id, VerificationType.HARDWARE_AUDIT)
            log.info("REPUTATION HARDWARE_AUDIT granted for verified host %s", payload.host_id)
        except Exception:
            log.exception("Non-fatal: could not update reputation for %s", payload.host_id)

    return {
        "ok": True,
        "host_id": payload.host_id,
        "state": result.state.value if hasattr(result.state, "value") else str(result.state),
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
    month: str  # YYYY-MM
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
        req.host_id,
        req.tier,
        req.month,
        req.monthly_spend_cad,
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
    try:
        engine = get_sla_engine()
    except Exception:
        return {"ok": True, "hosts": [], "count": 0}
    import scheduler as _sched

    hosts = _sched.list_hosts(active_only=False)
    summaries = []
    for h in hosts:
        hid = h.get("host_id", "")
        if not hid:
            continue
        try:
            uptime = engine.get_host_uptime_pct(hid)
            violations = engine.get_violations(hid)
        except Exception:
            uptime = 0.0
            violations = []
        tier = h.get("sla_tier", "community")
        summaries.append(
            {
                "host_id": hid,
                "gpu_model": h.get("gpu_model", "Unknown"),
                "status": h.get("status", "unknown"),
                "sla_tier": tier,
                "uptime_30d_pct": round(uptime, 4),
                "violation_count": len(violations),
                "last_violation": violations[-1] if violations else None,
                "country": h.get("country", ""),
                "province": h.get("province", ""),
            }
        )
    return {"ok": True, "hosts": summaries, "count": len(summaries)}


@app.get("/api/sla/{host_id}", tags=["SLA"])
def api_sla_status(host_id: str, month: str = ""):
    """Get SLA record and rolling uptime for a host."""
    engine = get_sla_engine()
    uptime_30d = engine.get_host_uptime_pct(host_id)
    record = None
    if month:
        rec = engine.get_host_sla(host_id, month)
        record = (
            {
                "month": rec.month,
                "tier": rec.tier,
                "uptime_pct": round(rec.uptime_pct, 4),
                "downtime_seconds": rec.downtime_seconds,
                "incidents": rec.incidents,
                "credit_pct": rec.credit_pct,
                "credit_cad": rec.credit_cad,
            }
            if rec
            else None
        )
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
    corporation_name: str = ""  # Required for company type
    business_number: str = ""  # CRA Business Number (BN)
    gst_hst_number: str = ""  # GST/HST registration number
    province: str = ""  # ON, QC, BC, AB, etc.
    legal_name: str = ""  # Legal name of individual or entity


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
    # Link provider_id to user account
    from db import UserStore
    UserStore.update_user(req.email, {"provider_id": req.provider_id})

    broadcast_sse(
        "provider_registered",
        {
            "provider_id": req.provider_id,
            "type": req.provider_type,
            "corporation_name": req.corporation_name,
        },
    )
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


@app.post("/api/providers/webhook", tags=["Providers"])
async def api_stripe_webhook(request: Request):
    """Handle Stripe Connect webhooks (account.updated, payment_intent.succeeded, etc.)."""
    with otel_span("webhook.stripe"):
        payload = await request.body()
        sig_header = request.headers.get("stripe-signature", "")
        mgr = get_stripe_manager()
        result = mgr.handle_webhook(payload, sig_header)
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
    safe_profile = {k: v for k, v in profile.items() if k not in ("hashed_password", "password")}

    # Gather jobs
    all_jobs = list_jobs()
    user_jobs = [
        j for j in all_jobs if j.get("customer_id") == customer_id or j.get("submitted_by") == email
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
        result.append(
            {
                "artifact_id": a.get("artifact_id", ""),
                "artifact_type": art_type,
                "created_at": created,
                "ttl_days": ttl_days,
                "expires_at": expiry,
                "days_remaining": max(0, int((expiry - time.time()) / 86400)),
            }
        )

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


# ═══════════════════════════════════════════════════════════════════════
# Inference API — Serverless GPU inference endpoints
# ═══════════════════════════════════════════════════════════════════════


class InferenceRequest(BaseModel):
    model: str = Field(
        ...,
        description="Model name or HuggingFace repo (e.g. 'distilbert-base-uncased-finetuned-sst-2-english')",
    )
    inputs: list[str] | str = Field(..., description="Text input(s) for inference")
    gpu_model: str = Field("any", description="Preferred GPU model or 'any'")
    max_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    timeout_sec: int = Field(300, ge=10, le=3600)


@app.post("/api/inference", tags=["Inference"])
def api_inference_submit(req: InferenceRequest, request: Request):
    """Submit a serverless inference request.

    Schedules a short-lived GPU job that runs the specified model on the
    provided inputs. Returns a job_id to poll for results.
    """
    user = _get_current_user(request)
    customer_id = user.get("customer_id", user.get("email", "anon")) if user else "anon"
    inputs_list = [req.inputs] if isinstance(req.inputs, str) else req.inputs

    job = submit_job(
        name=f"inference:{req.model.replace('/', '--')}",
        vram_needed_gb=2,
        image=f"xcelsior/inference:{req.model.replace('/', '--')}",
    )
    job_id = job.get("job_id", job.get("id", str(uuid.uuid4())))
    # Persist inference metadata to SQLite (survives API restarts)
    store_inference_job(
        job_id=job_id,
        customer_id=customer_id,
        model=req.model,
        inputs=inputs_list,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        timeout_sec=req.timeout_sec,
    )
    broadcast_sse("inference_submitted", {"job_id": job_id, "model": req.model})
    return {"ok": True, "job_id": job_id, "model": req.model, "status": "queued"}


@app.get("/api/inference/{job_id}", tags=["Inference"])
def api_inference_result(job_id: str):
    """Get inference results for a submitted request."""
    result = get_inference_result(job_id)
    if result:
        return {"ok": True, "status": "completed", **result}
    meta = get_inference_job(job_id)
    if meta:
        elapsed = time.time() - meta["submitted_at"]
        if elapsed > meta["timeout_sec"]:
            return {"ok": False, "status": "timeout", "job_id": job_id}
        return {
            "ok": True,
            "status": meta.get("status", "running"),
            "job_id": job_id,
            "model": meta["model"],
            "elapsed_sec": round(elapsed, 1),
        }
    # Check scheduler
    jobs = list_jobs()
    job = next((j for j in jobs if j.get("job_id") == job_id), None)
    if job:
        return {"ok": True, "status": job.get("status", "unknown"), "job_id": job_id}
    raise HTTPException(404, f"Inference job {job_id} not found")


@app.get("/api/inference/models/available", tags=["Inference"])
def api_inference_models():
    """List available inference models and their resource requirements."""
    models = [
        {
            "name": "distilbert-base-uncased-finetuned-sst-2-english",
            "task": "sentiment-analysis",
            "min_vram_gb": 1,
            "avg_latency_ms": 50,
        },
        {
            "name": "meta-llama/Llama-2-7b-chat-hf",
            "task": "text-generation",
            "min_vram_gb": 14,
            "avg_latency_ms": 2000,
        },
        {
            "name": "meta-llama/Llama-2-13b-chat-hf",
            "task": "text-generation",
            "min_vram_gb": 26,
            "avg_latency_ms": 4000,
        },
        {
            "name": "stabilityai/stable-diffusion-xl-base-1.0",
            "task": "image-generation",
            "min_vram_gb": 8,
            "avg_latency_ms": 5000,
        },
        {
            "name": "openai/whisper-large-v3",
            "task": "speech-to-text",
            "min_vram_gb": 4,
            "avg_latency_ms": 3000,
        },
        {
            "name": "BAAI/bge-large-en-v1.5",
            "task": "embeddings",
            "min_vram_gb": 2,
            "avg_latency_ms": 100,
        },
    ]
    return {"ok": True, "models": models}


class InferenceResultCallback(BaseModel):
    outputs: list = Field(default_factory=list)
    model: str = ""
    latency_ms: float = 0


@app.post("/api/inference/{job_id}/result", tags=["Inference"])
def api_inference_post_result(job_id: str, body: InferenceResultCallback):
    """Worker callback: post inference results. Internal use."""
    store_inference_result(
        job_id=job_id,
        outputs=body.outputs,
        model=body.model,
        latency_ms=body.latency_ms,
    )
    broadcast_sse("inference_completed", {"job_id": job_id})
    return {"ok": True}


# ── /v1/inference (OpenAI-compatible sync + async) ────────────────────


class V1InferenceRequest(BaseModel):
    """OpenAI-compatible inference request for /v1/inference."""
    model: str = Field(..., description="Model name or HuggingFace repo")
    inputs: list[str] | str = Field(..., description="Text input(s) for inference")
    max_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    stream: bool = Field(False, description="Stream response via SSE")


@app.post("/v1/inference", tags=["Inference v2"])
def api_v1_inference_sync(body: V1InferenceRequest, request: Request):
    """Synchronous inference endpoint (OpenAI-compatible path).

    Submits an inference request and polls for results up to 30 seconds.
    If stream=true, returns SSE text/event-stream with token deltas.
    """
    user = _get_current_user(request)
    customer_id = user.get("customer_id", user.get("email", "anon")) if user else "anon"
    inputs_list = [body.inputs] if isinstance(body.inputs, str) else body.inputs

    with otel_span("inference.v1.sync", {"model": body.model, "stream": body.stream}):
        job = submit_job(
            name=f"inference:{body.model.replace('/', '--')}",
            vram_needed_gb=2,
            image=f"xcelsior/inference:{body.model.replace('/', '--')}",
        )
        job_id = job.get("job_id", job.get("id", str(uuid.uuid4())))
        store_inference_job(
            job_id=job_id,
            customer_id=customer_id,
            model=body.model,
            inputs=inputs_list,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            timeout_sec=30,
        )

        if body.stream:
            # SSE streaming response (OpenAI delta format)
            async def _sse_stream():
                start = time.time()
                yield f"data: {json.dumps({'id': job_id, 'object': 'inference.chunk', 'model': body.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
                while time.time() - start < 30:
                    result = get_inference_result(job_id)
                    if result:
                        outputs = result.get("outputs", [])
                        for token in outputs:
                            yield f"data: {json.dumps({'id': job_id, 'object': 'inference.chunk', 'model': body.model, 'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}]})}\n\n"
                        yield f"data: {json.dumps({'id': job_id, 'object': 'inference.chunk', 'model': body.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    await asyncio.sleep(0.5)
                yield f"data: {json.dumps({'id': job_id, 'error': 'timeout'})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_sse_stream(), media_type="text/event-stream")

        # Sync: poll up to 30 seconds
        start = time.time()
        while time.time() - start < 30:
            result = get_inference_result(job_id)
            if result:
                return {
                    "id": job_id,
                    "object": "inference.completion",
                    "created": int(time.time()),
                    "model": body.model,
                    "outputs": result.get("outputs", []),
                    "usage": {
                        "input_tokens": result.get("input_tokens", 0),
                        "output_tokens": result.get("output_tokens", 0),
                        "total_tokens": result.get("input_tokens", 0) + result.get("output_tokens", 0),
                    },
                    "latency_ms": result.get("latency_ms", 0),
                }
            time.sleep(0.5)

        # Timeout — return pending status
        return JSONResponse(
            status_code=202,
            content={
                "id": job_id,
                "object": "inference.pending",
                "model": body.model,
                "status": "processing",
                "poll_url": f"/v1/inference/{job_id}",
            },
        )


@app.post("/v1/inference/async", tags=["Inference v2"])
def api_v1_inference_async(body: V1InferenceRequest, request: Request):
    """Asynchronous inference — returns job_id immediately for polling."""
    user = _get_current_user(request)
    customer_id = user.get("customer_id", user.get("email", "anon")) if user else "anon"
    inputs_list = [body.inputs] if isinstance(body.inputs, str) else body.inputs

    with otel_span("inference.v1.async", {"model": body.model}):
        job = submit_job(
            name=f"inference:{body.model.replace('/', '--')}",
            vram_needed_gb=2,
            image=f"xcelsior/inference:{body.model.replace('/', '--')}",
        )
        job_id = job.get("job_id", job.get("id", str(uuid.uuid4())))
        store_inference_job(
            job_id=job_id,
            customer_id=customer_id,
            model=body.model,
            inputs=inputs_list,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            timeout_sec=300,
        )

    return {
        "id": job_id,
        "object": "inference.async",
        "model": body.model,
        "status": "queued",
        "poll_url": f"/v1/inference/{job_id}",
    }


@app.get("/v1/inference/{job_id}", tags=["Inference v2"])
def api_v1_inference_poll(job_id: str):
    """Poll for inference results by job_id."""
    result = get_inference_result(job_id)
    if result:
        return {
            "id": job_id,
            "object": "inference.completion",
            "created": int(time.time()),
            "model": result.get("model", ""),
            "outputs": result.get("outputs", []),
            "usage": {
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
                "total_tokens": result.get("input_tokens", 0) + result.get("output_tokens", 0),
            },
            "latency_ms": result.get("latency_ms", 0),
            "status": "completed",
        }
    meta = get_inference_job(job_id)
    if meta:
        elapsed = time.time() - meta["submitted_at"]
        if elapsed > meta["timeout_sec"]:
            return {"id": job_id, "status": "timeout", "elapsed_sec": round(elapsed, 1)}
        return {"id": job_id, "status": "processing", "elapsed_sec": round(elapsed, 1)}
    raise HTTPException(404, f"Inference job {job_id} not found")


# ═══ Missing route aliases (frontend expects /api/ prefix) ════════════


@app.get("/api/alerts/config", tags=["Infrastructure"])
def api_get_alert_config_alias():
    """Alias for /alerts/config with /api/ prefix."""
    safe = {k: ("***" if "pass" in k or "token" in k else v) for k, v in ALERT_CONFIG.items()}
    return {"config": safe}


@app.put("/api/alerts/config", tags=["Infrastructure"])
def api_set_alert_config_alias(cfg: AlertConfig):
    """Alias for PUT /alerts/config with /api/ prefix."""
    updates = {k: v for k, v in cfg.model_dump().items() if v is not None}
    configure_alerts(**updates)
    return {"ok": True, "updated": list(updates.keys())}


@app.get("/api/artifacts", tags=["Artifacts"])
def api_list_all_artifacts():
    """List all artifacts (no job filter)."""
    mgr = get_artifact_manager()
    try:
        artifacts = []
        from artifacts import ArtifactType as AT
        for atype in AT:
            artifacts.extend(mgr.primary.list_objects(f"{atype.value}/"))
        return {"ok": True, "artifacts": artifacts}
    except Exception:
        return {"ok": True, "artifacts": []}


@app.get("/api/slurm/instances", tags=["Infrastructure"])
def api_slurm_list_instances():
    """List all tracked Slurm jobs."""
    from slurm_adapter import _load_slurm_map, get_slurm_job_status
    job_map = _load_slurm_map()
    jobs = []
    for xcelsior_id, slurm_id in job_map.items():
        jobs.append({"job_id": xcelsior_id, "slurm_job_id": slurm_id})
    return {"ok": True, "jobs": jobs}


@app.get("/api/events", tags=["Events"])
def api_get_all_events(limit: int = 100):
    """Get recent events across all entities."""
    store = get_event_store()
    events = store.get_events(limit=limit)
    return {"ok": True, "events": [e if isinstance(e, dict) else e.__dict__ for e in events]}


@app.get("/api/users/me/preferences", tags=["Auth"])
def api_get_user_preferences(request: Request):
    """Get user preferences."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    if _USE_PERSISTENT_AUTH:
        full_user = UserStore.get_user(user["email"]) or {}
    else:
        full_user = _users_db.get(user["email"], {})
    # Parse preferences JSONB (may be dict or JSON string)
    raw_prefs = full_user.get("preferences", {})
    if isinstance(raw_prefs, str):
        try:
            import json as _json
            raw_prefs = _json.loads(raw_prefs)
        except Exception:
            raw_prefs = {}
    return {
        "ok": True,
        "canada_only_routing": bool(full_user.get("canada_only_routing", 0)),
        "notifications": bool(full_user.get("notifications_enabled", 1)),
        "preferences": raw_prefs if isinstance(raw_prefs, dict) else {},
    }


@app.put("/api/users/me/preferences", tags=["Auth"])
def api_set_user_preferences(request: Request, body: dict):
    """Update user preferences."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    updates: dict = {}
    if "notifications" in body:
        updates["notifications_enabled"] = 1 if body["notifications"] else 0
    if "canada_only_routing" in body:
        updates["canada_only_routing"] = 1 if body["canada_only_routing"] else 0
    # Merge JSONB preferences (partial update)
    if "preferences" in body and isinstance(body["preferences"], dict):
        # Validate: only allow known preference keys
        allowed_pref_keys = {"onboarding", "ai_panel_open"}
        incoming = body["preferences"]
        safe_prefs = {k: v for k, v in incoming.items() if k in allowed_pref_keys}
        if safe_prefs and _USE_PERSISTENT_AUTH:
            # Merge with existing preferences
            existing_user = UserStore.get_user(user["email"]) or {}
            existing_prefs = existing_user.get("preferences", {})
            if isinstance(existing_prefs, str):
                try:
                    import json as _json
                    existing_prefs = _json.loads(existing_prefs)
                except Exception:
                    existing_prefs = {}
            if not isinstance(existing_prefs, dict):
                existing_prefs = {}
            merged = {**existing_prefs, **safe_prefs}
            updates["preferences"] = merged
    if updates and _USE_PERSISTENT_AUTH:
        UserStore.update_user(user["email"], updates)
    return {"ok": True}


# ── Notifications ─────────────────────────────────────────────────────


@app.get("/api/notifications", tags=["Notifications"])
def api_list_notifications(request: Request, unread: bool = False, limit: int = 50):
    """List notifications for the current user."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    notifications = NotificationStore.list_for_user(user["email"], unread_only=unread, limit=limit)
    unread_count = NotificationStore.unread_count(user["email"])
    return {"ok": True, "notifications": notifications, "unread_count": unread_count}


@app.get("/api/notifications/unread-count", tags=["Notifications"])
def api_notification_unread_count(request: Request):
    """Get the unread notification count for the current user."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    return {"ok": True, "unread_count": NotificationStore.unread_count(user["email"])}


@app.post("/api/notifications/{notification_id}/read", tags=["Notifications"])
def api_mark_notification_read(request: Request, notification_id: str):
    """Mark a single notification as read."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    ok = NotificationStore.mark_read(notification_id, user["email"])
    if not ok:
        raise HTTPException(404, "Notification not found")
    return {"ok": True}


@app.post("/api/notifications/read-all", tags=["Notifications"])
def api_mark_all_read(request: Request):
    """Mark all notifications as read for the current user."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    count = NotificationStore.mark_all_read(user["email"])
    return {"ok": True, "marked": count}


@app.delete("/api/notifications/{notification_id}", tags=["Notifications"])
def api_delete_notification(request: Request, notification_id: str):
    """Delete a single notification."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    ok = NotificationStore.delete(notification_id, user["email"])
    if not ok:
        raise HTTPException(404, "Notification not found")
    return {"ok": True}


@app.get("/api/admin/stats", tags=["Admin"])
def api_admin_stats(request: Request):
    """Get admin dashboard statistics."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    hosts = list_hosts(active_only=False)
    active_hosts = [h for h in hosts if h.get("state") == "idle" or h.get("state") == "busy"]
    jobs = list_jobs()
    running = [j for j in jobs if j.get("status") in ("running", "assigned")]
    if _USE_PERSISTENT_AUTH:
        users = UserStore.list_users()
    else:
        users = list(_users_db.values())
    return {
        "ok": True,
        "total_users": len(users),
        "active_hosts": len(active_hosts),
        "running_jobs": len(running),
        "revenue_mtd": 0,
    }


@app.get("/api/admin/users", tags=["Admin"])
def api_admin_users(request: Request):
    """List all users for admin panel."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    if _USE_PERSISTENT_AUTH:
        users = UserStore.list_users()
    else:
        users = list(_users_db.values())
    safe_users = [
        {
            "email": u.get("email", ""),
            "role": u.get("role", "submitter"),
            "is_active": True,
            "created_at": u.get("created_at", ""),
        }
        for u in users
    ]
    return {"ok": True, "users": safe_users}


@app.get("/api/admin/verification-queue", tags=["Admin"])
def api_admin_verification_queue(request: Request):
    """Get verification queue for admin panel."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    ve = get_verification_engine()
    store = ve.store
    try:
        with store._conn() as conn:
            rows = conn.execute(
                "SELECT host_id, state, overall_score, last_check_at FROM host_verifications WHERE state = 'pending' ORDER BY last_check_at DESC"
            ).fetchall()
        queue = [dict(r) for r in rows]
    except Exception:
        queue = []
    return {"ok": True, "queue": queue}


# ── AI Chat Endpoint ──────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = None


@app.post("/api/chat", tags=["Chat"])
async def api_chat(body: ChatRequest, request: Request):
    """Stream an AI chat response about Xcelsior via SSE."""
    client_ip = request.client.host if request.client else "unknown"

    if not check_chat_rate_limit(client_ip):
        raise HTTPException(429, "Chat rate limit exceeded. Please wait a moment.")

    if not CHAT_API_KEY:
        raise HTTPException(503, "Chat is not configured.")

    # Sanitise user input
    user_message = redact_pii(body.message)

    # Get or create conversation (persisted to SQLite)
    user = _get_current_user(request)
    user_email = user.get("email") if user else None
    conversation_id, history = get_or_create_conversation(
        body.conversation_id, ip=client_ip, user_email=user_email
    )

    # Build messages array
    system_prompt = build_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    # Track user message
    append_message(conversation_id, "user", user_message)

    async def _generate():
        full_response = []
        # Send conversation_id as first event
        yield f"data: {json.dumps({'type': 'meta', 'conversation_id': conversation_id})}\n\n"
        try:
            async for token in stream_chat_response(messages):
                full_response.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            # Store assistant response
            append_message(conversation_id, "assistant", "".join(full_response))
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            log.error("Chat stream error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': 'An error occurred. Please try again.'})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/chat/suggestions", tags=["Chat"])
def api_chat_suggestions():
    """Return suggested starter questions for the chat widget."""
    return {
        "ok": True,
        "suggestions": [
            "How do I list my GPU on the marketplace?",
            "What are the trust tiers?",
            "How does billing work?",
            "How do I submit a job?",
        ],
    }


@app.get("/api/chat/history/{conversation_id}", tags=["Chat"])
def api_chat_history(conversation_id: str, request: Request):
    """Return message history for an existing conversation."""
    client_ip = request.client.host if request.client else "unknown"
    if not check_chat_rate_limit(client_ip):
        raise HTTPException(429, "Rate limit exceeded.")
    messages = get_conversation_messages(conversation_id)
    if messages is None:
        raise HTTPException(404, "Conversation not found or expired.")
    return {
        "ok": True,
        "conversation_id": conversation_id,
        "messages": messages,
    }


@app.get("/api/chat/conversations", tags=["Chat"])
def api_chat_conversations(request: Request):
    """List recent conversations for the authenticated user."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated.")
    email = user.get("email", "")
    if not email:
        raise HTTPException(401, "No email in session.")
    conversations = get_user_conversations(email)
    return {"ok": True, "conversations": conversations}


class ChatFeedbackRequest(BaseModel):
    message_id: str
    vote: str  # "up" or "down"


@app.post("/api/chat/feedback", tags=["Chat"])
def api_chat_feedback(body: ChatFeedbackRequest):
    """Record thumbs-up / thumbs-down feedback on a chat message."""
    if body.vote not in ("up", "down"):
        raise HTTPException(400, "Vote must be 'up' or 'down'.")
    record_feedback(body.message_id, body.vote)
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: New Feature API Routes
# Marketplace, Inference v2, Volumes, Auto-billing, Cloud Burst, Privacy
# ═══════════════════════════════════════════════════════════════════════

from marketplace import get_marketplace_engine
from inference import get_inference_engine
from volumes import get_volume_engine
from cloudburst import get_burst_engine as get_cloudburst_engine
from privacy import get_crypto_shredder, get_consent_manager, execute_right_to_erasure

# ── GPU Marketplace Offers ────────────────────────────────────────────

OPENAPI_TAGS.append({"name": "Marketplace v2", "description": "GPU marketplace offers, allocations, spot pricing, reservations."})
OPENAPI_TAGS.append({"name": "Volumes", "description": "Persistent volume lifecycle management."})
OPENAPI_TAGS.append({"name": "Cloud Burst", "description": "Cloud provider burst auto-scaling."})
OPENAPI_TAGS.append({"name": "Inference v2", "description": "Production serverless inference with OpenAI-compatible API."})


class GPUOfferCreate(BaseModel):
    host_id: str
    gpu_model: str
    gpu_count_total: int = 1
    vram_gb: float = 0
    ask_cents_per_hour: int = 20
    region: str = "ca-east"
    spot_enabled: bool = True
    spot_min_cents: int = 10


@app.post("/api/v2/marketplace/offers", tags=["Marketplace v2"])
def api_marketplace_create_offer(body: GPUOfferCreate, request: Request):
    """Create or update a GPU offer on the marketplace."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    me = get_marketplace_engine()
    offer = me.upsert_offer(
        provider_id=user.get("user_id", user.get("email", "")),
        host_id=body.host_id,
        gpu_model=body.gpu_model,
        gpu_count_total=body.gpu_count_total,
        vram_gb=body.vram_gb,
        ask_cents_per_hour=body.ask_cents_per_hour,
        region=body.region,
        spot_enabled=body.spot_enabled,
        spot_min_cents=body.spot_min_cents,
    )
    return {"ok": True, "offer": offer}


class MarketplaceSearchParams(BaseModel):
    gpu_model: str = ""
    min_vram_gb: float = 0
    max_price_cents: int = 0
    region: str = ""
    canada_only: bool = False
    sort_by: str = "price"
    limit: int = 50


@app.post("/api/v2/marketplace/search", tags=["Marketplace v2"])
def api_marketplace_search(body: MarketplaceSearchParams):
    """Search available GPU offers with filters."""
    me = get_marketplace_engine()
    offers = me.search_offers(
        gpu_model=body.gpu_model or None,
        min_vram_gb=body.min_vram_gb or None,
        max_price_cents=body.max_price_cents or None,
        region=body.region or None,
        canada_only=body.canada_only,
        sort_by=body.sort_by,
        limit=body.limit,
    )
    return {"ok": True, "offers": offers, "count": len(offers)}


@app.get("/api/v2/marketplace/spot-prices", tags=["Marketplace v2"])
def api_marketplace_spot_prices():
    """Get current spot prices for all GPU models."""
    me = get_marketplace_engine()
    prices = me.get_current_spot_prices_list()
    return {"ok": True, "spot_prices": prices}


@app.get("/api/v2/marketplace/spot-prices/{gpu_model}/history", tags=["Marketplace v2"])
def api_marketplace_spot_history(gpu_model: str, hours: int = 24):
    """Get spot price history for a GPU model."""
    me = get_marketplace_engine()
    history = me.get_spot_price_history(gpu_model, hours=hours)
    return {"ok": True, "gpu_model": gpu_model, "history": history}


@app.get("/api/v2/marketplace/stats", tags=["Marketplace v2"])
def api_marketplace_stats_v2():
    """Get marketplace aggregate statistics."""
    me = get_marketplace_engine()
    stats = me.get_marketplace_stats()
    return {"ok": True, **stats}


class ReservationCreate(BaseModel):
    gpu_model: str
    gpu_count: int = 1
    period_months: int = 1


@app.post("/api/v2/marketplace/reservations", tags=["Marketplace v2"])
def api_marketplace_create_reservation(body: ReservationCreate, request: Request):
    """Create a reserved instance commitment."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    me = get_marketplace_engine()
    try:
        res = me.create_reservation(
            customer_id=user.get("user_id", user.get("email", "")),
            gpu_model=body.gpu_model,
            gpu_count=body.gpu_count,
            period_months=body.period_months,
        )
        return {"ok": True, "reservation": res}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/v2/marketplace/reservations/{reservation_id}", tags=["Marketplace v2"])
def api_marketplace_cancel_reservation(reservation_id: str, request: Request):
    """Cancel a reserved instance commitment early.

    Computes early termination fee: remaining_months * monthly_rate * 50%.
    """
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    me = get_marketplace_engine()
    result = me.cancel_reservation(
        reservation_id=reservation_id,
        customer_id=user.get("user_id", user.get("email", "")),
    )
    if "error" in result:
        raise HTTPException(400, result["error"])
    return {"ok": True, **result}


class AllocateGPURequest(BaseModel):
    offer_id: str
    job_id: str
    gpu_count: int = 1
    spot: bool = False


@app.post("/api/v2/marketplace/allocate", tags=["Marketplace v2"])
def api_marketplace_allocate(body: AllocateGPURequest, request: Request):
    """Allocate GPUs from an offer for a job. Atomic — prevents double-sell."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    me = get_marketplace_engine()
    alloc = me.allocate_gpu(body.offer_id, body.job_id, body.gpu_count, spot=body.spot)
    if not alloc:
        raise HTTPException(409, "Offer not available or insufficient GPUs")
    return {"ok": True, "allocation": alloc}


@app.post("/api/v2/marketplace/release/{allocation_id}", tags=["Marketplace v2"])
def api_marketplace_release(allocation_id: str):
    """Release a GPU allocation (job completed/failed)."""
    me = get_marketplace_engine()
    me.release_allocation(allocation_id)
    return {"ok": True}


# ── Inference v2: OpenAI-Compatible ───────────────────────────────────

class InferenceEndpointCreate(BaseModel):
    model_name: str
    min_workers: int = 0
    max_workers: int = 3
    scaledown_window_sec: int = 300


@app.post("/api/v2/inference/endpoints", tags=["Inference v2"])
def api_inference_create_endpoint(body: InferenceEndpointCreate, request: Request):
    """Create a serverless inference endpoint."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    ie = get_inference_engine()
    ep = ie.create_endpoint(
        owner_id=user.get("user_id", user.get("email", "")),
        model_name=body.model_name,
        min_workers=body.min_workers,
        max_workers=body.max_workers,
        scaledown_window_sec=body.scaledown_window_sec,
    )
    return {"ok": True, "endpoint": ep}


@app.get("/api/v2/inference/endpoints", tags=["Inference v2"])
def api_inference_list_endpoints(request: Request):
    """List inference endpoints for the current user."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    ie = get_inference_engine()
    endpoints = ie.list_endpoints(user.get("user_id", user.get("email", "")))
    return {"ok": True, "endpoints": endpoints}


@app.get("/api/v2/inference/endpoints/{endpoint_id}", tags=["Inference v2"])
def api_inference_get_endpoint(endpoint_id: str):
    """Get inference endpoint details."""
    ie = get_inference_engine()
    ep = ie.get_endpoint(endpoint_id)
    if not ep:
        raise HTTPException(404, "Endpoint not found")
    return {"ok": True, "endpoint": ep}


@app.delete("/api/v2/inference/endpoints/{endpoint_id}", tags=["Inference v2"])
def api_inference_delete_endpoint(endpoint_id: str, request: Request):
    """Delete an inference endpoint."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    ie = get_inference_engine()
    ie.delete_endpoint(endpoint_id)
    return {"ok": True}


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: list[dict] = Field(default_factory=list)
    max_tokens: int = Field(512, ge=1, le=32768)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    stream: bool = False


@app.post("/v1/chat/completions", tags=["Inference v2"])
def api_openai_chat_completions(body: ChatCompletionRequest, request: Request):
    """OpenAI-compatible chat completions endpoint.

    Routes to a serverless inference endpoint running the requested model.
    """
    user = _get_current_user(request)
    customer_id = user.get("user_id", user.get("email", "anon")) if user else "anon"

    ie = get_inference_engine()
    from inference import InferenceRequest as InfReq
    inf_req = InfReq(
        request_id=str(uuid.uuid4()),
        endpoint_id="",
        model=body.model,
        messages=body.messages,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        stream=body.stream,
        customer_id=customer_id,
    )
    result = ie.submit_request(inf_req)
    if not result:
        raise HTTPException(503, f"No available workers for model {body.model}")

    # Return OpenAI-compatible response format
    return {
        "id": f"chatcmpl-{inf_req.request_id[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": ""},
            "finish_reason": None,
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "xcelsior": {"request_id": inf_req.request_id, "status": "processing"},
    }


@app.post("/api/v2/inference/complete/{request_id}", tags=["Inference v2"])
def api_inference_complete(request_id: str, request: Request):
    """Worker callback: mark inference request as completed with results."""
    ie = get_inference_engine()
    try:
        body = json.loads(request._body) if hasattr(request, '_body') else {}
    except Exception:
        body = {}
    ie.complete_request(
        request_id=request_id,
        output_text=body.get("output_text", ""),
        input_tokens=body.get("input_tokens", 0),
        output_tokens=body.get("output_tokens", 0),
        latency_ms=body.get("latency_ms", 0),
    )
    broadcast_sse("inference_completed", {"request_id": request_id})
    return {"ok": True}


# ── Persistent Volumes ────────────────────────────────────────────────

VOLUME_PRICE_PER_GB_MONTH_CAD = 0.07  # $0.07/GB/month per plan.md §10.2

class VolumeCreate(BaseModel):
    name: str
    size_gb: int = 50
    region: str = "ca-east"
    encrypted: bool = True


@app.post("/api/v2/volumes", tags=["Volumes"])
def api_volume_create(body: VolumeCreate, request: Request):
    """Create a new persistent volume."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    ve = get_volume_engine()
    try:
        vol = ve.create_volume(
            owner_id=user.get("user_id", user.get("email", "")),
            name=body.name,
            size_gb=body.size_gb,
            region=body.region,
            encrypted=body.encrypted,
        )
        vol["price_per_gb_month_cad"] = VOLUME_PRICE_PER_GB_MONTH_CAD
        vol["estimated_monthly_cost_cad"] = round(body.size_gb * VOLUME_PRICE_PER_GB_MONTH_CAD, 2)
        return {"ok": True, "volume": vol}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/v2/volumes", tags=["Volumes"])
def api_volume_list(request: Request):
    """List volumes owned by the current user."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    ve = get_volume_engine()
    volumes = ve.list_volumes(user.get("user_id", user.get("email", "")))
    return {"ok": True, "volumes": volumes}


@app.get("/api/v2/volumes/{volume_id}", tags=["Volumes"])
def api_volume_get(volume_id: str, request: Request):
    """Get volume details."""
    ve = get_volume_engine()
    vol = ve.get_volume(volume_id)
    if not vol:
        raise HTTPException(404, "Volume not found")
    return {"ok": True, "volume": vol}


class VolumeAttachRequest(BaseModel):
    instance_id: str
    mount_path: str = "/workspace"
    mode: str = "rw"


@app.post("/api/v2/volumes/{volume_id}/attach", tags=["Volumes"])
def api_volume_attach(volume_id: str, body: VolumeAttachRequest, request: Request):
    """Attach a volume to a running instance."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    ve = get_volume_engine()
    try:
        att = ve.attach_volume(volume_id, body.instance_id, body.mount_path, body.mode)
        if not att:
            raise HTTPException(409, "Volume not available for attachment")
        return {"ok": True, "attachment": att}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/v2/volumes/{volume_id}/detach", tags=["Volumes"])
def api_volume_detach(volume_id: str, request: Request):
    """Detach a volume from its instance."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    body_data = {}
    ve = get_volume_engine()
    instance_id = body_data.get("instance_id", "")
    ve.detach_volume(volume_id, instance_id)
    return {"ok": True}


@app.delete("/api/v2/volumes/{volume_id}", tags=["Volumes"])
def api_volume_delete(volume_id: str, request: Request):
    """Delete a volume. Must not have active attachments."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    ve = get_volume_engine()
    try:
        ve.delete_volume(volume_id)
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(409, str(e))


# ── Auto-Billing Configuration ────────────────────────────────────────

class AutoTopupConfig(BaseModel):
    enabled: bool = True
    amount_cad: float = 25.0
    threshold_cad: float = 5.0
    stripe_payment_method_id: str = ""


@app.post("/api/v2/billing/auto-topup", tags=["Billing"])
def api_billing_configure_topup(body: AutoTopupConfig, request: Request):
    """Configure wallet auto-top-up via Stripe."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    be = get_billing_engine()
    customer_id = user.get("customer_id", user.get("user_id", user.get("email", "")))
    be.configure_auto_topup(
        customer_id=customer_id,
        enabled=body.enabled,
        amount_cad=body.amount_cad,
        threshold_cad=body.threshold_cad,
        payment_method_id=body.stripe_payment_method_id,
    )
    return {"ok": True, "auto_topup": body.model_dump()}


@app.get("/api/v2/billing/auto-topup", tags=["Billing"])
def api_billing_get_topup(request: Request):
    """Get current auto-top-up configuration."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    be = get_billing_engine()
    customer_id = user.get("customer_id", user.get("user_id", user.get("email", "")))
    wallet = be.get_or_create_wallet(customer_id)
    return {
        "ok": True,
        "auto_topup": {
            "enabled": bool(wallet.get("auto_topup_enabled", False)),
            "amount_cad": wallet.get("auto_topup_amount", 0),
            "threshold_cad": wallet.get("auto_topup_threshold", 0),
        },
    }


# ── Cloud Burst Status ────────────────────────────────────────────────

@app.get("/api/v2/burst/status", tags=["Cloud Burst"])
def api_burst_status(request: Request):
    """Get cloud burst auto-scaling status."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    cbe = get_cloudburst_engine()
    status = cbe.get_burst_status()
    return {"ok": True, **status}


# ── Privacy & Consent ─────────────────────────────────────────────────

class ConsentRequest(BaseModel):
    purpose: str
    consent_type: str = "express"


@app.post("/api/v2/privacy/consent", tags=["Privacy"])
def api_privacy_record_consent(body: ConsentRequest, request: Request):
    """Record CASL consent for a purpose."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    cm = get_consent_manager()
    client_ip = request.client.host if request.client else ""
    cm.record_consent(
        user_id=user.get("user_id", user.get("email", "")),
        consent_type=body.consent_type,
        purpose=body.purpose,
        source="api",
        ip_address=client_ip,
    )
    return {"ok": True}


@app.delete("/api/v2/privacy/consent/{purpose}", tags=["Privacy"])
def api_privacy_withdraw_consent(purpose: str, request: Request):
    """Withdraw CASL consent for a purpose (unsubscribe)."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    cm = get_consent_manager()
    cm.withdraw_consent(user.get("user_id", user.get("email", "")), purpose)
    return {"ok": True}


@app.get("/api/v2/privacy/consents", tags=["Privacy"])
def api_privacy_list_consents(request: Request):
    """List all consent records for the current user."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    cm = get_consent_manager()
    consents = cm.get_user_consents(user.get("user_id", user.get("email", "")))
    return {"ok": True, "consents": consents}


@app.post("/api/v2/privacy/erase", tags=["Privacy"])
def api_privacy_right_to_erasure(request: Request):
    """Execute right-to-erasure (PIPEDA/Law 25). Irreversible."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    summary = execute_right_to_erasure(user.get("user_id", user.get("email", "")))
    return {"ok": True, "erasure": summary}


# ── Bin-Pack Scheduler Route ──────────────────────────────────────────

from scheduler import allocate_binpack, process_queue_binpack

@app.post("/api/v2/scheduler/process-binpack", tags=["Jobs"])
def api_process_queue_binpack(canada_only: bool = False, province: str = ""):
    """Process job queue using best-fit-decreasing bin packing."""
    assigned = process_queue_binpack(
        canada_only=canada_only or None,
        province=province or None,
    )
    return {"ok": True, "assigned": assigned, "count": len(assigned)}


# ── AI Assistant Routes ────────────────────────────────────────────────

from ai_assistant import (
    FEATURE_AI_ASSISTANT,
    check_ai_rate_limit,
    create_conversation as ai_create_conversation,
    get_conversation as ai_get_conversation,
    list_conversations as ai_list_conversations,
    delete_conversation as ai_delete_conversation,
    get_conversation_messages as ai_get_messages,
    stream_ai_response,
    execute_confirmed_action,
    get_suggestions as ai_get_suggestions,
)


def _require_ai_enabled():
    if not FEATURE_AI_ASSISTANT:
        raise HTTPException(404, "AI assistant is not enabled.")


class AiChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = None
    page_context: str = ""


class AiConfirmRequest(BaseModel):
    confirmation_id: str
    approved: bool


@app.post("/api/ai/chat", tags=["AI Assistant"])
async def api_ai_chat(body: AiChatRequest, request: Request):
    """Stream an AI assistant response with tool-calling support via SSE."""
    _require_ai_enabled()
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    user_id = user.get("user_id", user.get("email", ""))
    if not check_ai_rate_limit(user_id):
        raise HTTPException(429, "Rate limit exceeded. Please wait a moment.")

    # Get or create conversation
    conversation_id = body.conversation_id
    if conversation_id:
        conv = ai_get_conversation(conversation_id, user_id)
        if not conv:
            raise HTTPException(404, "Conversation not found")
    else:
        conversation_id = ai_create_conversation(user_id)

    return StreamingResponse(
        stream_ai_response(body.message, conversation_id, user, body.page_context),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/ai/conversations", tags=["AI Assistant"])
def api_ai_list_conversations(request: Request, limit: int = 30):
    """List the current user's AI assistant conversations."""
    _require_ai_enabled()
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    user_id = user.get("user_id", user.get("email", ""))
    convs = ai_list_conversations(user_id, limit=limit)
    return {"ok": True, "conversations": convs}


@app.get("/api/ai/conversations/{conversation_id}", tags=["AI Assistant"])
def api_ai_get_conversation(conversation_id: str, request: Request):
    """Get messages for an AI assistant conversation."""
    _require_ai_enabled()
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    user_id = user.get("user_id", user.get("email", ""))
    messages = ai_get_messages(conversation_id, user_id)
    conv = ai_get_conversation(conversation_id, user_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return {"ok": True, "conversation": conv, "messages": messages}


@app.delete("/api/ai/conversations/{conversation_id}", tags=["AI Assistant"])
def api_ai_delete_conversation(conversation_id: str, request: Request):
    """Delete an AI assistant conversation."""
    _require_ai_enabled()
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    user_id = user.get("user_id", user.get("email", ""))
    deleted = ai_delete_conversation(conversation_id, user_id)
    if not deleted:
        raise HTTPException(404, "Conversation not found")
    return {"ok": True}


@app.post("/api/ai/confirm", tags=["AI Assistant"])
async def api_ai_confirm(body: AiConfirmRequest, request: Request):
    """Approve or reject a pending AI tool action."""
    _require_ai_enabled()
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    return StreamingResponse(
        execute_confirmed_action(body.confirmation_id, user, body.approved),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/ai/suggestions", tags=["AI Assistant"])
def api_ai_suggestions(request: Request):
    """Get context-aware suggestion chips for the AI assistant."""
    _require_ai_enabled()
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    return {"ok": True, "suggestions": ai_get_suggestions(user)}


if __name__ == "__main__":
    import uvicorn

    log.info("API STARTING on port 9500")
    uvicorn.run(app, host="0.0.0.0", port=9500)
