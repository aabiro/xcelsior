# routes/_deps.py — shared cross-cutting helpers for all route modules
"""Authentication helpers, SSE bus, constants, and rate-limit utilities."""

import asyncio
import hmac
import hashlib as _hashlib
import json
import os
import secrets
import time
import threading as _threading
from collections import defaultdict, deque

from fastapi import HTTPException, Request
from db import UserStore, NotificationStore
from scheduler import list_jobs, API_TOKEN, log


# ── Environment & Auth Config ─────────────────────────────────────────

XCELSIOR_ENV = os.environ.get("XCELSIOR_ENV", "dev").lower()
AUTH_REQUIRED = XCELSIOR_ENV not in {"dev", "development", "test"}
RATE_LIMIT_REQUESTS = int(os.environ.get("XCELSIOR_RATE_LIMIT_REQUESTS", "300"))
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("XCELSIOR_RATE_LIMIT_WINDOW_SEC", "60"))
_RATE_BUCKETS: dict[str, deque] = defaultdict(deque)

_AUTH_RATE_LIMIT_REQUESTS = int(os.environ.get("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "10"))
_AUTH_RATE_LIMIT_WINDOW_SEC = 300
_AUTH_RATE_BUCKETS: dict[str, deque] = defaultdict(deque)

_USE_PERSISTENT_AUTH = os.environ.get("XCELSIOR_PERSISTENT_AUTH", "true").lower() != "false"

SESSION_EXPIRY = 86400 * 30  # 30 days
MAX_SESSION_LIFETIME = 86400 * 30  # 30 days (must match SESSION_EXPIRY)

_AUTH_COOKIE_NAME = "xcelsior_session"

VALID_ACCOUNT_ROLES = {"submitter", "provider"}

PUBLIC_PATHS = {
    "/", "/docs", "/redoc", "/openapi.json", "/llms.txt",
    "/dashboard", "/legacy", "/healthz", "/readyz", "/metrics",
    "/api/stream", "/api/transparency/report",
}
PUBLIC_PATH_PREFIXES = ("/api/auth/", "/api/chat", "/legacy/")
AGENT_RATE_LIMIT_EXEMPT_PREFIXES = ("/host", "/agent/")


# ── In-Memory Auth Stores ─────────────────────────────────────────────

_users_db: dict[str, dict] = {}
_sessions: dict[str, dict] = {}
_api_keys: dict[str, dict] = {}
_user_lock = _threading.Lock()


# ── SSE Infrastructure ───────────────────────────────────────────────

_sse_subscribers: list[asyncio.Queue] = []
_sse_lock = _threading.Lock()


def _sse_message_text(event_type: str, data: dict) -> str:
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
    template = _NOTIF_EVENT_MAP.get(event_type)
    if not template:
        return
    try:
        notif_type, title_tmpl, body_tmpl = template
        title = title_tmpl.format_map(defaultdict(str, **data))
        body = body_tmpl.format_map(defaultdict(str, **data))

        if _USE_PERSISTENT_AUTH:
            if event_type in ("job_submitted", "job_status", "job_completed", "job_failed",
                              "preemption_scheduled"):
                job_id = data.get("job_id", "")
                jobs = list_jobs()
                job = next((j for j in jobs if j.get("job_id") == job_id), None)
                owner_email = job.get("owner_email", job.get("user_email", "")) if job else ""
                if owner_email:
                    user = UserStore.get_user(owner_email)
                    if user and user.get("notifications_enabled", 1):
                        NotificationStore.create(owner_email, notif_type, title, body, data)
                if event_type == "job_failed":
                    for u in UserStore.list_users():
                        if u.get("role") == "admin" and u["email"] != owner_email:
                            if u.get("notifications_enabled", 1):
                                NotificationStore.create(u["email"], notif_type, title, body, data)
            else:
                for u in UserStore.list_users():
                    if u.get("role") == "admin" and u.get("notifications_enabled", 1):
                        NotificationStore.create(u["email"], notif_type, title, body, data)
    except Exception as e:
        log.debug("Notification delivery error: %s", e)


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
    _threading.Thread(target=_deliver_notifications, args=(event_type, data), daemon=True).start()


# ── Auth Helpers ──────────────────────────────────────────────────────

def _get_real_client_ip(request: Request) -> str:
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_auth_rate_limit(request: Request) -> None:
    now = time.time()
    client_ip = _get_real_client_ip(request)
    bucket = _AUTH_RATE_BUCKETS[client_ip]
    while bucket and bucket[0] <= now - _AUTH_RATE_LIMIT_WINDOW_SEC:
        bucket.popleft()
    if len(bucket) >= _AUTH_RATE_LIMIT_REQUESTS:
        raise HTTPException(429, "Too many attempts. Please try again later.")
    bucket.append(now)


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = _hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return hashed, salt


def _admin_flag(value) -> int:
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes", "on"} else 0
    return 1 if value else 0


def _is_platform_admin(user: dict | None) -> bool:
    if not user:
        return False
    return _admin_flag(user.get("is_admin")) == 1 or user.get("role") == "admin"


def _merge_auth_user(base: dict, full_user: dict | None = None) -> dict:
    merged = dict(base or {})
    if full_user:
        merged["role"] = full_user.get("role", merged.get("role", "submitter"))
        merged["name"] = full_user.get("name", merged.get("name", ""))
        merged["customer_id"] = full_user.get("customer_id", merged.get("customer_id"))
        merged["provider_id"] = full_user.get("provider_id", merged.get("provider_id"))
        merged["is_admin"] = True if _is_platform_admin(full_user) else False
    else:
        merged["is_admin"] = True if _is_platform_admin(merged) else False
    return merged


def _set_auth_cookie(response, token: str):
    _base = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca")
    is_prod = _base.startswith("https")
    kwargs: dict = dict(
        key=_AUTH_COOKIE_NAME, value=token, max_age=SESSION_EXPIRY,
        httponly=True, secure=is_prod, samesite="lax", path="/",
    )
    if is_prod:
        kwargs["domain"] = ".xcelsior.ca"
    response.set_cookie(**kwargs)
    return response


def _clear_auth_cookie(response):
    _base = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca")
    kwargs: dict = dict(key=_AUTH_COOKIE_NAME, path="/")
    if _base.startswith("https"):
        kwargs["domain"] = ".xcelsior.ca"
    response.delete_cookie(**kwargs)
    return response


def _create_session(email: str, user: dict, request: Request | None = None) -> dict:
    token = secrets.token_urlsafe(48)
    now = time.time()
    session = {
        "token": token, "email": email,
        "user_id": user.get("user_id", email),
        "role": user.get("role", "submitter"),
        "is_admin": True if _is_platform_admin(user) else False,
        "name": user.get("name", ""),
        "created_at": now, "expires_at": now + SESSION_EXPIRY,
        "ip_address": _get_real_client_ip(request) if request else None,
        "user_agent": request.headers.get("user-agent", "")[:512] if request else None,
        "last_active": now,
    }
    if _USE_PERSISTENT_AUTH:
        UserStore.create_session(session)
    else:
        with _user_lock:
            _sessions[token] = session
    return session


def _get_current_user(request: Request) -> dict | None:
    auth = request.headers.get("authorization", "")
    token = ""
    if auth.startswith("Bearer "):
        token = auth[7:]
    if not token:
        token = request.cookies.get(_AUTH_COOKIE_NAME, "")
    if not token:
        return None
    master = os.environ.get("XCELSIOR_API_TOKEN", API_TOKEN)
    if master and hmac.compare_digest(token, master):
        return {
            "email": "api-token@xcelsior.ca", "user_id": "api-admin",
            "role": "admin", "is_admin": True, "name": "API Token",
        }
    if _USE_PERSISTENT_AUTH:
        session = UserStore.get_session(token)
        if session:
            created = session.get("created_at") or 0
            if time.time() - created > MAX_SESSION_LIFETIME:
                UserStore.delete_session(token)
                return None
            try:
                last = session.get("last_active") or 0
                if time.time() - last > 300:
                    UserStore.update_session_last_active(token)
            except Exception as e:
                log.debug("session last_active update failed: %s", e)
            full_user = UserStore.get_user(session["email"])
            return _merge_auth_user(dict(session), full_user)
        api_key = UserStore.get_api_key(token)
        if api_key:
            full_user = UserStore.get_user(api_key["email"])
            return _merge_auth_user(
                {
                    "email": api_key["email"], "user_id": api_key["user_id"],
                    "role": api_key.get("role", "submitter"),
                    "is_admin": api_key.get("is_admin", False),
                    "name": api_key.get("name", ""),
                    "scope": api_key.get("scope", "full-access"),
                },
                full_user,
            )
    else:
        with _user_lock:
            session = _sessions.get(token)
            full_user = _users_db.get(session["email"]) if session else None
        if session and session["expires_at"] > time.time():
            return _merge_auth_user(session, full_user)
        with _user_lock:
            api_key = _api_keys.get(token)
            full_user = _users_db.get(api_key["email"]) if api_key else None
        if api_key:
            api_key["last_used"] = time.time()
            return _merge_auth_user(
                {
                    "email": api_key["email"], "user_id": api_key["user_id"],
                    "role": api_key.get("role", "submitter"),
                    "is_admin": api_key.get("is_admin", False),
                    "name": api_key.get("name", ""),
                    "scope": api_key.get("scope", "full-access"),
                },
                full_user,
            )
    return None


def _require_auth(request: Request) -> dict:
    if not AUTH_REQUIRED:
        return {"email": "anonymous", "user_id": "anonymous", "role": "admin", "is_admin": True}
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Authentication required")
    return user


def _require_admin(request: Request) -> dict:
    user = _require_auth(request)
    if not _is_platform_admin(user):
        raise HTTPException(403, "Admin access required")
    return user


def _require_provider_or_admin(request: Request) -> dict:
    user = _require_auth(request)
    if user.get("role") != "provider" and not _is_platform_admin(user):
        raise HTTPException(403, "Provider or admin access required")
    return user


def _require_write_access(request: Request):
    user = _get_current_user(request)
    if (
        user
        and user.get("scope") == "read-only"
        and request.method not in ("GET", "HEAD", "OPTIONS")
    ):
        raise HTTPException(403, "This API key has read-only scope")
    return user


# ── OpenTelemetry Span Helper ─────────────────────────────────────────
# _otel_tracer is set by api.py after OTEL setup completes.
# Route modules import otel_span from here to avoid circular imports.

_otel_tracer = None  # populated by api.py


def otel_span(name: str, attributes: dict | None = None):
    """Create a custom OpenTelemetry span (context manager)."""
    if _otel_tracer is None:
        from contextlib import nullcontext
        return nullcontext()
    span = _otel_tracer.start_as_current_span(name, attributes=attributes or {})
    return span
