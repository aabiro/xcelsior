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
import urllib.parse
from collections import defaultdict, deque

from fastapi import HTTPException, Request, WebSocket
from db import DatabaseOps, UserStore, NotificationStore, get_engine
from oauth_service import (
    ACCESS_TOKEN_TTL_SEC,
    AuthCacheUnavailableError,
    REFRESH_COOKIE_NAME,
    REFRESH_TOKEN_TTL_SEC,
    build_deprecation_headers,
    is_oauth_access_token,
    resolve_opaque_access_token,
    validate_client_credentials_jwt,
)
from scheduler import list_jobs, API_TOKEN, log

try:
    from prometheus_client import Counter as _PromCounter

    _deprecated_api_key_requests = _PromCounter(
        "xcelsior_deprecated_api_key_requests_total",
        "Requests authenticated with deprecated API keys",
        ["email"],
    )
except Exception:
    class _NoopCounter:
        def labels(self, *a, **kw):
            return self
        def inc(self, *a, **kw):
            pass
    _deprecated_api_key_requests = _NoopCounter()


# ── Environment & Auth Config ─────────────────────────────────────────

XCELSIOR_ENV = os.environ.get("XCELSIOR_ENV", "dev").lower()
AUTH_REQUIRED = XCELSIOR_ENV not in {"dev", "development", "test"}
RATE_LIMIT_REQUESTS = int(os.environ.get("XCELSIOR_RATE_LIMIT_REQUESTS", "300"))
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("XCELSIOR_RATE_LIMIT_WINDOW_SEC", "60"))
_RATE_BUCKETS: dict[str, deque] = defaultdict(deque)

_AUTH_RATE_LIMIT_REQUESTS = int(os.environ.get("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "10"))
_AUTH_RATE_LIMIT_WINDOW_SEC = 300
_AUTH_RATE_BUCKETS: dict[str, deque] = defaultdict(deque)

_WS_CONNECT_RATE_LIMIT_REQUESTS = int(
    os.environ.get("XCELSIOR_WS_CONNECT_RATE_LIMIT_REQUESTS", "60")
)
_WS_CONNECT_RATE_LIMIT_WINDOW_SEC = int(
    os.environ.get("XCELSIOR_WS_CONNECT_RATE_LIMIT_WINDOW_SEC", "60")
)
_USE_SHARED_RUNTIME_LIMITS = os.environ.get(
    "XCELSIOR_SHARED_RUNTIME_LIMITS",
    "true",
).lower() not in {"0", "false", "no", "off"}
_WS_CONNECT_STATE_NAMESPACE = os.environ.get(
    "XCELSIOR_WS_CONNECT_STATE_NAMESPACE",
    "runtime.ws_connect_rate_limit",
)
_WS_TICKET_STATE_NAMESPACE = os.environ.get(
    "XCELSIOR_WS_TICKET_STATE_NAMESPACE",
    "runtime.ws_tickets",
)
_WS_CONNECT_BUCKETS: dict[str, deque] = defaultdict(deque)
_WS_CONNECT_LOCK = _threading.Lock()
_WS_TICKET_TTL_SEC = int(os.environ.get("XCELSIOR_WS_TICKET_TTL_SEC", "60"))
_WS_TICKETS: dict[str, dict] = {}
_WS_TICKET_LOCK = _threading.Lock()

_USE_PERSISTENT_AUTH = os.environ.get("XCELSIOR_PERSISTENT_AUTH", "true").lower() != "false"

SESSION_EXPIRY = 86400 * 30  # 30 days
MAX_SESSION_LIFETIME = 86400 * 30  # 30 days (must match SESSION_EXPIRY)

_AUTH_COOKIE_NAME = "xcelsior_session"

VALID_ACCOUNT_ROLES = {"submitter", "provider"}

PUBLIC_PATHS = {
    "/", "/docs", "/redoc", "/openapi.json", "/llms.txt",
    "/dashboard", "/legacy", "/healthz", "/readyz", "/metrics",
    "/api/stream", "/api/transparency/report",
    "/api/providers/webhook",
    "/.well-known/oauth-authorization-server",
}
PUBLIC_PATH_PREFIXES = ("/api/auth/", "/api/chat", "/legacy/", "/oauth/")
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
    "user_registered": {
        "notif_type": "system",
        "title": "New User Registered",
        "body": "{email} has joined the platform.",
    },
    "job_submitted": {
        "notif_type": "instance",
        "title": "Instance Submitted",
        "body": "Your instance {name} has been submitted.",
        "action_url": "/dashboard/instances/{job_id}",
        "entity_type": "job",
        "entity_id": "{job_id}",
    },
    "job_status": {
        "notif_type": "instance",
        "title": "Instance {status}",
        "body": "Instance {job_id} is now {status}.",
        "action_url": "/dashboard/instances/{job_id}",
        "entity_type": "job",
        "entity_id": "{job_id}",
    },
    "host_registered": {
        "notif_type": "host",
        "title": "Host Registered",
        "body": "A new host has been registered.",
        "action_url": "/dashboard/hosts/{host_id}",
        "entity_type": "host",
        "entity_id": "{host_id}",
    },
    "host_removed": {
        "notif_type": "host",
        "title": "Host Removed",
        "body": "Host {host_id} has been removed.",
        "action_url": "/dashboard/hosts/{host_id}",
        "entity_type": "host",
        "entity_id": "{host_id}",
        "priority": 1,
    },
    "job_completed": {
        "notif_type": "instance",
        "title": "Instance Completed",
        "body": "Instance {job_id} completed successfully.",
        "action_url": "/dashboard/instances/{job_id}",
        "entity_type": "job",
        "entity_id": "{job_id}",
    },
    "job_failed": {
        "notif_type": "instance",
        "title": "Instance Failed",
        "body": "Instance {job_id} has failed.",
        "action_url": "/dashboard/instances/{job_id}",
        "entity_type": "job",
        "entity_id": "{job_id}",
        "priority": 1,
    },
    "preemption_scheduled": {
        "notif_type": "instance",
        "title": "Preemption Scheduled",
        "body": "Instance {job_id} is being preempted.",
        "action_url": "/dashboard/instances/{job_id}",
        "entity_type": "job",
        "entity_id": "{job_id}",
        "priority": 1,
    },
}


def _format_notif_value(template: str, data: dict) -> str:
    return template.format_map(defaultdict(str, **data))


def _build_notification_payload(event_type: str, data: dict) -> dict | None:
    config = _NOTIF_EVENT_MAP.get(event_type)
    if not config:
        return None

    notif_data = dict(data)
    notif_data.setdefault("source_event", event_type)

    return {
        "notif_type": config["notif_type"],
        "title": _format_notif_value(config["title"], data),
        "body": _format_notif_value(config["body"], data),
        "data": notif_data,
        "action_url": _format_notif_value(config.get("action_url", ""), data),
        "entity_type": _format_notif_value(config.get("entity_type", ""), data),
        "entity_id": _format_notif_value(config.get("entity_id", ""), data),
        "priority": int(config.get("priority", 0)),
    }


def _deliver_notifications(event_type: str, data: dict):
    notification = _build_notification_payload(event_type, data)
    if not notification:
        return
    try:
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
                        NotificationStore.create(owner_email, **notification)
                if event_type == "job_failed":
                    for u in UserStore.list_users():
                        if u.get("role") == "admin" and u["email"] != owner_email:
                            if u.get("notifications_enabled", 1):
                                NotificationStore.create(u["email"], **notification)
            else:
                for u in UserStore.list_users():
                    if u.get("role") == "admin" and u.get("notifications_enabled", 1):
                        NotificationStore.create(u["email"], **notification)
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
    # Check cf-connecting-ip first so the resolved IP matches
    # _get_ws_client_ip() — both functions must agree when
    # a ticket is pinned during HTTP issuance and consumed on WS.
    for header in ("cf-connecting-ip", "x-real-ip", "x-forwarded-for"):
        val = request.headers.get(header, "")
        if val:
            return val.split(",")[0].strip()
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
        merged["mfa_enabled"] = bool(full_user.get("mfa_enabled"))
        merged["email_verified"] = bool(full_user.get("email_verified", merged.get("email_verified")))
        merged["is_admin"] = True if _is_platform_admin(full_user) else False
    else:
        merged["is_admin"] = True if _is_platform_admin(merged) else False
    return merged


def _set_auth_cookie(response, token: str, *, max_age: int = SESSION_EXPIRY):
    _base = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca")
    is_prod = _base.startswith("https")
    kwargs: dict = dict(
        key=_AUTH_COOKIE_NAME, value=token, max_age=max_age,
        httponly=True, secure=is_prod, samesite="lax", path="/",
    )
    if is_prod:
        kwargs["domain"] = ".xcelsior.ca"
    response.set_cookie(**kwargs)
    return response


def _set_refresh_cookie(response, token: str, *, max_age: int = REFRESH_TOKEN_TTL_SEC):
    _base = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca")
    is_prod = _base.startswith("https")
    kwargs: dict = dict(
        key=REFRESH_COOKIE_NAME, value=token, max_age=max_age,
        httponly=True, secure=is_prod, samesite="lax", path="/",
    )
    if is_prod:
        kwargs["domain"] = ".xcelsior.ca"
    response.set_cookie(**kwargs)
    return response


def _set_session_cookies(response, token_bundle: dict):
    _set_auth_cookie(
        response,
        str(token_bundle.get("access_token", "")),
        max_age=int(token_bundle.get("expires_in", ACCESS_TOKEN_TTL_SEC) or ACCESS_TOKEN_TTL_SEC),
    )
    refresh_token = str(token_bundle.get("refresh_token", "")).strip()
    if refresh_token:
        _set_refresh_cookie(
            response,
            refresh_token,
            max_age=int(
                token_bundle.get("refresh_expires_in", REFRESH_TOKEN_TTL_SEC)
                or REFRESH_TOKEN_TTL_SEC
            ),
        )
    return response


def _clear_auth_cookie(response):
    _base = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca")
    kwargs: dict = dict(key=_AUTH_COOKIE_NAME, path="/")
    if _base.startswith("https"):
        kwargs["domain"] = ".xcelsior.ca"
    response.delete_cookie(**kwargs)
    refresh_kwargs: dict = dict(key=REFRESH_COOKIE_NAME, path="/")
    if _base.startswith("https"):
        refresh_kwargs["domain"] = ".xcelsior.ca"
    response.delete_cookie(**refresh_kwargs)
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
            "auth_type": "master_token",
        }

    if is_oauth_access_token(token):
        try:
            oauth_user = resolve_opaque_access_token(token)
        except AuthCacheUnavailableError as exc:
            log.error("oauth.auth_cache.unavailable token_lookup_failed: %s", exc)
            raise HTTPException(503, "Dedicated auth cache unavailable") from exc
        if oauth_user:
            full_user = UserStore.get_user(oauth_user["email"]) if oauth_user.get("email") else None
            merged = _merge_auth_user(dict(oauth_user), full_user)
            merged["auth_type"] = "oauth_access_token"
            merged["scopes"] = list(oauth_user.get("scopes") or [])
            merged["client_id"] = oauth_user.get("client_id")
            merged["session_token"] = oauth_user.get("session_token")
            merged["session_type"] = oauth_user.get("session_type", "browser")
            return merged

    machine_principal = validate_client_credentials_jwt(token)
    if machine_principal:
        return machine_principal

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
            merged = _merge_auth_user(dict(session), full_user)
            merged["auth_type"] = "legacy_session"
            return merged
        api_key = UserStore.get_api_key(token)
        if api_key:
            full_user = UserStore.get_user(api_key["email"])
            merged = _merge_auth_user(
                {
                    "email": api_key["email"], "user_id": api_key["user_id"],
                    "role": api_key.get("role", "submitter"),
                    "is_admin": api_key.get("is_admin", False),
                    "name": api_key.get("name", ""),
                    "scope": api_key.get("scope", "full-access"),
                },
                full_user,
            )
            merged["auth_type"] = "api_key"
            merged["deprecation_headers"] = build_deprecation_headers("API keys")
            _deprecated_api_key_requests.labels(api_key["email"]).inc()
            log.warning(
                "deprecated.api_key.used email=%s key_name=%s",
                api_key["email"],
                api_key.get("name", "unknown"),
            )
            return merged
    else:
        with _user_lock:
            session = _sessions.get(token)
            full_user = _users_db.get(session["email"]) if session else None
        if session and session["expires_at"] > time.time():
            merged = _merge_auth_user(session, full_user)
            merged["auth_type"] = "legacy_session"
            return merged
        with _user_lock:
            api_key = _api_keys.get(token)
            full_user = _users_db.get(api_key["email"]) if api_key else None
        if api_key:
            api_key["last_used"] = time.time()
            merged = _merge_auth_user(
                {
                    "email": api_key["email"], "user_id": api_key["user_id"],
                    "role": api_key.get("role", "submitter"),
                    "is_admin": api_key.get("is_admin", False),
                    "name": api_key.get("name", ""),
                    "scope": api_key.get("scope", "full-access"),
                },
                full_user,
            )
            merged["auth_type"] = "api_key"
            merged["deprecation_headers"] = build_deprecation_headers("API keys")
            _deprecated_api_key_requests.labels(api_key["email"]).inc()
            log.warning(
                "deprecated.api_key.used email=%s key_name=%s",
                api_key["email"],
                api_key.get("name", "unknown"),
            )
            return merged
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


def _require_user_grant(request: Request, *, allow_api_key: bool = False) -> dict:
    """Require an interactive user session.

    Rejects client_credentials (machine) tokens outright and rejects
    API-key tokens unless *allow_api_key* is True.  This should guard
    every endpoint that mutates user-owned state: MFA, password, profile,
    sessions, preferences, privacy/consent, and account deletion.
    """
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    auth_type = str(user.get("auth_type", ""))
    if auth_type == "client_credentials":
        raise HTTPException(403, "Interactive user authentication required")
    if auth_type == "api_key" and not allow_api_key:
        raise HTTPException(403, "Interactive session authentication required")
    return user


def _require_scope(user: dict, *required: str) -> None:
    """Raise 403 if *user* is a scoped machine principal missing any of *required*.

    No-op for interactive user sessions (browser, authorization_code),
    legacy sessions, admin tokens, and principals without explicit scopes.
    Only OAuth client_credentials JWTs and machine tokens are scope-checked.
    """
    # Interactive users always pass — scopes only gate machine-to-machine access.
    grant_type = user.get("grant_type", "")
    if grant_type != "client_credentials":
        return  # browser/user session — implicit full access

    scopes = user.get("scopes")
    if scopes is None:
        return
    granted = set(scopes)
    missing = [s for s in required if s not in granted]
    if missing:
        raise HTTPException(
            403,
            f"Insufficient scope — required: {', '.join(required)}; "
            f"granted: {', '.join(sorted(granted))}",
        )


def _require_write_access(request: Request):
    user = _get_current_user(request)
    if (
        user
        and user.get("scope") == "read-only"
        and request.method not in ("GET", "HEAD", "OPTIONS")
    ):
        raise HTTPException(403, "This API key has read-only scope")
    return user


# ── WebSocket Auth ────────────────────────────────────────────────────

def _get_ws_client_ip(websocket: WebSocket) -> str:
    """Best-effort client IP extraction for WebSocket rate limiting."""
    for header in ("cf-connecting-ip", "x-forwarded-for", "x-real-ip"):
        raw = websocket.headers.get(header, "")
        if raw:
            return raw.split(",")[0].strip() or "unknown"
    client = getattr(websocket, "client", None)
    if client and getattr(client, "host", None):
        return client.host
    return "unknown"


def _lock_shared_state(
    conn,
    namespace: str,
    default_state: dict,
    *,
    backend: str,
) -> dict:
    """Load a shared state namespace under transaction/row lock."""
    if backend == "postgres":
        from psycopg.types.json import Jsonb

        conn.execute(
            """
            INSERT INTO state(namespace, payload) VALUES (%s, %s)
            ON CONFLICT(namespace) DO NOTHING
            """,
            (namespace, Jsonb(default_state)),
        )
        row = conn.execute(
            "SELECT payload FROM state WHERE namespace = %s FOR UPDATE",
            (namespace,),
        ).fetchone()
    else:
        conn.execute(
            "INSERT OR IGNORE INTO state(namespace, payload) VALUES (?, ?)",
            (namespace, json.dumps(default_state)),
        )
        row = conn.execute(
            "SELECT payload FROM state WHERE namespace = ?",
            (namespace,),
        ).fetchone()
    payload = row["payload"] if isinstance(row, dict) else row[0]
    return DatabaseOps.decode_payload(payload) or {}


def _shared_state_update(namespace: str, default_factory, mutator) -> tuple[bool, object]:
    """Atomically mutate shared runtime state, mirrored in dual-write mode."""
    if not _USE_SHARED_RUNTIME_LIMITS:
        return False, None

    engine = get_engine()
    try:
        with engine.transaction() as (conn, backend):
            state = _lock_shared_state(
                conn,
                namespace,
                default_factory(),
                backend=backend,
            )
            new_state, result = mutator(state)
            DatabaseOps.upsert_state(conn, namespace, new_state, backend=backend)
        engine.mirror_to_secondary(DatabaseOps.upsert_state, namespace, new_state)
        return True, result
    except Exception as exc:
        log.warning("shared.state.update_failed namespace=%s error=%s", namespace, exc)
        return False, None


def _purge_ws_tickets_locked(now: float | None = None) -> None:
    now = time.time() if now is None else now
    expired = [
        token
        for token, payload in _WS_TICKETS.items()
        if payload.get("expires_at", 0) <= now
    ]
    for token in expired:
        _WS_TICKETS.pop(token, None)


def _purge_shared_ws_tickets(
    tickets: dict[str, dict] | None,
    now: float,
) -> dict[str, dict]:
    pruned: dict[str, dict] = {}
    for token, payload in (tickets or {}).items():
        expires_at = float((payload or {}).get("expires_at", 0) or 0)
        if expires_at > now:
            pruned[token] = dict(payload or {})
    return pruned


def _issue_ws_ticket(
    user: dict,
    *,
    request: Request | None = None,
    purpose: str = "websocket",
    target: str = "",
    ttl_sec: int | None = None,
    client_ip: str | None = None,
) -> dict:
    """Create a short-lived one-time WebSocket ticket."""
    now = time.time()
    ticket = secrets.token_urlsafe(32)
    payload = {
        "ticket": ticket,
        "purpose": purpose,
        "target": target,
        "user": dict(user),
        "ip_address": client_ip or (_get_real_client_ip(request) if request else "unknown"),
        "expires_at": now + max(1, ttl_sec or _WS_TICKET_TTL_SEC),
    }
    shared_ok, shared_payload = _shared_state_update(
        _WS_TICKET_STATE_NAMESPACE,
        lambda: {"tickets": {}},
        lambda state: _mutate_shared_ws_tickets_issue(state, payload, now),
    )
    if shared_ok:
        return dict(shared_payload)
    with _WS_TICKET_LOCK:
        _purge_ws_tickets_locked(now)
        _WS_TICKETS[ticket] = payload
    return payload


def _consume_ws_ticket(
    ticket: str,
    websocket: WebSocket,
    *,
    purpose: str = "websocket",
    target: str = "",
) -> dict | None:
    """Consume a previously issued WebSocket ticket exactly once."""
    now = time.time()
    ws_ip = _get_ws_client_ip(websocket)
    shared_ok, shared_user = _shared_state_update(
        _WS_TICKET_STATE_NAMESPACE,
        lambda: {"tickets": {}},
        lambda state: _mutate_shared_ws_tickets_consume(
            state,
            ticket,
            purpose=purpose,
            target=target,
            ws_ip=ws_ip,
            now=now,
        ),
    )
    if shared_ok:
        return dict(shared_user or {}) or None
    with _WS_TICKET_LOCK:
        _purge_ws_tickets_locked(now)
        payload = _WS_TICKETS.get(ticket)
        if not payload:
            return None
        if payload.get("purpose") != purpose:
            return None
        if target and payload.get("target") not in ("", target):
            return None
        pinned_ip = payload.get("ip_address", "unknown")
        if pinned_ip not in ("", "unknown", ws_ip):
            return None
        _WS_TICKETS.pop(ticket, None)
        return dict(payload.get("user") or {})


def _mutate_shared_ws_tickets_issue(
    state: dict,
    payload: dict,
    now: float,
) -> tuple[dict, dict]:
    tickets = _purge_shared_ws_tickets(state.get("tickets"), now)
    tickets[payload["ticket"]] = dict(payload)
    return {"tickets": tickets, "updated_at": now}, payload


def _mutate_shared_ws_tickets_consume(
    state: dict,
    ticket: str,
    *,
    purpose: str,
    target: str,
    ws_ip: str,
    now: float,
) -> tuple[dict, dict | None]:
    tickets = _purge_shared_ws_tickets(state.get("tickets"), now)
    payload = tickets.get(ticket)
    if not payload:
        return {"tickets": tickets, "updated_at": now}, None
    if payload.get("purpose") != purpose:
        return {"tickets": tickets, "updated_at": now}, None
    if target and payload.get("target") not in ("", target):
        return {"tickets": tickets, "updated_at": now}, None
    pinned_ip = payload.get("ip_address", "unknown")
    if pinned_ip not in ("", "unknown", ws_ip):
        return {"tickets": tickets, "updated_at": now}, None
    tickets.pop(ticket, None)
    return {"tickets": tickets, "updated_at": now}, dict(payload.get("user") or {})


def _ws_has_explicit_query_auth(
    websocket: WebSocket,
    *,
    allow_query_token: bool = True,
) -> bool:
    return bool(
        websocket.query_params.get("ticket", "").strip()
        or (
            allow_query_token
            and websocket.query_params.get("token", "").strip()
        )
    )


def _get_ws_allowed_origin_hosts() -> set[str]:
    origins = [
        origin.strip()
        for origin in os.environ.get(
            "XCELSIOR_CORS_ORIGINS",
            "https://xcelsior.ca,https://www.xcelsior.ca",
        ).split(",")
        if origin.strip()
    ]
    return {
        urllib.parse.urlparse(origin).netloc
        for origin in origins
        if urllib.parse.urlparse(origin).netloc
    }


def _validate_ws_origin(
    websocket: WebSocket,
    *,
    require_for_cookie_auth: bool = False,
    allow_query_token: bool = True,
) -> bool:
    """Allow only configured browser origins when required by the auth mode."""
    origin = websocket.headers.get("origin", "").strip()
    if not origin:
        if not require_for_cookie_auth:
            return True
        cookie_auth = bool(websocket.cookies.get(_AUTH_COOKIE_NAME, "").strip())
        return not (
            cookie_auth
            and not _ws_has_explicit_query_auth(
                websocket,
                allow_query_token=allow_query_token,
            )
        )
    origin_host = urllib.parse.urlparse(origin).netloc
    if not origin_host:
        return False
    return origin_host in _get_ws_allowed_origin_hosts()


def _check_ws_connect_rate_limit(websocket: WebSocket, *, bucket: str = "default") -> bool:
    """Simple in-memory connection throttling per IP and WebSocket bucket."""
    if _WS_CONNECT_RATE_LIMIT_REQUESTS <= 0:
        return True

    key = f"{bucket}:{_get_ws_client_ip(websocket)}"
    now = time.time()

    shared_ok, shared_allowed = _shared_state_update(
        _WS_CONNECT_STATE_NAMESPACE,
        lambda: {"buckets": {}},
        lambda state: _mutate_shared_ws_connect_buckets(state, key, now),
    )
    if shared_ok:
        return bool(shared_allowed)

    with _WS_CONNECT_LOCK:
        history = _WS_CONNECT_BUCKETS[key]
        while history and history[0] <= now - _WS_CONNECT_RATE_LIMIT_WINDOW_SEC:
            history.popleft()
        if len(history) >= _WS_CONNECT_RATE_LIMIT_REQUESTS:
            return False
        history.append(now)
    return True


def _mutate_shared_ws_connect_buckets(state: dict, key: str, now: float) -> tuple[dict, bool]:
    """Prune + update shared per-IP connection buckets."""
    cutoff = now - _WS_CONNECT_RATE_LIMIT_WINDOW_SEC
    buckets = state.get("buckets", {}) or {}
    pruned: dict[str, list[float]] = {}
    limit = max(1, _WS_CONNECT_RATE_LIMIT_REQUESTS)

    for bucket_key, entries in buckets.items():
        cleaned = [
            float(ts)
            for ts in (entries or [])
            if float(ts) > cutoff
        ]
        if cleaned:
            pruned[bucket_key] = cleaned[-limit:]

    history = pruned.get(key, [])
    if len(history) >= _WS_CONNECT_RATE_LIMIT_REQUESTS:
        return {"buckets": pruned, "updated_at": now}, False

    history.append(now)
    pruned[key] = history[-limit:]
    return {"buckets": pruned, "updated_at": now}, True


def _validate_ws_auth(
    websocket: WebSocket,
    *,
    allow_query_token: bool = True,
) -> dict | None:
    """Validate auth for WebSocket connections (mirrors TokenAuthMiddleware).

    Checks (in order):
    1. Cookie ``xcelsior_session``
    2. Query param ``?token=``
    3. Master API token (constant-time compare)
    4. Persistent session / API key lookup

    Returns the user dict on success, ``None`` on failure.
    """
    if not AUTH_REQUIRED:
        return {"email": "anonymous", "user_id": "anonymous", "role": "admin", "is_admin": True}
    api_token = os.environ.get("XCELSIOR_API_TOKEN", API_TOKEN)
    token: str = websocket.cookies.get(_AUTH_COOKIE_NAME, "")
    if not token and allow_query_token:
        token = websocket.query_params.get("token", "")
    if not token:
        return None
    if api_token and hmac.compare_digest(token, api_token):
        return {"email": "api-token", "user_id": "api-token", "role": "admin", "is_admin": True}
    if _USE_PERSISTENT_AUTH:
        session = UserStore.get_session(token)
        if session:
            full_user = UserStore.get_user(session["email"])
            return _merge_auth_user(dict(session), full_user)
        api_key = UserStore.get_api_key(token)
        if api_key:
            return {
                "email": api_key["email"],
                "user_id": api_key["user_id"],
                "role": api_key.get("role", "submitter"),
                "is_admin": api_key.get("is_admin", False),
            }
    else:
        with _user_lock:
            if token in _sessions and _sessions[token]["expires_at"] > time.time():
                return _sessions[token]
            if token in _api_keys:
                return _api_keys[token]
    return None


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
