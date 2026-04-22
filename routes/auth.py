"""Routes: auth."""

import base64
import hmac
import os
import re
import secrets
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from routes._deps import (
    AUTH_REQUIRED,
    MAX_SESSION_LIFETIME,
    SESSION_EXPIRY,
    VALID_ACCOUNT_ROLES,
    XCELSIOR_ENV,
    _AUTH_COOKIE_NAME,
    _USE_PERSISTENT_AUTH,
    _admin_flag,
    _api_keys,
    _check_auth_rate_limit as _deps_check_auth_rate_limit,
    _clear_auth_cookie as _deps_clear_auth_cookie,
    _create_session as _deps_create_session,
    _get_current_user as _deps_get_current_user,
    _get_real_client_ip as _deps_get_real_client_ip,
    _hash_password as _deps_hash_password,
    _is_platform_admin as _deps_is_platform_admin,
    _merge_auth_user as _deps_merge_auth_user,
    _require_admin as _deps_require_admin,
    _require_auth as _deps_require_auth,
    _require_user_grant as _deps_require_user_grant,
    _require_write_access as _deps_require_write_access,
    _sessions,
    _set_auth_cookie as _deps_set_auth_cookie,
    _set_session_cookies,
    _user_lock,
    _users_db,
    broadcast_sse,
    log,
)
from scheduler import (
    API_TOKEN,
    alert,
    list_jobs,
    log,
)
from db import MfaStore, NotificationStore, UserStore
from billing import get_billing_engine
from reputation import get_reputation_engine
import hashlib as _hashlib
import httpx as _httpx
import urllib.parse as _urllib_parse
from routes.teams import _send_team_email
from oauth_service import (
    ACCESS_TOKEN_TTL_SEC,
    DEVICE_CODE_INTERVAL_SEC,
    REFRESH_COOKIE_NAME,
    REFRESH_TOKEN_TTL_SEC,
    OAuthGrantError,
    approve_device_code,
    authenticate_client,
    build_deprecation_headers,
    create_oauth_client,
    exchange_authorization_code,
    exchange_device_code,
    get_client,
    hash_secret,
    issue_authorization_code,
    issue_client_credentials_jwt,
    issue_device_authorization,
    issue_user_tokens,
    revoke_refresh_session,
    rotate_refresh_token,
)

router = APIRouter()

# OAuth configuration
VALID_KEY_SCOPES = {"full-access", "read-only"}
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
_oauth_states: dict[str, dict] = {}
_OAUTH_STATE_TTL = 600


# ── Helper: _set_auth_cookie ──


def _set_auth_cookie(response, token: str):
    """Add httpOnly session cookie to response."""
    return _deps_set_auth_cookie(response, token)


# ── Helper: _clear_auth_cookie ──


def _clear_auth_cookie(response):
    """Remove session cookie."""
    return _deps_clear_auth_cookie(response)


# ── Helper: _hash_password ──


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Hash a password with PBKDF2-HMAC-SHA256."""
    return _deps_hash_password(password, salt)


# ── Helper: _is_platform_admin ──


def _is_platform_admin(user: dict | None) -> bool:
    """Platform admin privilege is independent from the account role."""
    return _deps_is_platform_admin(user)


# ── Helper: _merge_auth_user ──


def _merge_auth_user(base: dict, full_user: dict | None = None) -> dict:
    """Overlay live user/account data onto a session or API-key auth payload."""
    return _deps_merge_auth_user(base, full_user)


# ── Helper: _create_session ──


def _create_session(email: str, user: dict, request: Request | None = None) -> dict:
    """Create a session token for a user."""
    return _deps_create_session(email, user, request)


# ── Helper: _get_current_user ──


def _get_current_user(request: Request) -> dict | None:
    """Extract user from Authorization header or session cookie."""
    return _deps_get_current_user(request)


# ── Helper: _require_auth ──


def _require_auth(request: Request) -> dict:
    """Return the current user or raise 401."""
    return _deps_require_auth(request)


# ── Helper: _require_admin ──


def _require_admin(request: Request) -> dict:
    """Return the current user or raise 403 if they lack platform admin access."""
    return _deps_require_admin(request)


# ── Helper: _require_write_access ──


def _require_write_access(request: Request):
    """Raise 403 if the current user is using a read-only API key on a mutating request."""
    return _deps_require_write_access(request)


def _check_auth_rate_limit(request: Request) -> None:
    _deps_check_auth_rate_limit(request)


def _require_user_grant(request: Request, *, allow_api_key: bool = False) -> dict:
    return _deps_require_user_grant(request, allow_api_key=allow_api_key)


def _build_auth_response(
    token_bundle: dict,
    *,
    user: dict | None = None,
    status_code: int = 200,
    extra: dict | None = None,
    deprecated_surface: str | None = None,
) -> JSONResponse:
    body = {
        "ok": True,
        "access_token": token_bundle["access_token"],
        "token_type": token_bundle.get("token_type", "Bearer"),
        "expires_in": token_bundle.get("expires_in", ACCESS_TOKEN_TTL_SEC),
    }
    if user is not None:
        body["user"] = {
            "user_id": user.get("user_id", ""),
            "email": user.get("email", ""),
            "name": user.get("name", ""),
            "role": user.get("role", "submitter"),
            "is_admin": True if _is_platform_admin(user) else False,
            "customer_id": user.get("customer_id"),
            "provider_id": user.get("provider_id"),
        }
    if extra:
        body.update(extra)
    resp = JSONResponse(content=body, status_code=status_code)
    _set_session_cookies(resp, token_bundle)
    if deprecated_surface:
        for key, value in build_deprecation_headers(deprecated_surface).items():
            resp.headers[key] = value
    return resp


def _oauth_error_response(
    exc: OAuthGrantError, *, deprecated_surface: str | None = None
) -> JSONResponse:
    resp = JSONResponse(status_code=exc.status_code, content=exc.payload())
    for key, value in exc.headers.items():
        resp.headers[key] = value
    if deprecated_surface:
        for key, value in build_deprecation_headers(deprecated_surface).items():
            resp.headers[key] = value
    return resp


async def _parse_request_body(request: Request) -> dict:
    content_type = request.headers.get("content-type", "").lower()
    if "application/x-www-form-urlencoded" in content_type:
        raw = (await request.body()).decode()
        return {key: value for key, value in _urllib_parse.parse_qsl(raw, keep_blank_values=True)}
    try:
        body = await request.json()
    except Exception:
        body = {}
    return body if isinstance(body, dict) else {}


def _extract_client_credentials(request: Request, body: dict) -> tuple[str, str | None]:
    auth = request.headers.get("authorization", "")
    if auth.startswith("Basic "):
        try:
            raw = base64.b64decode(auth[6:]).decode()
            client_id, client_secret = raw.split(":", 1)
            return client_id, client_secret
        except Exception:
            raise OAuthGrantError("invalid_client", "Invalid client authorization", status_code=401)
    client_id = str(body.get("client_id", "")).strip()
    client_secret = str(body.get("client_secret", "")).strip() or None
    return client_id, client_secret


def _current_refresh_token(request: Request, user: dict | None = None) -> str:
    refresh_token = request.cookies.get(REFRESH_COOKIE_NAME, "").strip()
    if refresh_token:
        return refresh_token
    if user and str(user.get("session_token", "")).strip():
        return str(user.get("session_token", "")).strip()
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return ""


def _validate_oauth_client_redirect(client: dict, redirect_uri: str) -> None:
    allowed = list(client.get("redirect_uris") or [])
    if allowed and redirect_uri not in allowed:
        raise OAuthGrantError("invalid_request", "redirect_uri is not registered for this client")


async def _oauth_token_grant_response(
    request: Request,
    body: dict,
    *,
    deprecated_surface: str | None = None,
) -> JSONResponse:
    try:
        grant_type = str(body.get("grant_type", "")).strip()
        client_id, client_secret = _extract_client_credentials(request, body)
        client = authenticate_client(client_id, client_secret)

        if grant_type == "authorization_code":
            if "authorization_code" not in list(client.get("grant_types") or []):
                raise OAuthGrantError(
                    "unauthorized_client", "Client is not allowed to use authorization_code"
                )
            redirect_uri = str(body.get("redirect_uri", "")).strip()
            code = str(body.get("code", "")).strip()
            code_verifier = str(body.get("code_verifier", "")).strip()
            if not redirect_uri or not code or not code_verifier:
                raise OAuthGrantError(
                    "invalid_request", "code, redirect_uri, and code_verifier are required"
                )
            token_bundle = exchange_authorization_code(
                client=client,
                code=code,
                redirect_uri=redirect_uri,
                code_verifier=code_verifier,
                request=request,
            )
        elif grant_type == "refresh_token":
            refresh_token = str(body.get("refresh_token", "")).strip()
            if not refresh_token:
                raise OAuthGrantError("invalid_request", "refresh_token is required")
            token_bundle = rotate_refresh_token(refresh_token, request)
        elif grant_type == "urn:ietf:params:oauth:grant-type:device_code":
            if grant_type not in list(client.get("grant_types") or []):
                raise OAuthGrantError(
                    "unauthorized_client", "Client is not allowed to use device_code"
                )
            device_code = str(body.get("device_code", "")).strip()
            if not device_code:
                raise OAuthGrantError("invalid_request", "device_code is required")
            token_bundle = exchange_device_code(
                client=client, device_code=device_code, request=request
            )
        elif grant_type == "client_credentials":
            if client.get("client_type") != "confidential":
                raise OAuthGrantError(
                    "unauthorized_client", "Public clients cannot use client_credentials"
                )
            if grant_type not in list(client.get("grant_types") or []):
                raise OAuthGrantError(
                    "unauthorized_client", "Client is not allowed to use client_credentials"
                )
            scopes = str(body.get("scope", "")).strip()
            token_bundle = issue_client_credentials_jwt(
                client,
                [scope for scope in scopes.split() if scope] or list(client.get("scopes") or []),
            )
        else:
            raise OAuthGrantError("unsupported_grant_type", "Unsupported grant_type")
    except OAuthGrantError as exc:
        return _oauth_error_response(exc, deprecated_surface=deprecated_surface)

    resp = JSONResponse(content=token_bundle)
    if (
        grant_type in {"authorization_code", "refresh_token"}
        and client.get("client_id") == "xcelsior-web"
    ):
        _set_session_cookies(resp, token_bundle)
    if deprecated_surface:
        for key, value in build_deprecation_headers(deprecated_surface).items():
            resp.headers[key] = value
    return resp


# ── Model: RegisterRequest ──


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=128)
    password: str = Field(..., min_length=8, max_length=128)
    name: str = Field("", max_length=64)
    role: str = Field("submitter", max_length=32)  # submitter | provider


# ── Model: LoginRequest ──


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=128)
    password: str = Field(..., min_length=8, max_length=128)


# ── Model: ProfileUpdateRequest ──


class ProfileUpdateRequest(BaseModel):
    name: str | None = Field(None, max_length=64)
    role: str | None = Field(None, max_length=32)
    country: str | None = Field(None, max_length=32)
    province: str | None = Field(None, max_length=32)


class OAuthClientCreateRequest(BaseModel):
    client_name: str = Field(min_length=1, max_length=128)
    redirect_uris: list[str] = Field(default_factory=list, max_length=20)
    grant_types: list[str] = Field(default_factory=lambda: ["client_credentials"], max_length=10)
    scopes: list[str] = Field(default_factory=lambda: ["api"], max_length=20)
    client_type: str = Field(default="confidential", pattern="^(confidential|public)$")
    is_first_party: bool = False


class DeviceVerificationRequest(BaseModel):
    user_code: str = Field(min_length=8, max_length=32)


@router.get("/.well-known/oauth-authorization-server", tags=["Auth"])
def oauth_authorization_server_metadata(request: Request):
    base_url = str(request.base_url).rstrip("/")
    return {
        "issuer": _OAUTH_BASE_URL.rstrip("/"),
        "authorization_endpoint": f"{base_url}/oauth/authorize",
        "token_endpoint": f"{base_url}/oauth/token",
        "device_authorization_endpoint": f"{base_url}/oauth/device/authorize",
        "grant_types_supported": [
            "authorization_code",
            "refresh_token",
            "client_credentials",
            "urn:ietf:params:oauth:grant-type:device_code",
        ],
        "response_types_supported": ["code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": [
            "client_secret_post",
            "client_secret_basic",
            "none",
        ],
        "scopes_supported": [
            "profile",
            "email",
            "offline_access",
            "instances:read",
            "instances:write",
            "billing:read",
            "billing:write",
            "hosts:read",
            "hosts:write",
            "api",
        ],
        "service_documentation": f"{_OAUTH_BASE_URL.rstrip('/')}/docs",
    }


@router.get("/oauth/authorize", tags=["Auth"])
def oauth_authorize(request: Request):
    user = _get_current_user(request)
    if not user:
        wants_html = "text/html" in request.headers.get("accept", "").lower()
        if wants_html or not request.headers.get("authorization"):
            redirect_target = _urllib_parse.quote(
                str(request.url.path) + (f"?{request.url.query}" if request.url.query else "")
            )
            return RedirectResponse(f"/login?redirect={redirect_target}", status_code=302)
        raise HTTPException(401, "Not authenticated")
    auth_type = str(user.get("auth_type", ""))
    if auth_type == "client_credentials":
        raise HTTPException(403, "Interactive user authentication required")
    if auth_type == "api_key":
        raise HTTPException(403, "Interactive session authentication required")
    response_type = str(request.query_params.get("response_type", "")).strip()
    client_id = str(request.query_params.get("client_id", "")).strip()
    redirect_uri = str(request.query_params.get("redirect_uri", "")).strip()
    state = str(request.query_params.get("state", "")).strip()
    code_challenge = str(request.query_params.get("code_challenge", "")).strip()
    code_challenge_method = str(request.query_params.get("code_challenge_method", "S256")).strip()
    if response_type != "code":
        raise HTTPException(400, "response_type=code is required")
    if not client_id or not redirect_uri or not code_challenge:
        raise HTTPException(400, "client_id, redirect_uri, and code_challenge are required")

    client = get_client(client_id)
    if not client:
        raise HTTPException(400, "Unknown OAuth client")
    if "authorization_code" not in list(client.get("grant_types") or []):
        raise HTTPException(403, "Client is not allowed to use authorization_code")
    _validate_oauth_client_redirect(client, redirect_uri)
    scopes = str(request.query_params.get("scope", "")).strip().split()
    try:
        code = issue_authorization_code(
            client=client,
            user=user,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            scopes=scopes or list(client.get("scopes") or []),
        )
    except OAuthGrantError as exc:
        raise HTTPException(exc.status_code, exc.description) from exc

    sep = "&" if "?" in redirect_uri else "?"
    location = f"{redirect_uri}{sep}code={code}"
    if state:
        location += f"&state={state}"
    return RedirectResponse(location, status_code=302)


@router.post("/oauth/token", tags=["Auth"])
async def oauth_token(request: Request):
    body = await _parse_request_body(request)
    return await _oauth_token_grant_response(request, body)


@router.post("/oauth/device/authorize", tags=["Auth"])
async def oauth_device_authorize(request: Request):
    body = await _parse_request_body(request)
    try:
        client_id, client_secret = _extract_client_credentials(request, body)
        client = authenticate_client(client_id, client_secret)
        grant_types = list(client.get("grant_types") or [])
        if "urn:ietf:params:oauth:grant-type:device_code" not in grant_types:
            raise OAuthGrantError("unauthorized_client", "Client is not allowed to use device_code")
        data = issue_device_authorization(
            client=client,
            scopes=str(body.get("scope", "")).strip().split(),
            base_url=str(request.base_url).rstrip("/"),
        )
    except OAuthGrantError as exc:
        return _oauth_error_response(exc)
    return JSONResponse(content=data)


@router.post("/api/auth/device", tags=["Auth"], deprecated=True)
async def oauth_device_authorize_compat(request: Request):
    body = await _parse_request_body(request)
    try:
        client_id, client_secret = _extract_client_credentials(request, body)
        client = authenticate_client(client_id, client_secret)
        data = issue_device_authorization(
            client=client,
            scopes=str(body.get("scope", "")).strip().split(),
            base_url=str(request.base_url).rstrip("/"),
        )
    except OAuthGrantError as exc:
        return _oauth_error_response(exc, deprecated_surface="/api/auth/device")
    resp = JSONResponse(content=data)
    for key, value in build_deprecation_headers("/api/auth/device").items():
        resp.headers[key] = value
    return resp


@router.post("/api/auth/token", tags=["Auth"], deprecated=True)
async def oauth_token_compat(request: Request):
    body = await _parse_request_body(request)
    return await _oauth_token_grant_response(request, body, deprecated_surface="/api/auth/token")


@router.get("/api/auth/verify", tags=["Auth"], deprecated=True)
def oauth_verify_page():
    from fastapi.responses import HTMLResponse

    html = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Xcelsior Device Authorization</title></head>
<body><main style="max-width:480px;margin:3rem auto;font-family:system-ui,sans-serif">
<h1>Authorize Device</h1>
<p>Sign in, then submit your device code with <code>POST /api/auth/verify</code>.</p>
</main></body></html>"""
    return HTMLResponse(content=html)


@router.post("/api/auth/verify", tags=["Auth"], deprecated=True)
def oauth_verify_device(body: DeviceVerificationRequest, request: Request):
    user = _require_user_grant(request)
    try:
        result = approve_device_code(user=user, user_code=body.user_code.strip().upper())
    except OAuthGrantError as exc:
        return _oauth_error_response(exc, deprecated_surface="/api/auth/verify")
    resp = JSONResponse(content={"message": "Device authorized", **result})
    for key, value in build_deprecation_headers("/api/auth/verify").items():
        resp.headers[key] = value
    return resp


@router.post("/api/oauth/clients", tags=["Auth"])
def api_create_oauth_client(body: OAuthClientCreateRequest, request: Request):
    user = _require_user_grant(request)
    if body.client_type not in {"public", "confidential"}:
        raise HTTPException(400, "client_type must be public or confidential")
    if body.is_first_party and not _is_platform_admin(user):
        raise HTTPException(403, "Only admins can create first-party OAuth clients")
    client = create_oauth_client(
        client_name=body.client_name.strip(),
        redirect_uris=list(body.redirect_uris),
        grant_types=list(body.grant_types),
        scopes=list(body.scopes),
        created_by_email=user["email"],
        client_type=body.client_type,
        is_first_party=body.is_first_party,
    )
    return {"ok": True, "client": client}


@router.get("/api/oauth/clients", tags=["Auth"])
def api_list_oauth_clients(request: Request):
    user = _require_user_grant(request)
    from db import OAuthStore

    clients = OAuthStore.list_clients(None if _is_platform_admin(user) else user["email"])
    safe_clients = [
        {
            "client_id": client["client_id"],
            "client_name": client["client_name"],
            "client_type": client["client_type"],
            "redirect_uris": list(client.get("redirect_uris") or []),
            "grant_types": list(client.get("grant_types") or []),
            "scopes": list(client.get("scopes") or []),
            "is_first_party": bool(client.get("is_first_party")),
            "created_by_email": client.get("created_by_email"),
            "status": client.get("status", "active"),
            "created_at": client.get("created_at"),
            "updated_at": client.get("updated_at"),
            "last_used": client.get("last_used"),
        }
        for client in clients
    ]
    return {"ok": True, "clients": safe_clients}


@router.delete("/api/oauth/clients/{client_id}", tags=["Auth"])
def api_delete_oauth_client(client_id: str, request: Request):
    user = _require_user_grant(request)
    from db import OAuthStore

    deleted = OAuthStore.delete_client(
        client_id, None if _is_platform_admin(user) else user["email"]
    )
    if not deleted:
        raise HTTPException(404, "OAuth client not found")
    return {"ok": True}


@router.post("/api/auth/register", tags=["Auth"])
def api_auth_register(body: RegisterRequest, request: Request):
    """Register a new user with email and password.

    Creates an account and sends a verification email. The user must verify
    their email before they can log in.
    """
    _check_auth_rate_limit(request)
    email = body.email.strip().lower()
    requested_role = (body.role or "submitter").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(400, "Invalid email address")
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if requested_role == "admin":
        raise HTTPException(403, "Platform admin access cannot be self-assigned")
    if requested_role not in VALID_ACCOUNT_ROLES:
        raise HTTPException(400, "Role must be submitter or provider")

    # Check for existing accounts
    existing_user = None
    if _USE_PERSISTENT_AUTH:
        existing_user = UserStore.get_user(email)
    else:
        with _user_lock:
            existing_user = _users_db.get(email)

    # If account exists with a password already set, reject
    if existing_user and existing_user.get("password_hash"):
        raise HTTPException(409, "Email already registered")

    password_hash, salt = _hash_password(body.password)

    if existing_user and not existing_user.get("password_hash"):
        # OAuth-only account — link password to existing account
        if _USE_PERSISTENT_AUTH:
            UserStore.update_user(
                email,
                {
                    "password_hash": password_hash,
                    "salt": salt,
                    "email_verified": 1,  # Already verified via OAuth
                },
            )
        else:
            existing_user["password_hash"] = password_hash
            existing_user["salt"] = salt
            existing_user["email_verified"] = 1
            with _user_lock:
                _users_db[email] = existing_user
        user = existing_user
        user["password_hash"] = password_hash
        user["salt"] = salt

        token_bundle = issue_user_tokens(
            user, request, client_id="xcelsior-web", session_type="browser"
        )
        return _build_auth_response(token_bundle, user=user)

    user_id = f"user-{uuid.uuid4().hex[:12]}"
    customer_id = f"cust-{uuid.uuid4().hex[:8]}"

    # Generate email verification token
    verification_token = secrets.token_urlsafe(32)
    verification_expires = time.time() + 86400  # 24 hours

    user = {
        "user_id": user_id,
        "email": email,
        "name": body.name or email.split("@")[0],
        "password_hash": password_hash,
        "salt": salt,
        "role": requested_role,
        "is_admin": False,
        "customer_id": customer_id,
        "provider_id": None,
        "country": "CA",
        "province": "ON",
        "created_at": time.time(),
    }

    if _USE_PERSISTENT_AUTH:
        UserStore.create_user(user)
        # Set verification fields
        UserStore.update_user(
            email,
            {
                "email_verified": 0,
                "email_verification_token": verification_token,
                "email_verification_expires": verification_expires,
            },
        )
    else:
        user["email_verified"] = 0
        user["email_verification_token"] = verification_token
        user["email_verification_expires"] = verification_expires
        with _user_lock:
            _users_db[email] = user

    # Create initial wallet
    try:
        be = get_billing_engine()
        be.deposit(customer_id, 0.0, "Account created")
    except Exception as e:
        log.debug("initial deposit creation failed: %s", e)

    broadcast_sse("user_registered", {"email": email, "user_id": user_id})

    # Telegram alert for new signup
    try:
        alert("New Signup", f"{user['name']} ({email}) — role: {user.get('role', 'submitter')}")
    except Exception as e:
        log.debug("signup alert failed: %s", e)

    # Welcome notification for the new user
    try:
        NotificationStore.create(
            email,
            "system",
            "Welcome to Xcelsior!",
            "Welcome to your Notifications Inbox! This is your new go-to destination for important account updates and personalized recommendations. Check back here to stay informed and maximize your Xcelsior experience!",
            {"user_id": user_id},
        )
    except Exception as e:
        log.debug("welcome notification failed: %s", e)

    # Send verification email
    base_url = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca")
    verify_url = f"{base_url}/verify-email?token={verification_token}"
    try:
        display_name = user["name"]
        _send_team_email(
            email,
            f"Verify your email, {display_name}",
            f"Hi {display_name},\n\n"
            "Thanks for signing up for Xcelsior — Canada's sovereign GPU compute marketplace.\n\n"
            "Please verify your email address by clicking the button below. This link expires in 24 hours.\n\n"
            f"{verify_url}\n\n"
            "If you didn't create this account, you can safely ignore this email.",
            cta_url=verify_url,
            cta_label="Verify Email",
        )
    except Exception as e:
        log.debug("verification email send failed: %s", e)

    # In test mode, auto-verify and return session (no email service)
    if XCELSIOR_ENV == "test":
        if _USE_PERSISTENT_AUTH:
            UserStore.update_user(
                email,
                {
                    "email_verified": 1,
                    "email_verification_token": None,
                    "email_verification_expires": None,
                },
            )
        else:
            with _user_lock:
                if email in _users_db:
                    _users_db[email]["email_verified"] = 1
        token_bundle = issue_user_tokens(
            user, request, client_id="xcelsior-web", session_type="browser"
        )
        return _build_auth_response(token_bundle, user=user)

    return JSONResponse(
        content={
            "ok": True,
            "email_verification_required": True,
            "message": "Account created. Please check your email to verify your address.",
        }
    )


@router.post("/api/auth/login", tags=["Auth"])
def api_auth_login(body: LoginRequest, request: Request):
    """Authenticate with email and password.

    Returns a Bearer token valid for 30 days.
    If MFA is enabled, returns mfa_required=True with a challenge_id instead.
    """
    _check_auth_rate_limit(request)
    email = body.email.strip().lower()

    if _USE_PERSISTENT_AUTH:
        user = UserStore.get_user(email)
    else:
        with _user_lock:
            user = _users_db.get(email)

    if not user:
        raise HTTPException(401, "Invalid email or password")

    # OAuth-only account — no password has been set yet
    if not user.get("password_hash") or not user.get("salt"):
        provider = user.get("oauth_provider", "OAuth")
        return JSONResponse(
            status_code=403,
            content={
                "ok": False,
                "oauth_account": True,
                "oauth_provider": provider,
                "error": {
                    "code": "oauth_only",
                    "message": f"This account was created with {provider.title()}. Please sign in with {provider.title()}, or use 'Forgot password' to set a password.",
                },
            },
        )

    password_hash, _ = _hash_password(body.password, user["salt"])
    if not hmac.compare_digest(password_hash, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")

    # Check email verification
    if not user.get("email_verified"):
        return JSONResponse(
            status_code=403,
            content={
                "ok": False,
                "email_verification_required": True,
                "email": email,
                "error": {
                    "code": "email_not_verified",
                    "message": "Please verify your email address before logging in.",
                },
            },
        )

    # ── MFA check ──
    enabled_methods: list[dict] = []
    if _USE_PERSISTENT_AUTH:
        all_methods = MfaStore.list_methods(email)
        enabled_methods = [m for m in all_methods if m.get("enabled")]
        log.info(
            f"MFA check for {email}: {len(all_methods)} total methods, {len(enabled_methods)} enabled methods"
        )
        if enabled_methods:
            log.info(
                f"Enabled MFA methods for {email}: {[m['method_type'] for m in enabled_methods]}"
            )
        live_mfa_enabled = bool(enabled_methods)
        if live_mfa_enabled != bool(user.get("mfa_enabled")):
            log.info(f"Syncing mfa_enabled flag for {email}: {live_mfa_enabled}")
            UserStore.update_user(email, {"mfa_enabled": 1 if live_mfa_enabled else 0})
            user["mfa_enabled"] = 1 if live_mfa_enabled else 0

    if enabled_methods:
        challenge_id = secrets.token_urlsafe(32)
        # Store a temporary partial session token
        partial_token = secrets.token_urlsafe(48)
        MfaStore.create_challenge(
            {
                "challenge_id": challenge_id,
                "email": email,
                "session_token": partial_token,
                "created_at": time.time(),
                "expires_at": time.time() + 300,  # 5-minute window
            }
        )
        return JSONResponse(
            content={
                "ok": True,
                "mfa_required": True,
                "challenge_id": challenge_id,
                "methods": [m["method_type"] for m in enabled_methods],
            }
        )

    token_bundle = issue_user_tokens(
        user, request, client_id="xcelsior-web", session_type="browser"
    )

    # Ensure a welcome notification exists (checks ALL notifications, not just unread,
    # so marking-all-read won't re-trigger on next login)
    try:
        if _USE_PERSISTENT_AUTH and not NotificationStore.list_for_user(email, limit=1):
            NotificationStore.create(
                email,
                "system",
                "Welcome to Xcelsior!",
                "Welcome to your Notifications Inbox! This is your new go-to destination for important account updates and personalized recommendations. Check back here to stay informed and maximize your Xcelsior experience!",
                {"user_id": user["user_id"]},
            )
    except Exception as e:
        log.debug("welcome notification (OAuth) failed: %s", e)

    return _build_auth_response(token_bundle, user=user)


@router.post("/api/auth/oauth/{provider}", tags=["Auth"])
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


@router.get("/api/auth/oauth/{provider}/callback", tags=["Auth"])
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
        # Do NOT log token_data — may contain refresh_token/id_token even when
        # access_token is missing. Only log the set of returned keys + any error field.
        returned_keys = sorted(token_data.keys()) if isinstance(token_data, dict) else []
        err_field = token_data.get("error") if isinstance(token_data, dict) else None
        log.error(
            "No access_token in OAuth response for %s (keys=%s, error=%s)",
            provider,
            returned_keys,
            err_field,
        )
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
            except Exception as e:
                log.debug("GitHub email extraction failed: %s", e)
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
            "is_admin": False,
            "customer_id": customer_id,
            "provider_id": None,
            "country": "CA",
            "province": "ON",
            "oauth_provider": provider,
            "created_at": time.time(),
        }
        if _USE_PERSISTENT_AUTH:
            UserStore.create_user(user)
            # OAuth users are auto-verified (provider verified their email)
            UserStore.update_user(email, {"email_verified": 1})
        else:
            user["email_verified"] = 1
            with _user_lock:
                _users_db[email] = user
    else:
        # Update name if it was empty
        if not user.get("name") and name:
            user["name"] = name
        # Always update oauth_provider on login so we track the most recent
        if _USE_PERSISTENT_AUTH:
            UserStore.update_user(email, {"oauth_provider": provider})
        else:
            user["oauth_provider"] = provider
            with _user_lock:
                _users_db[email] = user

    try:
        token_bundle = issue_user_tokens(
            user, request, client_id="xcelsior-web", session_type="browser"
        )
    except Exception as e:
        log.error("OAuth token issuance failed for %s (%s): %s", email, provider, e)
        return RedirectResponse("/dashboard?error=oauth_token_failed")

    # Ensure welcome notification exists
    try:
        if _USE_PERSISTENT_AUTH and not NotificationStore.list_for_user(email, limit=1):
            NotificationStore.create(
                email,
                "system",
                "Welcome to Xcelsior!",
                "Welcome to your Notifications Inbox! This is your new go-to destination for important account updates and personalized recommendations. Check back here to stay informed and maximize your Xcelsior experience!",
                {"user_id": user["user_id"]},
            )
    except Exception as e:
        log.debug("welcome notification (OAuth) failed: %s", e)

    # Set httpOnly cookie and redirect to dashboard
    resp = RedirectResponse("/dashboard", status_code=302)
    _set_session_cookies(resp, token_bundle)
    # Non-httpOnly cookie so login page JS can show "Last used" badge
    _base = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca")
    _oauth_kw: dict = dict(
        key="xcelsior_last_oauth",
        value=provider,
        max_age=86400 * 365,
        httponly=False,
        secure=_base.startswith("https"),
        samesite="lax",
        path="/",
    )
    if _base.startswith("https"):
        _oauth_kw["domain"] = ".xcelsior.ca"
    resp.set_cookie(**_oauth_kw)
    return resp


@router.get("/api/auth/me", tags=["Auth"])
def api_auth_me(request: Request):
    """Get the currently authenticated user's profile.

    Requires Authorization: Bearer <token> header.
    """
    user = _require_user_grant(request)

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
            "is_admin": True if _is_platform_admin(full_user or user) else False,
            "customer_id": full_user.get("customer_id", ""),
            "provider_id": full_user.get("provider_id"),
            "country": full_user.get("country", "CA"),
            "province": full_user.get("province", "ON"),
            "team_id": full_user.get("team_id"),
            "oauth_provider": full_user.get("oauth_provider"),
            "created_at": full_user.get("created_at", 0),
            "auth_type": user.get("auth_type", "legacy_session"),
            "session_type": user.get("session_type", "browser"),
        },
    }


@router.patch("/api/auth/me", tags=["Auth"])
def api_auth_update_profile(body: ProfileUpdateRequest, request: Request):
    """Update the current user's profile fields."""
    user = _require_user_grant(request)
    requested_role = None
    if body.role is not None:
        requested_role = body.role.strip().lower()
        if requested_role == "admin":
            raise HTTPException(403, "Platform admin access cannot be self-assigned")
        if requested_role not in VALID_ACCOUNT_ROLES:
            raise HTTPException(400, "Role must be submitter or provider")

    if _USE_PERSISTENT_AUTH:
        updates = {}
        if body.name is not None:
            updates["name"] = body.name
        if requested_role is not None:
            updates["role"] = requested_role
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
            if requested_role is not None:
                full_user["role"] = requested_role
            if body.country is not None:
                full_user["country"] = body.country
            if body.province is not None:
                full_user["province"] = body.province

    return {"ok": True, "message": "Profile updated"}


@router.post("/api/auth/refresh", tags=["Auth"], deprecated=True)
def api_auth_refresh(request: Request):
    """Refresh an existing session token.

    Returns a new token with a fresh 30-day expiry.
    """
    user = _get_current_user(request)
    if user and user.get("auth_type") == "api_key":
        raise HTTPException(403, "Interactive session authentication required")
    if user and user.get("auth_type") == "client_credentials":
        raise HTTPException(403, "Interactive user authentication required")

    refresh_token = _current_refresh_token(request, user)
    if refresh_token:
        try:
            token_bundle = rotate_refresh_token(refresh_token, request)
        except OAuthGrantError as exc:
            return _oauth_error_response(exc, deprecated_surface="/api/auth/refresh")

        refreshed_user = None
        email = str((user or {}).get("email", "")).strip()
        if email:
            refreshed_user = (
                UserStore.get_user(email) if _USE_PERSISTENT_AUTH else _users_db.get(email)
            )
        refreshed_user = refreshed_user or user or {}
        return _build_auth_response(
            token_bundle,
            user=refreshed_user,
            deprecated_surface="/api/auth/refresh",
        )

    if not user:
        raise HTTPException(401, "Session expired or invalid")

    if _USE_PERSISTENT_AUTH:
        full_user = UserStore.get_user(user["email"]) or user
    else:
        with _user_lock:
            full_user = _users_db.get(user["email"], user)

    # Invalidate old token (check header first, then cookie — same as logout)
    auth = request.headers.get("authorization", "")
    old_token = auth[7:] if auth.startswith("Bearer ") else ""
    if not old_token:
        old_token = request.cookies.get(_AUTH_COOKIE_NAME, "")
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
    for key, value in build_deprecation_headers("/api/auth/refresh").items():
        resp.headers[key] = value
    return resp


@router.post("/api/auth/logout", tags=["Auth"])
def api_auth_logout(request: Request):
    """Logout — invalidate session and clear cookie."""
    user = _get_current_user(request)
    if user:
        auth_type = str(user.get("auth_type", ""))
        if auth_type in {"api_key", "client_credentials"}:
            raise HTTPException(403, "Interactive session authentication required")
        refresh_token = _current_refresh_token(request, user)
        if refresh_token:
            revoke_refresh_session(refresh_token)
        else:
            auth = request.headers.get("authorization", "")
            token = auth[7:] if auth.startswith("Bearer ") else ""
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


@router.delete("/api/auth/me", tags=["Auth"])
def api_auth_delete_account(request: Request):
    """Delete the current user's account."""
    user = _require_user_grant(request)

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


@router.post("/api/keys/generate", tags=["Auth"], deprecated=True)
async def api_generate_api_key(request: Request):
    """Generate a named API key for the authenticated user.

    API keys can be used as Bearer tokens for programmatic access.
    Scope: 'full-access' (default) or 'read-only' (GET requests only).
    """
    user = _require_user_grant(request)
    body = await request.json()
    name = body.get("name", "default")
    scope = body.get("scope", "full-access")
    if scope not in VALID_KEY_SCOPES:
        raise HTTPException(
            400, f"Invalid scope. Must be one of: {', '.join(sorted(VALID_KEY_SCOPES))}"
        )

    key = f"xcel_{secrets.token_urlsafe(32)}"
    key_data = {
        "key": key,
        "name": name,
        "email": user["email"],
        "user_id": user["user_id"],
        "role": user.get("role", "submitter"),
        "is_admin": True if _is_platform_admin(user) else False,
        "scope": scope,
        "created_at": time.time(),
        "last_used": None,
    }
    with _user_lock:
        _api_keys[key] = key_data
    if _USE_PERSISTENT_AUTH:
        UserStore.create_api_key(key_data)

    resp = JSONResponse(
        content={
            "ok": True,
            "key": key,
            "name": name,
            "scope": scope,
            "preview": key[:12] + "..." + key[-4:],
            "note": "Save this key — it will not be shown again.",
        }
    )
    for hdr, value in build_deprecation_headers("/api/keys").items():
        resp.headers[hdr] = value
    return resp


@router.get("/api/keys", tags=["Auth"], deprecated=True)
def api_list_keys(request: Request):
    """List all API keys for the authenticated user (keys are redacted)."""
    user = _require_user_grant(request)

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

    resp = JSONResponse(content={"ok": True, "keys": keys})
    for hdr, value in build_deprecation_headers("/api/keys").items():
        resp.headers[hdr] = value
    return resp


@router.delete("/api/keys/{key_preview}", tags=["Auth"], deprecated=True)
def api_revoke_key(key_preview: str, request: Request):
    """Revoke an API key by its preview string."""
    user = _require_user_grant(request)

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
    resp = JSONResponse(content={"ok": True, "message": "API key revoked"})
    for hdr, value in build_deprecation_headers("/api/keys").items():
        resp.headers[hdr] = value
    return resp


# ── Model: PasswordResetRequest ──


class PasswordResetRequest(BaseModel):
    email: str


@router.post("/api/auth/password-reset", tags=["Auth"])
def api_auth_password_reset(req: PasswordResetRequest, request: Request):
    """Initiate a password reset. Sends email with reset link."""
    _check_auth_rate_limit(request)
    reset_token = None
    if _USE_PERSISTENT_AUTH:
        user = UserStore.get_user(req.email)
        if not user:
            return {
                "ok": False,
                "account_exists": False,
                "message": "No account found for this email address.",
            }
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
                return {
                    "ok": False,
                    "account_exists": False,
                    "message": "No account found for this email address.",
                }
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
        "account_exists": True,
        "message": "Password reset instructions have been sent to your email.",
        "reset_token": reset_token if os.environ.get("XCELSIOR_ENV") == "test" else None,
    }


# ── Model: PasswordResetConfirm ──


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str


@router.post("/api/auth/password-reset/confirm", tags=["Auth"])
def api_auth_password_reset_confirm(req: PasswordResetConfirm, request: Request):
    """Confirm password reset with token and set new password."""
    _check_auth_rate_limit(request)
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


# ── Model: ChangePasswordRequest ──


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@router.post("/api/auth/change-password", tags=["Auth"])
def api_auth_change_password(request: Request, req: ChangePasswordRequest):
    """Change password for the authenticated user."""
    user = _require_user_grant(request)
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


# ── Model: VerifyEmailRequest ──


class VerifyEmailRequest(BaseModel):
    token: str


# ── Model: ResendVerificationRequest ──


class ResendVerificationRequest(BaseModel):
    email: str


@router.post("/api/auth/verify-email", tags=["Auth"])
def api_auth_verify_email(req: VerifyEmailRequest, request: Request):
    """Verify a user's email address with the token sent during registration."""
    _check_auth_rate_limit(request)
    if _USE_PERSISTENT_AUTH:
        from db import auth_connection

        with auth_connection() as conn:
            row = conn.execute(
                "SELECT email, email_verification_expires FROM users WHERE email_verification_token = %s",
                (req.token,),
            ).fetchone()
            if not row:
                raise HTTPException(400, "Invalid verification token")
            if time.time() > (row["email_verification_expires"] or 0):
                raise HTTPException(400, "Verification link has expired. Please request a new one.")
            conn.execute(
                "UPDATE users SET email_verified = 1, email_verification_token = NULL, email_verification_expires = NULL WHERE email = %s",
                (row["email"],),
            )
        email = row["email"]
    else:
        with _user_lock:
            for em, u in _users_db.items():
                if u.get("email_verification_token") == req.token:
                    if time.time() > u.get("email_verification_expires", 0):
                        raise HTTPException(
                            400, "Verification link has expired. Please request a new one."
                        )
                    u["email_verified"] = 1
                    u.pop("email_verification_token", None)
                    u.pop("email_verification_expires", None)
                    email = em
                    break
            else:
                raise HTTPException(400, "Invalid verification token")

    # Auto-login after verification
    if _USE_PERSISTENT_AUTH:
        user = UserStore.get_user(email)
    else:
        with _user_lock:
            user = _users_db.get(email)

    if not user:
        raise HTTPException(400, "User not found")

    token_bundle = issue_user_tokens(
        user, request, client_id="xcelsior-web", session_type="browser"
    )

    # Send welcome email now that they're verified
    try:
        display_name = user.get("name", email.split("@")[0])
        _send_team_email(
            email,
            f"Welcome to Xcelsior, {display_name}!",
            f"Hi {display_name},\n\n"
            "Your email is verified and your account is active.\n\n"
            "From your dashboard you can browse available GPU hosts, "
            "launch compute instances, and track your usage — all billed in CAD with full Canadian data residency.\n\n"
            "We're currently in early access, so if you run into anything or have questions, just reply to this email. "
            "We'd love to hear from you.",
            cta_url="https://xcelsior.ca/dashboard",
            cta_label="Go to Dashboard",
        )
    except Exception as e:
        log.debug("email notification send failed: %s", e)

    return _build_auth_response(token_bundle, user=user)


@router.post("/api/auth/resend-verification", tags=["Auth"])
def api_auth_resend_verification(req: ResendVerificationRequest, request: Request):
    """Resend the email verification link."""
    _check_auth_rate_limit(request)
    email = req.email.strip().lower()

    if _USE_PERSISTENT_AUTH:
        user = UserStore.get_user(email)
    else:
        with _user_lock:
            user = _users_db.get(email)

    if not user:
        # Don't reveal whether the email exists
        return {
            "ok": True,
            "message": "If that email is registered, a verification link has been sent.",
        }

    if user.get("email_verified"):
        return {"ok": True, "message": "Email is already verified."}

    verification_token = secrets.token_urlsafe(32)
    verification_expires = time.time() + 86400

    if _USE_PERSISTENT_AUTH:
        UserStore.update_user(
            email,
            {
                "email_verification_token": verification_token,
                "email_verification_expires": verification_expires,
            },
        )
    else:
        with _user_lock:
            u = _users_db.get(email)
            if u:
                u["email_verification_token"] = verification_token
                u["email_verification_expires"] = verification_expires

    base_url = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca")
    verify_url = f"{base_url}/verify-email?token={verification_token}"
    try:
        display_name = user.get("name", email.split("@")[0])
        _send_team_email(
            email,
            f"Verify your email, {display_name}",
            f"Hi {display_name},\n\n"
            "Please verify your email address by clicking the button below. This link expires in 24 hours.\n\n"
            f"{verify_url}\n\n"
            "If you didn't create this account, you can safely ignore this email.",
            cta_url=verify_url,
            cta_label="Verify Email",
        )
    except Exception as e:
        log.debug("verification email send failed: %s", e)

    return {
        "ok": True,
        "message": "If that email is registered, a verification link has been sent.",
    }


@router.get("/api/auth/sessions", tags=["Auth"])
def api_auth_list_sessions(request: Request):
    """List active sessions for the current user."""
    user = _require_user_grant(request)
    current_token = _current_refresh_token(request, user)

    if _USE_PERSISTENT_AUTH:
        sessions = UserStore.list_user_sessions(user["email"])
    else:
        with _user_lock:
            sessions = [
                s
                for s in _sessions.values()
                if s.get("email") == user["email"] and s["expires_at"] > time.time()
            ]

    result = []
    for s in sessions:
        token = s["token"]
        result.append(
            {
                "token_prefix": token[:8],
                "is_current": token == current_token,
                "ip_address": s.get("ip_address", ""),
                "user_agent": s.get("user_agent", ""),
                "created_at": s.get("created_at"),
                "last_active": s.get("last_active"),
                "expires_at": s.get("expires_at"),
            }
        )

    return {"ok": True, "sessions": result}


@router.delete("/api/auth/sessions/{token_prefix}", tags=["Auth"])
def api_auth_revoke_session(token_prefix: str, request: Request):
    """Revoke a specific session by its token prefix."""
    user = _require_user_grant(request)

    if _USE_PERSISTENT_AUTH:
        sessions = UserStore.list_user_sessions(user["email"])
        for s in sessions:
            if s["token"][:8] == token_prefix:
                revoke_refresh_session(s["token"])
                return {"ok": True, "message": "Session revoked"}
    else:
        with _user_lock:
            for tok, s in list(_sessions.items()):
                if s.get("email") == user["email"] and tok[:8] == token_prefix:
                    del _sessions[tok]
                    return {"ok": True, "message": "Session revoked"}

    raise HTTPException(404, "Session not found")


@router.get("/api/auth/me/data-export", tags=["Auth"])
def api_data_export(request: Request):
    """Export all personal data for the current user (PIPEDA right).

    Returns a JSON bundle of all user data: profile, jobs, billing,
    reputation, artifacts, and consent records.
    """
    user = _require_user_grant(request)
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
        except Exception as e:
            log.debug("billing history fetch failed: %s", e)

    # Gather reputation
    rep_data = {}
    try:
        re = get_reputation_engine()
        rep_data = re.store.get_score(customer_id or email) or {}
    except Exception as e:
        log.debug("reputation data fetch failed: %s", e)

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


@router.get("/api/users/me/preferences", tags=["Auth"])
def api_get_user_preferences(request: Request):
    """Get user preferences."""
    user = _require_user_grant(request)
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
        except Exception as e:
            raw_prefs = {}
    return {
        "ok": True,
        "canada_only_routing": bool(full_user.get("canada_only_routing", 0)),
        "notifications": bool(full_user.get("notifications_enabled", 1)),
        "preferences": raw_prefs if isinstance(raw_prefs, dict) else {},
    }


@router.put("/api/users/me/preferences", tags=["Auth"])
def api_set_user_preferences(request: Request, body: dict):
    """Update user preferences."""
    user = _require_user_grant(request)
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
                except Exception as e:
                    existing_prefs = {}
            if not isinstance(existing_prefs, dict):
                existing_prefs = {}
            merged = {**existing_prefs, **safe_prefs}
            updates["preferences"] = merged
    if updates and _USE_PERSISTENT_AUTH:
        UserStore.update_user(user["email"], updates)
    return {"ok": True}


# ── OAuth Client Management ──────────────────────────────────────────


class OAuthClientUpdateRequest(BaseModel):
    client_name: str | None = None
    redirect_uris: list[str] | None = None
    grant_types: list[str] | None = None
    scopes: list[str] | None = None
    status: str | None = Field(None, pattern="^(active|disabled)$")


@router.patch("/api/oauth/clients/{client_id}", tags=["Auth"])
def api_update_oauth_client(client_id: str, body: OAuthClientUpdateRequest, request: Request):
    """Update an OAuth client's metadata (name, redirect URIs, scopes, status)."""
    user = _require_user_grant(request)
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(400, "No fields to update")
    from db import OAuthStore

    ok = OAuthStore.update_client(
        client_id, updates, None if _is_platform_admin(user) else user["email"]
    )
    if not ok:
        raise HTTPException(404, "OAuth client not found or not permitted")
    return {"ok": True}


@router.post("/api/oauth/clients/{client_id}/rotate-secret", tags=["Auth"])
def api_rotate_oauth_client_secret(client_id: str, request: Request):
    """Rotate an OAuth client's secret. Returns the new plaintext secret once."""
    user = _require_user_grant(request)
    new_secret = base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip("=")
    hash_, salt = hash_secret(new_secret)
    from db import OAuthStore

    ok = OAuthStore.rotate_client_secret(
        client_id, hash_, salt, None if _is_platform_admin(user) else user["email"]
    )
    if not ok:
        raise HTTPException(404, "OAuth client not found or not permitted")
    return {"ok": True, "client_id": client_id, "client_secret": new_secret}
