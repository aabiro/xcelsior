"""Routes: auth."""

import base64
import hashlib
import hmac
import os
import re

import env_config
import secrets
import time
import uuid
from typing import Any, Literal

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
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
    _team_context_for_user as _deps_team_context_for_user,
    _require_admin as _deps_require_admin,
    _require_auth as _deps_require_auth,
    _require_user_grant as _deps_require_user_grant,
    _require_write_access as _deps_require_write_access,
    _canonical_owner_id,
    _oauth_workspace_customer_id,
    _oauth_workspace_team_id,
    _require_oauth_client_access,
    _require_oauth_client_write,
    _user_can_access_oauth_client,
    _require_team_instance_write,
    _sessions,
    _set_auth_cookie as _deps_set_auth_cookie,
    _set_mfa_challenge_cookie,
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
    revoke_access_token,
    revoke_client_credentials_jwt,
    ACCESS_TOKEN_TTL_SEC,
    DEVICE_CODE_INTERVAL_SEC,
    MCP_RESOURCE_AUDIENCE,
    OAUTH_AUDIENCE,
    REFRESH_COOKIE_NAME,
    REFRESH_TOKEN_TTL_SEC,
    OAuthGrantError,
    approve_device_code,
    device_verification_url,
    authenticate_client,
    build_deprecation_headers,
    canonical_mcp_resource,
    create_oauth_client,
    normalize_resource_indicator,
    exchange_authorization_code,
    exchange_device_code,
    get_client,
    hash_secret,
    issue_authorization_code,
    issue_client_credentials_jwt,
    issue_device_authorization,
    issue_user_tokens,
    redirect_uri_matches,
    revoke_refresh_session,
    rotate_refresh_token,
    consume_social_oauth_state,
    store_social_oauth_state,
)

router = APIRouter()


def _validated_resource_indicator(body: dict) -> str:
    return normalize_resource_indicator(body.get("resource"))


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
    "facebook": {
        "client_id": os.environ.get("FACEBOOK_CLIENT_ID", ""),
        "client_secret": os.environ.get("FACEBOOK_CLIENT_SECRET", ""),
        "authorize_url": "https://www.facebook.com/v25.0/dialog/oauth",
        "token_url": "https://graph.facebook.com/v25.0/oauth/access_token",
        "userinfo_url": "https://graph.facebook.com/v25.0/me?fields=id,name,email",
        "scopes": "public_profile email",
    },
}
_OAUTH_STATE_TTL = 600


def _sanitize_oauth_redirect(path: str | None) -> str:
    value = str(path or "").strip()
    if not value or not value.startswith("/") or value.startswith("//"):
        return "/dashboard"
    return value


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
        auth_user = {
            "user_id": user.get("user_id", ""),
            "email": user.get("email", ""),
            "name": user.get("name", ""),
            "role": user.get("role", "submitter"),
            "is_admin": True if _is_platform_admin(user) else False,
            "customer_id": user.get("customer_id"),
            "provider_id": user.get("provider_id"),
            "team_id": user.get("team_id"),
        }
        auth_user.update(_deps_team_context_for_user(auth_user))
        body["user"] = auth_user
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
    if allowed and not any(redirect_uri_matches(entry, redirect_uri) for entry in allowed):
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
            resource = normalize_resource_indicator(body.get("resource"), default="") or None
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
                resource=resource,
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
            resource = normalize_resource_indicator(body.get("resource"), default="") or None
            if not device_code:
                raise OAuthGrantError("invalid_request", "device_code is required")
            token_bundle = exchange_device_code(
                client=client, device_code=device_code, request=request, resource=resource
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
            resource = normalize_resource_indicator(body.get("resource"), default="") or None
            token_bundle = issue_client_credentials_jwt(
                client,
                [scope for scope in scopes.split() if scope] or list(client.get("scopes") or []),
                audience=resource,
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
    business_name: str | None = Field(None, max_length=200)
    gst_hst_number: str | None = Field(None, max_length=20)


def _business_profile_updates(body: ProfileUpdateRequest) -> dict:
    """Optional business name and GST/HST number. Blank clears the field."""
    from seller_identity import normalize_business_name, normalize_gst_hst

    updates: dict = {}
    try:
        if body.business_name is not None:
            updates["business_name"] = normalize_business_name(body.business_name)
        if body.gst_hst_number is not None:
            updates["gst_hst_number"] = normalize_gst_hst(body.gst_hst_number)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return updates


def _parse_user_preferences(full_user: dict) -> dict:
    raw = full_user.get("preferences", {})
    if isinstance(raw, str):
        try:
            import json as _json

            raw = _json.loads(raw)
        except Exception:
            raw = {}
    return raw if isinstance(raw, dict) else {}


def _merge_user_preferences(email: str, patch: dict) -> None:
    if not patch:
        return
    existing_user = UserStore.get_user(email) or {}
    merged = {**_parse_user_preferences(existing_user), **patch}
    UserStore.update_user(email, {"preferences": merged})


class OAuthClientCreateRequest(BaseModel):
    client_name: str = Field(min_length=1, max_length=128)
    redirect_uris: list[str] = Field(default_factory=list, max_length=20)
    grant_types: list[str] = Field(default_factory=lambda: ["client_credentials"], max_length=10)
    # No default grant. A client that registers without naming scopes gets
    # none, and is refused by every scope-checked route until it asks for
    # what it needs. The old default was ["api"], a wildcard.
    scopes: list[str] = Field(default_factory=list, max_length=20)
    client_type: str = Field(default="confidential", pattern="^(confidential|public)$")
    is_first_party: bool = False


class DeviceVerificationRequest(BaseModel):
    user_code: str = Field(min_length=8, max_length=32)


@router.get("/.well-known/oauth-authorization-server", tags=["Auth"])
def oauth_authorization_server_metadata(request: Request):
    from oauth_registration import CIMD_ENABLED, DCR_ENABLED

    base_url = str(request.base_url).rstrip("/")
    metadata = {
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
        # Deliberately NOT every scope this server enforces.
        #
        # Per the MCP authorization spec, `scopes_supported` "is intended to
        # represent the minimal set of scopes necessary for basic
        # functionality", and a client that receives no `scope` in a
        # `WWW-Authenticate` challenge falls back to requesting *everything*
        # listed here. Publishing the operator scopes therefore meant an
        # ordinary MCP client's first connection asked the user to grant
        # `hosts:evict` and `control_plane:operate` — platform-operator
        # authority, for a chat client. The spec names this as a common mistake
        # under Scope Minimization.
        #
        # Operator scopes are still real, still enforced, and still described in
        # `SCOPE_DESCRIPTIONS` so a consent screen can render them when an admin
        # deliberately grants one. They are simply not advertised as a baseline.
        # `tests/test_advertised_scopes_are_minimal.py` asserts the disjointness.
        "scopes_supported": [
            "profile",
            "email",
            "offline_access",
            "instances:read",
            "instances:write",
            "instances:operate",
            "billing:read",
            "billing:write",
            "hosts:read",
            "hosts:write",
            "mcp_actions:approve",
            "gpu:read",
            "marketplace:read",
            "events:read",
            "inference:read",
            "inference:write",
            # What the MCP tools need beyond the above (the quick-connect token
            # set). Clients that pick scopes from this list rather than from
            # the resource metadata otherwise sign in without SSH, volume and
            # terminal access and every one of those tools refuses them.
            "instances:connect",
            "ssh:write",
            "volumes:read",
            "volumes:write",
            "artifacts:read",
            "providers:read",
            "reputation:read",
            # In the connector ceiling since plan §8.3. Clients that read this
            # list instead of the resource metadata (VS Code) would otherwise
            # never ask for it.
            "ssh:read",
        ],
        "service_documentation": f"{_OAUTH_BASE_URL.rstrip('/')}/docs",
        "jwks_uri": f"{base_url}/.well-known/jwks.json",
        "resource_indicators_supported": True,
    }
    # Client identification, advertised in the order clients should prefer it.
    # CIMD first: it is what Anthropic and OpenAI recommend for a directory
    # connector, and it keeps a high-traffic listing from turning into an
    # unbounded client-registration table.
    if CIMD_ENABLED:
        metadata["client_id_metadata_document_supported"] = True
    if DCR_ENABLED:
        metadata["registration_endpoint"] = f"{base_url}/oauth/register"
        # RFC 7591 §1.5: an open registration endpoint says so, so a client
        # does not go hunting for an initial access token it will never get.
        metadata["registration_endpoint_auth_methods_supported"] = ["none"]
    return metadata


@router.get("/.well-known/jwks.json", tags=["Auth"])
def oauth_jwks_document():
    from oauth_service import oauth_jwks

    return oauth_jwks()


def _oauth_error_page(title: str, detail: str, status: int) -> str:
    """Branded, self-contained HTML error page for browser-facing OAuth failures."""
    import html as _html

    safe_title = _html.escape(title)
    safe_detail = _html.escape(detail)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title} — Xcelsior</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
    color: #e6edf3; padding: 24px;
    background: radial-gradient(1200px 600px at 50% -10%, rgba(124,58,237,0.18), transparent 60%),
                radial-gradient(900px 500px at 80% 110%, rgba(34,211,238,0.12), transparent 55%),
                #070b12;
  }}
  .card {{
    width: 100%; max-width: 460px; padding: 40px 36px; text-align: center;
    background: rgba(14,20,31,0.85); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px; box-shadow: 0 30px 80px -20px rgba(0,0,0,0.7);
    backdrop-filter: blur(12px);
  }}
  .glyph {{
    width: 56px; height: 56px; margin: 0 auto 20px; border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, rgba(34,211,238,0.18), rgba(124,58,237,0.18));
    border: 1px solid rgba(255,255,255,0.1);
  }}
  .glyph svg {{ width: 28px; height: 28px; stroke: #38bdf8; }}
  .code {{ font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase; color: #6b7689; font-weight: 600; }}
  h1 {{ margin: 8px 0 12px; font-size: 22px; font-weight: 700;
        background: linear-gradient(120deg, #e6edf3, #9bd2ff 60%, #c4b5fd);
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }}
  p {{ margin: 0 0 28px; font-size: 15px; line-height: 1.6; color: #9aa6b8; }}
  .actions {{ display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }}
  a.btn {{
    display: inline-flex; align-items: center; justify-content: center;
    padding: 10px 20px; border-radius: 10px; font-size: 14px; font-weight: 600;
    text-decoration: none; transition: transform .12s ease, filter .12s ease;
  }}
  a.btn:hover {{ transform: translateY(-1px); filter: brightness(1.08); }}
  a.primary {{ color: #06121a; background: linear-gradient(120deg, #38bdf8, #818cf8); }}
  a.ghost {{ color: #cbd5e1; border: 1px solid rgba(255,255,255,0.12); }}
  .brand {{ margin-top: 28px; font-size: 12px; letter-spacing: 0.2em; text-transform: uppercase; color: #4a5568; }}
</style></head>
<body><main class="card">
  <div class="glyph"><svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
    <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></div>
  <div class="code">Error {status}</div>
  <h1>{safe_title}</h1>
  <p>{safe_detail}</p>
  <div class="actions">
    <a class="btn primary" href="/login">Sign in</a>
    <a class="btn ghost" href="/dashboard">Go to dashboard</a>
  </div>
  <div class="brand">Xcelsior</div>
</main></body></html>"""


def _oauth_browser_error(request: Request, status: int, title: str, detail: str):
    """Render a branded HTML page for browser callers; raise JSON HTTPException for API clients."""
    from fastapi.responses import HTMLResponse

    wants_html = "text/html" in request.headers.get("accept", "").lower()
    if not wants_html:
        raise HTTPException(status, detail)
    return HTMLResponse(content=_oauth_error_page(title, detail, status), status_code=status)


def _oauth_consent_page(
    *, client: dict, user: dict, scopes: list[str], resource: str, consent_key: str
) -> str:
    """The approve step of "paste URL → Connect → sign in → approve".

    Rendered server-side and self-contained so it works before any frontend
    asset loads and cannot be broken by a CDN or a client-side bundle — this
    page sits in the middle of every provider's connector onboarding, so it has
    to be the least fragile thing we serve.
    """
    import html as _html

    from oauth_service import summarize_consent_scopes

    def esc(value: object) -> str:
        return _html.escape(str(value or ""))

    client_name = esc(client.get("client_name") or client.get("client_id"))
    client_uri = str(client.get("client_uri") or "").strip()
    # Only ever link out to https; a javascript: or data: client_uri from a
    # metadata document must not become a clickable link on our origin.
    client_link = (
        f'<a class="clienturi" href="{esc(client_uri)}" rel="noopener noreferrer nofollow" '
        f'target="_blank">{esc(_urllib_parse.urlsplit(client_uri).hostname or client_uri)}</a>'
        if client_uri.lower().startswith("https://")
        else ""
    )
    summary_items = summarize_consent_scopes(scopes)
    rows = (
        "".join(
            f'<li><span class="dot"></span><span>{esc(item)}</span></li>'
            for item in summary_items
        )
        or '<li><span class="dot"></span><span>Sign you in only</span></li>'
    )
    resource_line = (
        f'<p class="resource">Access is limited to <code>{esc(resource)}</code>.</p>'
        if resource
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>Authorize {client_name} — Xcelsior</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
    color: #e6edf3; padding: 24px;
    background: radial-gradient(1200px 600px at 50% -10%, rgba(124,58,237,0.18), transparent 60%),
                radial-gradient(900px 500px at 80% 110%, rgba(34,211,238,0.12), transparent 55%),
                #070b12;
  }}
  .card {{
    width: 100%; max-width: 520px; padding: 36px 34px;
    background: rgba(14,20,31,0.85); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px; box-shadow: 0 30px 80px -20px rgba(0,0,0,0.7);
    backdrop-filter: blur(12px);
  }}
  .eyebrow {{ font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase;
              color: #6b7689; font-weight: 600; }}
  h1 {{ margin: 10px 0 6px; font-size: 22px; font-weight: 700; line-height: 1.3; }}
  .who {{ margin: 0 0 22px; font-size: 14px; color: #9aa6b8; }}
  .clienturi {{ color: #7dd3fc; text-decoration: none; }}
  ul {{ list-style: none; margin: 0 0 18px; padding: 16px 18px; border-radius: 14px;
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); }}
  li {{ display: flex; gap: 10px; align-items: flex-start; font-size: 14px;
        line-height: 1.5; color: #cbd5e1; padding: 5px 0; }}
  .dot {{ width: 6px; height: 6px; border-radius: 999px; margin-top: 8px; flex: none;
          background: linear-gradient(120deg, #38bdf8, #818cf8); }}
  code {{ display: block; font-size: 11px; color: #6b7689; margin-top: 2px; }}
  .resource {{ font-size: 13px; color: #9aa6b8; margin: 0 0 18px; }}
  .resource code {{ display: inline; color: #7dd3fc; font-size: 12px; }}
  form {{ display: flex; gap: 10px; }}
  button {{
    flex: 1; padding: 12px 18px; border-radius: 10px; font-size: 14px; font-weight: 600;
    border: 0; cursor: pointer; font-family: inherit;
    transition: transform .12s ease, filter .12s ease;
  }}
  button:hover {{ transform: translateY(-1px); filter: brightness(1.08); }}
  .approve {{ color: #06121a; background: linear-gradient(120deg, #38bdf8, #818cf8); }}
  .deny {{ color: #cbd5e1; background: transparent; border: 1px solid rgba(255,255,255,0.12); }}
  .foot {{ margin-top: 22px; font-size: 12px; color: #4a5568; line-height: 1.6; }}
  .foot a {{ color: #6b7689; }}
</style></head>
<body><main class="card">
  <div class="eyebrow">Authorize application</div>
  <h1>{client_name} wants to use your Xcelsior account</h1>
  <p class="who">Signed in as {esc(user.get("email"))}{" · " + client_link if client_link else ""}</p>
  <ul>{rows}</ul>
  {resource_line}
  <form method="post" action="/oauth/authorize">
    <input type="hidden" name="consent_key" value="{esc(consent_key)}">
    <input type="hidden" name="ui" value="1">
    <button class="deny" type="submit" name="decision" value="deny">Cancel</button>
    <button class="approve" type="submit" name="decision" value="approve">Approve</button>
  </form>
  <p class="foot">Destructive operations still require a separate, explicit approval at the
  time they run — approving here does not pre-authorize them. You can revoke this connection
  at any time in <a href="/dashboard/settings">Settings → AI Agents</a>.</p>
</main></body></html>"""


def _build_callback_url(redirect_uri: str, state: str, **params: str) -> str:
    """Append parameters to a client's callback URL, preserving any it already has."""
    sep = "&" if "?" in redirect_uri else "?"
    query = _urllib_parse.urlencode({**params, **({"state": state} if state else {})})
    return f"{redirect_uri}{sep}{query}"


def _redirect_with(redirect_uri: str, state: str, **params: str):
    """Append parameters to a client's callback, preserving any it already has."""
    return RedirectResponse(_build_callback_url(redirect_uri, state, **params), status_code=302)


def _resolve_desktop_app_links(client: dict) -> list[tuple[str, str]]:
    """Return `[(scheme_url, label), ...]` for focusing the originating IDE or desktop app.

    For VS Code and its forks (`vscode`, `vscode-insiders`, `cursor`, `windsurf`,
    `antigravity`), a bare scheme like `vscode://` has an empty authority —
    Electron's `--open-url` dispatcher either falls through to `startOpenWindow`
    (spawning a brand-new empty window) or `URLService.handleURL` returns
    unhandled without calling `focusWindow`. Addressing the built-in trusted
    `vscode.git` protocol handler with an empty path (`vscode://vscode.git`)
    routes via `URLHandlerRouter.getActiveClientId()` to the exact most-recently
    focused window, skips any trust prompt, performs a no-op in `GitProtocolHandler`,
    and triggers `nativeHostService.focusWindow` on that existing window.

    For Codex (`client_name="Codex"`, `client_uri="https://chatgpt.com/codex"`),
    the Codex VS Code extension shares the same client metadata as the standalone
    app, while `codex://` launches a separate ChatGPT desktop binary. We therefore
    default to focusing the existing VS Code window first and offer `codex://`
    as a secondary action.
    """
    haystack = " ".join(
        str(client.get(key) or "")
        for key in ("client_name", "client_uri", "software_id", "client_id")
    ).lower()

    vscode_focus_uri = "vscode://vscode.git"
    if "insiders" in haystack and any(
        token in haystack for token in ("vscode", "visual studio code", "vs code", "visualstudio")
    ):
        return [("vscode-insiders://vscode.git", "Open VS Code Insiders")]
    if any(
        token in haystack
        for token in ("vscode", "visual studio code", "vs code", "code.visualstudio.com")
    ):
        return [(vscode_focus_uri, "Open VS Code")]
    if "cursor" in haystack:
        return [("cursor://vscode.git", "Open Cursor"), (vscode_focus_uri, "Open VS Code")]
    if "windsurf" in haystack:
        return [("windsurf://vscode.git", "Open Windsurf"), (vscode_focus_uri, "Open VS Code")]
    if "antigravity" in haystack:
        return [("antigravity://vscode.git", "Open Antigravity"), (vscode_focus_uri, "Open VS Code")]
    if any(token in haystack for token in ("codex", "chatgpt", "openai")):
        return [(vscode_focus_uri, "Open VS Code"), ("codex://", "Open Codex app")]
    if any(token in haystack for token in ("claude", "anthropic")):
        return [("claude://", "Open Claude"), (vscode_focus_uri, "Open VS Code")]
    return [(vscode_focus_uri, "Open VS Code")]


def _oauth_decision_page(
    *, approved: bool, client: dict, callback_url: str, is_loopback: bool
) -> str:
    """Render a branded Xcelsior result screen for Approved / Denied OAuth decisions.

    For loopback CLI callbacks (`127.0.0.1` / `localhost`), local HTTP listeners
    like Codex CLI respond to both success and `error=access_denied` with a bare
    plain-text `"Authentication complete. You may close this window."` page on a
    white background. Delivering the callback via a hidden iframe notifies the
    local CLI immediately while keeping the user's browser tab on a proper
    Xcelsior dark-themed confirmation or cancellation screen with deep links
    back to VS Code or the originating desktop application.
    """
    import html as _html
    import json as _json

    def esc(value: object) -> str:
        return _html.escape(str(value or ""))

    raw_name = str(client.get("client_name") or client.get("client_id") or "Application")
    client_name = esc(raw_name)
    safe_callback = esc(callback_url)

    if approved:
        page_title = f"Authorized {client_name} — Xcelsior"
        eyebrow = "Authorization complete"
        heading = f"{client_name} is connected"
        detail = "Authentication complete. You can return to your application or close this window."
        glyph_bg = "rgba(16,185,129,0.12)"
        glyph_border = "rgba(16,185,129,0.32)"
        glyph_stroke = "#34d399"
        glyph_svg = '<polyline points="20 6 9 17 4 12"/>'
    else:
        page_title = "Access denied — Xcelsior"
        eyebrow = "Authorization cancelled"
        heading = "Access denied"
        detail = (
            f"You cancelled the authorization request. {client_name} was not granted "
            "access to your Xcelsior account. You may close this window."
        )
        glyph_bg = "rgba(244,63,94,0.12)"
        glyph_border = "rgba(244,63,94,0.3)"
        glyph_stroke = "#fb7185"
        glyph_svg = (
            '<circle cx="12" cy="12" r="10"/>'
            '<line x1="15" y1="9" x2="9" y2="15"/>'
            '<line x1="9" y1="9" x2="15" y2="15"/>'
        )

    def _close_btn(btn_class: str = "primary") -> str:
        return (
            f'<button type="button" class="btn {btn_class}" id="close-btn" '
            "onclick=\"window.close(); setTimeout(function(){ var b = document.getElementById('close-btn'); "
            "if (b) b.textContent = 'You may close this tab'; }, 150);\">Close window</button>"
        )

    if is_loopback:
        app_links = _resolve_desktop_app_links(client)
        primary_href, primary_label = app_links[0]
        app_buttons = [
            f'<a class="btn primary" id="open-app-btn" href="{esc(primary_href)}">{esc(primary_label)}</a>'
        ]
        for extra_href, extra_label in app_links[1:]:
            app_buttons.append(f'<a class="btn ghost" href="{esc(extra_href)}">{esc(extra_label)}</a>')
        actions_html = (
            "".join(app_buttons)
            + _close_btn("ghost")
            + '<a class="btn ghost" href="/dashboard">Go to dashboard</a>'
        )
        # Show the Accepted / Denied screen first, notify the local loopback
        # listener via the hidden iframe, then link back to the desktop app / IDE.
        auto_open_script = (
            "<script>(function(){"
            "var launched=false,cbDone=false,delayDone=false;"
            "function tryOpen(){if(launched||!cbDone||!delayDone)return;launched=true;"
            f"try{{window.location.href={_json.dumps(primary_href)};}}catch(e){{}}}}"
            "var f=document.getElementById('oauth-cb');"
            "if(f){f.addEventListener('load',function(){cbDone=true;tryOpen();});}"
            "else{cbDone=true;}"
            "setTimeout(function(){delayDone=true;tryOpen();},1500);"
            "setTimeout(function(){cbDone=true;delayDone=true;tryOpen();},2200);"
            "})();</script>"
        )
        callback_delivery = (
            f'<iframe id="oauth-cb" src="{safe_callback}" style="display:none" '
            'aria-hidden="true" tabindex="-1" title="OAuth callback"></iframe>'
            f"{auto_open_script}"
        )
    elif approved:
        actions_html = (
            f'<a class="btn primary" id="open-app-btn" href="{safe_callback}">Continue to {client_name}</a>'
            '<a class="btn ghost" href="/dashboard">Go to dashboard</a>'
        )
        callback_delivery = (
            f"<script>setTimeout(function(){{window.location.replace({_json.dumps(callback_url)});}},1500);</script>"
        )
    else:
        actions_html = (
            f'<a class="btn primary" id="open-app-btn" href="{safe_callback}">Return to {client_name}</a>'
            f"{_close_btn('ghost')}"
            '<a class="btn ghost" href="/dashboard">Go to dashboard</a>'
        )
        callback_delivery = (
            f"<script>setTimeout(function(){{window.location.replace({_json.dumps(callback_url)});}},1500);</script>"
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>{page_title}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
    color: #e6edf3; padding: 24px;
    background: radial-gradient(1200px 600px at 50% -10%, rgba(124,58,237,0.18), transparent 60%),
                radial-gradient(900px 500px at 80% 110%, rgba(34,211,238,0.12), transparent 55%),
                #070b12;
  }}
  .card {{
    width: 100%; max-width: 460px; padding: 36px 32px;
    background: rgba(14,20,31,0.85); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px; box-shadow: 0 30px 80px -20px rgba(0,0,0,0.7);
    backdrop-filter: blur(12px);
  }}
  .glyph {{
    width: 44px; height: 44px; border-radius: 12px; display: grid; place-items: center;
    background: {glyph_bg}; border: 1px solid {glyph_border};
    margin-bottom: 20px;
  }}
  .glyph svg {{ width: 22px; height: 22px; stroke: {glyph_stroke}; }}
  .eyebrow {{ font-size: 12px; letter-spacing: 0.14em; text-transform: uppercase;
              color: #6b7689; font-weight: 600; }}
  h1 {{ margin: 8px 0 10px; font-size: 22px; font-weight: 700; line-height: 1.3; }}
  p {{ margin: 0 0 24px; font-size: 14px; line-height: 1.6; color: #9aa6b8; }}
  .actions {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  .btn {{
    display: inline-flex; align-items: center; justify-content: center;
    padding: 10px 16px; border-radius: 10px; font-size: 14px; font-weight: 600;
    text-decoration: none; border: 0; cursor: pointer; font-family: inherit;
    transition: transform .12s ease, filter .12s ease;
  }}
  .btn:hover {{ transform: translateY(-1px); filter: brightness(1.08); }}
  .primary {{ color: #06121a; background: linear-gradient(120deg, #38bdf8, #818cf8); }}
  .ghost {{ color: #cbd5e1; background: transparent; border: 1px solid rgba(255,255,255,0.12); }}
  .brand {{ margin-top: 28px; font-size: 12px; letter-spacing: 0.2em; text-transform: uppercase; color: #4a5568; }}
</style></head>
<body><main class="card">
  <div class="glyph"><svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    {glyph_svg}</svg></div>
  <div class="eyebrow">{eyebrow}</div>
  <h1>{heading}</h1>
  <p>{detail}</p>
  <div class="actions">
    {actions_html}
  </div>
  <div class="brand">Xcelsior</div>
</main>{callback_delivery}</body></html>"""


def _prepare_authorization(request: Request, params) -> tuple[dict, dict, list[str], str]:
    """Validate an authorization request. Returns (user, client, scopes, resource).

    Raises `_AuthorizeRejection` carrying a rendered response for anything the
    user needs to see, so GET and POST share one validation path — two copies of
    this would be two places for the redirect or resource rules to drift.
    """
    response_type = str(params.get("response_type", "")).strip()
    client_id = str(params.get("client_id", "")).strip()
    redirect_uri = str(params.get("redirect_uri", "")).strip()
    code_challenge = str(params.get("code_challenge", "")).strip()
    if response_type != "code":
        raise _AuthorizeRejection(
            _oauth_browser_error(
                request,
                400,
                "Unsupported request",
                "This authorization link is malformed (response_type=code is required). "
                "Please start the connection again from your app.",
            )
        )
    if not client_id or not redirect_uri or not code_challenge:
        raise _AuthorizeRejection(
            _oauth_browser_error(
                request,
                400,
                "Incomplete request",
                "This authorization link is missing required parameters. "
                "Please start the connection again from your app.",
            )
        )

    try:
        client = get_client(client_id)
    except OAuthGrantError as exc:
        # A CIMD client id that resolves to a missing, malformed, or
        # self-inconsistent document lands here. Say which, rather than
        # "unknown client" — the difference is what a connector author needs.
        raise _AuthorizeRejection(
            _oauth_browser_error(
                request,
                400,
                "Untrusted application",
                exc.description,
            )
        ) from exc
    if not client:
        raise _AuthorizeRejection(
            _oauth_browser_error(
                request,
                400,
                "Unknown application",
                "We couldn't find an Xcelsior app matching that client ID. It may have been "
                "deleted, or the link is out of date. Create a new client in Settings → AI Agents.",
            )
        )
    if "authorization_code" not in list(client.get("grant_types") or []):
        raise _AuthorizeRejection(
            _oauth_browser_error(
                request,
                403,
                "Not permitted",
                "This application isn't allowed to sign you in interactively. "
                "Check its grant types in Settings → AI Agents.",
            )
        )
    try:
        _validate_oauth_client_redirect(client, redirect_uri)
    except OAuthGrantError:
        raise _AuthorizeRejection(
            _oauth_browser_error(
                request,
                400,
                "Redirect not allowed",
                "The redirect URL in this request isn't registered for this application. "
                "Update the client's allowed redirect URIs in Settings → AI Agents.",
            )
        ) from None

    raw_resource = str(params.get("resource", "")).strip()
    resource = canonical_mcp_resource(raw_resource) if raw_resource else None
    if raw_resource and resource is None:
        raise _AuthorizeRejection(
            _oauth_browser_error(
                request,
                400,
                "Unsupported resource",
                "This authorization request targets an unsupported protected resource.",
            )
        )
    # A client pinned to one protected resource gets that resource by default,
    # so a connector that omits the indicator still receives an MCP-bound token
    # instead of a general API token.
    pinned = str(client.get("resource_audience") or "").strip()
    if pinned and not resource:
        resource = pinned

    from oauth_service import client_scope_ceiling, is_connector_client

    allowed_scopes = list(client.get("scopes") or [])
    ceiling = client_scope_ceiling(client)
    requested = str(params.get("scope", "")).strip().split()
    if requested:
        invalid = [scope for scope in requested if ceiling and scope not in ceiling]
        if invalid and is_connector_client(client):
            # RFC 6749 §3.3 lets the server grant less than was asked; the
            # token response's `scope` says what was granted. A connector asks
            # for what discovery advertised (VS Code requests the
            # whole `scopes_supported`), and failing the sign-in over a scope it
            # can never hold left it with no way to connect at all.
            requested = [scope for scope in requested if scope not in invalid]
            invalid = []
        if invalid:
            raise _AuthorizeRejection(
                _oauth_browser_error(
                    request,
                    400,
                    "Unsupported permissions",
                    "This application asked for permissions it is not registered for: "
                    + ", ".join(sorted(set(invalid))),
                )
            )
        scopes = requested or allowed_scopes
    else:
        scopes = allowed_scopes
    # Signed-in last: a request for an unknown client, a bad redirect or a
    # foreign resource is refused before anyone is sent through login for it
    # (GX0 "an unresolvable metadata-document client id is rejected").
    user = _get_current_user(request)
    if not user:
        wants_html = "text/html" in request.headers.get("accept", "").lower()
        if wants_html or not request.headers.get("authorization"):
            redirect_target = _urllib_parse.quote(
                str(request.url.path) + (f"?{request.url.query}" if request.url.query else "")
            )
            raise _AuthorizeRejection(
                RedirectResponse(f"/login?redirect={redirect_target}", status_code=302)
            )
        raise HTTPException(401, "Not authenticated")
    auth_type = str(user.get("auth_type", ""))
    if auth_type == "client_credentials":
        raise HTTPException(403, "Interactive user authentication required")
    if auth_type == "api_key":
        raise HTTPException(403, "Interactive session authentication required")

    return user, client, sorted(set(scopes)), resource or ""


class _AuthorizeRejection(Exception):
    """Carries an already-rendered response out of shared authorize validation."""

    def __init__(self, response):
        super().__init__("authorization rejected")
        self.response = response


def _issue_code_and_redirect(
    *,
    request: Request,
    user: dict,
    client: dict,
    redirect_uri: str,
    state: str,
    code_challenge: str,
    code_challenge_method: str,
    scopes: list[str],
    resource: str,
    render_ui: bool = False,
):
    from db import ConsentStore, OAuthStore
    from fastapi.responses import HTMLResponse
    from oauth_registration import registration_expiry
    from oauth_service import is_loopback_redirect

    try:
        code = issue_authorization_code(
            client=client,
            user=user,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            scopes=scopes,
            audience=resource or OAUTH_AUDIENCE,
        )
    except OAuthGrantError as exc:
        raise HTTPException(exc.status_code, exc.description) from exc

    if not client.get("is_first_party"):
        user_id = str(user.get("user_id") or user.get("email") or "")
        try:
            ConsentStore.touch(user_id, str(client["client_id"]), resource)
        except Exception as exc:  # pragma: no cover - bookkeeping, never fatal
            log.warning("consent touch failed (non-fatal): %s", exc)
    if str(client.get("registration_source") or "") == "dcr":
        # A registration in active use must not expire out from under a working
        # connector, so every completed authorization renews its window.
        try:
            OAuthStore.touch_registration(str(client["client_id"]), registration_expiry())
        except Exception as exc:  # pragma: no cover - bookkeeping, never fatal
            log.warning("registration renewal failed (non-fatal): %s", exc)
    if render_ui:
        callback_url = _build_callback_url(redirect_uri, state, code=code)
        return HTMLResponse(
            content=_oauth_decision_page(
                approved=True,
                client=client,
                callback_url=callback_url,
                is_loopback=is_loopback_redirect(redirect_uri),
            )
        )
    return _redirect_with(redirect_uri, state, code=code)


@router.get("/oauth/authorize", tags=["Auth"])
def oauth_authorize(request: Request):
    from fastapi.responses import HTMLResponse
    from oauth_service import consent_covers, is_loopback_redirect, stash_consent_request

    params = request.query_params
    try:
        user, client, scopes, resource = _prepare_authorization(request, params)
    except _AuthorizeRejection as rejection:
        return rejection.response

    redirect_uri = str(params.get("redirect_uri", "")).strip()
    state = str(params.get("state", "")).strip()
    code_challenge = str(params.get("code_challenge", "")).strip()
    code_challenge_method = str(params.get("code_challenge_method", "S256")).strip()
    user_id = str(user.get("user_id") or user.get("email") or "")
    prompt_tokens = set(str(params.get("prompt", "")).lower().split())

    # First-party apps are us; a consent screen between our own login page and
    # our own dashboard would be theatre. Everything else — including the
    # pre-provisioned connector client and anything CIMD or DCR produced — asks.
    already_consented = "consent" not in prompt_tokens and (
        bool(client.get("is_first_party"))
        or consent_covers(user_id, str(client["client_id"]), resource, scopes)
    )
    if already_consented:
        is_browser_loopback = (
            not client.get("is_first_party")
            and is_loopback_redirect(redirect_uri)
            and request.headers.get("sec-fetch-mode", "").lower() == "navigate"
        )
        return _issue_code_and_redirect(
            request=request,
            user=user,
            client=client,
            redirect_uri=redirect_uri,
            state=state,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            scopes=scopes,
            resource=resource,
            render_ui=is_browser_loopback,
        )

    consent_key = stash_consent_request(
        {
            "user_id": user_id,
            "email": str(user.get("email") or ""),
            "client_id": str(client["client_id"]),
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
            "scopes": scopes,
            "resource": resource,
        }
    )
    return HTMLResponse(
        content=_oauth_consent_page(
            client=client,
            user=user,
            scopes=scopes,
            resource=resource,
            consent_key=consent_key,
        )
    )


@router.post("/oauth/authorize", tags=["Auth"])
async def oauth_authorize_decision(request: Request):
    """Record the user's decision on a staged authorization request."""
    from fastapi.responses import HTMLResponse
    from oauth_service import is_loopback_redirect, take_consent_request
    from oauth_registration import classify_surface, new_grant_id
    from db import ConsentStore

    body = await _parse_request_body(request)
    staged = take_consent_request(str(body.get("consent_key", "")).strip())
    if not staged:
        return _oauth_browser_error(
            request,
            400,
            "Request expired",
            "This approval screen is no longer valid. Start the connection again from your app.",
        )

    user = _get_current_user(request)
    user_id = str((user or {}).get("user_id") or (user or {}).get("email") or "")
    # The staged request belongs to one signed-in identity. A different session
    # submitting it would be consenting on someone else's behalf.
    if not user or user_id != str(staged.get("user_id") or ""):
        return _oauth_browser_error(
            request,
            403,
            "Sign-in changed",
            "The signed-in account changed while this approval was open. "
            "Start the connection again from your app.",
        )

    redirect_uri = str(staged.get("redirect_uri") or "")
    state = str(staged.get("state") or "")
    if not redirect_uri:  # pragma: no cover - staged requests always carry one
        return _oauth_browser_error(
            request,
            400,
            "Incomplete request",
            "This approval is missing its callback URL. Start the connection again from your app.",
        )
    wants_ui = (
        str(body.get("ui", "")).strip() == "1"
        or request.headers.get("sec-fetch-mode", "").lower() == "navigate"
    )
    if str(body.get("decision", "")).strip().lower() != "approve":
        if wants_ui:
            denied_client = get_client(str(staged.get("client_id") or "")) or {
                "client_name": staged.get("client_id") or "Application"
            }
            callback_url = _build_callback_url(
                redirect_uri,
                state,
                error="access_denied",
                error_description="The user declined the authorization request.",
            )
            return HTMLResponse(
                content=_oauth_decision_page(
                    approved=False,
                    client=denied_client,
                    callback_url=callback_url,
                    is_loopback=is_loopback_redirect(redirect_uri),
                )
            )
        return _redirect_with(
            redirect_uri,
            state,
            error="access_denied",
            error_description="The user declined the authorization request.",
        )

    client = get_client(str(staged.get("client_id") or ""))
    if not client:
        return _oauth_browser_error(
            request,
            400,
            "Unknown application",
            "That application is no longer registered. Start the connection again from your app.",
        )
    scopes = [str(scope) for scope in (staged.get("scopes") or [])]
    resource = str(staged.get("resource") or "")
    ConsentStore.record(
        grant_id=new_grant_id(),
        tenant_id=str(user.get("team_id") or user.get("customer_id") or user_id),
        user_id=user_id,
        email=str(user.get("email") or ""),
        client_id=str(client["client_id"]),
        scopes=scopes,
        resource=resource,
        # Attributable now, not later: the same connector client serves every
        # surface, so the callback host is the only thing that says which
        # directory produced this activation (X7.34).
        surface=classify_surface(redirect_uri, str(client["client_id"])),
    )
    from routes._deps import append_user_audit_event

    append_user_audit_event(
        "user.oauth.consent_granted",
        "oauth_client",
        str(client["client_id"]),
        user,
        data={
            "scopes": scopes,
            "resource": resource,
            "surface": classify_surface(redirect_uri, str(client["client_id"])),
        },
    )
    return _issue_code_and_redirect(
        request=request,
        user=user,
        client=client,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=str(staged.get("code_challenge") or ""),
        code_challenge_method=str(staged.get("code_challenge_method") or "S256"),
        scopes=scopes,
        resource=resource,
        render_ui=wants_ui,
    )


@router.post("/oauth/revoke", tags=["Auth"], include_in_schema=False)
async def oauth_revoke(request: Request):
    """RFC 7009. Revokes one presented token and always answers success.

    A client-credentials JWT is recorded by its `jti` until its own expiry.
    An opaque access token is removed from the auth cache. Callers who do not
    hold the token cannot revoke it here.
    """
    body = await _parse_request_body(request)
    token = str((body or {}).get("token") or "").strip()
    if token.count(".") == 2:
        revoke_client_credentials_jwt(token)
    elif token:
        revoke_access_token(token)
    return JSONResponse({"ok": True})


@router.post("/oauth/token", tags=["Auth"])
async def oauth_token(request: Request):
    # `_parse_request_body` accepts both `application/x-www-form-urlencoded`
    # (what RFC 6749 §4.1.3 mandates and what every connector client sends) and
    # JSON. A JSON-only token endpoint is a documented silent-failure cause for
    # connector onboarding; `test_token_endpoint_accepts_form_encoding` pins it.
    body = await _parse_request_body(request)
    return await _oauth_token_grant_response(request, body)


@router.post("/oauth/register", tags=["Auth"], status_code=201)
async def oauth_dynamic_client_registration(request: Request):
    """RFC 7591 dynamic client registration — the compatibility path.

    Open (no initial access token) because Microsoft Copilot Studio and several
    other clients discover this endpoint from authorization-server metadata and
    register unattended. Every gate that makes that safe lives in
    `oauth_registration.register_dynamic_client`: allowlisted redirect hosts,
    MCP-audience-only clients, read-biased default scopes, no operator scopes,
    per-IP and global rate limits, and expiry for registrations never used.
    """
    from oauth_registration import register_dynamic_client

    body = await _parse_request_body(request)
    try:
        registration = register_dynamic_client(
            body, client_ip=_deps_get_real_client_ip(request) or ""
        )
    except OAuthGrantError as exc:
        return _oauth_error_response(exc)
    log.info(
        "oauth dcr registration client_id=%s redirects=%s",
        registration["client_id"],
        len(registration["redirect_uris"]),
    )
    return JSONResponse(content=registration, status_code=201)


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
            audience=_validated_resource_indicator(body),
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
            audience=_validated_resource_indicator(body),
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


#: The shape issue_device_authorization gives user codes: XXXX-XXXX, hex.
_DEVICE_USER_CODE = re.compile(r"[0-9A-F]{4}-[0-9A-F]{4}")


@router.get("/api/auth/verify", tags=["Auth"], deprecated=True)
def oauth_verify_page(request: Request, user_code: str = ""):
    """Send browsers to the website's device sign-in page.

    This was the page itself: one line telling people to submit the code with
    `POST /api/auth/verify`. Wizards up to 0.1.5 still open this URL, so it
    redirects, carrying the code along when it looks like one.
    """
    code = user_code.strip().upper()
    target = device_verification_url(
        str(request.base_url), code if _DEVICE_USER_CODE.fullmatch(code) else None
    )
    return RedirectResponse(target, status_code=302)


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


def _safe_oauth_client_payload(client: dict) -> dict:
    team_id = str(client.get("team_id") or "").strip() or None
    return {
        "client_id": client["client_id"],
        "client_name": client["client_name"],
        "client_type": client["client_type"],
        "redirect_uris": list(client.get("redirect_uris") or []),
        "grant_types": list(client.get("grant_types") or []),
        "scopes": list(client.get("scopes") or []),
        "is_first_party": bool(client.get("is_first_party")),
        "created_by_email": client.get("created_by_email"),
        "status": client.get("status", "active"),
        "workspace_customer_id": client.get("workspace_customer_id"),
        "team_id": team_id,
        "workspace_label": "team" if team_id else "personal",
        "created_at": client.get("created_at"),
        "updated_at": client.get("updated_at"),
        "last_used": client.get("last_used"),
        "client_secret_preview": client.get("client_secret_preview"),
    }


# ── Operator-scope refusal (stopgap) ──────────────────────────────────
#
# Scopes reached the `oauth_clients.scopes` column with no check that the
# caller was entitled to delegate them. A machine principal is authorized on
# its scope alone (`control_plane_v1._require_host_operator`), so any principal
# able to write this column could obtain platform-operator authority.
#
# This is deliberately blunt: a literal list and a refusal, applied at the two
# routes that write the column. It depends on nothing outside this module so it
# can ship on its own.
#
# The durable fix puts the check on the three `OAuthStore` methods that persist
# the column, so every caller inherits it instead of each route remembering —
# there are five paths to those three methods, and one of them bypasses the
# service-layer funnel entirely. See #16. **Delete this block when that lands;
# do not build on it.**
_OPERATOR_SCOPES_STOPGAP = frozenset(
    {
        "control_plane:operate",
        "control_plane:read",
        "hosts:evict",
        "hosts:fleet",
        "hosts:operate",
        "transparency:read",
        "transparency:write",
    }
)


def _refuse_undelegatable_scopes(scopes: list[str] | None, user: dict) -> None:
    """Refuse operator scopes to a principal that is not a platform admin.

    `scopes is None` means the request did not mention scopes — a PATCH that
    only renames a client, say — and leaves the stored value untouched.
    """
    if scopes is None or _is_platform_admin(user):
        return
    refused = sorted(set(scopes) & _OPERATOR_SCOPES_STOPGAP)
    if refused:
        raise HTTPException(
            403,
            "These scopes may only be granted by a platform administrator: " + ", ".join(refused),
        )


@router.post("/api/oauth/clients", tags=["Auth"])
def api_create_oauth_client(body: OAuthClientCreateRequest, request: Request):
    user = _require_user_grant(request)
    _require_team_instance_write(user)
    if body.client_type not in {"public", "confidential"}:
        raise HTTPException(400, "client_type must be public or confidential")
    if body.is_first_party and not _is_platform_admin(user):
        raise HTTPException(403, "Only admins can create first-party OAuth clients")
    _refuse_undelegatable_scopes(list(body.scopes), user)
    workspace_customer_id = _oauth_workspace_customer_id(user)
    team_id = _oauth_workspace_team_id(user)
    client = create_oauth_client(
        actor=user,
        client_name=body.client_name.strip(),
        redirect_uris=list(body.redirect_uris),
        grant_types=list(body.grant_types),
        scopes=list(body.scopes),
        created_by_email=user["email"],
        client_type=body.client_type,
        is_first_party=body.is_first_party,
        workspace_customer_id=workspace_customer_id,
        team_id=team_id,
    )
    from routes._deps import append_user_audit_event

    append_user_audit_event(
        "user.oauth.client_created",
        "oauth_client",
        client["client_id"],
        user,
        data={"client_name": client.get("client_name")},
    )
    payload: dict = {"ok": True, "client": client}
    if client.get("client_secret"):
        payload["client_secret"] = client["client_secret"]
    return payload


@router.get("/api/oauth/clients", tags=["Auth"])
def api_list_oauth_clients(request: Request):
    user = _require_user_grant(request)
    from db import OAuthStore

    if _is_platform_admin(user):
        clients = OAuthStore.list_clients(None)
    else:
        clients = OAuthStore.list_clients(
            user["email"],
            workspace_customer_id=_oauth_workspace_customer_id(user),
            personal_customer_id=_canonical_owner_id(user),
        )
        clients = [c for c in clients if _user_can_access_oauth_client(user, c)]
    # System-managed clients (e.g. the /dashboard/mcp quick-connect client) are
    # provisioned automatically and shouldn't clutter the user's manual client list.
    clients = [c for c in clients if not c.get("is_system_managed")]
    safe_clients = [_safe_oauth_client_payload(client) for client in clients]
    return {
        "ok": True,
        "clients": safe_clients,
        "workspace_customer_id": _oauth_workspace_customer_id(user),
        "team_id": _oauth_workspace_team_id(user),
    }


@router.get("/api/mcp/quick-connect", tags=["Auth"])
def api_mcp_quick_connect(
    request: Request, regenerate: bool = False, surface: Literal["mcp", "cli"] = "mcp"
):
    """Return a dedicated MCP or CLI API key, in full on every visit.

    Each surface has its own client, audience and independently revocable key.
    These keys are kept encrypted so the page can show them again: showing a
    masked prefix after the first visit pushed people to regenerate, which
    revokes the key their agent is already using. Keys minted before that
    (no ciphertext) stay masked until regenerated once.
    """
    user = _require_user_grant(request)
    from oauth_service import (
        MCP_QUICK_CONNECT_SCOPES,
        get_or_create_mcp_quick_connect_client,
    )

    from db import auth_transaction

    # Serialize creation/rotation for this identity and surface across workers.
    # Client replacement, old-key revocation and minting commit together.
    lock_key = (
        f"quick-connect:{user['email']}:{_oauth_workspace_team_id(user) or 'personal'}:{surface}"
    )
    with auth_transaction(lock_key):
        client = get_or_create_mcp_quick_connect_client(
            created_by_email=user["email"],
            workspace_customer_id=_oauth_workspace_customer_id(user),
            team_id=_oauth_workspace_team_id(user),
            regenerate=regenerate,
            surface=surface,
        )
        from db import AgentKeyStore
        from oauth_service import (
            MCP_RESOURCE_AUDIENCE,
            OAUTH_AUDIENCE,
            issue_agent_api_key,
            reveal_agent_api_key,
        )

        scopes = list(client.get("scopes") or MCP_QUICK_CONNECT_SCOPES)
        user_id = str(user.get("user_id") or user.get("email") or "")
        client_id = client["client_id"]

        replaced_client_id = client.get("replaced_client_id")
        if replaced_client_id:
            AgentKeyStore.revoke_all_for_client(user_id, replaced_client_id)

        live = [k for k in AgentKeyStore.list_for_user(user_id) if k["client_id"] == client_id]
        existing = live[0] if live else None

        # A copied key may not have been used yet. Reads never invalidate it;
        # only an explicit regeneration replaces an existing credential.
        mint = existing is None or regenerate
        bundle = None
        if mint:
            bundle = issue_agent_api_key(
                user=user,
                client_id=client_id,
                scopes=scopes,
                audience=MCP_RESOURCE_AUDIENCE if surface == "mcp" else OAUTH_AUDIENCE,
                name="MCP Quick Connect" if surface == "mcp" else "CLI Skill",
                replace_existing=True,
                retrievable=True,
            )

    # `XCELSIOR_BASE_URL` is the plumbed name for this. `XCELSIOR_PUBLIC_URL`
    # was read here and **mapped in no compose file**, so the container never
    # received it and the literal fallback always won. On production that is
    # invisible, because the fallback *is* production — which is exactly why it
    # would have surfaced on staging first, handing a staging user a Quick
    # Connect config pointing at prod. `.env.staging` already sets
    # `XCELSIOR_BASE_URL`, so reading it fixes staging with a value that is
    # already there.
    #
    # `XCELSIOR_PUBLIC_URL` still wins when explicitly set, so any deployment
    # that has been setting it keeps working; it is no longer the *only* name
    # that works.
    mcp_base = (
        os.environ.get("XCELSIOR_PUBLIC_URL")
        or os.environ.get("XCELSIOR_BASE_URL")
        or "https://xcelsior.ca"
    ).rstrip("/")
    return {
        "ok": True,
        "client_id": client_id,
        # The canonical connector URL, not `${api}/mcp`: an agent key is bound
        # to the MCP resource, so a config pointing anywhere else authenticates
        # against a resource the token was never issued for.
        # Absent only for a key minted before keys were kept encrypted: the
        # UI then shows the masked prefix instead of a token it cannot produce.
        "access_token": (
            bundle["access_token"]
            if bundle
            else (reveal_agent_api_key(existing) if existing else None)
        ),
        "key_id": bundle["key_id"] if bundle else (existing["key_id"] if existing else None),
        "key_prefix": (
            bundle["key_prefix"] if bundle else (existing["key_prefix"] if existing else None)
        ),
        # A freshly minted key has never authenticated anything, so it is never
        # "in use" — that only becomes true once the agent actually calls.
        # NULL last_used_at means never used (companion §4.4.5 timestamps), so
        # "in use" is simply whether it has ever authenticated a request.
        "in_use": bool(not bundle and existing and existing["last_used_at"] is not None),
        "last_used_at": (
            None
            if bundle or not existing or existing["last_used_at"] is None
            else existing["last_used_at"].isoformat()
        ),
        # These keys do not expire; they are revoked. Reporting an expiry would
        # make clients schedule a refresh that has nothing to refresh.
        "expires_in": None,
        "scopes": scopes,
        "mcp_url": MCP_RESOURCE_AUDIENCE,
        "api_url": mcp_base,
    }


@router.get("/api/agent-keys", tags=["Auth"])
def api_list_agent_keys(request: Request):
    """List the caller's agent keys for the Settings management view."""
    user = _require_user_grant(request)
    from db import AgentKeyStore

    user_id = str(user.get("user_id") or user.get("email") or "")
    return {
        "ok": True,
        "keys": [
            {
                "key_id": k["key_id"],
                "name": k["name"],
                "key_prefix": k["key_prefix"],
                "client_id": k["client_id"],
                "scopes": (k["scopes"] or "").split(),
                # ISO 8601, matching /api/mcp/quick-connect. Serialised here
                # rather than left to FastAPI so both endpoints agree.
                "created_at": k["created_at"].isoformat(),
                "last_used_at": (
                    k["last_used_at"].isoformat() if k["last_used_at"] is not None else None
                ),
                # NULL means the key has never authenticated a request, so it
                # cannot be in anyone's config yet.
                "in_use": k["last_used_at"] is not None,
            }
            for k in AgentKeyStore.list_for_user(user_id)
        ],
    }


@router.delete("/api/agent-keys/{key_id}", tags=["Auth"])
def api_revoke_agent_key(key_id: str, request: Request):
    """Revoke a key immediately. Scoped to the owner."""
    user = _require_user_grant(request)
    from db import AgentKeyStore

    user_id = str(user.get("user_id") or user.get("email") or "")
    if not AgentKeyStore.revoke(key_id, user_id):
        raise HTTPException(404, "Key not found")
    return {"ok": True, "revoked": key_id}


class AgentKeyRename(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


@router.patch("/api/agent-keys/{key_id}", tags=["Auth"])
def api_rename_agent_key(key_id: str, body: AgentKeyRename, request: Request):
    """Rename a key so several are tellable apart in Settings."""
    user = _require_user_grant(request)
    from db import AgentKeyStore

    user_id = str(user.get("user_id") or user.get("email") or "")
    if not AgentKeyStore.rename(key_id, user_id, body.name.strip()):
        raise HTTPException(404, "Key not found")
    return {"ok": True, "key_id": key_id, "name": body.name.strip()}


@router.delete("/api/oauth/clients/{client_id}", tags=["Auth"])
def api_delete_oauth_client(client_id: str, request: Request):
    user = _require_user_grant(request)
    from db import OAuthStore

    if _is_platform_admin(user):
        deleted = OAuthStore.delete_client(client_id, None)
    else:
        client = OAuthStore.get_client(client_id)
        if not client:
            raise HTTPException(404, "OAuth client not found")
        _require_oauth_client_write(user, client)
        deleted = OAuthStore.delete_client_in_workspace(
            client_id,
            workspace_customer_id=_oauth_workspace_customer_id(user),
            personal_customer_id=_canonical_owner_id(user),
            user_email=user["email"],
        )
    if not deleted:
        raise HTTPException(404, "OAuth client not found")
    from routes._deps import append_user_audit_event

    append_user_audit_event("user.oauth.client_deleted", "oauth_client", client_id, user)
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

    # OAuth-only account — registering never claims it. Knowing the address
    # proves nothing; the emailed reset link is the path that adds a password.
    if existing_user:
        provider = existing_user.get("oauth_provider", "OAuth")
        return JSONResponse(
            status_code=409,
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

    password_hash, salt = _hash_password(body.password)

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

    try:
        from meta_capi import track_event

        track_event(
            "CompleteRegistration",
            f"reg_{user_id}",
            user=user,
            request=request,
            custom_data={"status": "registered"},
        )
    except Exception:
        log.debug("meta registration event skipped", exc_info=True)

    # Telegram alert for new signup
    try:
        alert("New Signup", f"{user['name']} ({email}) — role: {user.get('role', 'submitter')}")
    except Exception as e:
        log.debug("signup alert failed: %s", e)

    # Welcome notification for the new user. Providers get an expectations-first
    # message (weekly uptime goal + how earning works + the don't-drop-mid-job
    # rule) so the guidance is waiting in their inbox even before the email lands.
    try:
        if requested_role == "provider":
            from provider_expectations import provider_welcome_notification

            _notif_title, _notif_body = provider_welcome_notification()
        else:
            _notif_title = "Welcome to Xcelsior!"
            _notif_body = (
                "Welcome to your Notifications Inbox! This is your new go-to destination "
                "for important account updates and personalized recommendations. Check back "
                "here to stay informed and maximize your Xcelsior experience!"
            )
        NotificationStore.create(
            email,
            "system",
            _notif_title,
            _notif_body,
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
            "Thanks for signing up for Xcelsior — the global GPU compute marketplace.\n\n"
            "Please verify your email address by clicking the button below. This link expires in 24 hours.\n\n"
            f"{verify_url}\n\n"
            "If you didn't create this account, you can safely ignore this email.",
            cta_url=verify_url,
            cta_label="Verify Email",
        )
    except Exception as e:
        # `log.warning`, not debug: production does not emit debug, so a user
        # got "check your email" while the failure left no trace anywhere. This
        # is a user-visible failure, not diagnostic detail.
        log.warning("verification email send failed for %s: %s", email, e)

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


@router.get("/api/auth/demo-credentials", tags=["Auth"])
def api_auth_demo_credentials(request: Request):
    """Reveal the standing demo-account credentials, but ONLY to whitelisted IPs
    (the owner's public IP, Tailscale nodes, home LAN, localhost). The login page
    calls this on mount: 200 → show the one-click "Demo account" button and
    prefill the fields; anything else → hide it. See demo_account.py for the
    whitelist and the security caveats. Kill switch: XCELSIOR_DEMO_ENABLED=0."""
    from demo_account import (
        DEMO_EMAIL,
        DEMO_PASSWORD,
        demo_enabled,
        is_ip_whitelisted,
    )

    if not demo_enabled():
        raise HTTPException(404, "Not found")
    client_ip = _deps_get_real_client_ip(request)
    if not is_ip_whitelisted(client_ip):
        # 403, not 404: the endpoint exists, this network just isn't allowed.
        raise HTTPException(403, "Demo login is not available from this network.")
    return {"email": DEMO_EMAIL, "password": DEMO_PASSWORD}


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


class SocialOAuthInitBody(BaseModel):
    redirect: str = "/dashboard"


@router.get("/api/auth/oauth/providers", tags=["Auth"])
def oauth_providers():
    """Sign-in providers configured here, in display order.

    The login and register pages showed every provider they knew of, so the
    Facebook button (no app configured) answered every click with "OAuth
    provider facebook is not configured". They now show only these.
    """
    return {
        "ok": True,
        "providers": [name for name, cfg in _OAUTH_PROVIDERS.items() if cfg["client_id"]],
    }


@router.post("/api/auth/oauth/{provider}", tags=["Auth"])
def api_auth_oauth_initiate(provider: str, body: SocialOAuthInitBody | None = None):
    """Initiate OAuth flow — returns the provider's authorization URL.

    The frontend should redirect the user to the returned URL.
    """
    if provider not in _OAUTH_PROVIDERS:
        raise HTTPException(400, f"Unsupported OAuth provider: {provider}")

    cfg = _OAUTH_PROVIDERS[provider]
    if not cfg["client_id"]:
        raise HTTPException(503, f"OAuth provider {provider} is not configured")

    # Generate CSRF state token — stored in shared auth cache (Redis in prod).
    state = secrets.token_urlsafe(32)
    redirect_path = _sanitize_oauth_redirect((body or SocialOAuthInitBody()).redirect)
    store_social_oauth_state(
        state,
        provider=provider,
        redirect_path=redirect_path,
        ttl=_OAUTH_STATE_TTL,
    )

    redirect_uri = f"{_OAUTH_BASE_URL}/api/auth/oauth/{provider}/callback"
    params = {
        "client_id": cfg["client_id"],
        "redirect_uri": redirect_uri,
        "state": state,
        "response_type": "code",
        "scope": cfg["scopes"],
    }
    # A saved browser session otherwise skips the account picker. GitHub and
    # Google accept prompt=select_account. Hugging Face is OIDC and needs
    # prompt=login. Facebook uses auth_type=reauthenticate.
    if provider == "google":
        params["access_type"] = "offline"
        params["prompt"] = "select_account"
    elif provider == "github":
        params["prompt"] = "select_account"
    elif provider == "huggingface":
        params["prompt"] = "login"
    elif provider == "facebook":
        params["auth_type"] = "reauthenticate"

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

    # Validate CSRF state (shared cache — survives multi-worker deploys)
    state_data = consume_social_oauth_state(state, ttl=_OAUTH_STATE_TTL)
    if not state_data or state_data["provider"] != provider:
        return RedirectResponse("/login?error=oauth_invalid_state")
    redirect_path = _sanitize_oauth_redirect(state_data.get("redirect_path"))

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
    elif provider == "facebook":
        email = profile.get("email", "")
        name = profile.get("name", "")
        facebook_subject = str(profile.get("id") or "").strip()
    else:
        facebook_subject = ""

    if provider != "facebook":
        facebook_subject = ""

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
            "oauth_subject": facebook_subject,
            "created_at": time.time(),
        }
        if _USE_PERSISTENT_AUTH:
            UserStore.create_user(user)
            # OAuth users are auto-verified (provider verified their email)
            verified: dict[str, Any] = {"email_verified": 1}
            if facebook_subject:
                verified["oauth_subject"] = facebook_subject
            UserStore.update_user(email, verified)
        else:
            user["email_verified"] = 1
            with _user_lock:
                _users_db[email] = user
    else:
        # Update name if it was empty
        if not user.get("name") and name:
            user["name"] = name
        # Always update oauth_provider on login so we track the most recent.
        #
        # `email_verified` is set here for the same reason the new-account branch
        # sets it: the provider just verified this address. Without it an account
        # that registered by password and never verified stays at 0 forever —
        # refused on the password door with "please verify your email" while
        # signing in fine through this one. The flag was not a stale cache of a
        # fact; it was a fact the system had just been told and discarded.
        updates = {"oauth_provider": provider, "email_verified": 1}
        if facebook_subject:
            updates["oauth_subject"] = facebook_subject
        if _USE_PERSISTENT_AUTH:
            UserStore.update_user(email, updates)
        else:
            user.update(updates)
            with _user_lock:
                _users_db[email] = user

    mfa_challenge_id = ""
    if _USE_PERSISTENT_AUTH:
        try:
            enabled_methods = [m for m in MfaStore.list_methods(email) if m.get("enabled")]
        except Exception:
            log.exception("OAuth MFA lookup failed for provider %s", provider)
            return RedirectResponse("/login?error=oauth_mfa_unavailable")
        if enabled_methods:
            mfa_challenge_id = secrets.token_urlsafe(32)
            MfaStore.create_challenge(
                {
                    "challenge_id": mfa_challenge_id,
                    "email": email,
                    "session_token": secrets.token_urlsafe(48),
                    "created_at": time.time(),
                    "expires_at": time.time() + 300,
                }
            )

    if mfa_challenge_id:
        # Do not issue a browser session until the enrolled second factor passes.
        # Clear any leftover session so /login does not bounce past the prompt.
        dest = "/login?mfa=1&redirect=" + _urllib_parse.quote(redirect_path, safe="")
        resp = RedirectResponse(dest, status_code=302)
        _deps_clear_auth_cookie(resp)
        _set_mfa_challenge_cookie(resp, mfa_challenge_id)
    else:
        try:
            token_bundle = issue_user_tokens(
                user, request, client_id="xcelsior-web", session_type="browser"
            )
        except Exception as e:
            log.error("OAuth token issuance failed for %s (%s): %s", email, provider, e)
            return RedirectResponse("/dashboard?error=oauth_token_failed")

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

        resp = RedirectResponse(redirect_path, status_code=302)
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
    # Same rule as the session cookies in `routes/_deps.py`: the domain follows
    # the configured origin, not the scheme. Hardcoding `.xcelsior.ca` for any
    # https base meant a staging or preview deployment set a cookie for a domain
    # it is not on, which the browser discards — here that silently loses the
    # "which provider did you last use" hint rather than a session, so it would
    # have gone unnoticed indefinitely.
    from routes._deps import _auth_cookie_domain

    _oauth_domain = _auth_cookie_domain(_base)
    if _oauth_domain:
        _oauth_kw["domain"] = _oauth_domain
    resp.set_cookie(**_oauth_kw)
    return resp


# ── Facebook signed_request ───────────────────────────────────────────────
#
# Facebook signs the deauthorize and data-deletion callbacks with the app
# secret. Both callbacks had their own copy of the verification, and both
# copies had the same defect: the `HTTPException` raised on a signature
# mismatch was thrown *inside* a `try` whose `except Exception` sat two lines
# below it. `HTTPException` is an `Exception`, so the refusal was caught by the
# handler's own error path, logged as a parse failure, and the request went on
# to return `{"status": "success"}`. A forged signature was accepted exactly
# like a valid one, in production, with the secret correctly configured.
#
# Someone had already found and fixed the *other* bug in this block — the env
# check that only refused on a literal "production" — and that fix could never
# have taken effect. A test that forges a signature is the only thing that
# distinguishes the two states, and there was none.
#
# One copy now, because two copies is how one of them stays broken.


def _facebook_signed_request(signed_request: str, *, purpose: str) -> dict[str, Any]:
    """Verify Facebook's `signed_request` and return its payload.

    Raises `HTTPException(400)` on anything that does not verify. Callers must
    not wrap this in a bare `except Exception` — that is the original bug.

    Relaxed environments keep the permissive behaviour: local development has
    no app secret and needs to be able to exercise the route.
    """
    import base64
    import hashlib
    import hmac
    import json as _json

    relaxed = env_config.is_relaxed_env()

    def _refuse(reason: str) -> None:
        log.warning("Facebook %s signed_request rejected: %s", purpose, reason)
        if not relaxed:
            raise HTTPException(400, "Invalid signed request")

    if not signed_request or "." not in signed_request:
        _refuse("missing or malformed")
        return {}

    client_secret = _OAUTH_PROVIDERS.get("facebook", {}).get("client_secret")
    if not client_secret:
        # Fail closed. Without the secret there is nothing to verify against,
        # so accepting the payload means accepting anyone's — and this callback
        # names a user whose data is to be deleted.
        _refuse("FACEBOOK_CLIENT_SECRET is not configured, so nothing can be verified")
        return {}

    def _b64url(value: str) -> bytes:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))

    encoded_sig, payload = signed_request.split(".", 1)
    try:
        sig = _b64url(encoded_sig)
        data = _json.loads(_b64url(payload).decode("utf-8"))
    except (ValueError, TypeError) as exc:
        # Deliberately not `except Exception`: that is the construct that made
        # the refusal below unreachable. `binascii.Error`, `JSONDecodeError`
        # and `UnicodeDecodeError` are all `ValueError`, so this covers every
        # way the two decodes can fail without also covering `HTTPException`.
        _refuse(f"undecodable: {exc}")
        return {}

    expected = hmac.new(
        client_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).digest()
    if not hmac.compare_digest(sig, expected):
        _refuse("signature mismatch")
        return {}

    if not isinstance(data, dict):
        _refuse("payload is not an object")
        return {}
    return data


async def _facebook_signed_request_from(request: Request, *, purpose: str) -> dict[str, Any]:
    """Pull `signed_request` from the form body, falling back to the query."""
    signed_request = ""
    try:
        form = await request.form()
        raw_signed = form.get("signed_request", "")
        # form.get() may hand back an UploadFile; a signed request is text.
        signed_request = raw_signed if isinstance(raw_signed, str) else ""
    except Exception:
        pass
    if not signed_request:
        signed_request = request.query_params.get("signed_request", "")
    return _facebook_signed_request(signed_request, purpose=purpose)


def _public_base_url() -> str:
    return os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca").rstrip("/")


def _deletion_code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _iso(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _user_for_facebook(fb_user_id: str) -> dict | None:
    fb_user_id = str(fb_user_id or "").strip()
    if not fb_user_id:
        return None
    if _USE_PERSISTENT_AUTH:
        return UserStore.find_by_oauth_subject(fb_user_id)
    with _user_lock:
        for user in _users_db.values():
            if str(user.get("oauth_subject") or "") == fb_user_id:
                return user
    return None


def _unlink_facebook_user(fb_user_id: str) -> None:
    fb_user_id = str(fb_user_id or "").strip()
    if not fb_user_id:
        return
    if _USE_PERSISTENT_AUTH:
        UserStore.clear_oauth_subject(fb_user_id)
        return
    with _user_lock:
        for user in _users_db.values():
            if str(user.get("oauth_subject") or "") != fb_user_id:
                continue
            user["oauth_subject"] = ""
            if user.get("oauth_provider") == "facebook":
                user["oauth_provider"] = ""


def _remember_confirmation(code: str, request_id: str | None, outcome: str) -> None:
    digest = _deletion_code_hash(code)
    if not _USE_PERSISTENT_AUTH:
        _memory_deletion_confirmations[digest] = {
            "request_id": request_id,
            "outcome": outcome,
        }
        return
    UserStore.save_facebook_deletion_confirmation(digest, request_id, outcome)


def _confirmation_for_code(code: str) -> dict | None:
    digest = _deletion_code_hash(code)
    if not _USE_PERSISTENT_AUTH:
        return _memory_deletion_confirmations.get(digest)
    return UserStore.get_facebook_deletion_confirmation(digest)


def _remember_quietly(code: str, request_id: str | None, outcome: str) -> None:
    try:
        _remember_confirmation(code, request_id, outcome)
    except Exception:
        log.exception("Facebook deletion confirmation was not stored")


def _perform_facebook_deletion(fb_user_id: str) -> dict[str, str]:
    """Start durable deletion when this Facebook id matches an account.

    The confirmation code is the deletion status token when a request exists,
    so /delete-data can open that request without a session. A callback for an
    unknown id still gets a code. Nothing is deleted in that case.
    """
    code = secrets.token_urlsafe(32)
    try:
        user = _user_for_facebook(fb_user_id)
    except Exception:
        log.exception("Facebook deletion lookup failed")
        _remember_quietly(code, None, "unavailable")
        return {"code": code, "request_id": ""}

    if not user or not user.get("user_id") or not user.get("email"):
        _remember_quietly(code, None, "no_account")
        return {"code": code, "request_id": ""}

    try:
        from privacy_deletion import create_deletion_request

        customer_id = str(user.get("customer_id") or "").strip()
        receipt = create_deletion_request(
            user_id=str(user["user_id"]),
            email=str(user["email"]),
            customer_ids=[customer_id] if customer_id else [],
            idempotency_key=f"facebook-{fb_user_id}"[:128],
            requested_by="facebook-data-deletion",
            request_source="facebook_deletion",
        )
    except Exception:
        log.exception("Facebook deletion request failed")
        _remember_quietly(code, None, "unavailable")
        return {"code": code, "request_id": ""}

    code = receipt.status_token
    request_id = str(receipt.request_id)
    _remember_quietly(code, request_id, "started")
    return {"code": code, "request_id": request_id}


def _facebook_deletion_response(fb_user_id: str) -> dict[str, str]:
    from urllib.parse import quote

    base = _public_base_url()
    if not fb_user_id:
        code = secrets.token_urlsafe(32)
        _remember_quietly(code, None, "no_account")
        return {
            "url": f"{base}/delete-data?code={quote(code)}",
            "confirmation_code": code,
        }
    result = _perform_facebook_deletion(fb_user_id)
    code = result["code"]
    url = f"{base}/delete-data?code={quote(code)}"
    if result.get("request_id"):
        url += f"&request={quote(result['request_id'])}"
    return {"url": url, "confirmation_code": code}


def _public_deletion_body(request_id: str, code: str) -> dict[str, Any]:
    from privacy_deletion import (
        PrivacyDeletionAccessDenied,
        PrivacyDeletionNotFound,
        get_deletion_status,
    )

    try:
        status = get_deletion_status(request_id, status_token=code)
    except (PrivacyDeletionNotFound, PrivacyDeletionAccessDenied) as exc:
        raise HTTPException(404, "Deletion confirmation was not found") from exc
    return {
        "ok": True,
        "outcome": "started",
        "request_id": status["request_id"],
        "state": status["state"],
        "deadline_at": _iso(status.get("deadline_at")),
        "completed_at": _iso(status.get("completed_at")),
    }


# In-memory receipts for tests and the non-persistent auth mode. Production
# writes facebook_deletion_confirmations instead.
_memory_deletion_confirmations: dict[str, dict[str, Any]] = {}


@router.post("/api/auth/oauth/facebook/deauthorize", tags=["Auth"])
async def facebook_deauthorize(request: Request):
    """Facebook Deauthorize Callback.

    Called when a person removes Xcelsior from their Facebook account. This
    detaches the Facebook id. It does not delete the Xcelsior account. Data
    deletion is the separate callback.
    """
    log.info("Facebook deauthorize request received")
    data = await _facebook_signed_request_from(request, purpose="deauthorize")
    fb_user_id = str(data.get("user_id") or "").strip()
    if fb_user_id:
        try:
            _unlink_facebook_user(fb_user_id)
        except Exception:
            log.exception("Facebook deauthorize could not detach the account")
            raise HTTPException(500, "Facebook deauthorize could not be recorded") from None
    return {"status": "success", "action": "deauthorized"}


@router.post("/api/auth/oauth/facebook/delete-data", tags=["Auth"])
async def facebook_delete_data(request: Request):
    """Facebook Data Deletion Callback.

    Meta posts a signed_request here. The JSON response is the confirmation
    code and the public status page. https://xcelsior.ca/delete-data posts to
    this same handler.
    """
    log.info("Facebook data-deletion request received")
    data = await _facebook_signed_request_from(request, purpose="delete-data")
    fb_user_id = str(data.get("user_id") or "").strip()
    return _facebook_deletion_response(fb_user_id)


@router.get("/api/privacy/deletion-status", tags=["Auth"])
def api_privacy_deletion_status(
    request: Request,
    code: str = "",
    request_id: str = Query("", alias="request"),
):
    """Public status for a deletion confirmation code. No session required."""
    _check_auth_rate_limit(request)
    code = str(code or "").strip()
    request_id = str(request_id or "").strip()
    if not code or len(code) > 200 or len(request_id) > 80:
        raise HTTPException(400, "A confirmation code is required")
    if request_id:
        return _public_deletion_body(request_id, code)
    try:
        remembered = _confirmation_for_code(code)
    except Exception:
        log.exception("Deletion confirmation lookup failed")
        raise HTTPException(404, "Deletion confirmation was not found") from None
    if not remembered:
        raise HTTPException(404, "Deletion confirmation was not found")
    stored_request = str(remembered.get("request_id") or "").strip()
    if stored_request:
        return _public_deletion_body(stored_request, code)
    return {
        "ok": True,
        "outcome": remembered.get("outcome") or "no_account",
        "request_id": None,
        "state": None,
        "deadline_at": None,
        "completed_at": None,
    }


@router.get("/api/auth/me", tags=["Auth"])
def api_auth_me(request: Request):
    """Get the currently authenticated user's profile.

    Returns ``user: null`` when no session is present (HTTP 200) so public
    marketing pages can probe session state without console 401 noise.
    Invalid or machine credentials still return 401/403.
    """
    from routes._deps import _AUTH_COOKIE_NAME, _get_current_user

    auth = request.headers.get("authorization", "")
    bearer = auth[7:].strip() if auth.startswith("Bearer ") else ""
    cookie = request.cookies.get(_AUTH_COOKIE_NAME, "")
    if not bearer and not cookie:
        return {"ok": True, "user": None}

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    if user.get("auth_type") == "client_credentials":
        raise HTTPException(403, "Interactive user authentication required")

    email = user["email"]
    if _USE_PERSISTENT_AUTH:
        full_user = UserStore.get_user(email) or {}
    else:
        with _user_lock:
            full_user = _users_db.get(email, {})

    merged_user = {
        "user_id": user["user_id"],
        "email": email,
        "name": full_user.get("name", user.get("name", "")),
        "role": full_user.get("role", user.get("role", "submitter")),
        "is_admin": True if _is_platform_admin(full_user or user) else False,
        "customer_id": full_user.get("customer_id", ""),
        "provider_id": full_user.get("provider_id"),
        "country": full_user.get("country", "CA"),
        "province": full_user.get("province", "ON"),
        "business_name": full_user.get("business_name") or "",
        "gst_hst_number": full_user.get("gst_hst_number") or "",
        "team_id": full_user.get("team_id"),
        "oauth_provider": full_user.get("oauth_provider"),
        "created_at": full_user.get("created_at", 0),
        "auth_type": user.get("auth_type", "legacy_session"),
        "session_type": user.get("session_type", "browser"),
    }
    merged_user.update(_deps_team_context_for_user(merged_user))
    avatar_url = None
    try:
        from user_avatars import avatar_public_url

        avatar_url = avatar_public_url(_parse_user_preferences(full_user), merged_user["user_id"])
    except Exception:
        pass
    if avatar_url:
        merged_user["avatar_url"] = avatar_url
    return {"ok": True, "user": merged_user}


@router.get("/api/auth/introspect", tags=["Auth"], include_in_schema=False)
def api_auth_introspect(request: Request):
    """Validate any bearer token — interactive user OR machine/client_credentials —
    and return the principal. The MCP gateway uses this to authenticate agents,
    whose client_credentials tokens /api/auth/me intentionally rejects (403)."""
    from routes._deps import _get_current_user

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Invalid or expired token")
    return {
        "ok": True,
        "auth_type": user.get("auth_type"),
        "client_id": user.get("client_id"),
        "email": user.get("email"),
        "user_id": user.get("user_id"),
        "customer_id": user.get("customer_id"),
        "scopes": list(user.get("scopes") or []),
        "workspace_id": user.get("tenant_id")
        or user.get("workspace_customer_id")
        or user.get("customer_id"),
        "team_id": user.get("team_id"),
        "subject": user.get("user_id") or user.get("client_id"),
        "audience": user.get("audience"),
        "jti": user.get("jti"),
        "grant_type": user.get("grant_type"),
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
        updates.update(_business_profile_updates(body))
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
            full_user.update(_business_profile_updates(body))

    return {"ok": True, "message": "Profile updated"}


@router.post("/api/auth/me/avatar", tags=["Auth"])
async def api_auth_upload_avatar(request: Request, file: UploadFile = File(...)):
    """Upload a profile photo (JPEG, PNG, or WebP, max 2 MB)."""
    user = _require_user_grant(request)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        from user_avatars import save_avatar

        avatar_key = save_avatar(user["user_id"], data, file.content_type)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, "Could not save avatar") from exc
    except Exception as exc:
        raise HTTPException(500, "Could not save avatar") from exc

    now = time.time()
    pref_patch = {"avatar_key": avatar_key, "avatar_updated_at": now}
    if _USE_PERSISTENT_AUTH:
        _merge_user_preferences(user["email"], pref_patch)
    else:
        with _user_lock:
            full_user = _users_db.get(user["email"])
            if full_user is not None:
                merged = {**_parse_user_preferences(full_user), **pref_patch}
                full_user["preferences"] = merged
    return {
        "ok": True,
        "avatar_url": f"/api/auth/me/avatar?v={int(now)}",
    }


@router.get("/api/auth/me/avatar", tags=["Auth"])
def api_auth_get_avatar(request: Request):
    """Serve the authenticated user's profile photo."""
    user = _require_user_grant(request)

    # Primary store: Postgres (durable across redeploys).
    from db import AvatarStore

    rec = AvatarStore.get(user["user_id"])
    if rec:
        return Response(
            content=rec["data"],
            media_type=rec["content_type"],
            headers={"Cache-Control": "private, max-age=3600"},
        )

    from artifacts import StorageBackend, get_artifact_manager
    from user_avatars import legacy_avatar_file_for_user, local_avatar_path_for_key

    full_user = (
        UserStore.get_user(user["email"]) if _USE_PERSISTENT_AUTH else _users_db.get(user["email"])
    )
    prefs = _parse_user_preferences(full_user or {})
    avatar_key = (prefs.get("avatar_key") or "").strip()

    if avatar_key:
        local_path = local_avatar_path_for_key(avatar_key)
        if local_path:
            media = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }.get(local_path.suffix.lower(), "application/octet-stream")
            return FileResponse(
                local_path,
                media_type=media,
                headers={"Cache-Control": "private, max-age=3600"},
            )

        mgr = get_artifact_manager()
        if mgr.primary.config.backend != StorageBackend.LOCAL:
            try:
                signed = mgr.primary.generate_download_url(avatar_key, expiry_sec=3600)
                return RedirectResponse(
                    signed["url"],
                    headers={"Cache-Control": "private, max-age=300"},
                )
            except Exception:
                raise HTTPException(404, "Avatar not found")

    path = legacy_avatar_file_for_user(user["user_id"])
    if not path:
        raise HTTPException(404, "Avatar not found")
    media = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media, headers={"Cache-Control": "private, max-age=3600"})


@router.delete("/api/auth/me/avatar", tags=["Auth"])
def api_auth_delete_avatar(request: Request):
    """Remove the user's profile photo."""
    user = _require_user_grant(request)
    from user_avatars import delete_avatar_objects, delete_legacy_avatar_files

    full_user = (
        UserStore.get_user(user["email"]) if _USE_PERSISTENT_AUTH else _users_db.get(user["email"])
    )
    prefs = _parse_user_preferences(full_user or {})
    avatar_key = (prefs.get("avatar_key") or "").strip() or None
    from db import AvatarStore

    AvatarStore.delete(user["user_id"])
    delete_avatar_objects(user["user_id"], avatar_key)
    delete_legacy_avatar_files(user["user_id"])
    prefs.pop("avatar_key", None)
    prefs.pop("avatar_updated_at", None)
    if _USE_PERSISTENT_AUTH:
        UserStore.update_user(user["email"], {"preferences": prefs})
    else:
        with _user_lock:
            mem_user = _users_db.get(user["email"])
            if mem_user is not None:
                mem_user["preferences"] = prefs
    return {"ok": True}


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
                    # The bearer that was presented, revoked exactly. Deleting the
                    # session row does not reach it: an opaque access token resolves
                    # from the auth cache alone.
                    revoke_access_token(token)
                else:
                    with _user_lock:
                        _sessions.pop(token, None)
    resp = JSONResponse(content={"ok": True})
    _clear_auth_cookie(resp)
    return resp


@router.delete("/api/auth/me", tags=["Auth"])
def api_auth_delete_account(request: Request):
    """Start the durable cross-store account deletion workflow."""
    from privacy_deletion import (
        PrivacyDeletionError,
        create_deletion_request,
    )

    user = _require_user_grant(request)
    idempotency_key = request.headers.get("Idempotency-Key", "").strip()
    try:
        receipt = create_deletion_request(
            user_id=str(user.get("user_id") or ""),
            email=str(user.get("email") or ""),
            customer_ids=[str(user.get("customer_id") or "")],
            idempotency_key=idempotency_key,
            requested_by=str(user.get("user_id") or user.get("email") or ""),
            request_source="account_settings",
        )
    except PrivacyDeletionError as exc:
        raise HTTPException(428, str(exc)) from exc

    response = JSONResponse(
        status_code=202,
        content={
            "ok": True,
            "request_id": receipt.request_id,
            "state": receipt.state,
            "deadline_at": receipt.deadline_at.isoformat(),
            "status_token": receipt.status_token,
            "already_existed": receipt.already_existed,
            "message": (
                "Account deletion is in progress. Completion is reported only "
                "after every data store has recorded an outcome."
            ),
        },
    )
    _clear_auth_cookie(response)
    return response


@router.post("/api/keys/generate", tags=["Auth"], deprecated=True)
async def api_generate_api_key(request: Request):
    """API keys have been permanently disabled. Use OAuth 2.0 client_credentials."""
    raise HTTPException(
        410, "API keys are permanently disabled. Use OAuth 2.0: POST /api/oauth/clients"
    )


@router.get("/api/keys", tags=["Auth"], deprecated=True)
def api_list_keys(request: Request):
    """API keys have been permanently disabled. Use OAuth 2.0 client_credentials."""
    raise HTTPException(
        410, "API keys are permanently disabled. Use OAuth 2.0: POST /api/oauth/clients"
    )


@router.delete("/api/keys/{key_preview}", tags=["Auth"], deprecated=True)
def api_revoke_key(key_preview: str, request: Request):
    """API keys have been permanently disabled. Use OAuth 2.0 client_credentials."""
    raise HTTPException(
        410, "API keys are permanently disabled. Use OAuth 2.0: POST /api/oauth/clients"
    )


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

    # Send welcome email now that they're verified. Providers get a different
    # email that sets expectations "straight out" — the weekly uptime goal, how
    # scoring works, what reliability is worth, and the one thing that hurts the
    # whole network (dropping a job). Copy is derived from the real reputation
    # engine via provider_expectations, so it can never overstate the rewards.
    try:
        display_name = user.get("name", email.split("@")[0])
        if user.get("role") == "provider":
            from provider_expectations import provider_welcome_email_text

            _send_team_email(
                email,
                "Welcome to Xcelsior — here's how you earn",
                provider_welcome_email_text(display_name),
                cta_url="https://docs.xcelsior.ca/earn",
                cta_label="See how you earn",
            )
        else:
            _send_team_email(
                email,
                f"Welcome to Xcelsior, {display_name}!",
                f"Hi {display_name},\n\n"
                "Your email is verified and your account is active.\n\n"
                "From your dashboard you can browse available GPU hosts, "
                "launch compute instances, and track your usage — all billed in CAD.\n\n"
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
        # `log.warning`, not debug: production does not emit debug, so a user
        # got "check your email" while the failure left no trace anywhere. This
        # is a user-visible failure, not diagnostic detail.
        log.warning("verification email send failed for %s: %s", email, e)

    return {
        "ok": True,
        "message": "If that email is registered, a verification link has been sent.",
    }


# ── Profile + email change (self-service) ────────────────────────────────

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class EmailChangeRequest(BaseModel):
    new_email: str = Field(..., max_length=254)


class EmailChangeConfirmRequest(BaseModel):
    token: str = Field(..., max_length=128)


@router.put("/api/auth/me/profile", tags=["Auth"])
def api_auth_put_profile(body: ProfileUpdateRequest, request: Request):
    """Update the signed-in user's display name (and optional locale fields).

    Email is NOT changed here — that goes through the verified email-change flow.
    Neither is `role`; see the refusal below.

    Named distinctly from `api_auth_update_profile` (`PATCH /api/auth/me`),
    which it shadowed at module scope — two `def`s, one name, two different
    routes. Both routes worked, because the decorator had already captured each
    function object, but `routes.auth.api_auth_update_profile` resolved to
    whichever came last in the file, and the two behave differently on `role`.
    """
    user = _require_user_grant(request)
    email = str(user.get("email") or "")
    # `ProfileUpdateRequest` carries `role`, and this route quietly dropped it:
    # a client sending {"role": "provider"} got `{"ok": true}` and an unchanged
    # role. Refusing points at the route that does honour it — and keeps role
    # assignment, which `PATCH /api/auth/me` guards against self-granted admin,
    # from acquiring a second and less careful entrance.
    if body.role is not None:
        raise HTTPException(
            400,
            "Role cannot be changed here; use PATCH /api/auth/me",
        )
    updates: dict = {}
    if body.name is not None:
        updates["name"] = body.name.strip()[:64]
    if body.country is not None:
        updates["country"] = body.country.strip()[:32]
    if body.province is not None:
        updates["province"] = body.province.strip()[:32]
    updates.update(_business_profile_updates(body))
    if not updates:
        return {"ok": True}
    if _USE_PERSISTENT_AUTH:
        UserStore.update_user(email, updates)
    else:
        with _user_lock:
            if email in _users_db:
                _users_db[email].update(updates)
    return {"ok": True, **updates}


@router.post("/api/auth/me/email-change", tags=["Auth"])
def api_auth_email_change(body: EmailChangeRequest, request: Request):
    """Start a verified email change: emails a confirmation link to the NEW
    address. The account keeps its current email until the link is confirmed."""
    _check_auth_rate_limit(request)
    user = _require_user_grant(request)
    current_email = user.get("email") or ""
    new_email = body.new_email.strip().lower()
    if not _EMAIL_RE.match(new_email):
        raise HTTPException(400, "Enter a valid email address")
    if new_email == current_email.lower():
        raise HTTPException(400, "That is already your email address")

    if _USE_PERSISTENT_AUTH:
        existing = UserStore.get_user(new_email)
    else:
        with _user_lock:
            existing = _users_db.get(new_email)
    if existing:
        raise HTTPException(409, "That email is already registered to another account")

    token = secrets.token_urlsafe(32)
    expires = time.time() + 86400  # 24h
    change_fields = {
        "pending_email": new_email,
        "email_change_token": token,
        "email_change_expires": expires,
    }
    if _USE_PERSISTENT_AUTH:
        UserStore.update_user(current_email, change_fields)
    else:
        with _user_lock:
            if current_email in _users_db:
                _users_db[current_email].update(change_fields)

    base_url = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca")
    confirm_url = f"{base_url}/verify-email-change?token={token}"
    display_name = user.get("name") or new_email.split("@")[0]
    try:
        _send_team_email(
            new_email,
            "Confirm your new Xcelsior email",
            f"Hi {display_name},\n\n"
            f"We received a request to change the email on your Xcelsior account to {new_email}.\n\n"
            "Click the button below to confirm. This link expires in 24 hours, and your account "
            "keeps using your current email until you confirm.\n\n"
            f"{confirm_url}\n\n"
            "If you didn't request this, you can safely ignore this email — nothing will change.",
            cta_url=confirm_url,
            cta_label="Confirm New Email",
        )
    except Exception as e:
        log.debug("email-change verification send failed: %s", e)

    return {
        "ok": True,
        "pending_email": new_email,
        "message": f"We sent a confirmation link to {new_email}.",
    }


@router.post("/api/auth/me/email-change/confirm", tags=["Auth"])
def api_auth_email_change_confirm(body: EmailChangeConfirmRequest, request: Request):
    """Confirm a pending email change via the token from the verification email.
    Works without an active session (the link may be opened from any device)."""
    _check_auth_rate_limit(request)
    token = body.token.strip()
    if not token:
        raise HTTPException(400, "Invalid confirmation link")

    if _USE_PERSISTENT_AUTH:
        from db import auth_connection

        with auth_connection() as conn:
            row = conn.execute(
                "SELECT email, pending_email, email_change_expires FROM users "
                "WHERE email_change_token = %s",
                (token,),
            ).fetchone()
        if not row or not row.get("pending_email"):
            raise HTTPException(400, "This confirmation link is invalid or has already been used")
        if time.time() > (row["email_change_expires"] or 0):
            raise HTTPException(
                400, "This confirmation link has expired. Please request a new email change."
            )
        old_email = row["email"]
        new_email = row["pending_email"]
        try:
            UserStore.change_email(old_email, new_email)
        except ValueError:
            raise HTTPException(409, "That email is now registered to another account")
    else:
        old_email = None
        new_email = None
        with _user_lock:
            for em, u in list(_users_db.items()):
                if u.get("email_change_token") == token and u.get("pending_email"):
                    if time.time() > u.get("email_change_expires", 0):
                        raise HTTPException(
                            400,
                            "This confirmation link has expired. Please request a new email change.",
                        )
                    new_email = u["pending_email"]
                    if new_email in _users_db:
                        raise HTTPException(409, "That email is now registered to another account")
                    updated = {**u, "email": new_email, "email_verified": 1}
                    for k in ("pending_email", "email_change_token", "email_change_expires"):
                        updated.pop(k, None)
                    del _users_db[em]
                    _users_db[new_email] = updated
                    old_email = em
                    break
            else:
                raise HTTPException(
                    400, "This confirmation link is invalid or has already been used"
                )

    # In-app notification + confirmation email to the new address
    try:
        NotificationStore.create(
            new_email,
            "system",
            "Email address updated",
            f"Your account email was successfully changed to {new_email}.",
            {},
        )
    except Exception as e:
        log.debug("email-change notification failed: %s", e)
    try:
        _send_team_email(
            new_email,
            "Your Xcelsior email was changed",
            f"This confirms your Xcelsior account email is now {new_email}.\n\n"
            "If you didn't make this change, contact support immediately.",
            cta_url="https://xcelsior.ca/dashboard/settings#profile",
            cta_label="Go to Settings",
        )
    except Exception as e:
        log.debug("email-change confirmation send failed: %s", e)

    return {"ok": True, "email": new_email, "message": "Email address updated"}


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
    """Export all personal data for the current user.

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
    # Checked before the admin/non-admin split below, because both branches
    # write the scopes column.
    # No route-level scope check here any more: `OAuthStore.update_client*` enforce
    # delegation themselves (#16), so every writer inherits it instead of each route
    # remembering. Keeping both would be one rule in two places, which is precisely
    # what left this path unguarded when registration was fixed alone.
    from db import OAuthStore

    if _is_platform_admin(user):
        ok = OAuthStore.update_client(client_id, updates, None, actor=user)
    else:
        client = OAuthStore.get_client(client_id)
        if not client:
            raise HTTPException(404, "OAuth client not found or not permitted")
        _require_oauth_client_write(user, client)
        ok = OAuthStore.update_client_in_workspace(
            client_id,
            updates,
            workspace_customer_id=_oauth_workspace_customer_id(user),
            personal_customer_id=_canonical_owner_id(user),
            user_email=user["email"],
            actor=user,
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
    preview = f"{new_secret[:4]}...{new_secret[-4:]}"
    from db import OAuthStore

    if _is_platform_admin(user):
        ok = OAuthStore.rotate_client_secret(client_id, hash_, salt, preview, None)
    else:
        client = OAuthStore.get_client(client_id)
        if not client:
            raise HTTPException(404, "OAuth client not found or not permitted")
        _require_oauth_client_write(user, client)
        ok = OAuthStore.rotate_client_secret_in_workspace(
            client_id,
            hash_,
            salt,
            preview,
            workspace_customer_id=_oauth_workspace_customer_id(user),
            personal_customer_id=_canonical_owner_id(user),
            user_email=user["email"],
        )
    if not ok:
        raise HTTPException(404, "OAuth client not found or not permitted")
    return {
        "ok": True,
        "client_id": client_id,
        "client_secret": new_secret,
        "client_secret_preview": preview,
    }
