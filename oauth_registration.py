"""Connector client identification: CIMD (preferred) and RFC 7591 DCR (compat).

A hosted MCP connector has to let a provider it has never met identify itself
before an end user can authorize it. There are three ways to do that and this
module implements the two portable ones, in the order the adoption plan ranks
them (docs/mcp-enterprise-adoption-plan.md, BLOCKER 3):

1. **Client ID Metadata Documents.** The `client_id` *is* an HTTPS URL that
   serves the client's own metadata. Nothing is stored on our side, so a
   high-traffic directory connector cannot fill a client table with millions of
   rows. This is what Anthropic and OpenAI prefer.
2. **RFC 7591 dynamic client registration.** The compatibility path — Microsoft
   Copilot Studio documents DCR discovery as its simplest OAuth onboarding — and
   the reason this module is hard-gated rather than permissive.

The gates are the point. An open registration endpoint that any caller can
create clients at is an abuse surface, so:

  * redirect URIs must be on the connector allowlist, be same-origin with the
    client's own metadata document, or be loopback — an attacker can host a
    document but cannot receive the authorization code;
  * dynamically identified clients are pinned to the MCP resource and can never
    obtain a general-purpose API token;
  * they can never request operator scopes, and default to a read-biased set;
  * registration is rate limited per IP and globally;
  * a registration that is never used expires.
"""

from __future__ import annotations

import ipaddress
import json
import os
import secrets
import socket
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit

from oauth_service import (
    MCP_RESOURCE_AUDIENCE,
    OAuthGrantError,
    is_loopback_redirect,
)

# ── Policy ────────────────────────────────────────────────────────────────

#: Scopes a dynamically identified connector client may ever hold. Operator
#: authority (`hosts:*`, `control_plane:*`) and the blanket `api` scope are
#: absent by construction — `api` short-circuits every per-tool scope check, so
#: granting it to a self-registered client would erase the scope model.
CONNECTOR_ALLOWED_SCOPES: tuple[str, ...] = (
    "profile",
    "email",
    "offline_access",
    "instances:read",
    "instances:write",
    "instances:operate",
    "billing:read",
    "gpu:read",
    "marketplace:read",
    "inference:read",
    "inference:write",
    "events:read",
    "mcp_actions:approve",
    "instances:connect",
    "ssh:read",
    "ssh:write",
    "volumes:read",
    "volumes:write",
    "artifacts:read",
    "artifacts:write",
    "notifications:read",
    "reputation:read",
    "sla:read",
)

#: What a client that asks for nothing in particular gets. Read-biased on
#: purpose: a connector that only wants to answer questions should not arrive
#: holding the authority to spend the user's money.
CONNECTOR_DEFAULT_SCOPES: tuple[str, ...] = (
    "profile",
    "email",
    "offline_access",
    "instances:read",
    "billing:read",
    "gpu:read",
    "marketplace:read",
    "inference:read",
    "events:read",
)

CONNECTOR_GRANT_TYPES: tuple[str, ...] = ("authorization_code", "refresh_token")

#: Hosts whose OAuth callbacks we accept from a dynamically registered client.
#: Subdomains are matched, so `chatgpt.com` covers `www.chatgpt.com`.
DEFAULT_REDIRECT_HOST_ALLOWLIST: tuple[str, ...] = (
    "claude.ai",
    "claude.com",
    "anthropic.com",
    "chatgpt.com",
    "openai.com",
    "oaiusercontent.com",
    "grok.com",
    "x.ai",
    "microsoft.com",
    "microsoftonline.com",
    "powerplatform.com",
    "copilotstudio.com",
    "github.com",
    "githubusercontent.com",
    "vscode.dev",
    "cursor.com",
    "cursor.sh",
    "xcelsior.ca",
)


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return tuple(item.strip().lower() for item in raw.split(",") if item.strip())


REDIRECT_HOST_ALLOWLIST = _env_list(
    "XCELSIOR_OAUTH_DCR_REDIRECT_HOSTS", DEFAULT_REDIRECT_HOST_ALLOWLIST
)

DCR_ENABLED = os.environ.get("XCELSIOR_OAUTH_DCR_ENABLED", "1").lower() not in {"0", "false", "no"}
CIMD_ENABLED = os.environ.get("XCELSIOR_OAUTH_CIMD_ENABLED", "1").lower() not in {"0", "false", "no"}

#: A registration nobody completes a flow with disappears. Renewed on every
#: successful authorization, so a live connector never expires.
DCR_UNUSED_TTL_DAYS = int(os.environ.get("XCELSIOR_OAUTH_DCR_UNUSED_TTL_DAYS", "30"))
DCR_MAX_PER_IP_PER_HOUR = int(os.environ.get("XCELSIOR_OAUTH_DCR_MAX_PER_IP_PER_HOUR", "5"))
DCR_MAX_PER_HOUR = int(os.environ.get("XCELSIOR_OAUTH_DCR_MAX_PER_HOUR", "200"))
DCR_MAX_REDIRECT_URIS = 10

CIMD_FETCH_TIMEOUT_SEC = float(os.environ.get("XCELSIOR_OAUTH_CIMD_TIMEOUT_SEC", "5"))
CIMD_MAX_DOCUMENT_BYTES = int(os.environ.get("XCELSIOR_OAUTH_CIMD_MAX_BYTES", str(64 * 1024)))
CIMD_CACHE_TTL_SEC = int(os.environ.get("XCELSIOR_OAUTH_CIMD_CACHE_TTL_SEC", "600"))
CIMD_CACHE_MAX_ENTRIES = 512


# ── Shared validation ─────────────────────────────────────────────────────


def normalize_requested_scopes(requested: str | list[str] | None) -> list[str]:
    """Intersect a client's request with what connectors may hold.

    Unknown or forbidden scopes are rejected rather than silently dropped: a
    client that thinks it holds `hosts:evict` and is quietly given a read token
    produces a confusing runtime failure much later, at the tool call.
    """
    if isinstance(requested, list):
        items = [str(scope).strip() for scope in requested if str(scope).strip()]
    else:
        items = [scope for scope in str(requested or "").split() if scope]
    if not items:
        return list(CONNECTOR_DEFAULT_SCOPES)
    forbidden = [scope for scope in items if scope not in CONNECTOR_ALLOWED_SCOPES]
    if forbidden:
        raise OAuthGrantError(
            "invalid_scope",
            "Connector clients may not request: " + ", ".join(sorted(set(forbidden))),
        )
    # `offline_access` is what makes the refresh grant reachable; a connector
    # that omits it would be forced back through the browser every hour.
    if "offline_access" not in items:
        items.append("offline_access")
    return sorted(set(items))


def _host_allowed(host: str) -> bool:
    host = (host or "").lower().strip(".")
    if not host:
        return False
    return any(
        host == allowed or host.endswith(f".{allowed}") for allowed in REDIRECT_HOST_ALLOWLIST
    )


def validate_redirect_uris(
    redirect_uris: list[str], *, same_origin_host: str | None = None
) -> list[str]:
    """Every callback must be somewhere we are willing to send an auth code.

    Three ways to qualify, in decreasing order of how much we know about the
    holder: same origin as the client's own metadata document (proves domain
    control), a first-party provider callback on the allowlist, or loopback.
    """
    if not redirect_uris:
        raise OAuthGrantError("invalid_redirect_uri", "At least one redirect_uri is required")
    if len(redirect_uris) > DCR_MAX_REDIRECT_URIS:
        raise OAuthGrantError(
            "invalid_redirect_uri", f"At most {DCR_MAX_REDIRECT_URIS} redirect URIs are accepted"
        )
    cleaned: list[str] = []
    for raw in redirect_uris:
        uri = str(raw or "").strip()
        if not uri:
            continue
        parts = urlsplit(uri)
        if parts.fragment or "@" in parts.netloc:
            raise OAuthGrantError(
                "invalid_redirect_uri", f"redirect_uri must not carry a fragment or userinfo: {uri}"
            )
        host = (parts.hostname or "").lower()
        if is_loopback_redirect(uri):
            cleaned.append(uri)
            continue
        if parts.scheme != "https":
            raise OAuthGrantError(
                "invalid_redirect_uri", f"redirect_uri must use https or loopback http: {uri}"
            )
        if same_origin_host and host == same_origin_host.lower():
            cleaned.append(uri)
            continue
        if _host_allowed(host):
            cleaned.append(uri)
            continue
        raise OAuthGrantError(
            "invalid_redirect_uri",
            f"redirect_uri host is not an approved connector callback: {host or uri}",
        )
    if not cleaned:
        raise OAuthGrantError("invalid_redirect_uri", "At least one redirect_uri is required")
    return cleaned


def registration_expiry(now: datetime | None = None) -> datetime:
    return (now or datetime.now(UTC)) + timedelta(days=DCR_UNUSED_TTL_DAYS)


# ── CIMD ──────────────────────────────────────────────────────────────────

_cimd_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_cimd_lock = threading.Lock()


def is_cimd_client_id(client_id: str) -> bool:
    """Shape check only — cheap enough to run before any lookup.

    A bare `https://host/` is deliberately not a CIMD id: the path is what
    distinguishes "this document describes one client" from "this domain", and
    accepting the root would make every domain a single implicit client.
    """
    if not CIMD_ENABLED:
        return False
    value = str(client_id or "").strip()
    if not value.lower().startswith("https://"):
        return False
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.hostname:
        return False
    if parts.fragment or parts.query or "@" in parts.netloc:
        return False
    return bool(parts.path) and parts.path != "/"


def _reject_private_address(host: str) -> None:
    """Refuse to fetch a metadata document from inside our own network.

    Without this, `client_id` is a server-side request forgery primitive: any
    caller could point it at 169.254.169.254 or a service on the private
    network and use our error messages as an oracle.
    """
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise OAuthGrantError(
            "invalid_client_metadata", f"Cannot resolve client metadata host: {host}"
        ) from exc
    for info in infos:
        address = info[4][0]
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        if (
            parsed.is_private
            or parsed.is_loopback
            or parsed.is_link_local
            or parsed.is_reserved
            or parsed.is_multicast
            or parsed.is_unspecified
        ):
            raise OAuthGrantError(
                "invalid_client_metadata",
                "Client metadata documents must be hosted on a public address",
            )


def _fetch_cimd_document(client_id: str) -> dict[str, Any]:
    import httpx

    host = urlsplit(client_id).hostname or ""
    _reject_private_address(host)
    try:
        with httpx.Client(
            timeout=CIMD_FETCH_TIMEOUT_SEC,
            # A redirect would let a validated host hand the fetch to an
            # unvalidated one, which is the SSRF check undone.
            follow_redirects=False,
        ) as http:
            response = http.get(client_id, headers={"Accept": "application/json"})
    except Exception as exc:
        raise OAuthGrantError(
            "invalid_client_metadata", f"Client metadata document is unreachable: {exc}"
        ) from exc
    if response.status_code != 200:
        raise OAuthGrantError(
            "invalid_client_metadata",
            f"Client metadata document returned HTTP {response.status_code}",
        )
    if len(response.content) > CIMD_MAX_DOCUMENT_BYTES:
        raise OAuthGrantError("invalid_client_metadata", "Client metadata document is too large")
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type not in {"application/json", "application/ld+json"}:
        raise OAuthGrantError(
            "invalid_client_metadata",
            f"Client metadata document must be JSON, got {content_type or 'nothing'}",
        )
    try:
        document = json.loads(response.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OAuthGrantError(
            "invalid_client_metadata", "Client metadata document is not valid JSON"
        ) from exc
    if not isinstance(document, dict):
        raise OAuthGrantError(
            "invalid_client_metadata", "Client metadata document must be a JSON object"
        )
    return document


def _cimd_cached(client_id: str) -> dict[str, Any] | None:
    with _cimd_lock:
        entry = _cimd_cache.get(client_id)
        if not entry:
            return None
        expires_at, document = entry
        if expires_at <= time.time():
            _cimd_cache.pop(client_id, None)
            return None
        return document


def _cimd_store(client_id: str, client: dict[str, Any]) -> None:
    with _cimd_lock:
        if len(_cimd_cache) >= CIMD_CACHE_MAX_ENTRIES:
            # Oldest expiry first; the cache is a latency optimisation, not a
            # correctness dependency, so an approximate eviction is fine.
            oldest = min(_cimd_cache, key=lambda key: _cimd_cache[key][0])
            _cimd_cache.pop(oldest, None)
        _cimd_cache[client_id] = (time.time() + CIMD_CACHE_TTL_SEC, client)


def reset_cimd_cache() -> None:
    with _cimd_lock:
        _cimd_cache.clear()


def resolve_cimd_client(client_id: str) -> dict[str, Any]:
    """Resolve a metadata-document client id into a client record.

    Returns the same shape `OAuthStore.get_client` does, so every downstream
    consumer — authorize, token exchange, redirect validation — treats a CIMD
    client exactly like a stored one. Nothing is written to the database.
    """
    cached = _cimd_cached(client_id)
    if cached is not None:
        return cached

    document = _fetch_cimd_document(client_id)
    # The document must claim the id it was fetched from. Without this a single
    # hosted document could impersonate any other client id.
    if str(document.get("client_id", "")).strip() != client_id:
        raise OAuthGrantError(
            "invalid_client_metadata",
            "Client metadata document's client_id does not match the URL it was fetched from",
        )
    auth_method = str(document.get("token_endpoint_auth_method", "none")).strip() or "none"
    if auth_method != "none":
        raise OAuthGrantError(
            "invalid_client_metadata",
            "Metadata-document clients are public clients and must use "
            "token_endpoint_auth_method=none with PKCE",
        )
    redirect_uris = document.get("redirect_uris")
    if not isinstance(redirect_uris, list):
        raise OAuthGrantError(
            "invalid_client_metadata", "Client metadata document must list redirect_uris"
        )
    host = urlsplit(client_id).hostname or ""
    validated = validate_redirect_uris(
        [str(uri) for uri in redirect_uris], same_origin_host=host
    )
    grant_types = [
        grant
        for grant in (document.get("grant_types") or CONNECTOR_GRANT_TYPES)
        if grant in CONNECTOR_GRANT_TYPES
    ]
    if "authorization_code" not in grant_types:
        raise OAuthGrantError(
            "invalid_client_metadata",
            "Metadata-document clients must support the authorization_code grant",
        )
    scopes = normalize_requested_scopes(document.get("scope"))

    client = {
        "client_id": client_id,
        "client_name": str(document.get("client_name") or host)[:128],
        "client_type": "public",
        "redirect_uris": validated,
        "grant_types": sorted(set(grant_types) | {"refresh_token"}),
        "scopes": scopes,
        "client_secret_hash": None,
        "client_secret_salt": None,
        "status": "active",
        "is_first_party": 0,
        "is_system_managed": 0,
        "created_by_email": None,
        "workspace_customer_id": None,
        "team_id": None,
        "registration_source": "cimd",
        # Pinned: a metadata-document client exists to reach the MCP connector
        # and must not be usable to mint a general-purpose API token.
        "resource_audience": MCP_RESOURCE_AUDIENCE,
        "client_uri": str(document.get("client_uri") or "")[:512] or None,
        "logo_uri": str(document.get("logo_uri") or "")[:512] or None,
        "policy_uri": str(document.get("policy_uri") or "")[:512] or None,
        "tos_uri": str(document.get("tos_uri") or "")[:512] or None,
        "software_id": str(document.get("software_id") or "")[:128] or None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    _cimd_store(client_id, client)
    return client


# ── RFC 7591 dynamic client registration ──────────────────────────────────


def register_dynamic_client(body: dict[str, Any], *, client_ip: str) -> dict[str, Any]:
    """Create a hard-gated dynamic client and return the RFC 7591 response."""
    from db import OAuthStore
    from oauth_service import _cache_incr, create_oauth_client

    if not DCR_ENABLED:
        raise OAuthGrantError(
            "invalid_request", "Dynamic client registration is disabled", status_code=403
        )

    # Two ceilings: one so a single source cannot flood the table, one so the
    # table cannot be flooded from many sources either.
    per_ip = _cache_incr("dcr_register_ip", client_ip or "unknown", 3600)
    if per_ip > DCR_MAX_PER_IP_PER_HOUR:
        raise OAuthGrantError(
            "invalid_request",
            "Too many client registrations from this address; try again later",
            status_code=429,
            headers={"Retry-After": "3600"},
        )
    if OAuthStore.count_registrations_since(time.time() - 3600) >= DCR_MAX_PER_HOUR:
        raise OAuthGrantError(
            "invalid_request",
            "Client registration is temporarily rate limited",
            status_code=429,
            headers={"Retry-After": "3600"},
        )

    redirect_uris = body.get("redirect_uris")
    if not isinstance(redirect_uris, list):
        raise OAuthGrantError(
            "invalid_redirect_uri", "redirect_uris must be a list of absolute URIs"
        )
    validated = validate_redirect_uris([str(uri) for uri in redirect_uris])

    requested_grants = body.get("grant_types") or ["authorization_code", "refresh_token"]
    unsupported = [grant for grant in requested_grants if grant not in CONNECTOR_GRANT_TYPES]
    if unsupported:
        # client_credentials in particular: a self-registered client must never
        # be able to act without a user behind it.
        raise OAuthGrantError(
            "invalid_client_metadata",
            "Dynamically registered clients may only use authorization_code and "
            "refresh_token; rejected: " + ", ".join(sorted(set(unsupported))),
        )
    response_types = body.get("response_types") or ["code"]
    if [rt for rt in response_types if rt != "code"]:
        raise OAuthGrantError("invalid_client_metadata", "Only response_type=code is supported")

    auth_method = str(body.get("token_endpoint_auth_method", "none")).strip() or "none"
    if auth_method != "none":
        raise OAuthGrantError(
            "invalid_client_metadata",
            "Dynamically registered clients are public clients and must use "
            "token_endpoint_auth_method=none with S256 PKCE",
        )

    scopes = normalize_requested_scopes(body.get("scope"))
    client_name = str(body.get("client_name") or "Dynamically registered connector").strip()[:128]
    contacts = [str(item)[:128] for item in (body.get("contacts") or [])][:5]

    expires_at = registration_expiry()
    created = create_oauth_client(
        client_name=client_name,
        redirect_uris=validated,
        grant_types=sorted(set(CONNECTOR_GRANT_TYPES)),
        scopes=scopes,
        created_by_email=None,
        client_type="public",
        is_first_party=False,
        registration_source="dcr",
        resource_audience=MCP_RESOURCE_AUDIENCE,
        registration_expires_at=expires_at,
        client_uri=str(body.get("client_uri") or "")[:512] or None,
        logo_uri=str(body.get("logo_uri") or "")[:512] or None,
        policy_uri=str(body.get("policy_uri") or "")[:512] or None,
        tos_uri=str(body.get("tos_uri") or "")[:512] or None,
        software_id=str(body.get("software_id") or "")[:128] or None,
        software_version=str(body.get("software_version") or "")[:64] or None,
        contacts=contacts,
    )

    return {
        "client_id": created["client_id"],
        "client_id_issued_at": int(created["created_at"]),
        "client_name": client_name,
        "redirect_uris": validated,
        "grant_types": sorted(set(CONNECTOR_GRANT_TYPES)),
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
        "scope": " ".join(scopes),
        "resource": MCP_RESOURCE_AUDIENCE,
        # Not part of RFC 7591, but a client that knows when its registration
        # lapses can renew it deliberately instead of discovering it at 401.
        "xcelsior_registration_expires_at": expires_at.isoformat(),
    }


# ── Surface attribution ───────────────────────────────────────────────────
#
# Which product a user connected from. Recorded at consent time because it is
# knowable then and unrecoverable later: the same pre-provisioned connector
# client is used by every surface, so `client_id` alone cannot answer "which
# directory produced this activation" (adoption plan X7.34).

_SURFACE_HOSTS: tuple[tuple[str, str], ...] = (
    ("claude.ai", "claude"),
    ("claude.com", "claude"),
    ("anthropic.com", "claude"),
    ("chatgpt.com", "chatgpt"),
    ("openai.com", "chatgpt"),
    ("oaiusercontent.com", "chatgpt"),
    ("grok.com", "grok"),
    ("x.ai", "grok"),
    ("copilotstudio.com", "copilot-studio"),
    ("powerplatform.com", "copilot-studio"),
    ("microsoftonline.com", "microsoft"),
    ("microsoft.com", "microsoft"),
    ("githubusercontent.com", "github"),
    ("github.com", "github"),
    ("vscode.dev", "vscode"),
    ("cursor.com", "cursor"),
    ("cursor.sh", "cursor"),
    ("xcelsior.ca", "first-party"),
)


def classify_surface(redirect_uri: str, client_id: str = "") -> str:
    """Name the product a connection came from, or 'unknown'.

    Loopback is reported as `local` rather than guessed at: a native client
    binds 127.0.0.1 and its identity is genuinely not in the redirect. Where the
    client id is itself a metadata-document URL, its host is a better signal
    than the callback, so it is consulted first.
    """
    for candidate in (client_id, redirect_uri):
        host = (urlsplit(str(candidate or "")).hostname or "").lower()
        if not host:
            continue
        if is_loopback_redirect(str(candidate)) or host in _LOOPBACK_SURFACE_HOSTS:
            # Keep looking — a loopback callback with a CIMD client id is still
            # attributable through the id.
            continue
        for suffix, surface in _SURFACE_HOSTS:
            if host == suffix or host.endswith(f".{suffix}"):
                return surface
    if is_loopback_redirect(redirect_uri):
        return "local"
    return "unknown"


_LOOPBACK_SURFACE_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def new_grant_id() -> str:
    return f"cg_{uuid.uuid4().hex}"


def new_state_nonce() -> str:
    return secrets.token_urlsafe(24)
