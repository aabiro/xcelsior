"""Internal OAuth 2.0 service layer with dedicated cache-backed auth state."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from db import OAuthStore, UserStore

try:
    from prometheus_client import Counter

    _oauth_tokens_issued = Counter(
        "xcelsior_oauth_tokens_issued_total",
        "OAuth tokens issued by grant type",
        ["grant_type"],
    )
    _oauth_refresh_events = Counter(
        "xcelsior_oauth_refresh_events_total",
        "OAuth refresh outcomes",
        ["outcome"],
    )
    _oauth_device_poll_events = Counter(
        "xcelsior_oauth_device_poll_events_total",
        "OAuth device-code polling outcomes",
        ["outcome"],
    )
except Exception:
    class _NoopCounter:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            return None

    _oauth_tokens_issued = _NoopCounter()
    _oauth_refresh_events = _NoopCounter()
    _oauth_device_poll_events = _NoopCounter()


ACCESS_TOKEN_TTL_SEC = int(os.environ.get("XCELSIOR_OAUTH_ACCESS_TTL_SEC", "900"))
REFRESH_TOKEN_TTL_SEC = int(os.environ.get("XCELSIOR_OAUTH_REFRESH_TTL_SEC", str(86400 * 30)))
AUTH_CODE_TTL_SEC = int(os.environ.get("XCELSIOR_OAUTH_CODE_TTL_SEC", "300"))
DEVICE_CODE_TTL_SEC = int(os.environ.get("XCELSIOR_OAUTH_DEVICE_TTL_SEC", "900"))
DEVICE_CODE_INTERVAL_SEC = int(os.environ.get("XCELSIOR_OAUTH_DEVICE_INTERVAL_SEC", "5"))
CLIENT_CREDENTIALS_TTL_SEC = int(
    os.environ.get("XCELSIOR_OAUTH_CLIENT_CREDENTIALS_TTL_SEC", "900")
)
ACCESS_TOKEN_PREFIX = os.environ.get("XCELSIOR_OAUTH_ACCESS_TOKEN_PREFIX", "xoa_")
AUTH_CACHE_BACKEND = os.environ.get(
    "XCELSIOR_AUTH_CACHE_BACKEND",
    "memory" if os.environ.get("XCELSIOR_ENV", "dev").lower() == "test" else "redis",
).lower()
AUTH_REDIS_URL = os.environ.get("XCELSIOR_AUTH_REDIS_URL", "redis://localhost:6379/0")
AUTH_CACHE_PREFIX = os.environ.get("XCELSIOR_AUTH_CACHE_PREFIX", "xcelsior:oauth")
OAUTH_ISSUER = os.environ.get(
    "XCELSIOR_OAUTH_ISSUER",
    os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca").rstrip("/"),
)
OAUTH_AUDIENCE = os.environ.get("XCELSIOR_OAUTH_AUDIENCE", "xcelsior-api")
REFRESH_COOKIE_NAME = "xcelsior_refresh"
API_KEY_SUNSET_DATE = os.environ.get("XCELSIOR_API_KEY_SUNSET_DATE", "2026-07-07")

_cache_instance = None
_cache_lock = threading.Lock()
_defaults_lock = threading.Lock()
_defaults_ready = False


class AuthCacheUnavailableError(RuntimeError):
    """Raised when the configured dedicated auth cache is unavailable."""


class OAuthGrantError(Exception):
    """Structured OAuth grant error."""

    def __init__(
        self,
        error: str,
        description: str,
        *,
        status_code: int = 400,
        extra: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(description)
        self.error = error
        self.description = description
        self.status_code = status_code
        self.extra = dict(extra or {})
        self.headers = dict(headers or {})

    def payload(self) -> dict[str, Any]:
        body = {"error": self.error, "error_description": self.description}
        body.update(self.extra)
        return body


class MemoryAuthCache:
    """Small in-memory cache used in tests and local fallback."""

    def __init__(self):
        self._lock = threading.Lock()
        self._store: dict[str, tuple[str, float]] = {}

    def _purge(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        expired = [key for key, (_, expires_at) in self._store.items() if expires_at <= now]
        for key in expired:
            self._store.pop(key, None)

    def get(self, key: str) -> str | None:
        now = time.time()
        with self._lock:
            self._purge(now)
            item = self._store.get(key)
            return item[0] if item else None

    def set(self, key: str, value: str, ttl: int, *, nx: bool = False) -> bool:
        now = time.time()
        with self._lock:
            self._purge(now)
            if nx and key in self._store:
                return False
            self._store[key] = (value, now + max(1, ttl))
            return True

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def getdel(self, key: str) -> str | None:
        now = time.time()
        with self._lock:
            self._purge(now)
            item = self._store.pop(key, None)
            return item[0] if item else None

    def incr(self, key: str, ttl: int) -> int:
        now = time.time()
        with self._lock:
            self._purge(now)
            raw = self._store.get(key)
            value = int(raw[0]) if raw else 0
            value += 1
            self._store[key] = (str(value), now + max(1, ttl))
            return value


class RedisAuthCache:
    """Redis-backed cache for ephemeral OAuth state."""

    def __init__(self, url: str):
        try:
            import redis
        except ImportError as exc:
            raise AuthCacheUnavailableError("redis package is not installed") from exc
        try:
            self._client = redis.from_url(url, decode_responses=True)
            self._client.ping()
        except Exception as exc:
            raise AuthCacheUnavailableError(f"redis unavailable: {exc}") from exc

    def get(self, key: str) -> str | None:
        try:
            return self._client.get(key)
        except Exception as exc:
            raise AuthCacheUnavailableError(str(exc)) from exc

    def set(self, key: str, value: str, ttl: int, *, nx: bool = False) -> bool:
        try:
            return bool(self._client.set(key, value, ex=max(1, ttl), nx=nx))
        except Exception as exc:
            raise AuthCacheUnavailableError(str(exc)) from exc

    def delete(self, key: str) -> None:
        try:
            self._client.delete(key)
        except Exception as exc:
            raise AuthCacheUnavailableError(str(exc)) from exc

    def getdel(self, key: str) -> str | None:
        try:
            value = self._client.getdel(key)
        except AttributeError:
            pipe = self._client.pipeline()
            while True:
                try:
                    pipe.watch(key)
                    value = pipe.get(key)
                    pipe.multi()
                    pipe.delete(key)
                    pipe.execute()
                    break
                except Exception:
                    pipe.reset()
                    raise
        except Exception as exc:
            raise AuthCacheUnavailableError(str(exc)) from exc
        return value

    def incr(self, key: str, ttl: int) -> int:
        try:
            value = int(self._client.incr(key))
            if value == 1:
                self._client.expire(key, max(1, ttl))
            return value
        except Exception as exc:
            raise AuthCacheUnavailableError(str(exc)) from exc


def _cache_key(kind: str, identifier: str) -> str:
    return f"{AUTH_CACHE_PREFIX}:{kind}:{identifier}"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _json_loads(value: str | None) -> Any:
    if not value:
        return None
    return json.loads(value)


def get_auth_cache():
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance
    with _cache_lock:
        if _cache_instance is not None:
            return _cache_instance
        if AUTH_CACHE_BACKEND == "memory":
            _cache_instance = MemoryAuthCache()
        elif AUTH_CACHE_BACKEND == "redis":
            _cache_instance = RedisAuthCache(AUTH_REDIS_URL)
        else:
            raise AuthCacheUnavailableError(f"unsupported auth cache backend: {AUTH_CACHE_BACKEND}")
    return _cache_instance


def _cache_set_json(kind: str, identifier: str, payload: dict[str, Any], ttl: int, *, nx: bool = False) -> bool:
    return bool(
        get_auth_cache().set(
            _cache_key(kind, identifier),
            _json_dumps(payload),
            ttl,
            nx=nx,
        )
    )


def _cache_get_json(kind: str, identifier: str) -> dict[str, Any] | None:
    value = get_auth_cache().get(_cache_key(kind, identifier))
    data = _json_loads(value)
    return data if isinstance(data, dict) else None


def _cache_getdel_json(kind: str, identifier: str) -> dict[str, Any] | None:
    value = get_auth_cache().getdel(_cache_key(kind, identifier))
    data = _json_loads(value)
    return data if isinstance(data, dict) else None


def _cache_delete(kind: str, identifier: str) -> None:
    get_auth_cache().delete(_cache_key(kind, identifier))


def _cache_incr(kind: str, identifier: str, ttl: int) -> int:
    return int(get_auth_cache().incr(_cache_key(kind, identifier), ttl))


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def hash_secret(secret: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", secret.encode(), salt.encode(), 200_000).hex()
    return digest, salt


def verify_secret(secret: str, expected_hash: str, salt: str) -> bool:
    actual, _ = hash_secret(secret, salt)
    return hmac.compare_digest(actual, expected_hash)


def build_deprecation_headers(surface: str) -> dict[str, str]:
    sunset = datetime.strptime(API_KEY_SUNSET_DATE, "%Y-%m-%d").replace(tzinfo=UTC)
    return {
        "Deprecation": "true",
        "Sunset": sunset.strftime("%a, %d %b %Y 00:00:00 GMT"),
        "Warning": f'299 xcelsior "{surface} is deprecated; migrate to OAuth 2.0"',
    }


def is_oauth_access_token(token: str) -> bool:
    return str(token or "").startswith(ACCESS_TOKEN_PREFIX)


def _base64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _base64url_decode(raw: str) -> bytes:
    padding = "=" * ((4 - len(raw) % 4) % 4)
    return base64.urlsafe_b64decode(raw + padding)


def _get_jwt_signing_keys() -> tuple[str, dict[str, str]]:
    raw = os.environ.get("XCELSIOR_OAUTH_JWT_KEYS_JSON", "").strip()
    if raw:
        data = json.loads(raw)
        active_kid = str(data["active_kid"])
        keys = {str(item["kid"]): str(item["secret"]) for item in data.get("keys", [])}
        if active_kid not in keys:
            raise OAuthGrantError("server_error", "Active JWT signing key is missing", status_code=500)
        return active_kid, keys

    active_kid = os.environ.get("XCELSIOR_OAUTH_ACTIVE_KID", "default")
    secret = os.environ.get("XCELSIOR_OAUTH_JWT_SECRET", "")
    if not secret:
        env = os.environ.get("XCELSIOR_ENV", "dev").lower()
        if env in {"test", "dev", "development"}:
            secret = "xcelsior-dev-jwt-secret"
        else:
            raise OAuthGrantError(
                "server_error",
                "OAuth JWT signing secret is not configured",
                status_code=500,
            )
    return active_kid, {active_kid: secret}


def issue_client_credentials_jwt(client: dict, scopes: list[str]) -> dict[str, Any]:
    active_kid, keys = _get_jwt_signing_keys()
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT", "kid": active_kid}
    payload = {
        "iss": OAUTH_ISSUER,
        "sub": client["client_id"],
        "aud": OAUTH_AUDIENCE,
        "iat": now,
        "exp": now + CLIENT_CREDENTIALS_TTL_SEC,
        "scope": " ".join(scopes),
        "client_id": client["client_id"],
        "auth_type": "client_credentials",
        "jti": f"jwt-{uuid.uuid4().hex}",
    }
    signing_input = ".".join(
        [
            _base64url_encode(_json_dumps(header).encode()),
            _base64url_encode(_json_dumps(payload).encode()),
        ]
    )
    signature = hmac.new(
        keys[active_kid].encode(),
        signing_input.encode(),
        hashlib.sha256,
    ).digest()
    access_token = f"{signing_input}.{_base64url_encode(signature)}"
    _oauth_tokens_issued.labels("client_credentials").inc()
    return {
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": CLIENT_CREDENTIALS_TTL_SEC,
        "scope": " ".join(scopes),
    }


def validate_client_credentials_jwt(token: str) -> dict[str, Any] | None:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".", 2)
        header = json.loads(_base64url_decode(header_b64))
        payload = json.loads(_base64url_decode(payload_b64))
    except Exception:
        return None

    kid = str(header.get("kid", ""))
    alg = str(header.get("alg", ""))
    if alg != "HS256" or not kid:
        return None

    try:
        _, keys = _get_jwt_signing_keys()
    except OAuthGrantError:
        return None

    secret = keys.get(kid)
    if not secret:
        return None

    signing_input = f"{header_b64}.{payload_b64}"
    expected_sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(expected_sig, _base64url_decode(signature_b64)):
        return None

    now = int(time.time())
    if int(payload.get("exp", 0) or 0) <= now:
        return None
    if payload.get("iss") != OAUTH_ISSUER:
        return None
    if payload.get("aud") != OAUTH_AUDIENCE:
        return None

    client = OAuthStore.get_client(str(payload.get("client_id", "")))
    if not client:
        return None

    return {
        "auth_type": "client_credentials",
        "grant_type": "client_credentials",
        "client_id": client["client_id"],
        "user_id": f"client:{client['client_id']}",
        "email": "",
        "role": "machine",
        "name": client.get("client_name", client["client_id"]),
        "scopes": [scope for scope in str(payload.get("scope", "")).split() if scope],
        "is_admin": False,
    }


def _normalize_scopes(requested_scopes: str | list[str] | None, allowed_scopes: list[str]) -> list[str]:
    if isinstance(requested_scopes, list):
        requested = [str(scope).strip() for scope in requested_scopes if str(scope).strip()]
    else:
        requested = [scope for scope in str(requested_scopes or "").split() if scope]
    if not requested:
        return list(allowed_scopes)
    allowed = set(allowed_scopes)
    invalid = [scope for scope in requested if scope not in allowed]
    if invalid:
        raise OAuthGrantError(
            "invalid_scope",
            f"Invalid scope(s): {', '.join(invalid)}",
        )
    return requested


def ensure_default_oauth_clients() -> None:
    global _defaults_ready
    if _defaults_ready:
        return
    with _defaults_lock:
        if _defaults_ready:
            return
        now = time.time()
        base = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca").rstrip("/")
        defaults = [
            {
                "client_id": "xcelsior-web",
                "client_name": "Xcelsior Web",
                "client_type": "public",
                "redirect_uris": [
                    f"{base}/oauth/callback",
                    f"{base}/login/oauth/callback",
                    f"{base}/dashboard/oauth/callback",
                ],
                "grant_types": ["authorization_code", "refresh_token"],
                "scopes": ["profile", "email", "offline_access"],
                "created_by_email": None,
                "is_first_party": 1,
                "created_at": now,
                "updated_at": now,
            },
            {
                "client_id": "xcelsior-cli",
                "client_name": "Xcelsior CLI",
                "client_type": "public",
                "redirect_uris": [],
                "grant_types": [
                    "urn:ietf:params:oauth:grant-type:device_code",
                    "refresh_token",
                ],
                "scopes": ["profile", "email", "offline_access"],
                "created_by_email": None,
                "is_first_party": 1,
                "created_at": now,
                "updated_at": now,
            },
        ]
        for client in defaults:
            if not OAuthStore.get_client(client["client_id"]):
                OAuthStore.create_client(client)
        _defaults_ready = True


def get_client(client_id: str) -> dict | None:
    ensure_default_oauth_clients()
    return OAuthStore.get_client(client_id)


def create_oauth_client(
    *,
    client_name: str,
    redirect_uris: list[str],
    grant_types: list[str],
    scopes: list[str],
    created_by_email: str | None,
    client_type: str = "confidential",
    is_first_party: bool = False,
) -> dict[str, Any]:
    now = time.time()
    client_id = f"oauth_{uuid.uuid4().hex[:16]}"
    client_secret = ""
    secret_hash = None
    secret_salt = None
    if client_type == "confidential":
        client_secret = secrets.token_urlsafe(32)
        secret_hash, secret_salt = hash_secret(client_secret)

    client = {
        "client_id": client_id,
        "client_name": client_name,
        "client_type": client_type,
        "redirect_uris": redirect_uris,
        "grant_types": grant_types,
        "scopes": scopes,
        "client_secret_hash": secret_hash,
        "client_secret_salt": secret_salt,
        "created_by_email": created_by_email,
        "is_first_party": 1 if is_first_party else 0,
        "created_at": now,
        "updated_at": now,
    }
    OAuthStore.create_client(client)
    response = {
        "client_id": client_id,
        "client_name": client_name,
        "client_type": client_type,
        "redirect_uris": redirect_uris,
        "grant_types": grant_types,
        "scopes": scopes,
        "created_at": now,
    }
    if client_secret:
        response["client_secret"] = client_secret
    return response


def authenticate_client(client_id: str, client_secret: str | None = None) -> dict:
    client = get_client(client_id)
    if not client:
        raise OAuthGrantError("invalid_client", "Unknown OAuth client", status_code=401)
    if client.get("client_type") == "confidential":
        if not client_secret:
            raise OAuthGrantError("invalid_client", "Client secret is required", status_code=401)
        if not verify_secret(
            client_secret,
            str(client.get("client_secret_hash") or ""),
            str(client.get("client_secret_salt") or ""),
        ):
            raise OAuthGrantError("invalid_client", "Invalid client credentials", status_code=401)
    return client


def _pkce_challenge(verifier: str) -> str:
    return _base64url_encode(hashlib.sha256(verifier.encode()).digest())


def issue_authorization_code(
    *,
    client: dict,
    user: dict,
    redirect_uri: str,
    code_challenge: str,
    code_challenge_method: str,
    scopes: list[str],
) -> str:
    if code_challenge_method != "S256":
        raise OAuthGrantError("invalid_request", "Only S256 PKCE is supported")
    code = secrets.token_urlsafe(32)
    payload = {
        "client_id": client["client_id"],
        "redirect_uri": redirect_uri,
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
        "scopes": scopes,
        "email": user.get("email"),
        "user_id": user.get("user_id"),
        "role": user.get("role", "submitter"),
        "name": user.get("name", ""),
        "is_admin": bool(user.get("is_admin")),
        "customer_id": user.get("customer_id"),
        "provider_id": user.get("provider_id"),
        "session_type": "browser",
        "created_at": time.time(),
    }
    if not _cache_set_json("authorization_code", code, payload, AUTH_CODE_TTL_SEC):
        raise OAuthGrantError("server_error", "Failed to issue authorization code", status_code=500)
    return code


def _build_session_record(user: dict, request, *, refresh_token: str, client_id: str, session_type: str) -> dict:
    now = time.time()
    client_host = request.client.host if request and request.client else None
    user_agent = request.headers.get("user-agent", "")[:512] if request else ""
    return {
        "token": refresh_token,
        "email": user["email"],
        "user_id": user.get("user_id", user["email"]),
        "role": user.get("role", "submitter"),
        "is_admin": bool(user.get("is_admin")),
        "name": user.get("name", ""),
        "created_at": now,
        "expires_at": now + REFRESH_TOKEN_TTL_SEC,
        "ip_address": client_host,
        "user_agent": user_agent,
        "last_active": now,
        "session_type": session_type,
        "client_id": client_id,
    }


def _issue_opaque_access_token(
    user: dict,
    *,
    client_id: str,
    scopes: list[str],
    session_token: str,
    session_type: str,
) -> dict[str, Any]:
    token = f"{ACCESS_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    now = time.time()
    payload = {
        "auth_type": "oauth_access_token",
        "email": user.get("email", ""),
        "user_id": user.get("user_id", ""),
        "role": user.get("role", "submitter"),
        "name": user.get("name", ""),
        "is_admin": bool(user.get("is_admin")),
        "customer_id": user.get("customer_id"),
        "provider_id": user.get("provider_id"),
        "client_id": client_id,
        "scopes": list(scopes),
        "session_token": session_token,
        "session_type": session_type,
        "issued_at": now,
        "expires_at": now + ACCESS_TOKEN_TTL_SEC,
    }
    if not _cache_set_json("access_token", token, payload, ACCESS_TOKEN_TTL_SEC):
        raise OAuthGrantError(
            "server_error",
            "Dedicated auth cache unavailable for access token issuance",
            status_code=503,
        )
    return {
        "access_token": token,
        "expires_in": ACCESS_TOKEN_TTL_SEC,
        "payload": payload,
    }


def _create_refresh_token_record(
    *,
    user: dict,
    request,
    client_id: str,
    scopes: list[str],
    session_type: str,
    family_id: str | None = None,
    parent_token_id: str | None = None,
    created_at: float | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    refresh_token = secrets.token_urlsafe(48)
    token_id = f"rt_{uuid.uuid4().hex}"
    created_at = created_at or time.time()
    session = _build_session_record(
        user,
        request,
        refresh_token=refresh_token,
        client_id=client_id,
        session_type=session_type,
    )
    session["created_at"] = created_at
    session["last_active"] = time.time()
    session["expires_at"] = created_at + REFRESH_TOKEN_TTL_SEC
    refresh_record = {
        "token_id": token_id,
        "token_hash": hash_token(refresh_token),
        "family_id": family_id or f"rtfam_{uuid.uuid4().hex}",
        "parent_token_id": parent_token_id,
        "session_token": refresh_token,
        "client_id": client_id,
        "email": user.get("email"),
        "user_id": user.get("user_id"),
        "session_type": session_type,
        "scopes": list(scopes),
        "created_at": created_at,
        "expires_at": created_at + REFRESH_TOKEN_TTL_SEC,
    }
    return refresh_token, session, refresh_record


def issue_user_tokens(
    user: dict,
    request,
    *,
    client_id: str = "xcelsior-web",
    scopes: list[str] | None = None,
    session_type: str = "browser",
) -> dict[str, Any]:
    client = get_client(client_id)
    if not client:
        raise OAuthGrantError("invalid_client", "Unknown OAuth client", status_code=401)
    allowed_scopes = list(client.get("scopes") or ["profile", "email", "offline_access"])
    scopes = _normalize_scopes(scopes, allowed_scopes)
    refresh_token, session, refresh_record = _create_refresh_token_record(
        user=user,
        request=request,
        client_id=client_id,
        scopes=scopes,
        session_type=session_type,
    )
    UserStore.create_session(session)
    OAuthStore.create_refresh_token(refresh_record)
    access = _issue_opaque_access_token(
        user,
        client_id=client_id,
        scopes=scopes,
        session_token=refresh_token,
        session_type=session_type,
    )
    _oauth_tokens_issued.labels(session_type).inc()
    return {
        "access_token": access["access_token"],
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in": ACCESS_TOKEN_TTL_SEC,
        "refresh_expires_in": REFRESH_TOKEN_TTL_SEC,
        "scope": " ".join(scopes),
        "session_token": refresh_token,
        "session_type": session_type,
    }


def exchange_authorization_code(
    *,
    client: dict,
    code: str,
    redirect_uri: str,
    code_verifier: str,
    request,
) -> dict[str, Any]:
    payload = _cache_getdel_json("authorization_code", code)
    if not payload:
        raise OAuthGrantError("invalid_grant", "Authorization code is invalid or expired")
    if payload.get("client_id") != client["client_id"]:
        raise OAuthGrantError("invalid_grant", "Authorization code does not match client")
    if payload.get("redirect_uri") != redirect_uri:
        raise OAuthGrantError("invalid_grant", "redirect_uri mismatch")
    if payload.get("code_challenge_method") != "S256":
        raise OAuthGrantError("invalid_grant", "Unsupported PKCE method")
    if _pkce_challenge(code_verifier) != payload.get("code_challenge"):
        raise OAuthGrantError("invalid_grant", "PKCE verification failed")
    user = {
        "email": payload.get("email"),
        "user_id": payload.get("user_id"),
        "role": payload.get("role", "submitter"),
        "name": payload.get("name", ""),
        "is_admin": bool(payload.get("is_admin")),
        "customer_id": payload.get("customer_id"),
        "provider_id": payload.get("provider_id"),
    }
    _oauth_tokens_issued.labels("authorization_code").inc()
    return issue_user_tokens(
        user,
        request,
        client_id=client["client_id"],
        scopes=list(payload.get("scopes") or []),
        session_type="browser",
    )


def _lookup_user(user_code: str) -> dict[str, Any] | None:
    device_ref = _cache_get_json("device_user_code", user_code)
    if not device_ref:
        return None
    return _cache_get_json("device_code", str(device_ref.get("device_code", "")))


def issue_device_authorization(*, client: dict, scopes: list[str] | None = None, base_url: str) -> dict[str, Any]:
    scopes = _normalize_scopes(scopes, list(client.get("scopes") or ["profile", "email", "offline_access"]))
    device_code = secrets.token_urlsafe(32)
    user_code = f"{secrets.token_hex(2).upper()}-{secrets.token_hex(2).upper()}"
    now = time.time()
    payload = {
        "client_id": client["client_id"],
        "device_code": device_code,
        "user_code": user_code,
        "scopes": scopes,
        "status": "pending",
        "created_at": now,
        "expires_at": now + DEVICE_CODE_TTL_SEC,
        "interval": DEVICE_CODE_INTERVAL_SEC,
    }
    if not _cache_set_json("device_code", device_code, payload, DEVICE_CODE_TTL_SEC):
        raise OAuthGrantError("server_error", "Failed to issue device code", status_code=503)
    _cache_set_json(
        "device_user_code",
        user_code,
        {"device_code": device_code},
        DEVICE_CODE_TTL_SEC,
    )
    return {
        "device_code": device_code,
        "user_code": user_code,
        "verification_uri": f"{base_url.rstrip('/')}/api/auth/verify",
        "verification_uri_complete": f"{base_url.rstrip('/')}/api/auth/verify?user_code={user_code}",
        "expires_in": DEVICE_CODE_TTL_SEC,
        "interval": DEVICE_CODE_INTERVAL_SEC,
    }


def approve_device_code(*, user: dict, user_code: str) -> dict[str, Any]:
    device_ref = _cache_get_json("device_user_code", user_code)
    if not device_ref:
        raise OAuthGrantError("invalid_grant", "Invalid or expired device user code", status_code=404)
    device_code = str(device_ref.get("device_code", ""))
    payload = _cache_get_json("device_code", device_code)
    if not payload:
        raise OAuthGrantError("invalid_grant", "Device code is invalid or expired", status_code=404)
    if payload.get("status") != "pending":
        raise OAuthGrantError("invalid_grant", "Device code has already been used")
    payload["status"] = "approved"
    payload["approved_user"] = {
        "email": user.get("email"),
        "user_id": user.get("user_id"),
        "role": user.get("role", "submitter"),
        "name": user.get("name", ""),
        "is_admin": bool(user.get("is_admin")),
        "customer_id": user.get("customer_id"),
        "provider_id": user.get("provider_id"),
    }
    remaining = max(1, int(float(payload.get("expires_at", time.time())) - time.time()))
    _cache_set_json("device_code", device_code, payload, remaining)
    return {"ok": True, "user_code": user_code}


def exchange_device_code(*, client: dict, device_code: str, request) -> dict[str, Any]:
    payload = _cache_get_json("device_code", device_code)
    if not payload:
        raise OAuthGrantError("expired_token", "Device code is invalid or expired", status_code=410)
    if payload.get("client_id") != client["client_id"]:
        raise OAuthGrantError("invalid_grant", "Device code does not match client")

    ip = request.client.host if request and request.client else "unknown"
    poll_count = _cache_incr("device_poll", f"{device_code}:{ip}", DEVICE_CODE_INTERVAL_SEC)
    if poll_count > 1:
        _oauth_device_poll_events.labels("slow_down").inc()
        raise OAuthGrantError(
            "slow_down",
            "Poll interval exceeded",
            status_code=429,
            headers={"Retry-After": str(DEVICE_CODE_INTERVAL_SEC)},
        )

    expires_at = float(payload.get("expires_at", 0) or 0)
    if expires_at <= time.time():
        _cache_delete("device_code", device_code)
        _cache_delete("device_user_code", str(payload.get("user_code", "")))
        _oauth_device_poll_events.labels("expired").inc()
        raise OAuthGrantError("expired_token", "Device code has expired", status_code=410)

    status = payload.get("status")
    if status == "pending":
        _oauth_device_poll_events.labels("pending").inc()
        raise OAuthGrantError(
            "authorization_pending",
            "authorization_pending",
            status_code=428,
            headers={"Retry-After": str(DEVICE_CODE_INTERVAL_SEC)},
        )

    if status != "approved":
        raise OAuthGrantError("invalid_grant", "Unknown device authorization state")

    user = dict(payload.get("approved_user") or {})
    if not user.get("email"):
        raise OAuthGrantError("invalid_grant", "Approved device code has no user context")

    _cache_delete("device_code", device_code)
    _cache_delete("device_user_code", str(payload.get("user_code", "")))
    _oauth_device_poll_events.labels("approved").inc()
    _oauth_tokens_issued.labels("device_code").inc()
    return issue_user_tokens(
        user,
        request,
        client_id=client["client_id"],
        scopes=list(payload.get("scopes") or []),
        session_type="device",
    )


def resolve_opaque_access_token(token: str) -> dict[str, Any] | None:
    payload = _cache_get_json("access_token", token)
    if not payload:
        return None
    expires_at = float(payload.get("expires_at", 0) or 0)
    if expires_at <= time.time():
        _cache_delete("access_token", token)
        return None
    return payload


def rotate_refresh_token(refresh_token: str, request) -> dict[str, Any]:
    lock_key = hash_token(refresh_token)
    if not _cache_set_json("refresh_lock", lock_key, {"locked": True}, 10, nx=True):
        _oauth_refresh_events.labels("locked").inc()
        raise OAuthGrantError("temporarily_unavailable", "Refresh already in progress", status_code=409)

    try:
        token_hash = hash_token(refresh_token)
        current = OAuthStore.get_refresh_token(token_hash)
        if not current:
            _oauth_refresh_events.labels("invalid").inc()
            raise OAuthGrantError("invalid_grant", "Refresh token is invalid or expired", status_code=401)

        now = time.time()
        if float(current.get("expires_at", 0) or 0) <= now or current.get("revoked_at"):
            _oauth_refresh_events.labels("revoked").inc()
            raise OAuthGrantError("invalid_grant", "Refresh token is revoked or expired", status_code=401)

        if current.get("consumed_at"):
            OAuthStore.revoke_refresh_family(str(current["family_id"]), reuse_detected=True)
            _oauth_refresh_events.labels("reused").inc()
            raise OAuthGrantError(
                "invalid_grant",
                "Refresh token has been reused",
                status_code=401,
                extra={"code": "refresh_token_reused"},
            )

        user = UserStore.get_user(str(current.get("email", ""))) or {
            "email": current.get("email", ""),
            "user_id": current.get("user_id", ""),
            "role": "submitter",
            "name": "",
            "is_admin": False,
        }
        scopes = list(current.get("scopes") or [])
        family_id = str(current["family_id"])
        session_type = str(current.get("session_type") or "browser")
        client_id = str(current.get("client_id") or "xcelsior-web")
        created_at = float(current.get("created_at", now) or now)

        new_refresh_token, new_session, new_record = _create_refresh_token_record(
            user=user,
            request=request,
            client_id=client_id,
            scopes=scopes,
            session_type=session_type,
            family_id=family_id,
            parent_token_id=str(current["token_id"]),
            created_at=created_at,
        )

        OAuthStore.mark_refresh_token_rotated(str(current["token_id"]), str(new_record["token_id"]))
        OAuthStore.create_refresh_token(new_record)
        UserStore.rotate_session_token(refresh_token, new_session)

        access = _issue_opaque_access_token(
            user,
            client_id=client_id,
            scopes=scopes,
            session_token=new_refresh_token,
            session_type=session_type,
        )
        _oauth_refresh_events.labels("rotated").inc()
        return {
            "access_token": access["access_token"],
            "refresh_token": new_refresh_token,
            "token_type": "Bearer",
            "expires_in": ACCESS_TOKEN_TTL_SEC,
            "refresh_expires_in": REFRESH_TOKEN_TTL_SEC,
            "scope": " ".join(scopes),
            "session_type": session_type,
        }
    finally:
        _cache_delete("refresh_lock", lock_key)


def revoke_refresh_session(session_token: str | None) -> None:
    if not session_token:
        return
    current = OAuthStore.get_refresh_token(hash_token(session_token))
    if current:
        OAuthStore.revoke_refresh_family(str(current["family_id"]))
    UserStore.delete_session(session_token)


def reset_auth_cache_for_tests() -> None:
    global _cache_instance, _defaults_ready
    with _cache_lock:
        _cache_instance = None
    with _defaults_lock:
        _defaults_ready = False
