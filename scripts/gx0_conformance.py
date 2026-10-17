#!/usr/bin/env python3
"""GX0 — connector conformance, executed against a real endpoint over the internet.

Gate GX0 from docs/mcp-enterprise-adoption-plan.md §6. Every assertion below is
one a directory reviewer's client makes implicitly on first connect; running
them from an external vantage point is the whole point, because localhost is
precisely what hides WAF and TLS problems.

    python3 scripts/gx0_conformance.py --base https://mcp.xcelsior.ca/mcp

Assertions that need a signed-in browser session (the consent screen) are only
attempted when credentials are supplied; without them the script reports
BLOCKED(env) for those checks rather than passing them, because a gate that
cannot run is never green.

    python3 scripts/gx0_conformance.py \\
        --base https://mcp.xcelsior.ca/mcp \\
        --email reviewer@example.test  # prompts for password without echoing it

Exit status is 0 only when nothing FAILED. BLOCKED checks do not fail the run
but are reported, counted, and must be resolved before the gate closes.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import os
import re
import secrets
import signal
import socket
import ssl
import sys
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover - operator-facing script
    print("gx0_conformance requires httpx (pip install httpx)", file=sys.stderr)
    raise SystemExit(2) from None


PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED(env)"
DEFAULT_CIMD_CLIENT_ID = "https://downloads.xcelsior.ca/gx0/client-v1.json"
READ_SCOPES = "instances:read billing:read offline_access"


_SECRET_FIELD = re.compile(
    r'("(?:access[_-]?token|refresh[_-]?token|session[_-]?token|id[_-]?token|device[_-]?code|client[_-]?secret)"\s*:\s*")[^"]*',
    re.IGNORECASE,
)
_SECRET_PARAM = re.compile(
    r"([?&](?:code|access[_-]?token|refresh[_-]?token)=)[^&#\s\"]+", re.IGNORECASE
)
_BEARER_VALUE = re.compile(r"\b(xoa_|xrt_|xc_)[A-Za-z0-9_-]{8,}")


def redact(text: str) -> str:
    """Hide token and code values. The run's output is meant to be pasted into
    a ticket or a chat; a refresh token in it is a working 30-day credential
    for the reviewer account."""
    text = _SECRET_FIELD.sub(r"\1<redacted>", text)
    text = _SECRET_PARAM.sub(r"\1<redacted>", text)
    return _BEARER_VALUE.sub(r"\1<redacted>", text)


@dataclass
class Results:
    rows: list[tuple[str, str, str]] = field(default_factory=list)

    def record(self, status: str, name: str, detail: str = "") -> None:
        detail = redact(detail)
        self.rows.append((status, name, detail))
        marker = {PASS: "  ok  ", FAIL: " FAIL ", BLOCKED: "BLOCKED"}[status]
        print(f"[{marker}] {name}" + (f"\n          {detail}" if detail else ""))

    def ok(self, name: str, detail: str = "") -> None:
        self.record(PASS, name, detail)

    def fail(self, name: str, detail: str = "") -> None:
        self.record(FAIL, name, detail)

    def blocked(self, name: str, detail: str = "") -> None:
        self.record(BLOCKED, name, detail)

    def check(self, condition: bool, name: str, detail: str = "") -> bool:
        (self.ok if condition else self.fail)(name, detail)
        return condition

    @property
    def failures(self) -> int:
        return sum(1 for status, _, _ in self.rows if status == FAIL)

    @property
    def blocks(self) -> int:
        return sum(1 for status, _, _ in self.rows if status == BLOCKED)


@dataclass(repr=False)
class IssuedSession:
    label: str
    client_id: str
    access_token: str
    refresh_token: str
    browser: bool = False


@dataclass(repr=False)
class SessionCleanup:
    """Keep every grant, including pre-rotation tokens, until verified logout.

    Xcelsior's logout accepts the refresh cookie naming the session to revoke.
    Retain the web login until last to authenticate logout even for an expired
    connector access token. Requests are built without the consent cookie jar,
    so a stale cookie cannot select the wrong session or mask a rejected bearer.
    """

    results: Results
    http: httpx.Client
    issuer: str
    token_endpoint: str
    mcp_url: str
    sessions: list[IssuedSession] = field(default_factory=list)

    def track(
        self, bundle: dict[str, Any], client_id: str, label: str, *, browser: bool = False
    ) -> None:
        access = bundle.get("access_token", "")
        refresh = bundle.get("refresh_token") or bundle.get("session_token", "")
        if access or refresh:
            self.sessions.append(IssuedSession(label, client_id, access, refresh, browser))
        self.results.check(bool(access and refresh), f"{label}: access and refresh tokens issued")

    def cleanup(self) -> None:
        if not self.sessions:
            return
        # The login is tracked first; reverse order revokes rotated grants first
        # and the web login last. Also revoke the old token's cached access.
        sessions = list(reversed(self.sessions))
        login = next((s.access_token for s in self.sessions if s.browser), "")
        for session in sessions:
            try:
                response = self.http.send(
                    httpx.Request(
                        "POST",
                        f"{self.issuer.rstrip('/')}/api/auth/logout",
                        headers={"authorization": f"Bearer {login or session.access_token}"},
                        cookies={"xcelsior_refresh": session.refresh_token},
                    )
                )
                self.results.check(
                    response.status_code == 200,
                    f"cleanup {session.label}: logout accepted",
                    f"HTTP {response.status_code}",
                )
            except httpx.HTTPError as exc:
                self.results.fail(f"cleanup {session.label}: logout", type(exc).__name__)
        # Logout's 200 is not evidence of revocation: it can be a no-op.
        # Check every bearer and every refresh credential independently.
        for session in sessions:
            try:
                headers = {
                    "authorization": f"Bearer {session.access_token}",
                    "accept": "application/json, text/event-stream",
                }
                if session.browser:
                    request = httpx.Request(
                        "GET", f"{self.issuer.rstrip('/')}/api/auth/me", headers=headers
                    )
                else:
                    request = httpx.Request(
                        "POST",
                        self.mcp_url,
                        headers=headers,
                        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                    )
                response = self.http.send(request)
                self.results.check(
                    response.status_code == 401,
                    f"cleanup {session.label}: access token rejected",
                    f"HTTP {response.status_code}",
                )
            except httpx.HTTPError as exc:
                self.results.fail(f"cleanup {session.label}: access revocation", type(exc).__name__)
            try:
                response = self.http.send(
                    httpx.Request(
                        "POST",
                        self.token_endpoint,
                        data={
                            "grant_type": "refresh_token",
                            "client_id": session.client_id,
                            "refresh_token": session.refresh_token,
                        },
                    )
                )
                payload = response.json()
                rejected = (
                    response.status_code in (400, 401) and payload.get("error") == "invalid_grant"
                )
                self.results.check(
                    rejected,
                    f"cleanup {session.label}: refresh token rejected",
                    f"HTTP {response.status_code}",
                )
                if response.status_code == 200:
                    # A failed logout must not leave the verification's newly
                    # rotated session behind. Revoke it with its own bearer.
                    self.http.send(
                        httpx.Request(
                            "POST",
                            f"{self.issuer.rstrip('/')}/api/auth/logout",
                            headers={"authorization": f"Bearer {payload.get('access_token', '')}"},
                            cookies={"xcelsior_refresh": payload.get("refresh_token", "")},
                        )
                    )
            except (httpx.HTTPError, ValueError) as exc:
                self.results.fail(
                    f"cleanup {session.label}: refresh revocation", type(exc).__name__
                )
        self.http.cookies.clear()


def pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )
    return verifier, challenge


def parse_challenge(header: str) -> dict[str, str]:
    """Pull the auth-params out of a `Bearer k="v", k2="v2"` challenge."""
    if not header.startswith("Bearer"):
        return {}
    return dict(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', header[6:]))


# ── Individual checks ─────────────────────────────────────────────────────


#: Xcelsior's own servers (prod, Headscale, the edge proxy). A run from one of
#: them proves nothing about a provider's egress. Add more, such as the office
#: connection, with GX0_INTERNAL_IPS="1.2.3.4 5.6.7.8".
KNOWN_INTERNAL_IPS = {"46.225.20.97", "149.28.121.61", "45.76.3.128"}


def egress_ip(timeout: float = 10.0) -> str:
    """The public address this run reaches the internet from."""
    try:
        trace = httpx.get("https://1.1.1.1/cdn-cgi/trace", timeout=timeout).text
    except httpx.HTTPError:
        return ""
    match = re.search(r"^ip=(\S+)$", trace, re.MULTILINE)
    return match.group(1) if match else ""


def check_external_vantage(results: Results, host: str) -> None:
    """The run must not originate from inside our own network.

    A gate that silently ran from a machine on the tailnet would prove nothing
    about a provider's egress, so this is reported rather than assumed. It reads
    the public egress address: the hostname's own address is 127.0.1.1 on every
    Debian or Ubuntu machine, which blocked the check wherever it ran.
    """
    internal = KNOWN_INTERNAL_IPS | set(os.environ.get("GX0_INTERNAL_IPS", "").split())
    ip = egress_ip()
    if not ip:
        results.blocked("external vantage point", "could not determine the public egress IP")
    elif ip in internal:
        results.blocked(
            "external vantage point",
            f"egress {ip} is an Xcelsior address; run from a network Xcelsior does not own "
            "(a phone hotspot, Cloud Shell, a CI runner).",
        )
    else:
        results.ok("external vantage point", f"egress {ip}")


def check_tls(results: Results, host: str, port: int = 443) -> None:
    context = ssl.create_default_context()
    try:
        with socket.create_connection((host, port), timeout=10) as raw:
            with context.wrap_socket(raw, server_hostname=host) as tls:
                version = tls.version() or ""
                peer = tls.getpeercert() or {}
    except Exception as exc:
        results.fail("TLS handshake", f"{host}:{port} — {exc}")
        return
    results.check(
        version in {"TLSv1.2", "TLSv1.3"}, "TLS version is 1.2 or 1.3", f"negotiated {version}"
    )
    results.check(
        bool(peer.get("subject")), "TLS certificate chain validates", str(peer.get("issuer"))
    )


def check_challenge(results: Results, http: httpx.Client, mcp_url: str) -> str | None:
    """BLOCKER 1: the 401 must name where to authenticate."""
    response = http.post(
        mcp_url,
        headers={
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
        },
        content=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
    )
    if not results.check(
        response.status_code == 401,
        "unauthenticated initialize returns 401",
        f"got {response.status_code}",
    ):
        return None
    header = response.headers.get("www-authenticate", "")
    if not results.check(
        bool(header),
        "401 carries WWW-Authenticate",
        "" if header else "header absent — connectors cannot discover OAuth",
    ):
        return None
    params = parse_challenge(header)
    metadata_url = params.get("resource_metadata", "")
    results.check(bool(params.get("realm")), "challenge names a realm", header)
    results.check(bool(metadata_url), "challenge names resource_metadata", header)

    invalid = http.post(
        mcp_url,
        headers={
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
            "authorization": "Bearer gx0-definitely-not-a-token",
        },
        content=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
    )
    invalid_params = parse_challenge(invalid.headers.get("www-authenticate", ""))
    if invalid.status_code == 429:
        results.blocked(
            "invalid token is challenged with error=invalid_token", "MCP endpoint returned HTTP 429"
        )
    else:
        results.check(
            invalid.status_code == 401 and invalid_params.get("error") == "invalid_token",
            "invalid token is challenged with error=invalid_token",
            f"status {invalid.status_code}, challenge {invalid.headers.get('www-authenticate', '')!r}",
        )
    return metadata_url or None


def check_protected_resource(
    results: Results, http: httpx.Client, metadata_url: str, mcp_url: str
) -> dict[str, Any] | None:
    response = http.get(metadata_url, headers={"accept": "application/json"})
    if not results.check(
        response.status_code == 200,
        "following WWW-Authenticate reaches protected-resource metadata",
        f"{metadata_url} → {response.status_code}",
    ):
        return None
    document = response.json()
    results.check(
        document.get("resource", "").rstrip("/") == mcp_url.rstrip("/"),
        "metadata `resource` is the exact connector URL",
        f"metadata says {document.get('resource')!r}, connector is {mcp_url!r}",
    )
    results.check(
        bool(document.get("authorization_servers")),
        "metadata names an authorization server",
        json.dumps(document.get("authorization_servers")),
    )
    scopes = document.get("scopes_supported") or []
    results.check(
        not [s for s in scopes if s.startswith(("hosts:", "control_plane:"))],
        "public metadata advertises no operator scopes",
        json.dumps(scopes),
    )
    return document


def check_authorization_server(
    results: Results, http: httpx.Client, issuer: str
) -> dict[str, Any] | None:
    url = f"{issuer.rstrip('/')}/.well-known/oauth-authorization-server"
    response = http.get(url, headers={"accept": "application/json"})
    if not results.check(
        response.status_code == 200,
        "authorization-server metadata is reachable",
        f"{url} → {response.status_code}",
    ):
        return None
    metadata = response.json()
    results.check(
        metadata.get("code_challenge_methods_supported") == ["S256"],
        "authorization server is S256-PKCE only",
        json.dumps(metadata.get("code_challenge_methods_supported")),
    )
    results.check(
        metadata.get("resource_indicators_supported") is True,
        "authorization server supports RFC 8707 resource indicators",
    )
    results.check(
        "refresh_token" in (metadata.get("grant_types_supported") or []),
        "refresh_token grant is advertised",
    )
    results.check(
        metadata.get("client_id_metadata_document_supported") is True,
        "CIMD is advertised as the preferred client-identification path",
    )
    results.check(
        bool(metadata.get("registration_endpoint")),
        "RFC 7591 registration endpoint is advertised for compatibility clients",
    )
    return metadata


def check_dcr(
    results: Results, http: httpx.Client, metadata: dict[str, Any], resource: str
) -> dict[str, Any] | None:
    endpoint = metadata.get("registration_endpoint")
    if not endpoint:
        results.blocked("DCR register", "no registration_endpoint advertised")
        return None
    hostile = http.post(
        endpoint,
        json={"client_name": "gx0-hostile", "redirect_uris": ["https://gx0-evil.example/steal"]},
    )
    if _record_dcr_rejection(
        results, hostile, "DCR rejects an off-allowlist redirect host", "invalid_redirect_uri"
    ):
        results.blocked(
            "a DCR client cannot obtain operator scopes",
            "skipped because DCR is rate-limited",
        )
        results.blocked(
            "DCR registers a well-formed connector", "skipped because DCR is rate-limited"
        )
        return None
    operator = http.post(
        endpoint,
        json={
            "client_name": "gx0-operator-attempt",
            "redirect_uris": ["http://127.0.0.1:8765/cb"],
            "scope": "instances:read hosts:evict",
        },
    )
    if _record_dcr_rejection(
        results, operator, "a DCR client cannot obtain operator scopes", "invalid_scope"
    ):
        results.blocked(
            "DCR registers a well-formed connector", "skipped because DCR is rate-limited"
        )
        return None
    good = http.post(
        endpoint,
        json={
            "client_name": "gx0-conformance",
            "redirect_uris": ["http://127.0.0.1:8765/cb", "http://localhost:8766/cb"],
            "grant_types": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_method": "none",
        },
    )
    if good.status_code == 429:
        results.blocked(
            "DCR registers a well-formed connector",
            _dcr_rate_limit_detail(good),
        )
        return None
    if not results.check(
        good.status_code in (200, 201),
        "DCR registers a well-formed connector",
        f"expected HTTP 200 or 201; got HTTP {good.status_code}",
    ):
        return None
    registered = good.json()
    results.check(
        registered.get("resource", "").rstrip("/") == resource.rstrip("/"),
        "a DCR client is pinned to the MCP resource",
        json.dumps(registered.get("resource")),
    )
    return registered


def _dcr_rate_limit_detail(response: httpx.Response) -> str:
    retry_after = response.headers.get("retry-after", "")
    suffix = f" Retry-After: {retry_after} seconds." if retry_after.isdigit() else ""
    return "DCR returned HTTP 429; this check could not run." + suffix


def _record_dcr_rejection(
    results: Results, response: httpx.Response, name: str, expected_error: str
) -> bool:
    """Require the expected policy rejection; a throttle is an unrun gate."""
    if response.status_code == 429:
        results.blocked(name, _dcr_rate_limit_detail(response))
        return True
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    actual_error = payload.get("error") if isinstance(payload, dict) else None
    results.check(
        response.status_code in (400, 403) and actual_error == expected_error,
        name,
        f"expected HTTP 400/403 with error={expected_error}; got HTTP {response.status_code}"
        + (f" with error={actual_error}" if actual_error else " without an OAuth error code"),
    )
    return False


def check_cimd_rejection(results: Results, http: httpx.Client, metadata: dict[str, Any]) -> None:
    """An untrusted metadata-document client id must be refused."""
    authorize = metadata.get("authorization_endpoint", "")
    if not authorize:
        results.blocked("CIMD rejection", "no authorization_endpoint advertised")
        return
    _, challenge = pkce()
    response = http.get(
        authorize,
        params={
            "response_type": "code",
            "client_id": "https://gx0-nonexistent.invalid/client.json",
            "redirect_uri": "http://127.0.0.1:8765/cb",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        headers={"accept": "text/html"},
        follow_redirects=False,
    )
    if response.status_code == 429:
        results.blocked(
            "an unresolvable metadata-document client id is rejected",
            "authorization endpoint returned HTTP 429",
        )
        return
    results.check(
        response.status_code == 400,
        "an unresolvable metadata-document client id is rejected",
        f"expected HTTP 400; got HTTP {response.status_code}",
    )


def sign_in(
    http: httpx.Client, issuer: str, email: str, password: str, cleanup: SessionCleanup
) -> tuple[str, str]:
    response = http.post(
        f"{issuer.rstrip('/')}/api/auth/login", json={"email": email, "password": password}
    )
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if response.status_code == 200:
        if isinstance(payload, dict) and payload.get("mfa_required"):
            return (
                "blocked",
                "reviewer account requires MFA; this script cannot complete interactive MFA",
            )
        payload["refresh_token"] = payload.get("refresh_token") or response.cookies.get(
            "xcelsior_refresh", ""
        )
        cleanup.track(payload, "xcelsior-web", "reviewer login", browser=True)
        if not (payload.get("access_token") and payload.get("refresh_token")):
            return "fail", "login did not issue both access token and refresh cookie"
        return "pass", ""
    if response.status_code == 429:
        retry_after = response.headers.get("retry-after", "")
        detail = "reviewer login is rate-limited"
        if retry_after.isdigit():
            detail += f"; retry after {retry_after} seconds"
        return "blocked", detail
    error = payload.get("error") if isinstance(payload, dict) else None
    error_code = error.get("code") if isinstance(error, dict) else None
    if error_code == "oauth_only":
        return "blocked", "reviewer account is OAuth-only; use a password-enabled reviewer account"
    if error_code == "email_not_verified":
        return "blocked", "reviewer account email is not verified"
    if response.status_code == 401:
        return "fail", "reviewer email or password was rejected"
    return "fail", f"reviewer login returned HTTP {response.status_code}" + (
        f" ({error_code})" if error_code else ""
    )


def check_full_authorization(
    results: Results,
    http: httpx.Client,
    metadata: dict[str, Any],
    client_id: str,
    redirect_uri: str,
    resource: str,
    label: str,
    cleanup: SessionCleanup,
) -> dict[str, Any] | None:
    verifier, challenge = pkce()
    state = secrets.token_urlsafe(12)
    authorize = http.get(
        metadata["authorization_endpoint"],
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "scope": READ_SCOPES,
            "resource": resource,
        },
        headers={"accept": "text/html"},
        follow_redirects=False,
    )
    if authorize.status_code == 200:
        key = re.search(r'name="consent_key"\s+value="([^"]+)"', authorize.text)
        if not key:
            results.fail(f"{label}: consent screen", "no consent form in the authorize response")
            return None
        authorize = http.post(
            metadata["authorization_endpoint"],
            data={"consent_key": key.group(1), "decision": "approve"},
            follow_redirects=False,
        )
    if not results.check(
        authorize.status_code in (302, 303),
        f"{label}: authorize redirects to the client callback",
        f"got {authorize.status_code}",
    ):
        return None
    location = authorize.headers.get("location", "")
    callback = urllib.parse.urlsplit(location)
    expected = urllib.parse.urlsplit(redirect_uri)
    if not results.check(
        (callback.scheme, callback.netloc, callback.path)
        == (expected.scheme, expected.netloc, expected.path),
        f"{label}: redirect matches the registered callback",
    ):
        return None
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(location).query)
    if not results.check(query.get("state", [""])[0] == state, f"{label}: state is echoed back"):
        return None
    code = query.get("code", [""])[0]
    if not results.check(bool(code), f"{label}: an authorization code is issued", location):
        return None

    token = http.post(
        metadata["token_endpoint"],
        data={
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
            "resource": resource,
        },
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    if not results.check(
        token.status_code == 200,
        f"{label}: token endpoint accepts form encoding",
        f"HTTP {token.status_code}",
    ):
        return None
    bundle = token.json()
    cleanup.track(bundle, client_id, label)
    results.check(
        bundle.get("resource", "").rstrip("/") == resource.rstrip("/"),
        f"{label}: the token is bound to the MCP resource",
        json.dumps(bundle.get("resource")),
    )
    results.check(
        int(bundle.get("expires_in") or 0) >= 1800,
        f"{label}: access token lifetime is ~1h",
        f"expires_in={bundle.get('expires_in')}",
    )
    return bundle


def check_refresh(
    results: Results,
    http: httpx.Client,
    metadata: dict[str, Any],
    client_id: str,
    bundle: dict[str, Any],
    cleanup: SessionCleanup,
) -> dict[str, Any] | None:
    refresh_token = bundle.get("refresh_token")
    if not refresh_token:
        results.fail("refresh_token grant", "no refresh_token was issued")
        return
    response = http.post(
        metadata["token_endpoint"],
        data={
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
        },
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    if not results.check(
        response.status_code == 200,
        "refresh_token grant succeeds",
        f"HTTP {response.status_code}",
    ):
        return
    refreshed = response.json()
    cleanup.track(refreshed, client_id, "refreshed token")
    results.check(
        bool(refreshed.get("access_token"))
        and refreshed["access_token"] != bundle.get("access_token"),
        "refresh rotates the access token",
    )
    results.check(
        bool(refreshed.get("refresh_token")) and refreshed["refresh_token"] != refresh_token,
        "refresh rotates the refresh token",
    )
    return refreshed


def rpc_result(response: httpx.Response, request_id: int) -> dict[str, Any] | None:
    """Accept JSON or SSE, but never mistake HTTP 200 + RPC error for success."""
    if response.status_code != 200:
        return None
    messages: list[Any] = []
    try:
        messages.append(response.json())
    except ValueError:
        for event in re.split(r"\r?\n\r?\n", response.text):
            data = "\n".join(
                line[5:].lstrip() for line in event.splitlines() if line.startswith("data:")
            )
            if data:
                try:
                    messages.append(json.loads(data))
                except ValueError:
                    continue
    for message in messages:
        if (
            isinstance(message, dict)
            and message.get("jsonrpc") == "2.0"
            and message.get("id") == request_id
        ):
            result = message.get("result")
            return result if isinstance(result, dict) and "error" not in message else None
    return None


def check_authenticated_tools_list(
    results: Results, http: httpx.Client, mcp_url: str, access_token: str, label: str
) -> None:
    headers = {
        "authorization": f"Bearer {access_token}",
        "content-type": "application/json",
        "accept": "application/json, text/event-stream",
    }
    init = http.send(
        httpx.Request(
            "POST",
            mcp_url,
            headers=headers,
            content=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "gx0-conformance", "version": "2.0.0"},
                    },
                }
            ),
        )
    )
    initialized = rpc_result(init, 1)
    if not results.check(
        initialized is not None and bool(initialized.get("protocolVersion")),
        f"{label}: authenticated initialize succeeds",
        f"HTTP {init.status_code}",
    ):
        return
    headers["mcp-protocol-version"] = initialized["protocolVersion"]
    if init.headers.get("mcp-session-id"):
        headers["mcp-session-id"] = init.headers["mcp-session-id"]
    notified = http.send(
        httpx.Request(
            "POST",
            mcp_url,
            headers=headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
    )
    results.check(
        notified.status_code == 202,
        f"{label}: initialized notification accepted",
        f"HTTP {notified.status_code}",
    )
    listed = http.send(
        httpx.Request(
            "POST",
            mcp_url,
            headers=headers,
            content=json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
        )
    )
    payload = rpc_result(listed, 2)
    tool_list = payload.get("tools") if payload is not None else None
    if not results.check(
        isinstance(tool_list, list)
        and bool(tool_list)
        and all(isinstance(t, dict) and isinstance(t.get("name"), str) for t in tool_list),
        f"{label}: authenticated tools/list succeeds",
        f"HTTP {listed.status_code}",
    ):
        return
    tools = tool_list
    names = {tool["name"] for tool in tools}
    results.ok(f"{label}: tools/list returns the public customer profile", f"{len(names)} tools")
    leaked = names & {
        "drain_host",
        "undrain_host",
        "evict_host_workloads",
        "retry_agent_command",
        "get_scheduler_health",
        "get_host_capacity",
        "list_reconciliation_findings",
    }
    results.check(
        not leaked, f"{label}: no operator tools in the public profile", ", ".join(sorted(leaked))
    )
    missing_annotations = [t["name"] for t in tools if not t.get("annotations")]
    results.check(
        not missing_annotations,
        f"{label}: every tool carries annotations",
        ", ".join(missing_annotations),
    )
    missing_output = [t["name"] for t in tools if not t.get("outputSchema")]
    results.check(
        not missing_output,
        f"{label}: every tool carries an output schema",
        ", ".join(missing_output),
    )
    if not results.check("list_instances" in names, f"{label}: read-only probe tool is advertised"):
        return
    called = http.send(
        httpx.Request(
            "POST",
            mcp_url,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "list_instances", "arguments": {"limit": 1}},
            },
        )
    )
    result = rpc_result(called, 3)
    data = result.get("structuredContent") if result else None
    if result and not isinstance(data, dict):
        for content in result.get("content", []):
            if content.get("type") == "text":
                try:
                    data = json.loads(content["text"])
                except (KeyError, ValueError):
                    continue
                break
    success = (
        bool(result)
        and not result.get("isError")
        and isinstance(data, dict)
        and not data.get("error")
        and data.get("ok") is not False
        and isinstance(data.get("instances"), list)
    )
    results.check(
        success,
        f"{label}: tools/call list_instances succeeds",
        f"HTTP {called.status_code}; instance data omitted",
    )


def check_cimd_connection(
    results: Results,
    http: httpx.Client,
    metadata: dict[str, Any],
    client_id: str,
    resource: str,
    cleanup: SessionCleanup,
) -> None:
    response = http.get(client_id, headers={"accept": "application/json"})
    if not results.check(
        response.status_code == 200,
        "CIMD: public client document is reachable",
        f"HTTP {response.status_code}",
    ):
        return
    document = response.json()
    if not results.check(
        urllib.parse.urlsplit(client_id).scheme == "https"
        and document.get("client_id") == client_id
        and response.headers.get("content-type", "").split(";")[0]
        in {"application/json", "application/ld+json"},
        "CIMD: HTTPS JSON document names its exact client ID",
    ):
        return
    redirect_uri = "http://127.0.0.1:8765/cb"
    if not results.check(
        redirect_uri in document.get("redirect_uris", []), "CIMD: probe callback is registered"
    ):
        return
    bundle = check_full_authorization(
        results, http, metadata, client_id, redirect_uri, resource, "CIMD client", cleanup
    )
    if bundle:
        check_authenticated_tools_list(
            results, http, resource, bundle.get("access_token", ""), "CIMD client"
        )


def check_resource_substitution(
    results: Results,
    http: httpx.Client,
    metadata: dict[str, Any],
    client_id: str,
    redirect_uri: str,
) -> None:
    _, challenge = pkce()
    response = http.get(
        metadata["authorization_endpoint"],
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "resource": "https://gx0-attacker.example/mcp",
        },
        headers={"accept": "text/html"},
        follow_redirects=False,
    )
    if response.status_code == 429:
        results.blocked(
            "resource substitution is rejected at authorize",
            "authorization endpoint returned HTTP 429",
        )
        return
    results.check(
        response.status_code == 400,
        "resource substitution is rejected at authorize",
        f"expected HTTP 400; got HTTP {response.status_code}",
    )


# ── Entry point ───────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base", default="https://mcp.xcelsior.ca/mcp", help="canonical connector URL"
    )
    parser.add_argument(
        "--issuer", default="", help="authorization server origin (default: from metadata)"
    )
    parser.add_argument(
        "--email", default=os.environ.get("GX0_EMAIL", ""), help="reviewer account (or GX0_EMAIL)"
    )
    parser.add_argument("--client-id", default="xcelsior-connector")
    parser.add_argument(
        "--cimd-client-id",
        default=os.environ.get("GX0_CIMD_CLIENT_ID", DEFAULT_CIMD_CLIENT_ID),
        help="public HTTPS client metadata document",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    mcp_url = args.base.rstrip("/")
    host = urllib.parse.urlsplit(mcp_url).hostname or ""
    results = Results()

    print(f"GX0 connector conformance against {mcp_url}\n")
    check_external_vantage(results, host)
    check_tls(results, host)

    with httpx.Client(
        timeout=args.timeout,
        follow_redirects=False,
        headers={"user-agent": f"gx0-conformance/2 run/{secrets.token_hex(8)}"},
    ) as http:
        metadata_url = check_challenge(results, http, mcp_url)
        if not metadata_url:
            return _summarise(results)
        resource_document = check_protected_resource(results, http, metadata_url, mcp_url)
        if not resource_document:
            return _summarise(results)
        issuer = args.issuer or (resource_document.get("authorization_servers") or [""])[0]
        as_metadata = check_authorization_server(results, http, issuer)
        if not as_metadata:
            return _summarise(results)
        resource = resource_document.get("resource", mcp_url)

        check_cimd_rejection(results, http, as_metadata)
        registered = check_dcr(results, http, as_metadata, resource)

        password = os.environ.get("GX0_PASSWORD", "")
        if args.email and not password and sys.stdin.isatty():
            password = getpass.getpass("Reviewer password (hidden): ")

        if not (args.email and password):
            results.blocked(
                "authorization_code + PKCE round trip",
                "provide --email and enter a password interactively, or set GX0_EMAIL/GX0_PASSWORD",
            )
            results.blocked("loopback callback on two random ports", "same reason")
            results.blocked("refresh_token grant", "same reason")
            results.blocked("authenticated tools/list", "same reason")
            results.blocked("CIMD connection and tool execution", "same reason")
            results.blocked("DCR client tool execution", "same reason")
            return _summarise(results)

        cleanup = SessionCleanup(results, http, issuer, as_metadata["token_endpoint"], mcp_url)
        previous_termination_handler = signal.getsignal(signal.SIGTERM)

        def terminate(_signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, terminate)
        try:
            run_authenticated_checks(
                results,
                http,
                issuer,
                args.email.strip(),
                password,
                as_metadata,
                resource,
                args.client_id,
                args.cimd_client_id,
                registered,
                cleanup,
            )
        except KeyboardInterrupt:
            results.fail("run interrupted", "revoking this run's sessions before exit")
        except Exception as exc:
            # Exception text can contain an authorization URL or a token body.
            results.fail("authenticated checks aborted", type(exc).__name__)
        finally:
            try:
                cleanup.cleanup()
            finally:
                signal.signal(signal.SIGTERM, previous_termination_handler)

    return _summarise(results)


def run_authenticated_checks(
    results: Results,
    http: httpx.Client,
    issuer: str,
    email: str,
    password: str,
    as_metadata: dict[str, Any],
    resource: str,
    client_id: str,
    cimd_client_id: str,
    registered: dict[str, Any] | None,
    cleanup: SessionCleanup,
) -> None:
    login_status, login_detail = sign_in(http, issuer, email, password, cleanup)
    if login_status == "blocked":
        results.blocked("reviewer sign-in", login_detail)
        return
    if login_status == "fail":
        results.fail("reviewer sign-in", login_detail)
        return
    results.ok("reviewer sign-in")

    check_resource_substitution(results, http, as_metadata, client_id, "http://127.0.0.1:8765/cb")

    bundle = check_full_authorization(
        results,
        http,
        as_metadata,
        client_id,
        "https://claude.ai/api/mcp/auth_callback",
        resource,
        "provider callback",
        cleanup,
    )
    # Two different random ports in the same run: RFC 8252 §7.3, and the
    # single most common reason a native client fails on its second attempt.
    for port in (secrets.randbelow(10000) + 40000, secrets.randbelow(10000) + 50000):
        check_full_authorization(
            results,
            http,
            as_metadata,
            client_id,
            f"http://127.0.0.1:{port}/callback",
            resource,
            f"loopback :{port}",
            cleanup,
        )
    if registered:
        dcr_bundle = check_full_authorization(
            results,
            http,
            as_metadata,
            registered["client_id"],
            "http://127.0.0.1:8765/cb",
            resource,
            "DCR client",
            cleanup,
        )
        if dcr_bundle:
            check_authenticated_tools_list(
                results, http, resource, dcr_bundle.get("access_token", ""), "DCR client"
            )
    else:
        results.blocked("DCR client tool execution", "no newly registered client")
    check_cimd_connection(results, http, as_metadata, cimd_client_id, resource, cleanup)
    if bundle:
        check_authenticated_tools_list(
            results, http, resource, bundle.get("access_token", ""), "provider callback"
        )
        refreshed = check_refresh(results, http, as_metadata, client_id, bundle, cleanup)
        if refreshed:
            check_authenticated_tools_list(
                results, http, resource, refreshed.get("access_token", ""), "refreshed token"
            )


def _summarise(results: Results) -> int:
    passed = sum(1 for status, _, _ in results.rows if status == PASS)
    print(f"\nGX0: {passed} passed, {results.failures} failed, {results.blocks} blocked")
    if results.blocks:
        print("A blocked gate is not a passed gate — resolve the environment and re-run.")
    return 1 if results.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
