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
        --email reviewer@example.test --password ...

Exit status is 0 only when nothing FAILED. BLOCKED checks do not fail the run
but are reported, counted, and must be resolved before the gate closes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import secrets
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


@dataclass
class Results:
    rows: list[tuple[str, str, str]] = field(default_factory=list)

    def record(self, status: str, name: str, detail: str = "") -> None:
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


def check_external_vantage(results: Results, host: str) -> None:
    """The run must not originate from inside our own network.

    A gate that silently ran from a machine on the tailnet would prove nothing
    about a provider's egress, so this is reported rather than assumed.
    """
    try:
        local = socket.gethostbyname(socket.gethostname())
    except OSError:
        local = ""
    private = local.startswith(("10.", "192.168.", "127.", "100.64."))
    if private:
        results.blocked(
            "external vantage point",
            f"resolved local address {local} looks internal; GX0 requires a non-Xcelsior "
            f"egress IP. Run this from a scheduled GitHub Actions job or a Cloud Run job.",
        )
    else:
        results.ok("external vantage point", f"local address {local or 'unknown'}")


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
    results.check(bool(peer.get("subject")), "TLS certificate chain validates", str(peer.get("issuer")))


def check_challenge(results: Results, http: httpx.Client, mcp_url: str) -> str | None:
    """BLOCKER 1: the 401 must name where to authenticate."""
    response = http.post(
        mcp_url,
        headers={"content-type": "application/json", "accept": "application/json, text/event-stream"},
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
        bool(header), "401 carries WWW-Authenticate", "header absent — connectors cannot discover OAuth"
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
    results.check(
        invalid.status_code in (401, 429) and (
            invalid.status_code == 429 or invalid_params.get("error") == "invalid_token"
        ),
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
        response.status_code == 200, "authorization-server metadata is reachable", f"{url} → {response.status_code}"
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
    results.check(
        hostile.status_code >= 400,
        "DCR rejects an off-allowlist redirect host",
        f"got {hostile.status_code}: {hostile.text[:200]}",
    )
    operator = http.post(
        endpoint,
        json={
            "client_name": "gx0-operator-attempt",
            "redirect_uris": ["http://127.0.0.1:8765/cb"],
            "scope": "instances:read hosts:evict",
        },
    )
    results.check(
        operator.status_code >= 400,
        "a DCR client cannot obtain operator scopes",
        f"got {operator.status_code}: {operator.text[:200]}",
    )
    good = http.post(
        endpoint,
        json={
            "client_name": "gx0-conformance",
            "redirect_uris": ["http://127.0.0.1:8765/cb", "http://localhost:8766/cb"],
            "grant_types": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_method": "none",
        },
    )
    if not results.check(
        good.status_code in (200, 201), "DCR registers a well-formed connector", f"{good.status_code}: {good.text[:300]}"
    ):
        return None
    registered = good.json()
    results.check(
        registered.get("resource", "").rstrip("/") == resource.rstrip("/"),
        "a DCR client is pinned to the MCP resource",
        json.dumps(registered.get("resource")),
    )
    return registered


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
    results.check(
        response.status_code >= 400,
        "an unresolvable metadata-document client id is rejected",
        f"got {response.status_code}",
    )


def sign_in(http: httpx.Client, issuer: str, email: str, password: str) -> bool:
    response = http.post(
        f"{issuer.rstrip('/')}/api/auth/login", json={"email": email, "password": password}
    )
    return response.status_code == 200


def check_full_authorization(
    results: Results,
    http: httpx.Client,
    metadata: dict[str, Any],
    client_id: str,
    redirect_uri: str,
    resource: str,
    label: str,
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
            "scope": "instances:read billing:read offline_access",
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
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(location).query)
    results.check(query.get("state", [""])[0] == state, f"{label}: state is echoed back")
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
        f"{token.status_code}: {token.text[:300]}",
    ):
        return None
    bundle = token.json()
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
    results: Results, http: httpx.Client, metadata: dict[str, Any], client_id: str, bundle: dict[str, Any]
) -> None:
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
        "refresh_token grant returns a working access token",
        f"{response.status_code}: {response.text[:300]}",
    ):
        return
    results.check(
        response.json().get("access_token") != bundle.get("access_token"),
        "refresh rotates the access token",
    )


def check_authenticated_tools_list(
    results: Results, http: httpx.Client, mcp_url: str, access_token: str
) -> None:
    headers = {
        "authorization": f"Bearer {access_token}",
        "content-type": "application/json",
        "accept": "application/json, text/event-stream",
    }
    init = http.post(
        mcp_url,
        headers=headers,
        content=json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "gx0-conformance", "version": "1.0.0"},
            },
        }),
    )
    if not results.check(
        init.status_code == 200, "authenticated initialize succeeds", f"{init.status_code}: {init.text[:300]}"
    ):
        return
    listed = http.post(
        mcp_url,
        headers=headers,
        content=json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
    )
    if not results.check(
        listed.status_code == 200, "authenticated tools/list succeeds", f"{listed.status_code}"
    ):
        return
    body = listed.text
    payload = body
    if body.lstrip().startswith("event:"):  # SSE framing
        payload = "".join(
            line[5:].strip() for line in body.splitlines() if line.startswith("data:")
        )
    try:
        tools = json.loads(payload)["result"]["tools"]
    except Exception as exc:
        results.fail("tools/list returns a tool array", f"{exc}: {body[:300]}")
        return
    names = {tool["name"] for tool in tools}
    results.ok("tools/list returns the public customer profile", f"{len(names)} tools")
    leaked = names & {
        "drain_host", "undrain_host", "evict_host_workloads", "retry_agent_command",
        "get_scheduler_health", "get_host_capacity", "list_reconciliation_findings",
    }
    results.check(not leaked, "no operator tools in the public profile", ", ".join(sorted(leaked)))
    missing_annotations = [t["name"] for t in tools if not t.get("annotations")]
    results.check(not missing_annotations, "every tool carries annotations", ", ".join(missing_annotations))
    missing_output = [t["name"] for t in tools if not t.get("outputSchema")]
    results.check(not missing_output, "every tool carries an output schema", ", ".join(missing_output))


def check_resource_substitution(
    results: Results, http: httpx.Client, metadata: dict[str, Any], client_id: str, redirect_uri: str
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
    results.check(
        response.status_code >= 400,
        "resource substitution is rejected at authorize",
        f"got {response.status_code}",
    )


# ── Entry point ───────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="https://mcp.xcelsior.ca/mcp", help="canonical connector URL")
    parser.add_argument("--issuer", default="", help="authorization server origin (default: from metadata)")
    parser.add_argument("--email", default="", help="reviewer account for the consent leg")
    parser.add_argument("--password", default="")
    parser.add_argument("--client-id", default="xcelsior-connector")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    mcp_url = args.base.rstrip("/")
    host = urllib.parse.urlsplit(mcp_url).hostname or ""
    results = Results()

    print(f"GX0 connector conformance against {mcp_url}\n")
    check_external_vantage(results, host)
    check_tls(results, host)

    with httpx.Client(timeout=args.timeout, follow_redirects=False) as http:
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

        if not (args.email and args.password):
            results.blocked(
                "authorization_code + PKCE round trip",
                "no --email/--password supplied; the consent leg needs a signed-in session",
            )
            results.blocked("loopback callback on two random ports", "same reason")
            results.blocked("refresh_token grant", "same reason")
            results.blocked("authenticated tools/list", "same reason")
            return _summarise(results)

        if not sign_in(http, issuer, args.email, args.password):
            results.fail("reviewer sign-in", "credentials rejected by /api/auth/login")
            return _summarise(results)
        results.ok("reviewer sign-in")

        check_resource_substitution(
            results, http, as_metadata, args.client_id, "http://127.0.0.1:8765/cb"
        )

        bundle = check_full_authorization(
            results, http, as_metadata, args.client_id,
            "https://claude.ai/api/mcp/auth_callback", resource, "provider callback",
        )
        # Two different random ports in the same run: RFC 8252 §7.3, and the
        # single most common reason a native client fails on its second attempt.
        for port in (secrets.randbelow(10000) + 40000, secrets.randbelow(10000) + 50000):
            check_full_authorization(
                results, http, as_metadata, args.client_id,
                f"http://127.0.0.1:{port}/callback", resource, f"loopback :{port}",
            )
        if registered:
            check_full_authorization(
                results, http, as_metadata, registered["client_id"],
                "http://127.0.0.1:8765/cb", resource, "DCR client",
            )
        if bundle:
            check_refresh(results, http, as_metadata, args.client_id, bundle)
            check_authenticated_tools_list(results, http, mcp_url, bundle["access_token"])

    return _summarise(results)


def _summarise(results: Results) -> int:
    passed = sum(1 for status, _, _ in results.rows if status == PASS)
    print(
        f"\nGX0: {passed} passed, {results.failures} failed, {results.blocks} blocked"
    )
    if results.blocks:
        print("A blocked gate is not a passed gate — resolve the environment and re-run.")
    return 1 if results.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
