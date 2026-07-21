"""Worker / agent identity admission (blueprint §19.2, Phase 10).

Production rule: every worker mutation maps to a unique *admitted* host
identity. Shared platform bearer remains a migration path only when the
caller is authenticated and the host_id is a registered, admitted host.

Untrusted inbound identity headers (``X-Worker-*``, ``X-Spiffe-Id``) are
stripped at the edge and never trusted from the public network — only
values set by a private gateway after mTLS/SPIFFE verification are
honored (via ``XCELSIOR_TRUSTED_AGENT_GATEWAY=1``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping


# Headers a public client must never be allowed to inject / spoof.
# Includes the gateway marker itself — only a private edge that also
# presents XCELSIOR_AGENT_GATEWAY_SECRET may re-set these.
UNTRUSTED_IDENTITY_HEADERS = (
    "x-worker-host-id",
    "x-worker-spiffe-id",
    "x-worker-provider-id",
    "x-spiffe-id",
    "x-worker-identity",
    "x-forwarded-client-cert",
    "x-xcelsior-agent-gateway",
    "x-xcelsior-gateway-auth",
)

# Shared secret the private agent gateway injects after mTLS. Public clients
# must not know it; without it gateway identity headers are ignored/rejected.
_GATEWAY_AUTH_HEADER = "x-xcelsior-gateway-auth"


@dataclass(frozen=True, slots=True)
class AdmittedWorkerIdentity:
    """Resolved production worker principal."""

    host_id: str
    source: str  # gateway_header | bearer_host | test
    spiffe_id: str | None = None
    provider_id: str | None = None


class IdentityAdmissionError(Exception):
    """Fail-closed identity rejection."""

    def __init__(self, code: str, message: str, *, http_status: int = 403):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


def trusted_gateway_enabled() -> bool:
    return os.environ.get("XCELSIOR_TRUSTED_AGENT_GATEWAY", "").lower() in (
        "1",
        "true",
        "yes",
    )


def agent_gateway_secret() -> str:
    """Shared secret the private agent gateway must present (may be empty)."""
    return (os.environ.get("XCELSIOR_AGENT_GATEWAY_SECRET") or "").strip()


def strip_untrusted_identity_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of headers with public-injectable identity keys removed.

    Call at the API edge before any identity resolution when the request did
    not arrive from a secret-authenticated private agent gateway.
    """
    out: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in UNTRUSTED_IDENTITY_HEADERS:
            continue
        out[key] = value
    return out


def gateway_headers_authenticated(headers: Mapping[str, str]) -> bool:
    """True only when the private gateway shared secret matches.

    A bare ``X-Xcelsior-Agent-Gateway: 1`` is forgeable on public ingress —
    blueprint §19.2 requires identity headers only after private-network
    gateway verification. The secret is that proof.
    """
    secret = agent_gateway_secret()
    if not secret:
        return False
    presented = (headers.get(_GATEWAY_AUTH_HEADER) or "").strip()
    if not presented:
        return False
    # Constant-time compare for the shared secret.
    import hmac

    return hmac.compare_digest(presented, secret)


def spiffe_id_for_host(host_id: str, *, trust_domain: str | None = None) -> str:
    """Canonical SPIFFE ID shape for an admitted host (scaffold for SPIRE)."""
    domain = (trust_domain or os.environ.get("XCELSIOR_SPIFFE_TRUST_DOMAIN") or "xcelsior.ca").strip()
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in host_id)
    return f"spiffe://{domain}/worker/host/{safe}"


def resolve_gateway_identity(
    headers: Mapping[str, str],
    *,
    required_host_id: str | None = None,
) -> AdmittedWorkerIdentity | None:
    """If trusted gateway mode is on, map gateway-set headers to an identity.

    Returns None when gateway mode is off (caller falls through to bearer).
    Raises IdentityAdmissionError when gateway mode is on but:
      - gateway secret is not configured (misconfig fail-closed),
      - request lacks matching gateway auth secret (public spoof rejected),
      - host/SPIFFE headers are missing/mismatched.
    """
    if not trusted_gateway_enabled():
        return None

    # Fail closed if operators enabled gateway mode without a secret —
    # otherwise any client could set X-Xcelsior-Agent-Gateway: 1.
    if not agent_gateway_secret():
        raise IdentityAdmissionError(
            "gateway_secret_unconfigured",
            "XCELSIOR_TRUSTED_AGENT_GATEWAY requires XCELSIOR_AGENT_GATEWAY_SECRET",
            http_status=503,
        )

    if not gateway_headers_authenticated(headers):
        raise IdentityAdmissionError(
            "gateway_auth_required",
            "Agent gateway authentication failed",
            http_status=401,
        )

    if headers.get("x-xcelsior-agent-gateway", "").lower() not in ("1", "true", "yes"):
        raise IdentityAdmissionError(
            "gateway_required",
            "Agent gateway identity required",
            http_status=401,
        )

    host_id = (headers.get("x-worker-host-id") or "").strip()
    spiffe = (headers.get("x-worker-spiffe-id") or headers.get("x-spiffe-id") or "").strip() or None
    if not host_id:
        raise IdentityAdmissionError(
            "missing_host_identity",
            "Gateway did not set worker host identity",
            http_status=401,
        )
    if required_host_id and host_id != required_host_id:
        raise IdentityAdmissionError(
            "host_mismatch",
            "Gateway host identity does not match request host_id",
            http_status=403,
        )
    if spiffe:
        expected_prefix = f"spiffe://{os.environ.get('XCELSIOR_SPIFFE_TRUST_DOMAIN', 'xcelsior.ca').strip()}/worker/"
        if not spiffe.startswith("spiffe://") or (
            expected_prefix
            and not spiffe.startswith(expected_prefix)
            and "/worker/" not in spiffe
        ):
            raise IdentityAdmissionError(
                "invalid_spiffe",
                "Gateway SPIFFE ID is not a worker identity",
                http_status=403,
            )
    return AdmittedWorkerIdentity(
        host_id=host_id,
        source="gateway_header",
        spiffe_id=spiffe,
        provider_id=(headers.get("x-worker-provider-id") or "").strip() or None,
    )


def shared_bearer_migration_enabled() -> bool:
    """Time-bounded migration flag for the platform shared fleet bearer.

    When false (default), production rejects ``api-token`` / ``api-admin``
    principals acting on a host they do not own — blueprint §19.2 rotate-away.
    """
    return os.environ.get("XCELSIOR_AGENT_SHARED_BEARER_MIGRATION", "").lower() in (
        "1",
        "true",
        "yes",
    )


def shared_bearer_host_allowlisted(host_id: str) -> bool:
    """Optional CSV allowlist of host_ids the shared bearer may still touch."""
    raw = (os.environ.get("XCELSIOR_AGENT_SHARED_BEARER_HOSTS") or "").strip()
    if not raw:
        # Empty allowlist while migration is on = all admitted hosts (broad migration).
        return True
    allowed = {h.strip() for h in raw.split(",") if h.strip()}
    return host_id in allowed


def is_shared_fleet_bearer(principal: Mapping[str, Any] | None) -> bool:
    """True for the platform shared agent bearer principals (migration path)."""
    if not principal:
        return False
    uid = str(principal.get("user_id") or "")
    return uid in ("api-token", "api-admin")


def allow_shared_bearer_for_host(
    principal: Mapping[str, Any] | None,
    *,
    host_id: str,
    host_owner: str | None,
) -> tuple[bool, str]:
    """Decide whether a principal may mutate ``host_id`` under production rules.

    Returns ``(allowed, reason_code)``.

    Order is intentional (Phase 10 skeptic fix):
      1. Owner match always allows.
      2. Shared fleet bearer (``api-admin`` / ``api-token`` master token shape
         from ``_get_current_user`` — often ``is_admin=True``) is **never**
         treated as a generic human admin. It requires the migration flag
         (+ optional host allowlist). Checking ``is_admin`` first made the
         gate dead for the real master token principal.
      3. Human platform admins (``is_admin`` and *not* shared fleet bearer)
         may act.
      4. Otherwise deny.
    """
    if not principal:
        return False, "no_principal"
    uid = str(principal.get("user_id") or "")
    email = str(principal.get("email") or "")
    owner = (host_owner or "").strip()
    if owner and (owner == uid or owner == email):
        return True, "owner"
    # Shared fleet bearer BEFORE is_admin — master token sets is_admin=True.
    if is_shared_fleet_bearer(principal):
        if not shared_bearer_migration_enabled():
            return False, "shared_bearer_migration_disabled"
        if not shared_bearer_host_allowlisted(host_id):
            return False, "shared_bearer_host_not_allowlisted"
        return True, "shared_bearer_migration"
    if principal.get("is_admin") or str(principal.get("role") or "") == "admin":
        return True, "admin"
    return False, "not_owner"


def _is_explicitly_admitted(value: Any) -> bool:
    """True only for an explicit affirmative admission flag (fail closed)."""
    if value is True or value == 1:
        return True
    if isinstance(value, str) and value.strip().lower() in ("true", "1", "yes"):
        return True
    return False


def require_admitted_host(
    host_row: Mapping[str, Any] | None,
    *,
    host_id: str,
) -> AdmittedWorkerIdentity:
    """Fail closed unless the host is registered and *explicitly* admitted.

    Missing/None/empty ``admitted`` is treated as not admitted — production
    never fail-opens on an absent flag (Phase 10 / skeptic gate).
    """
    if host_row is None:
        raise IdentityAdmissionError(
            "unknown_host",
            f"Host {host_id!r} is not registered",
            http_status=403,
        )
    if not _is_explicitly_admitted(host_row.get("admitted")):
        raise IdentityAdmissionError(
            "host_not_admitted",
            f"Host {host_id!r} is not admitted",
            http_status=403,
        )
    return AdmittedWorkerIdentity(
        host_id=str(host_row.get("host_id") or host_id),
        source="bearer_host",
        spiffe_id=spiffe_id_for_host(str(host_row.get("host_id") or host_id)),
        provider_id=(host_row.get("owner") or host_row.get("provider_id") or None),
    )
