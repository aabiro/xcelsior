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


# Headers a public client must never be allowed to inject.
UNTRUSTED_IDENTITY_HEADERS = (
    "x-worker-host-id",
    "x-worker-spiffe-id",
    "x-spiffe-id",
    "x-worker-identity",
    "x-forwarded-client-cert",
)


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


def strip_untrusted_identity_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of headers with public-injectable identity keys removed.

    Call at the API edge before any identity resolution when the request did
    not arrive from the private agent gateway.
    """
    out: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in UNTRUSTED_IDENTITY_HEADERS:
            continue
        out[key] = value
    return out


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
    Raises IdentityAdmissionError when gateway mode is on but headers are
    missing/mismatched — fail closed.
    """
    if not trusted_gateway_enabled():
        return None

    # Only accept identity headers when the gateway marker is present.
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


def require_admitted_host(
    host_row: Mapping[str, Any] | None,
    *,
    host_id: str,
) -> AdmittedWorkerIdentity:
    """Fail closed unless the host is registered and admitted."""
    if host_row is None:
        raise IdentityAdmissionError(
            "unknown_host",
            f"Host {host_id!r} is not registered",
            http_status=403,
        )
    admitted = host_row.get("admitted")
    if admitted is False or admitted == 0 or str(admitted).lower() in ("false", "0", "no"):
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
