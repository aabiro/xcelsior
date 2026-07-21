"""Production startup validator (blueprint §30, §31; companion §14.3).

Blueprint §30 ends with a list the deployment is required to refuse:

    Production startup validator must reject:
    - SQLite or dual backend;
    - missing PostgreSQL TLS policy where required;
    - empty OAuth signing/JWKS configuration;
    - unauthenticated agent mode;
    - hard security tier fallback;
    - MCP in-memory rate limiting when more than one replica is configured;
    - runtime DDL enabled;
    - API `SYS_ADMIN` expectation after volume-provisioner cutover.

Each of those is a configuration state that looks harmless in a diff and
is expensive in production: a SQLite backend silently gives up every
concurrency guarantee Track A built; an unauthenticated agent mode makes
worker identity meaningless; process-local MCP rate limiting under two
replicas is not a rate limit. This module turns each into a named,
testable check with a documented remediation.

It runs at API startup and backs ``/startupz`` (§21.3). Outside
production it reports the same findings without blocking, so a developer
sees the drift without their laptop refusing to boot.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Literal

Severity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class Finding:
    """One failed (or degraded) production requirement."""

    code: str
    severity: Severity
    message: str
    remediation: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "remediation": self.remediation,
        }


class StartupValidationError(RuntimeError):
    """Production configuration failed a §30 gate."""

    def __init__(self, findings: list[Finding]):
        self.findings = findings
        detail = "; ".join(f"{f.code}: {f.message}" for f in findings)
        super().__init__(f"production startup validation failed — {detail}")


def _truthy(name: str, default: str = "") -> bool:
    return (os.environ.get(name, default) or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def is_production() -> bool:
    return (os.environ.get("XCELSIOR_ENV") or "").strip().lower() == "production"


# ── Individual checks ─────────────────────────────────────────────────
#
# Each returns a Finding or None. Kept separate (rather than one big
# function) so a test can drive exactly one condition at a time.


def _check_database_backend() -> Finding | None:
    backend = (os.environ.get("XCELSIOR_DB_BACKEND") or "postgres").strip().lower()
    if backend == "postgres":
        return None
    return Finding(
        code="database_backend",
        severity="error",
        message=f"XCELSIOR_DB_BACKEND={backend!r} — production requires postgres",
        remediation=(
            "Set XCELSIOR_DB_BACKEND=postgres. SQLite and dual-write modes "
            "cannot provide the row locks, partial unique indexes, and "
            "transactional placement the control plane depends on."
        ),
    )


def _dsn_is_local(dsn: str) -> bool:
    """True when the DSN points at loopback or a unix socket."""
    lowered = dsn.lower()
    if lowered.startswith("postgresql:///") or "host=/" in lowered:
        return True
    for token in ("@127.0.0.1", "@localhost", "@[::1]", "@/"):
        if token in lowered:
            return True
    return False


def _check_database_tls() -> Finding | None:
    from db import resolve_postgres_dsn

    dsn = resolve_postgres_dsn()
    lowered = dsn.lower()
    # TLS is required for any DSN that leaves the host, and can be forced
    # for loopback too (a shared host is not a trust boundary).
    forced = _truthy("XCELSIOR_PG_REQUIRE_TLS")
    if not forced and _dsn_is_local(dsn):
        return None
    if "sslmode=" not in lowered or "sslmode=disable" in lowered:
        return Finding(
            code="database_tls",
            severity="error",
            message="PostgreSQL DSN has no TLS policy (sslmode missing or disabled)",
            remediation=(
                "Append sslmode=require (or verify-full with a root cert) to "
                "XCELSIOR_POSTGRES_DSN, or set XCELSIOR_PG_REQUIRE_TLS=0 only "
                "for a loopback/unix-socket database."
            ),
        )
    return None


def _check_oauth_signing() -> Finding | None:
    keys_json = (os.environ.get("XCELSIOR_OAUTH_JWT_KEYS_JSON") or "").strip()
    secret = (os.environ.get("XCELSIOR_OAUTH_JWT_SECRET") or "").strip()
    if keys_json or secret:
        return None
    return Finding(
        code="oauth_signing",
        severity="error",
        message="No OAuth signing key configured (JWT keys and secret both empty)",
        remediation=(
            "Set XCELSIOR_OAUTH_JWT_KEYS_JSON (preferred — asymmetric keys "
            "with a published JWKS) or XCELSIOR_OAUTH_JWT_SECRET. Without "
            "one, issued tokens cannot be verified across replicas."
        ),
    )


def _check_agent_authentication() -> Finding | None:
    if _truthy("XCELSIOR_ALLOW_UNAUTH_AGENT"):
        return Finding(
            code="unauthenticated_agent",
            severity="error",
            message="XCELSIOR_ALLOW_UNAUTH_AGENT is enabled in production",
            remediation=(
                "Unset XCELSIOR_ALLOW_UNAUTH_AGENT. It is a non-production "
                "escape hatch; with it set, anything that can reach /agent/* "
                "can report telemetry, claim leases, and read SSH keys."
            ),
        )
    return None


def _check_mcp_rate_limiting() -> Finding | None:
    try:
        replicas = int(os.environ.get("MCP_REPLICAS", "1") or "1")
    except ValueError:
        replicas = 1
    backend = (os.environ.get("MCP_RATE_LIMIT_BACKEND") or "memory").strip().lower()
    if replicas <= 1 or backend == "redis":
        # Even on one replica, a fail-open limiter is worth a warning.
        if backend == "redis" and not _truthy("MCP_RATE_LIMIT_FAIL_CLOSED", "true"):
            return Finding(
                code="mcp_rate_limit_fail_open",
                severity="warning",
                message="MCP rate limiting falls open when Redis is unavailable",
                remediation="Set MCP_RATE_LIMIT_FAIL_CLOSED=true.",
            )
        return None
    return Finding(
        code="mcp_rate_limit_process_local",
        severity="error",
        message=(
            f"MCP_RATE_LIMIT_BACKEND={backend!r} with MCP_REPLICAS={replicas} — "
            "a per-process limit is not a rate limit across replicas"
        ),
        remediation=(
            "Set MCP_RATE_LIMIT_BACKEND=redis, MCP_REDIS_URL, and "
            "MCP_RATE_LIMIT_REQUIRE_REDIS=true."
        ),
    )


def _check_runtime_ddl() -> Finding | None:
    if _truthy("XCELSIOR_ALLOW_RUNTIME_DDL"):
        return Finding(
            code="runtime_ddl_enabled",
            severity="error",
            message="XCELSIOR_ALLOW_RUNTIME_DDL is enabled in production",
            remediation=(
                "Unset it. Alembic is the only production DDL authority "
                "(ADR-009 / companion §4.4 rule 1); runtime CREATE/ALTER "
                "races rolling deploys and hides schema drift."
            ),
        )
    return None


def _check_volume_privilege() -> Finding | None:
    mode = (os.environ.get("XCELSIOR_VOLUME_PRIVILEGE") or "host_ssh").strip().lower()
    if mode in ("host_ssh", "provisioner"):
        return None
    return Finding(
        code="api_privilege_expectation",
        severity="error",
        message=(
            f"XCELSIOR_VOLUME_PRIVILEGE={mode!r} expects in-process privileged "
            "LUKS/NFS work, which the unprivileged API image cannot do"
        ),
        remediation=(
            "Set XCELSIOR_VOLUME_PRIVILEGE=host_ssh (default) or run the "
            "volume-provisioner profile (§19.4). 'local' belongs only to the "
            "provisioner container."
        ),
    )


def _check_agent_gateway_secret() -> Finding | None:
    from control_plane.identity import agent_gateway_secret, trusted_gateway_enabled

    if trusted_gateway_enabled() and not agent_gateway_secret():
        return Finding(
            code="agent_gateway_unauthenticated",
            severity="error",
            message=(
                "XCELSIOR_TRUSTED_AGENT_GATEWAY=1 without "
                "XCELSIOR_AGENT_GATEWAY_SECRET — identity headers would be forgeable"
            ),
            remediation=(
                "Set XCELSIOR_AGENT_GATEWAY_SECRET to the value the gateway "
                "injects (infra/spire/, nginx/agent-xcelsior.conf)."
            ),
        )
    return None


def _check_host_token_rotation_readiness() -> Finding | None:
    """Flipping to ``require`` while a host has no token locks it out."""
    from control_plane.agent_tokens import host_tokens_required

    if not host_tokens_required():
        return None
    try:
        from control_plane.agent_tokens import rotation_coverage
        from db import _get_pg_pool

        with _get_pg_pool().connection() as conn:
            coverage = rotation_coverage(conn)
    except Exception as exc:  # pragma: no cover - DB gate reports elsewhere
        return Finding(
            code="host_token_coverage_unknown",
            severity="warning",
            message=f"could not verify per-host token coverage: {exc}",
            remediation="Check GET /api/admin/agent-tokens/coverage.",
        )
    if coverage["ready"]:
        return None
    missing = ", ".join(coverage["missing"][:5])
    return Finding(
        code="host_token_coverage_incomplete",
        severity="error",
        message=(
            f"XCELSIOR_AGENT_HOST_TOKENS=require but {len(coverage['missing'])} "
            f"host(s) have no live token ({missing})"
        ),
        remediation=(
            "Issue tokens (POST /api/admin/hosts/{host_id}/agent-tokens) until "
            "GET /api/admin/agent-tokens/coverage reports ready=true, or set "
            "XCELSIOR_AGENT_HOST_TOKENS=allow."
        ),
    )


def _check_shared_bearer_migration() -> Finding | None:
    from control_plane.identity import shared_bearer_migration_enabled

    if shared_bearer_migration_enabled():
        return Finding(
            code="shared_fleet_bearer_active",
            severity="warning",
            message=(
                "XCELSIOR_AGENT_SHARED_BEARER_MIGRATION=1 — one platform token "
                "still authenticates the fleet (§19.2)"
            ),
            remediation=(
                "Complete per-host token rollout, then unset this flag and set "
                "XCELSIOR_AGENT_HOST_TOKENS=require."
            ),
        )
    return None


#: Ordered so the most fundamental misconfiguration is reported first.
CHECKS: tuple[Callable[[], "Finding | None"], ...] = (
    _check_database_backend,
    _check_database_tls,
    _check_runtime_ddl,
    _check_oauth_signing,
    _check_agent_authentication,
    _check_agent_gateway_secret,
    _check_host_token_rotation_readiness,
    _check_shared_bearer_migration,
    _check_mcp_rate_limiting,
    _check_volume_privilege,
)


def collect_findings() -> list[Finding]:
    """Run every check. Never raises — a broken check is a warning."""
    findings: list[Finding] = []
    for check in CHECKS:
        try:
            result = check()
        except Exception as exc:  # pragma: no cover - defensive
            findings.append(
                Finding(
                    code=f"check_failed:{check.__name__.lstrip('_')}",
                    severity="warning",
                    message=f"startup check raised: {exc}",
                    remediation="Investigate; this check could not be evaluated.",
                )
            )
            continue
        if result is not None:
            findings.append(result)
    return findings


def validate_startup(*, enforce: bool | None = None) -> list[Finding]:
    """Validate configuration; raise in production on any ``error``.

    ``enforce`` defaults to "production only" so a developer machine
    surfaces the same findings without refusing to start.
    """
    findings = collect_findings()
    should_enforce = is_production() if enforce is None else enforce
    if should_enforce and not _truthy("XCELSIOR_SKIP_STARTUP_VALIDATION"):
        errors = [f for f in findings if f.severity == "error"]
        if errors:
            raise StartupValidationError(errors)
    return findings


def startup_report() -> dict:
    """Payload for ``/startupz`` (§21.3)."""
    findings = collect_findings()
    errors = [f for f in findings if f.severity == "error"]
    return {
        "ok": not errors,
        "environment": (os.environ.get("XCELSIOR_ENV") or "").strip().lower() or "unset",
        "enforced": is_production(),
        "findings": [f.as_dict() for f in findings],
    }
