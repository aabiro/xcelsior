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

import env_config

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
    """Is this the production deployment itself?

    Kept for callers that genuinely mean production. Do not use it to decide
    whether to *enforce* — see `enforcement_enabled`.
    """
    return env_config.is_production()


def enforcement_enabled(raw: str | None = None) -> bool:
    """Should an error finding refuse the boot?

    This used to be `is_production()`, implemented as an exact match on the
    literal "production". `prod`, `staging`, a typo, or an unset variable all
    returned False, so every error finding — SQLite backend, unauthenticated
    agent mode, no asymmetric signing key — degraded to a log line. The gate
    meant to catch fail-open configuration was fail-open on the same variable.

    Enforcement is now the default and only an explicitly relaxed environment is
    exempt. Staging enforces deliberately: it holds real data.
    """
    return not env_config.is_relaxed_env(raw)


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
    import json

    keys_json = (os.environ.get("XCELSIOR_OAUTH_JWT_KEYS_JSON") or "").strip()
    secret = (os.environ.get("XCELSIOR_OAUTH_JWT_SECRET") or "").strip()
    if env_config.is_relaxed_env() and secret:
        return None
    if keys_json:
        try:
            data = json.loads(keys_json)
            active = str(data.get("active_kid") or "")
            keys = {str(item.get("kid")): item for item in data.get("keys", [])}
            if active and active in keys and keys[active].get("private_key_pem"):
                return None
        except (TypeError, ValueError):
            pass
    return Finding(
        code="oauth_signing",
        severity="error",
        message="No valid asymmetric OAuth signing key configured",
        remediation=(
            "Set XCELSIOR_OAUTH_JWT_KEYS_JSON with active_kid and a matching "
            "RS256 private_key_pem/public_key_pem pair. Symmetric shared "
            "secrets are forbidden in production."
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


def _check_privacy_deletion_credentials() -> Finding | None:
    reference_secret = (
        os.environ.get("XCELSIOR_PRIVACY_REFERENCE_SECRET") or ""
    ).strip()
    if not reference_secret:
        return Finding(
            code="privacy_reference_secret_missing",
            severity="error",
            message=(
                "XCELSIOR_PRIVACY_REFERENCE_SECRET is empty; completed "
                "deletion evidence cannot use a keyed subject reference"
            ),
            remediation=(
                "Generate a dedicated high-entropy secret, store it in the "
                "production secret manager, and expose it only to the API and "
                "privacy worker."
            ),
        )

    posthog_enabled = any(
        (os.environ.get(name) or "").strip()
        for name in (
            "NEXT_PUBLIC_POSTHOG_PROJECT_TOKEN",
            "XCELSIOR_MCP_POSTHOG_PROJECT_API_KEY",
            "POSTHOG_PROJECT_API_KEY",
        )
    )
    if not posthog_enabled:
        return None
    personal_key = (
        os.environ.get("XCELSIOR_POSTHOG_PERSONAL_API_KEY") or ""
    ).strip()
    project_id = (
        os.environ.get("XCELSIOR_POSTHOG_PROJECT_ID") or ""
    ).strip()
    if personal_key and project_id:
        return None
    return Finding(
        code="posthog_deletion_credentials_missing",
        severity="error",
        message=(
            "PostHog identification is enabled without the credentials needed "
            "to delete persons, events, and recordings"
        ),
        remediation=(
            "Set XCELSIOR_POSTHOG_PERSONAL_API_KEY with person:read and "
            "person:write scopes plus XCELSIOR_POSTHOG_PROJECT_ID, or disable "
            "PostHog identification."
        ),
    )


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


def _check_stripe_webhook_secret() -> Finding | None:
    """A live Stripe integration with no signing secret loses events silently.

    Without a secret, no event can be verified. `POST /api/providers/webhook`
    now answers 503 in that state so Stripe retries — but retries only help if
    someone notices, and nothing else would tell them: the failure looks like
    "payments are quiet".

    This is worth refusing the boot over rather than warning about, because the
    events it drops are the ones that confirm money moved. Auto-top-up
    completion, SCA recovery, and Connect payout onboarding all take their
    completion signal from a webhook and nowhere else.
    """
    try:
        from stripe_connect import STRIPE_ENABLED, _webhook_secret_candidates
    except Exception:  # pragma: no cover - import failure is its own finding
        return None

    if not STRIPE_ENABLED:
        return None
    if _webhook_secret_candidates():
        return None
    return Finding(
        code="stripe_webhook_secret_missing",
        severity="error",
        message=(
            "Stripe is enabled but no webhook signing secret is configured — "
            "no event can be verified, and every one is refused"
        ),
        remediation=(
            "Set XCELSIOR_STRIPE_WEBHOOK_SECRET (and "
            "XCELSIOR_STRIPE_CONNECT_WEBHOOK_SECRET / "
            "XCELSIOR_STRIPE_THIN_WEBHOOK_SECRET if those destinations exist) "
            "to the signing secret from the Stripe dashboard for each endpoint."
        ),
    )


def _check_compatibility_session_secret() -> Finding | None:
    """Host admission (082) derives submit tokens from this secret.

    Without it, host_admission falls back to a hard-coded development value.
    That is fine on a laptop and fatal in production: the submit token and the
    proof-of-possession challenge become predictable, so anyone could post
    compatibility evidence for a host they do not control. The service already
    refuses to start a session in production without it — this surfaces the
    problem at boot rather than at the first provider's first attempt.
    """
    if (os.environ.get("XCELSIOR_COMPAT_SESSION_SECRET") or "").strip():
        return None
    if not enforcement_enabled():
        # host_admission falls back to a development constant in a relaxed
        # environment so a laptop still boots. Mirror that here rather than
        # failing every dev and test run — and mirror it exactly, which an
        # exact match on "production" did not: it skipped this check on
        # staging, where host_admission does *not* fall back.
        return None
    return Finding(
        code="compat_session_secret_missing",
        severity="error",
        message=(
            "XCELSIOR_COMPAT_SESSION_SECRET is unset — host compatibility "
            "sessions would derive submit tokens from a public development "
            "constant, making provider evidence forgeable"
        ),
        remediation=(
            "Generate a dedicated high-entropy secret "
            "(python -c 'import secrets; print(secrets.token_urlsafe(32))') "
            "and store it in the production secret manager."
        ),
    )


def _check_audit_signing_key() -> Finding | None:
    """Audit checkpoints must not be signed with the development key.

    control_plane/audit_checkpoints falls back to the literal
    "dev-audit-key-not-for-prod" when neither XCELSIOR_AUDIT_SIGNING_KEYS nor
    XCELSIOR_AUDIT_SIGNING_KEY is set. That string is in the public source, so
    in production the Merkle checkpoint signature would be forgeable by anyone
    who has read the repository — which defeats the point of a tamper-evident
    audit trail.
    """
    if not enforcement_enabled():
        return None
    if (os.environ.get("XCELSIOR_AUDIT_SIGNING_KEYS") or "").strip():
        return None
    if (os.environ.get("XCELSIOR_AUDIT_SIGNING_KEY") or "").strip():
        return None
    return Finding(
        code="audit_signing_key_default",
        severity="error",
        message=(
            "Neither XCELSIOR_AUDIT_SIGNING_KEYS nor XCELSIOR_AUDIT_SIGNING_KEY "
            "is set — audit checkpoints would be signed with the public "
            "development key and the audit trail would be forgeable"
        ),
        remediation=(
            "Set XCELSIOR_AUDIT_SIGNING_KEYS to a JSON map of key-id to secret "
            '(for example {"v1": "<random>"}) and XCELSIOR_AUDIT_SIGNING_ACTIVE '
            "to the active id, so keys can be rotated without invalidating "
            "previously signed checkpoints."
        ),
    )


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


def _check_artifact_backend() -> Finding | None:
    backend = (os.environ.get("XCELSIOR_ARTIFACT_BACKEND") or "").strip().lower()
    if not backend:
        backend = (os.environ.get("XCELSIOR_STORAGE_BACKEND") or "").strip().lower()
    if not backend:
        if not enforcement_enabled():
            return None
        return Finding(
            code="artifact_backend_missing",
            severity="error",
            message="XCELSIOR_ARTIFACT_BACKEND is unset in production — production requires 's3' or 'gcs'",
            remediation=(
                "Set XCELSIOR_ARTIFACT_BACKEND to 's3' or 'gcs' with corresponding bucket and credentials."
            ),
        )
    if backend in ("local", "file", "filesystem"):
        if not enforcement_enabled():
            return None
        return Finding(
            code="artifact_backend_local",
            severity="error",
            message=f"XCELSIOR_ARTIFACT_BACKEND={backend!r} — production requires cloud object storage ('s3' or 'gcs')",
            remediation=(
                "Set XCELSIOR_ARTIFACT_BACKEND to 's3' or 'gcs'. Local filesystem artifact "
                "storage cannot be shared across replicas and is prohibited in production."
            ),
        )
    if backend not in ("s3", "gcs"):
        return Finding(
            code="artifact_backend_invalid",
            severity="error",
            message=f"XCELSIOR_ARTIFACT_BACKEND={backend!r} is unsupported (must be 's3' or 'gcs')",
            remediation="Set XCELSIOR_ARTIFACT_BACKEND to 's3' or 'gcs'.",
        )
    return None


def _check_auth_cache() -> Finding | None:
    backend = (os.environ.get("XCELSIOR_AUTH_CACHE_BACKEND") or "redis").strip().lower()
    if backend == "redis":
        return None
    if backend == "memory":
        if not enforcement_enabled():
            return None
        return Finding(
            code="auth_cache_memory",
            severity="error",
            message=(
                "XCELSIOR_AUTH_CACHE_BACKEND='memory' is prohibited in production — "
                "in-memory auth cache does not share tokens and sessions across replicas"
            ),
            remediation="Set XCELSIOR_AUTH_CACHE_BACKEND=redis and configure XCELSIOR_AUTH_REDIS_URL.",
        )
    return Finding(
        code="auth_cache_invalid",
        severity="error",
        message=f"XCELSIOR_AUTH_CACHE_BACKEND={backend!r} is unsupported (must be 'redis')",
        remediation="Set XCELSIOR_AUTH_CACHE_BACKEND=redis.",
    )


def _check_rate_limiting_backend() -> Finding | None:
    backend = (os.environ.get("XCELSIOR_RATE_LIMIT_BACKEND") or "").strip().lower()
    shared_limits = (os.environ.get("XCELSIOR_SHARED_RUNTIME_LIMITS") or "true").strip().lower()
    if shared_limits in ("0", "false", "no", "off"):
        if enforcement_enabled():
            return Finding(
                code="rate_limit_process_local",
                severity="error",
                message=(
                    "XCELSIOR_SHARED_RUNTIME_LIMITS is disabled in production — "
                    "process-local rate limiting does not protect multi-worker deployments"
                ),
                remediation="Set XCELSIOR_SHARED_RUNTIME_LIMITS=true.",
            )
    if backend in ("memory", "process", "local"):
        if enforcement_enabled():
            return Finding(
                code="rate_limit_process_local",
                severity="error",
                message=(
                    f"XCELSIOR_RATE_LIMIT_BACKEND={backend!r} — production requires "
                    "a shared Redis rate limit store"
                ),
                remediation="Set XCELSIOR_RATE_LIMIT_BACKEND=redis and provide XCELSIOR_RATE_LIMIT_REDIS_URL.",
            )
    return None


def _check_analytics_configuration() -> Finding | None:
    if not _truthy("XCELSIOR_ANALYTICS_EXPORT_ENABLED"):
        return None
    project = (os.environ.get("XCELSIOR_ANALYTICS_GCP_PROJECT") or "").strip()
    location = (os.environ.get("XCELSIOR_ANALYTICS_BQ_LOCATION") or "").strip()
    dataset = (os.environ.get("XCELSIOR_ANALYTICS_BQ_RAW_DATASET") or "").strip()
    bucket = (os.environ.get("XCELSIOR_ANALYTICS_GCS_LANDING_BUCKET") or "").strip()
    workload_id = (
        (os.environ.get("XCELSIOR_ANALYTICS_WORKLOAD_IDENTITY") or "").strip()
        or (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    )
    missing = []
    if not project:
        missing.append("XCELSIOR_ANALYTICS_GCP_PROJECT")
    if not location:
        missing.append("XCELSIOR_ANALYTICS_BQ_LOCATION")
    if not dataset:
        missing.append("XCELSIOR_ANALYTICS_BQ_RAW_DATASET")
    if not bucket:
        missing.append("XCELSIOR_ANALYTICS_GCS_LANDING_BUCKET")
    if not workload_id:
        missing.append("XCELSIOR_ANALYTICS_WORKLOAD_IDENTITY")
    if missing:
        return Finding(
            code="analytics_configuration_incomplete",
            severity="error",
            message=(
                "Analytics export is enabled (XCELSIOR_ANALYTICS_EXPORT_ENABLED=true) but "
                f"required configuration is missing: {', '.join(missing)}"
            ),
            remediation=(
                "Set the missing variables or set XCELSIOR_ANALYTICS_EXPORT_ENABLED=false."
            ),
        )
    return None


def _check_retrieval_configuration() -> Finding | None:
    if not _truthy("XCELSIOR_RETRIEVAL_ENABLED"):
        return None
    embed_path = (os.environ.get("XCELSIOR_RETRIEVAL_EMBED_MODEL_PATH") or "").strip()
    code_path = (os.environ.get("XCELSIOR_RETRIEVAL_CODE_MODEL_PATH") or "").strip()
    rerank_path = (os.environ.get("XCELSIOR_RETRIEVAL_RERANK_MODEL_PATH") or "").strip()
    sha256 = (os.environ.get("XCELSIOR_RETRIEVAL_EXPECTED_MODEL_SHA256") or "").strip()
    dimension = (os.environ.get("XCELSIOR_RETRIEVAL_EXPECTED_DIMENSION") or "").strip()
    revision = (
        (os.environ.get("XCELSIOR_RETRIEVAL_REVISION") or "").strip()
        or (os.environ.get("XCELSIOR_RETRIEVAL_MODEL_REVISION") or "").strip()
    )
    missing = []
    if not embed_path:
        missing.append("XCELSIOR_RETRIEVAL_EMBED_MODEL_PATH")
    if not code_path:
        missing.append("XCELSIOR_RETRIEVAL_CODE_MODEL_PATH")
    if not rerank_path:
        missing.append("XCELSIOR_RETRIEVAL_RERANK_MODEL_PATH")
    if not sha256:
        missing.append("XCELSIOR_RETRIEVAL_EXPECTED_MODEL_SHA256")
    if not dimension:
        missing.append("XCELSIOR_RETRIEVAL_EXPECTED_DIMENSION")
    if not revision:
        missing.append("XCELSIOR_RETRIEVAL_REVISION")
    if missing:
        return Finding(
            code="retrieval_configuration_incomplete",
            severity="error",
            message=(
                "Retrieval service is enabled (XCELSIOR_RETRIEVAL_ENABLED=true) but "
                f"required model probes and registered revision are missing: {', '.join(missing)}"
            ),
            remediation=(
                "Populate model paths, expected SHA256, dimension (after verified probe), "
                "and registered revision, or set XCELSIOR_RETRIEVAL_ENABLED=false."
            ),
        )
    return None


def _check_lightning_tls() -> Finding | None:
    if not _truthy("XCELSIOR_LN_ENABLED"):
        return None
    url = (os.environ.get("XCELSIOR_LN_CLNREST_URL") or "https://127.0.0.1:3010").strip().lower()
    if url.startswith("http://"):
        return Finding(
            code="lightning_tls_insecure",
            severity="error",
            message=f"XCELSIOR_LN_CLNREST_URL={url!r} uses plaintext HTTP — Lightning clnrest requires TLS (https://)",
            remediation="Change URL scheme to https:// and provide XCELSIOR_LN_CA_CERT.",
        )
    if enforcement_enabled():
        ca_cert = (os.environ.get("XCELSIOR_LN_CA_CERT") or "").strip()
        if not ca_cert:
            return Finding(
                code="lightning_ca_cert_missing",
                severity="error",
                message="XCELSIOR_LN_CA_CERT is required in production when Lightning is enabled",
                remediation="Set XCELSIOR_LN_CA_CERT to the absolute path of the CLN CA certificate.",
            )
        if not os.path.exists(ca_cert):
            return Finding(
                code="lightning_ca_cert_not_found",
                severity="error",
                message=f"XCELSIOR_LN_CA_CERT points to nonexistent file: {ca_cert}",
                remediation="Ensure the CA certificate file exists and is readable.",
            )
    return None


def _check_secret_manager_discipline() -> Finding | None:
    sm = (
        (os.environ.get("XCELSIOR_SECRET_MANAGER") or "").strip().lower()
        or (os.environ.get("XCELSIOR_SECRETS_BACKEND") or "").strip().lower()
    )
    if not sm or sm in ("none", "false", "0"):
        return None
    env_file = os.environ.get("XCELSIOR_ENV_FILE") or ".env"
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                content = f.read()
            sensitive_tokens = ("_SECRET=", "_KEY=", "_PASSWORD=", "_RUNE=")
            found_tokens = [tok for tok in sensitive_tokens if tok in content]
            if found_tokens:
                return Finding(
                    code="secrets_in_plain_env_file",
                    severity="error",
                    message=(
                        f"Secret manager ({sm}) is active but plain environment file '{env_file}' "
                        f"contains plaintext secrets ({', '.join(found_tokens)})"
                    ),
                    remediation=(
                        "Remove plaintext secrets from .env files when using a secret manager. "
                        "Inject secrets directly through the secret manager daemon or orchestrator."
                    ),
                )
        except Exception:
            pass
    return None


#: Ordered so the most fundamental misconfiguration is reported first.
CHECKS: tuple[Callable[[], "Finding | None"], ...] = (
    _check_database_backend,
    _check_database_tls,
    _check_runtime_ddl,
    _check_oauth_signing,
    _check_auth_cache,
    _check_artifact_backend,
    _check_rate_limiting_backend,
    _check_privacy_deletion_credentials,
    _check_agent_authentication,
    _check_agent_gateway_secret,
    _check_host_token_rotation_readiness,
    _check_compatibility_session_secret,
    _check_stripe_webhook_secret,
    _check_audit_signing_key,
    _check_shared_bearer_migration,
    _check_mcp_rate_limiting,
    _check_volume_privilege,
    _check_lightning_tls,
    _check_analytics_configuration,
    _check_retrieval_configuration,
    _check_secret_manager_discipline,
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
    should_enforce = enforcement_enabled() if enforce is None else enforce
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
