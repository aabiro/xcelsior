"""Per-service PostgreSQL roles, grants, and connection identity.

Data-architecture companion §4.2 ("Logical schemas and roles") and §4.3
("Connection pools and workload isolation"). The end state the companion
requires is explicit:

    "The end state should avoid a single unrestricted application role."

This module is the in-repo authority for three things:

1. **Domain ownership of every table.** :data:`TABLE_DOMAINS` assigns each
   relation to exactly one logical domain (``identity``/``control``/
   ``billing``/``marketplace``/``storage``/``audit``/``serverless``/
   ``retrieval``). A drift test asserts the live database contains no
   table this map does not claim, so a new migration cannot silently
   create an ungranted (and therefore unreachable) relation.

2. **Least-privilege role definitions.** :data:`SERVICE_ROLES` encodes the
   companion's recommended roles together with the *denial* properties
   that make them meaningful — ``xcelsior_scheduler`` has no arbitrary
   billing mutation, ``xcelsior_billing`` has no host-command rights,
   ``xcelsior_readonly`` writes nothing, and **no runtime role holds
   ``CREATE`` on a schema**, which is what makes companion §4.4 rule 1
   ("Alembic is the only production DDL authority") enforceable by the
   database rather than by convention.

3. **Runtime connection identity.** :func:`resolve_service_dsn` maps a
   service name to its own DSN (falling back to the shared one when an
   operator has not cut that service over yet) and always stamps
   ``application_name`` plus the §4.3 session timeouts, so pool
   saturation and long transactions are attributable per service.

Rollout is deliberately opt-in: a service uses its dedicated role only
once ``XCELSIOR_POSTGRES_DSN_<SERVICE>`` is set. Provisioning the roles
(``scripts/provision_db_roles.py``) is safe to run before any service is
cut over — it only creates roles and grants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Mapping

# ── Logical domains (companion §4.2) ──────────────────────────────────
#
# One authority per fact: every relation belongs to exactly one domain.
# Kept as literal names (not prefixes) so an unclassified new table is a
# test failure instead of an accidental grant.

IDENTITY_TABLES = frozenset(
    {
        "users",
        "teams",
        "team_members",
        "team_invites",
        "sessions",
        "api_keys",
        "oauth_clients",
        "oauth_refresh_tokens",
        "mfa_methods",
        "mfa_challenges",
        "mfa_backup_codes",
        "user_ssh_keys",
        "user_avatars",
        "user_images",
        "web_push_subscriptions",
        "notifications",
        "casl_consent",
        "consent_records",
        "privacy_configs",
        "data_disclosures",
        "legal_requests",
        "retention_records",
        "fintrac_reports",
        "verification_history",
        "host_verifications",
    }
)

CONTROL_TABLES = frozenset(
    {
        "jobs",
        "hosts",
        "job_attempts",
        "job_logs",
        "job_failure_log",
        "host_gpu_devices",
        "gpu_device_allocations",
        "placement_leases",
        "leases",
        "agent_commands",
        "agent_rollouts",
        "host_agent_tokens",
        "node_versions",
        "host_observations",
        "observed_workloads",
        "reconciliation_queue",
        "reconciliation_findings",
        "scheduled_tasks",
        "scheduler_shadow_decisions",
        "service_heartbeats",
        "telemetry_latest",
        "telemetry_samples",
        "telemetry_snapshots",
        "state",
        "api_idempotency_keys",
        "volumes",
        "volume_attachments",
        "volume_snapshots",
        "worker_model_cache",
        "registry_health_cache",
        "benchmarks",
        "cloud_burst_instances",
        "slurm_job_mappings",
    }
)

BILLING_TABLES = frozenset(
    {
        "wallets",
        "wallet_transactions",
        "wallet_holds",
        "usage_meters",
        "invoices",
        "billing_cycles",
        "payout_ledger",
        "payout_splits",
        "payment_intents",
        "connect_accounts",
        "connect_products",
        "crypto_deposits",
        "ln_deposits",
        "stripe_event_inbox",
        "storage_billing_rates",
    }
)

MARKETPLACE_TABLES = frozenset(
    {
        "gpu_offers",
        "gpu_allocations",
        "gpu_pricing",
        "reservations",
        "reserved_commitments",
        "spot_prices",
        "spot_price_history",
        "marketplace_sales",
        "provider_accounts",
        "reputation_events",
        "reputation_scores",
        "sla_downtime",
        "sla_monthly",
        "sla_violations",
    }
)

AUDIT_TABLES = frozenset(
    {
        "events",
        "events_archive",
        "event_snapshots",
        "outbox_events",
        "alembic_version",
    }
)

SERVERLESS_TABLES = frozenset(
    {
        "serverless_endpoints",
        "serverless_workers",
        "serverless_jobs",
        "serverless_job_stream_events",
        "serverless_api_keys",
        "serverless_batches",
        "serverless_semantic_cache",
        "serverless_cache_savings",
        "serverless_token_ledger",
        "serverless_kv_cache_samples",
        "serverless_prefix_affinity",
        "serverless_lora_adapters",
        "inference_endpoints",
        "inference_jobs",
        "inference_results",
    }
)

RETRIEVAL_TABLES = frozenset(
    {
        "ai_docs",
        "ai_conversations",
        "ai_messages",
        "ai_confirmations",
        "chat_conversations",
        "chat_messages",
        "chat_feedback",
    }
)

# The ``storage`` PostgreSQL schema (migration 064) is wholly the storage
# domain; membership is by schema, not by an enumerated name list.
STORAGE_SCHEMA = "storage"

TABLE_DOMAINS: dict[str, frozenset[str]] = {
    "identity": IDENTITY_TABLES,
    "control": CONTROL_TABLES,
    "billing": BILLING_TABLES,
    "marketplace": MARKETPLACE_TABLES,
    "audit": AUDIT_TABLES,
    "serverless": SERVERLESS_TABLES,
    "retrieval": RETRIEVAL_TABLES,
}

ALL_DOMAINS = tuple(TABLE_DOMAINS) + ("storage",)


def domain_of(table: str) -> str | None:
    """Return the logical domain owning ``table`` (schema-qualified aware).

    Partitions of a partitioned table (``telemetry_samples_202607``,
    ``telemetry_samples_default``) resolve to their parent's domain.
    """
    name = table.strip()
    if "." in name:
        schema, _, bare = name.partition(".")
        if schema == STORAGE_SCHEMA:
            return "storage"
        name = bare
    for dom, tables in TABLE_DOMAINS.items():
        if name in tables:
            return dom
    # Declarative partition children inherit the parent's domain.
    for dom, tables in TABLE_DOMAINS.items():
        for parent in tables:
            if name.startswith(parent + "_"):
                return dom
    return None


# ── Role definitions (companion §4.2) ─────────────────────────────────


@dataclass(frozen=True, slots=True)
class ServiceRole:
    """A PostgreSQL role and the domains it may read/write.

    ``write`` implies ``read``. ``sequences`` is derived: any domain a
    role may INSERT into also needs USAGE on that domain's sequences.
    """

    name: str
    description: str
    read: frozenset[str] = field(default_factory=frozenset)
    write: frozenset[str] = field(default_factory=frozenset)
    #: DDL authority. Exactly one role (the migrator) may hold this.
    ddl: bool = False
    #: Roles that must never be granted to a long-running runtime service.
    runtime: bool = True

    @property
    def readable(self) -> frozenset[str]:
        return frozenset(self.read | self.write)


SERVICE_ROLES: dict[str, ServiceRole] = {
    "migrator": ServiceRole(
        name="xcelsior_migrator",
        description="DDL only; used by the release migration job, never by a runtime service.",
        read=frozenset(ALL_DOMAINS),
        write=frozenset(ALL_DOMAINS),
        ddl=True,
        runtime=False,
    ),
    "api": ServiceRole(
        name="xcelsior_api",
        description="Tenant-scoped API reads/writes. No DDL — Alembic is the only DDL authority.",
        write=frozenset(
            {
                "identity",
                "control",
                "billing",
                "storage",
                "serverless",
                "retrieval",
                "audit",
            }
        ),
        read=frozenset({"marketplace"}),
    ),
    "scheduler": ServiceRole(
        name="xcelsior_scheduler",
        description="Claim and placement rights; no arbitrary billing mutation, no identity writes.",
        write=frozenset({"control", "audit"}),
        read=frozenset({"identity", "billing", "marketplace", "serverless"}),
    ),
    "reconciler": ServiceRole(
        name="xcelsior_reconciler",
        description="Observation and drift-repair rights over control state only.",
        write=frozenset({"control", "audit"}),
        read=frozenset({"identity", "billing", "marketplace", "serverless", "storage"}),
    ),
    "billing": ServiceRole(
        name="xcelsior_billing",
        description="Ledger and meter rights; no host command rights.",
        write=frozenset({"billing", "marketplace", "audit"}),
        read=frozenset({"identity", "control", "serverless", "storage"}),
    ),
    "retrieval": ServiceRole(
        name="xcelsior_retrieval",
        description="Retrieval tables plus read-only approved document sources.",
        write=frozenset({"retrieval"}),
        read=frozenset({"identity", "storage"}),
    ),
    "projector": ServiceRole(
        name="xcelsior_projector",
        description="Outbox claim, checkpoint, and projection state.",
        write=frozenset({"audit"}),
        read=frozenset(ALL_DOMAINS),
    ),
    "readonly": ServiceRole(
        name="xcelsior_readonly",
        description="Audited operational queries; writes nothing.",
        read=frozenset(ALL_DOMAINS),
    ),
}

#: Explicit denial contract asserted by tests. ``(role_key, table, verb)``
#: triples that MUST be rejected by PostgreSQL once roles are provisioned.
#: These are the properties that make the split meaningful rather than
#: cosmetic — each one maps to a sentence in companion §4.2.
DENIAL_CONTRACT: tuple[tuple[str, str, str], ...] = (
    # "no access to secrets or arbitrary billing mutation"
    ("scheduler", "wallets", "UPDATE"),
    ("scheduler", "wallet_transactions", "INSERT"),
    ("scheduler", "users", "UPDATE"),
    ("scheduler", "api_keys", "UPDATE"),
    # "ledger and meter rights; no host command rights"
    ("billing", "agent_commands", "INSERT"),
    ("billing", "agent_commands", "UPDATE"),
    ("billing", "placement_leases", "UPDATE"),
    # reconciler repairs control state, never money
    ("reconciler", "wallets", "UPDATE"),
    ("reconciler", "usage_meters", "INSERT"),
    # read-only means read-only
    ("readonly", "jobs", "UPDATE"),
    ("readonly", "events", "INSERT"),
    ("readonly", "wallets", "UPDATE"),
    # retrieval never touches the control plane
    ("retrieval", "jobs", "UPDATE"),
    ("retrieval", "agent_commands", "INSERT"),
    # projector reads everything but only settles the outbox
    ("projector", "jobs", "UPDATE"),
)


def runtime_role_keys() -> tuple[str, ...]:
    """Role keys that may be handed to a long-running service."""
    return tuple(k for k, r in SERVICE_ROLES.items() if r.runtime)


# ── Grant SQL generation ──────────────────────────────────────────────


def _qualified(domain: str, table: str) -> str:
    if domain == "storage":
        return f'"{STORAGE_SCHEMA}"."{table}"'
    return f'public."{table}"'


def _domain_tables(domain: str, storage_tables: Iterable[str]) -> list[str]:
    if domain == "storage":
        return sorted(storage_tables)
    return sorted(TABLE_DOMAINS[domain])


def grant_statements(
    role: ServiceRole,
    *,
    storage_tables: Iterable[str] = (),
    existing_tables: Iterable[str] | None = None,
) -> list[str]:
    """Idempotent GRANT/REVOKE statements bringing ``role`` to its contract.

    ``existing_tables`` (schema-qualified, lowercase) filters the output to
    relations that actually exist, so provisioning a database that has not
    reached the latest migration does not fail on a missing table.
    """
    present: set[str] | None = None
    if existing_tables is not None:
        present = {t.lower() for t in existing_tables}

    def exists(domain: str, table: str) -> bool:
        if present is None:
            return True
        key = f"{STORAGE_SCHEMA}.{table}" if domain == "storage" else f"public.{table}"
        return key in present

    stmts: list[str] = [
        # Start from zero every run so removing a domain from the contract
        # actually removes the privilege (idempotent convergence, not drift).
        f'REVOKE ALL ON ALL TABLES IN SCHEMA public FROM "{role.name}"',
        f'REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM "{role.name}"',
        f'REVOKE ALL ON SCHEMA public FROM "{role.name}"',
        f'GRANT USAGE ON SCHEMA public TO "{role.name}"',
    ]
    if storage_tables or role.readable & {"storage"}:
        stmts += [
            f'REVOKE ALL ON ALL TABLES IN SCHEMA "{STORAGE_SCHEMA}" FROM "{role.name}"',
            f'REVOKE ALL ON ALL SEQUENCES IN SCHEMA "{STORAGE_SCHEMA}" FROM "{role.name}"',
            f'REVOKE ALL ON SCHEMA "{STORAGE_SCHEMA}" FROM "{role.name}"',
            f'GRANT USAGE ON SCHEMA "{STORAGE_SCHEMA}" TO "{role.name}"',
        ]

    if role.ddl:
        # The migrator — and only the migrator — may create objects.
        stmts.append(f'GRANT CREATE ON SCHEMA public TO "{role.name}"')
        if storage_tables:
            stmts.append(f'GRANT CREATE ON SCHEMA "{STORAGE_SCHEMA}" TO "{role.name}"')

    for domain in sorted(role.write):
        for table in _domain_tables(domain, storage_tables):
            if not exists(domain, table):
                continue
            stmts.append(
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON {_qualified(domain, table)} "
                f'TO "{role.name}"'
            )
    for domain in sorted(role.read - role.write):
        for table in _domain_tables(domain, storage_tables):
            if not exists(domain, table):
                continue
            stmts.append(f'GRANT SELECT ON {_qualified(domain, table)} TO "{role.name}"')

    if role.write:
        stmts.append(f'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO "{role.name}"')
        if "storage" in role.write and storage_tables:
            stmts.append(
                f'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA "{STORAGE_SCHEMA}" '
                f'TO "{role.name}"'
            )
    return stmts


# ── Runtime connection identity (companion §4.3) ──────────────────────

#: Logical service names that resolve their own DSN. Keys mirror the
#: ``XCELSIOR_POSTGRES_DSN_<SERVICE>`` environment contract.
SERVICE_POOLS = (
    "api",
    "scheduler",
    "reconciler",
    "billing",
    "projector",
    "retrieval",
    "readonly",
    "migrator",
)

_DEFAULT_SESSION_TIMEOUTS: Mapping[str, str] = {
    # §4.3: bound every runtime session. Individual control-plane
    # transactions still narrow these further with SET LOCAL.
    "statement_timeout": "30000",
    "lock_timeout": "5000",
    "idle_in_transaction_session_timeout": "60000",
}

#: The scheduler reservation pool is explicitly "small, low timeout, no
#: long queries" (§4.3).
_SERVICE_SESSION_OVERRIDES: Mapping[str, Mapping[str, str]] = {
    "scheduler": {"statement_timeout": "15000", "lock_timeout": "3000"},
    "readonly": {"statement_timeout": "120000", "idle_in_transaction_session_timeout": "30000"},
    # Migrations legitimately run long DDL and must never be killed
    # mid-statement by a pool default.
    "migrator": {
        "statement_timeout": "0",
        "lock_timeout": "0",
        "idle_in_transaction_session_timeout": "0",
    },
}


def current_service() -> str:
    """The logical service this process is (``XCELSIOR_SERVICE`` or ``api``)."""
    raw = (os.environ.get("XCELSIOR_SERVICE") or "").strip().lower()
    return raw if raw in SERVICE_POOLS else "api"


def application_name(service: str | None = None) -> str:
    """``application_name`` stamped on every connection (§4.3)."""
    svc = (service or current_service()).strip().lower()
    explicit = (os.environ.get("XCELSIOR_PG_APPLICATION_NAME") or "").strip()
    if explicit:
        return explicit[:63]
    return f"xcelsior-{svc}"[:63]


def session_options(service: str | None = None) -> str:
    """libpq ``options`` string carrying the §4.3 session timeouts.

    Returned in ``-c key=value`` form so the settings apply to the
    physical session at connect time — they survive pool checkout/return
    and cannot be forgotten by a call site.
    """
    svc = (service or current_service()).strip().lower()
    merged = dict(_DEFAULT_SESSION_TIMEOUTS)
    merged.update(_SERVICE_SESSION_OVERRIDES.get(svc, {}))
    for key in list(merged):
        env_key = f"XCELSIOR_PG_{key.upper()}"
        override = (os.environ.get(env_key) or "").strip()
        if override:
            merged[key] = override
    return " ".join(f"-c {k}={v}" for k, v in sorted(merged.items()))


def service_dsn_env_var(service: str) -> str:
    return f"XCELSIOR_POSTGRES_DSN_{service.strip().upper()}"


def resolve_service_dsn(service: str | None = None, *, fallback: str | None = None) -> str:
    """DSN for ``service``, falling back to the shared application DSN.

    A service uses its dedicated least-privilege role only once an
    operator sets ``XCELSIOR_POSTGRES_DSN_<SERVICE>``. Until then this
    returns the shared DSN unchanged, so provisioning roles is not a
    breaking change for any deployment.
    """
    svc = (service or current_service()).strip().lower()
    specific = (os.environ.get(service_dsn_env_var(svc)) or "").strip()
    if specific:
        return specific
    if fallback is not None:
        return fallback
    # The *shared* DSN, never the service-aware one — resolving another
    # service's DSN must not silently inherit this process's override.
    from db import resolve_shared_postgres_dsn

    return resolve_shared_postgres_dsn()


def connection_kwargs(service: str | None = None) -> dict[str, str]:
    """psycopg connect kwargs stamping identity and session limits."""
    svc = (service or current_service()).strip().lower()
    return {
        "application_name": application_name(svc),
        "options": session_options(svc),
    }
