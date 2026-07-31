"""Event contract registry (Track B B4.3, companion §12.1/§13.4, §16.2/§DA8.3).

The single source of truth for every domain event's name, version, field
classifications, and the sinks it fans out to. Two invariants it enforces:

  1. **No secret in an audit event.** A field classified `credential_secret`
     may never be part of an event contract — audit rows are redacted, and a
     secret must not be persisted or fanned out. `validate_contract` rejects it.
  2. **Every sink mapping is classified.** A downstream sink cannot receive an
     event without a declared data classification, so residency/redaction
     decisions are never made by omission. `validate_sink_mapping` rejects a
     missing/unknown classification.

The canonical §16.2/DA§8.3 names live here alongside the names Track A already
emits — those are *registered as-is*, never renamed mid-flight.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# ── Data-classification vocabulary (companion §13.4) ──────────────────
CLASSIFICATIONS: frozenset[str] = frozenset(
    {"public", "internal", "pii", "financial", "credential_secret"}
)
# Classifications that must never appear inside an event contract / audit sink.
FORBIDDEN_IN_EVENTS: frozenset[str] = frozenset({"credential_secret"})

# Known fan-out sinks (companion §12.1). A mapping to any of these must carry a
# classification.
SINKS: frozenset[str] = frozenset({"audit_log", "warehouse", "sse", "webhook"})


class ContractViolation(ValueError):
    """A contract or sink mapping that violates §13.4."""


@dataclass(frozen=True)
class EventContract:
    event_type: str
    version: int
    classification: str
    # field name -> classification
    fields: dict[str, str] = field(default_factory=dict)
    compatibility_mode: str = "backward"

    def schema(self) -> dict[str, Any]:
        """A minimal JSON-schema-ish document carrying per-field classification."""
        return {
            "event_type": self.event_type,
            "version": self.version,
            "properties": {
                name: {"classification": cls} for name, cls in sorted(self.fields.items())
            },
        }

    def schema_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.schema(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def validate_contract(contract: EventContract) -> None:
    """Reject a contract whose event-level or field-level classification is
    unknown, or that carries a `credential_secret` field (§13.4)."""
    if contract.classification not in CLASSIFICATIONS:
        raise ContractViolation(
            f"{contract.event_type} v{contract.version}: unknown classification "
            f"{contract.classification!r}"
        )
    if contract.classification in FORBIDDEN_IN_EVENTS:
        raise ContractViolation(
            f"{contract.event_type} v{contract.version} is classified "
            f"{contract.classification!r}; a secret event may never be persisted"
        )
    for name, cls in contract.fields.items():
        if cls not in CLASSIFICATIONS:
            raise ContractViolation(f"{contract.event_type}.{name}: unknown classification {cls!r}")
        if cls in FORBIDDEN_IN_EVENTS:
            raise ContractViolation(
                f"{contract.event_type}.{name} is classified {cls!r}; a secret may "
                f"never be part of an event contract (§13.4) — redact it upstream"
            )


def validate_sink_mapping(event_type: str, sink: str, classification: str | None) -> None:
    """Reject a sink mapping with no (or unknown) classification (§13.4)."""
    if sink not in SINKS:
        raise ContractViolation(f"{event_type}: unknown sink {sink!r}")
    if not classification:
        raise ContractViolation(
            f"{event_type} → {sink}: sink mapping has no classification (§13.4)"
        )
    if classification not in CLASSIFICATIONS:
        raise ContractViolation(f"{event_type} → {sink}: unknown classification {classification!r}")


# ── The registry ──────────────────────────────────────────────────────
# Canonical §16.2/DA§8.3 domain events + the names Track A already emits.
# Nothing carries a credential_secret field — that is the point.
def _job(evt: str) -> EventContract:
    return EventContract(evt, 1, "internal", {"job_id": "internal", "tenant_id": "internal"})


CONTRACTS: tuple[EventContract, ...] = (
    # Canonical lifecycle (§16.2 / DA§8.3)
    _job("job.v1.created"),
    _job("job.v1.placement_reserved"),
    _job("job.v1.lease_claimed"),
    _job("job.v1.running_observed"),
    _job("job.v1.terminal"),
    EventContract("host.v1.condition_changed", 1, "internal", {"host_id": "internal"}),
    EventContract(
        "command.v1.dead_lettered", 1, "internal", {"command_id": "internal", "host_id": "internal"}
    ),
    EventContract(
        "billing.v1.meter_started", 1, "financial", {"attempt_id": "internal", "owner": "pii"}
    ),
    EventContract(
        "billing.v1.usage_interval_closed",
        1,
        "financial",
        {"attempt_id": "internal", "cost_cad": "financial"},
    ),
    EventContract(
        "billing.v1.wallet_ledger_posted",
        1,
        "financial",
        {"customer_id": "pii", "amount_micros": "financial"},
    ),
    EventContract(
        "billing.v1.invoice_finalized",
        1,
        "financial",
        {"invoice_id": "internal", "total_cad": "financial"},
    ),
    EventContract(
        "billing.v1.provider_payout_posted",
        1,
        "financial",
        {"provider_id": "pii", "amount_cad": "financial"},
    ),
    EventContract(
        "serverless.v1.request_completed",
        1,
        "internal",
        {"endpoint_id": "internal", "job_id": "internal"},
    ),
    EventContract(
        "artifact.v1.available", 1, "internal", {"artifact_id": "internal", "job_id": "internal"}
    ),
    EventContract("artifact.v1.deleted", 1, "internal", {"artifact_id": "internal"}),
    EventContract(
        "privacy.v1.deletion_requested",
        1,
        "pii",
        {
            "request_id": "internal",
            "subject_reference_hash": "pii",
            "deadline_at": "internal",
        },
    ),
    EventContract(
        "privacy.v1.authority_anonymized",
        1,
        "pii",
        {
            "request_id": "internal",
            "subject_reference_hash": "pii",
            "evidence": "internal",
        },
    ),
    EventContract(
        "privacy.v1.deletion_completed",
        1,
        "pii",
        {
            "request_id": "internal",
            "subject_reference_hash": "pii",
            "deadline_at": "internal",
        },
    ),
    EventContract(
        "mcp.v1.action_approved", 1, "internal", {"plan_id": "internal", "client_id": "internal"}
    ),
    EventContract(
        "mcp.v1.tool_completed",
        1,
        "internal",
        {
            "audit_id": "internal",
            "tool_name": "internal",
            "outcome": "internal",
        },
    ),
    # Names Track A already emits — registered as-is, never renamed.
    _job("job.v1.submitted"),
    _job("job.v1.legacy_status_changed"),
    _job("job.v1.queue_blocked"),
    _job("job.v1.attempt_status_changed"),
    _job("job.v1.cancelled"),
    _job("job.v1.preempted"),
    _job("job.v1.lease_expired"),
    _job("job.v1.instance_started"),
    _job("job.v1.instance_stopped"),
    _job("job.v1.instance_restarted"),
    _job("job.v1.instance_terminated"),
    EventContract("host.v1.status_changed", 1, "internal", {"host_id": "internal"}),
    EventContract("host.v1.removed", 1, "internal", {"host_id": "internal"}),
    EventContract("pricing.v1.spot_prices_updated", 1, "public", {"gpu_model": "public"}),
)

def register_all(conn: Any) -> int:
    """Upsert every contract into `event_contracts`. Validates first — an invalid
    contract is a build/deploy failure, never a silently-registered secret."""
    n = 0
    for c in CONTRACTS:
        validate_contract(c)
        conn.execute(
            """
            INSERT INTO event_contracts
                (event_type, version, schema, schema_sha256, classification,
                 compatibility_mode, active, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, TRUE, clock_timestamp())
            ON CONFLICT (event_type, version) DO UPDATE
               SET schema = EXCLUDED.schema,
                   schema_sha256 = EXCLUDED.schema_sha256,
                   classification = EXCLUDED.classification,
                   compatibility_mode = EXCLUDED.compatibility_mode,
                   active = TRUE,
                   updated_at = clock_timestamp()
            """,
            (
                c.event_type,
                c.version,
                json.dumps(c.schema()),
                c.schema_sha256(),
                c.classification,
                c.compatibility_mode,
            ),
        )
        n += 1
    return n
