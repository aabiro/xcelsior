"""Event contract registry (Track B B4.3).

Companion §12.1 / §13.4: the authoritative registry of every domain event's
type, version, JSON schema, schema hash, data classification, and compatibility
mode. Downstream projections and sinks read this to know an event's shape and
its sensitivity — a `credential_secret`-classified field may never reach an
audit sink, and a sink mapping without a classification is rejected.

The companion's illustrative DDL names an `audit` schema; this repo keeps every
audit-domain table in `public` (events, mcp_tool_audit, audit_events_v2), so the
table lands in `public.event_contracts` and is claimed by the audit domain in
db_roles — matching repo reality rather than forcing a new schema (B4.4 note).

Expand-only; the registry is populated and read by `analytics/contracts.py`.

Revision ID: 073
Revises: 072
Create Date: 2026-07-24
"""

from typing import Sequence, Union

from alembic import op

revision: str = "073"
down_revision: Union[str, None] = "072"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Data-classification vocabulary (companion §13.4). `credential_secret` is the
# one that must never appear in an audit payload.
_CLASSIFICATIONS = ("public", "internal", "pii", "financial", "credential_secret")
_COMPAT_MODES = ("backward", "forward", "full", "none")


def upgrade() -> None:
    classes = ", ".join(f"'{c}'" for c in _CLASSIFICATIONS)
    modes = ", ".join(f"'{m}'" for m in _COMPAT_MODES)
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS event_contracts (
            event_type          TEXT NOT NULL,
            version             INTEGER NOT NULL DEFAULT 1,
            schema              JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            schema_sha256       TEXT NOT NULL,
            classification      TEXT NOT NULL
                                CHECK (classification IN ({classes})),
            compatibility_mode  TEXT NOT NULL DEFAULT 'backward'
                                CHECK (compatibility_mode IN ({modes})),
            active              BOOLEAN NOT NULL DEFAULT TRUE,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            updated_at          TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (event_type, version)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_event_contracts_active "
        "ON event_contracts (event_type) WHERE active"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS event_contracts")
