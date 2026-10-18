"""Per-sink projection delivery + checkpoints (Track B B4.4).

Companion §12.1: extend the **existing** outbox (never a second outbox
authority) with the event metadata a projection needs, and add two tables that
track fan-out per sink so a dispatcher crash converges on exactly one logical
delivery per (event, sink):

  * `projection_deliveries` — one row per (event_id, sink). PK makes fan-out
    preparation idempotent; a partial unique on (sink, external_id) makes the
    success record idempotent by the sink's stable external id.
  * `projection_checkpoints` — per-sink cursor + active flag + an explicit
    `backfilled_from` bound, so a sink added later only receives events inside a
    stated backfill range, never the entire history by accident.

Two durable stages (implemented in control_plane/projection_delivery.py):
  1. claim un-prepared outbox rows → INSERT the per-sink delivery rows and set
     `fanout_prepared_at` in one short transaction (`fanout_prepared_at` means
     obligations were materialized, NOT that any sink succeeded);
  2. claim delivery rows, do external I/O outside the transaction, record success
     by stable external id.

Expand-only: every added column is nullable or defaulted.

Revision ID: 074
Revises: 073
Create Date: 2026-07-24
"""

from typing import Sequence, Union

from alembic import op

revision: str = "074"
down_revision: Union[str, None] = "073"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE outbox_events
            ADD COLUMN IF NOT EXISTS event_version      INTEGER NOT NULL DEFAULT 1,
            ADD COLUMN IF NOT EXISTS tenant_id          TEXT,
            ADD COLUMN IF NOT EXISTS occurred_at        TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS classification     TEXT,
            ADD COLUMN IF NOT EXISTS payload_sha256     TEXT,
            ADD COLUMN IF NOT EXISTS correlation_id     TEXT,
            ADD COLUMN IF NOT EXISTS causation_id       TEXT,
            ADD COLUMN IF NOT EXISTS trace_id           TEXT,
            ADD COLUMN IF NOT EXISTS fanout_prepared_at TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS fanout_attempts    INTEGER NOT NULL DEFAULT 0
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_outbox_events_unprepared "
        "ON outbox_events (created_at) WHERE fanout_prepared_at IS NULL"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS projection_deliveries (
            event_id      UUID NOT NULL,
            sink          TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'pending'
                          CHECK (status IN ('pending', 'delivered', 'dead_lettered')),
            external_id   TEXT,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            max_attempts  INTEGER NOT NULL DEFAULT 10,
            last_error    TEXT,
            prepared_at   TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            available_at  TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            delivered_at  TIMESTAMPTZ,
            claim_owner   TEXT,
            claim_expires_at TIMESTAMPTZ,
            PRIMARY KEY (event_id, sink)
        )
        """
    )
    # Success is recorded by the sink's stable external id — a replayed delivery
    # with the same external id collapses, so at-least-once I/O becomes
    # exactly-once logical delivery.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_projection_deliveries_external "
        "ON projection_deliveries (sink, external_id) WHERE external_id IS NOT NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_projection_deliveries_pending "
        "ON projection_deliveries (sink, available_at) WHERE status = 'pending'"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS projection_checkpoints (
            sink             TEXT PRIMARY KEY,
            active           BOOLEAN NOT NULL DEFAULT TRUE,
            last_prepared_at TIMESTAMPTZ,
            backfilled_from  TIMESTAMPTZ,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            updated_at       TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS projection_checkpoints")
    op.execute("DROP TABLE IF EXISTS projection_deliveries")
    op.execute("DROP INDEX IF EXISTS idx_outbox_events_unprepared")
    op.execute(
        """
        ALTER TABLE outbox_events
            DROP COLUMN IF EXISTS event_version,
            DROP COLUMN IF EXISTS tenant_id,
            DROP COLUMN IF EXISTS occurred_at,
            DROP COLUMN IF EXISTS classification,
            DROP COLUMN IF EXISTS payload_sha256,
            DROP COLUMN IF EXISTS correlation_id,
            DROP COLUMN IF EXISTS causation_id,
            DROP COLUMN IF EXISTS trace_id,
            DROP COLUMN IF EXISTS fanout_prepared_at,
            DROP COLUMN IF EXISTS fanout_attempts
        """
    )
