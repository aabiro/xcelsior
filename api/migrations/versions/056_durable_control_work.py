"""Durable commands, transactional outbox, idempotency, reconcile queue.

Blueprint §13.3 (migration 056): the durable-work backbone for Track A.

- ``agent_commands`` base table is created with ``CREATE TABLE IF NOT
  EXISTS`` (A1.6 — historically only runtime DDL in ``db._ensure_pg_tables``)
  then *evolved in place* (expand-only) with the claim/ACK lifecycle
  (§9.4: pending → claimed → acknowledged | failed → pending |
  dead_letter; pending → cancelled; claimed → pending on claim timeout),
  attempt/fence references, retry budget, idempotency key, and result
  fields. The v1 drain path (DELETE ... RETURNING in routes/agent.py) keeps
  working untouched: it only ever writes ``status='pending'`` rows through
  an explicit column list. The v2 protocol switches to claim + ACK in a
  later change; nothing reads the new columns yet.
- ``outbox_events``: at-least-once side-effect intents committed in the
  same transaction as the state mutation that implies them (ADR-006/§16.1).
- ``api_idempotency_keys``: one durable response per
  (principal, tenant, route, key) so repeated API/MCP calls return the
  original resource instead of creating another (§8 invariant 9).
- ``reconciliation_queue`` (coalesced by resource) and
  ``reconciliation_findings`` (§12.3).
- ``scheduled_tasks``: durable periodic work claims replacing
  process-local timers (§6.1 maintenance scheduler).

Revision ID: 056
Revises: 055
Create Date: 2026-07-17
"""

from alembic import op

revision = "056"
down_revision = "055"
branch_labels = None
depends_on = None

# Command lifecycle vocabulary (§9.4). 'pending' doubles as the legacy v1
# value so existing rows validate without rewrite.
COMMAND_STATUSES = (
    "pending",
    "claimed",
    "acknowledged",
    "failed",
    "dead_letter",
    "cancelled",
)


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    # ── agent_commands base table (A1.6) ─────────────────────────────
    # Historically created only by db._ensure_pg_tables() runtime DDL.
    # Pure `alembic upgrade head` from an empty DB failed here because the
    # expand ALTERs below assume the table exists. CREATE IF NOT EXISTS is
    # a no-op on already-bootstrapped environments (same pattern as
    # migration 030 for gpu_pricing).
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_commands (
            id BIGSERIAL PRIMARY KEY,
            host_id TEXT NOT NULL,
            command TEXT NOT NULL,
            args JSONB NOT NULL DEFAULT '{}'::jsonb,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at DOUBLE PRECISION NOT NULL
                DEFAULT EXTRACT(EPOCH FROM NOW()),
            expires_at DOUBLE PRECISION NOT NULL
                DEFAULT EXTRACT(EPOCH FROM NOW()) + 900,
            created_by TEXT,
            result JSONB
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_commands_host_pending
        ON agent_commands (host_id, status, created_at)
        WHERE status = 'pending'
        """
    )

    # ── agent_commands: claim/ACK lifecycle (expand in place) ────────
    # Stable identity for the v2 protocol; the legacy BIGSERIAL id stays
    # the PK so v1 queries and FK-free joins are untouched.
    op.execute(
        "ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS command_id UUID "
        "NOT NULL DEFAULT gen_random_uuid()"
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                 WHERE conname = 'uq_agent_commands_command_id'
                   AND conrelid = 'agent_commands'::regclass
            ) THEN
                ALTER TABLE agent_commands
                    ADD CONSTRAINT uq_agent_commands_command_id UNIQUE (command_id);
            END IF;
        END$$;
        """
    )
    # Authority references (§11.1 payload): which attempt/fence this
    # command belongs to. NULL for legacy admin commands.
    op.execute("ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS job_id TEXT")
    op.execute("ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS attempt_id UUID")
    op.execute("ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS fencing_token BIGINT")
    op.execute("ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS spec_hash TEXT")
    # Delivery scheduling.
    op.execute(
        "ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS priority INTEGER "
        "NOT NULL DEFAULT 0"
    )
    op.execute("ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS not_before TIMESTAMPTZ")
    # Claim ownership (claimed → pending redelivery on claim expiry).
    op.execute("ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS claim_owner TEXT")
    op.execute("ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS claim_session TEXT")
    op.execute(
        "ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS claim_expires_at TIMESTAMPTZ"
    )
    # Bounded retry budget.
    op.execute(
        "ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS attempt_count INTEGER "
        "NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS max_attempts INTEGER "
        "NOT NULL DEFAULT 5"
    )
    op.execute(
        "ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ"
    )
    # Idempotency + durable outcome (duplicate ACK returns original result).
    op.execute("ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS idempotency_key TEXT")
    op.execute("ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS acked_at TIMESTAMPTZ")
    op.execute("ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS ack_result JSONB")
    op.execute("ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS error_code TEXT")
    op.execute("ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS error_details JSONB")
    # Trace + retention.
    op.execute("ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS trace_id TEXT")
    op.execute(
        "ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS retention_expires_at TIMESTAMPTZ"
    )

    status_list = ", ".join(f"'{s}'" for s in COMMAND_STATUSES)
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                 WHERE conname = 'ck_agent_commands_status'
                   AND conrelid = 'agent_commands'::regclass
            ) THEN
                ALTER TABLE agent_commands
                    ADD CONSTRAINT ck_agent_commands_status
                    CHECK (status IN ({status_list})) NOT VALID;
            END IF;
        END$$;
        """
    )
    op.execute("ALTER TABLE agent_commands VALIDATE CONSTRAINT ck_agent_commands_status")
    # A claimed command must say who claimed it and until when.
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                 WHERE conname = 'ck_agent_commands_claim_shape'
                   AND conrelid = 'agent_commands'::regclass
            ) THEN
                ALTER TABLE agent_commands
                    ADD CONSTRAINT ck_agent_commands_claim_shape
                    CHECK (
                        status <> 'claimed'
                        OR (claim_owner IS NOT NULL AND claim_expires_at IS NOT NULL)
                    ) NOT VALID;
            END IF;
        END$$;
        """
    )
    op.execute(
        "ALTER TABLE agent_commands VALIDATE CONSTRAINT ck_agent_commands_claim_shape"
    )
    # §10.6: duplicate enqueue of the same intent for one host is refused.
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_command_idempotency
        ON agent_commands (host_id, idempotency_key)
        WHERE idempotency_key IS NOT NULL
        """
    )
    # v2 fetch scan: deliverable work per host in priority order.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_commands_deliverable
        ON agent_commands (host_id, priority DESC, created_at)
        WHERE status = 'pending'
        """
    )
    # Claim-expiry redelivery sweep.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_commands_claim_expiry
        ON agent_commands (claim_expires_at)
        WHERE status = 'claimed'
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_commands_attempt "
        "ON agent_commands (attempt_id) WHERE attempt_id IS NOT NULL"
    )

    # ── outbox_events ────────────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS outbox_events (
            event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            aggregate_type TEXT NOT NULL,
            aggregate_id TEXT NOT NULL,
            aggregate_version BIGINT NOT NULL DEFAULT 0,
            event_type TEXT NOT NULL,
            payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            headers JSONB NOT NULL DEFAULT '{}'::jsonb,
            destination_class TEXT NOT NULL DEFAULT 'default',
            idempotency_key TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            available_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            claim_owner TEXT,
            claim_expires_at TIMESTAMPTZ,
            published_at TIMESTAMPTZ,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 10,
            last_error TEXT,
            dead_lettered_at TIMESTAMPTZ,
            CONSTRAINT uq_outbox_idempotency
                UNIQUE (destination_class, idempotency_key)
        )
        """
    )
    # Dispatcher claim scan: unpublished, not dead-lettered, due now.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_outbox_dispatch
        ON outbox_events (destination_class, available_at)
        WHERE published_at IS NULL AND dead_lettered_at IS NULL
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_outbox_aggregate "
        "ON outbox_events (aggregate_type, aggregate_id, created_at)"
    )

    # ── api_idempotency_keys ─────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS api_idempotency_keys (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            principal TEXT NOT NULL,
            tenant_id TEXT NOT NULL DEFAULT '',
            route TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            request_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'in_progress'
                CHECK (status IN ('in_progress', 'succeeded', 'failed')),
            response_status INTEGER,
            response_body JSONB,
            resource_id TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            expires_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT uq_api_idempotency
                UNIQUE (principal, tenant_id, route, idempotency_key)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_api_idempotency_expiry "
        "ON api_idempotency_keys (expires_at)"
    )

    # ── reconciliation_queue (coalesced per resource, §12.3) ─────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS reconciliation_queue (
            resource_type TEXT NOT NULL,
            resource_id TEXT NOT NULL,
            due_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            priority INTEGER NOT NULL DEFAULT 0,
            reason TEXT NOT NULL DEFAULT '',
            requested_by TEXT,
            claim_owner TEXT,
            claim_expires_at TIMESTAMPTZ,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (resource_type, resource_id)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_reconciliation_queue_due
        ON reconciliation_queue (due_at, priority DESC)
        """
    )

    # ── reconciliation_findings ──────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS reconciliation_findings (
            finding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            resource_type TEXT NOT NULL,
            resource_id TEXT NOT NULL,
            tenant_id TEXT,
            finding_type TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'info'
                CHECK (severity IN ('info', 'warning', 'error', 'critical')),
            summary TEXT NOT NULL DEFAULT '',
            desired JSONB,
            observed JSONB,
            action_taken TEXT,
            action_result JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            resolved_at TIMESTAMPTZ
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_reconciliation_findings_open
        ON reconciliation_findings (resource_type, resource_id, created_at)
        WHERE resolved_at IS NULL
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_reconciliation_findings_tenant "
        "ON reconciliation_findings (tenant_id, created_at) "
        "WHERE tenant_id IS NOT NULL"
    )

    # ── scheduled_tasks (durable periodic work, §6.1) ────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            task_name TEXT PRIMARY KEY,
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            interval_seconds INTEGER NOT NULL CHECK (interval_seconds > 0),
            next_run_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            last_run_at TIMESTAMPTZ,
            last_status TEXT
                CHECK (last_status IS NULL OR last_status IN ('succeeded', 'failed')),
            last_error TEXT,
            claim_owner TEXT,
            claim_expires_at TIMESTAMPTZ,
            payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_due
        ON scheduled_tasks (next_run_at)
        WHERE enabled
        """
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")

    op.execute("DROP TABLE IF EXISTS scheduled_tasks")
    op.execute("DROP TABLE IF EXISTS reconciliation_findings")
    op.execute("DROP TABLE IF EXISTS reconciliation_queue")
    op.execute("DROP TABLE IF EXISTS api_idempotency_keys")
    op.execute("DROP TABLE IF EXISTS outbox_events")

    op.execute("DROP INDEX IF EXISTS idx_agent_commands_attempt")
    op.execute("DROP INDEX IF EXISTS idx_agent_commands_claim_expiry")
    op.execute("DROP INDEX IF EXISTS idx_agent_commands_deliverable")
    op.execute("DROP INDEX IF EXISTS uq_command_idempotency")
    op.execute(
        "ALTER TABLE agent_commands DROP CONSTRAINT IF EXISTS ck_agent_commands_claim_shape"
    )
    op.execute(
        "ALTER TABLE agent_commands DROP CONSTRAINT IF EXISTS ck_agent_commands_status"
    )
    op.execute(
        "ALTER TABLE agent_commands DROP CONSTRAINT IF EXISTS uq_agent_commands_command_id"
    )
    for col in (
        "retention_expires_at",
        "trace_id",
        "error_details",
        "error_code",
        "ack_result",
        "acked_at",
        "idempotency_key",
        "next_attempt_at",
        "max_attempts",
        "attempt_count",
        "claim_expires_at",
        "claim_session",
        "claim_owner",
        "not_before",
        "priority",
        "spec_hash",
        "fencing_token",
        "attempt_id",
        "job_id",
        "command_id",
    ):
        op.execute(f"ALTER TABLE agent_commands DROP COLUMN IF EXISTS {col}")
