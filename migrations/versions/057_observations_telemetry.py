"""Host observations, observed workloads, shared telemetry, heartbeats.

Blueprint §13.4 (migration 057): the observed-state half of the
desired-vs-observed reconciliation model (§12), plus shared "latest
telemetry" persistence so a multi-worker API stops keeping the fleet's
current state in one process's memory (§5.8).

- ``host_observations``: immutable snapshot per (host, session,
  inventory_generation). ``received_at`` (API receipt, DB clock) is the
  authoritative freshness signal; worker-reported time is diagnostic only
  (§12.2).
- ``observed_workloads``: what the agent actually saw running, keyed to
  attempt/fence so the reconciler can detect stale-fence containers.
- ``telemetry_latest``: one upserted row per (host, gpu) — replaces the
  process-local latest-telemetry dictionary in ``routes/agent.py``.
- ``telemetry_samples``: partitioned history (monthly range partitions,
  created ahead by a durable scheduled task seeded here — §13.4 forbids
  ad-hoc partition creation in request handlers; a DEFAULT partition
  catches writes that outrun maintenance instead of erroring).
- ``service_heartbeats``: scheduler/reconciler/outbox/maintenance replica
  liveness + schema revision for the §21.3 readiness contract.

Expand-only: nothing reads these tables yet.

Revision ID: 057
Revises: 056
Create Date: 2026-07-17
"""

import datetime as _dt

from alembic import op

revision = "057"
down_revision = "056"
branch_labels = None
depends_on = None

# Monthly partitions created at migration time (current + next two).
# After that, the seeded 'telemetry_partition_maintenance' scheduled task
# owns partition lifecycle.
_INITIAL_PARTITION_MONTHS = 3


def _month_bounds(start: _dt.date, offset: int) -> tuple[str, str, str]:
    """(suffix, from_iso, to_iso) for the month `offset` months after start."""
    year = start.year + (start.month - 1 + offset) // 12
    month = (start.month - 1 + offset) % 12 + 1
    frm = _dt.date(year, month, 1)
    if month == 12:
        to = _dt.date(year + 1, 1, 1)
    else:
        to = _dt.date(year, month + 1, 1)
    return f"{year:04d}{month:02d}", frm.isoformat(), to.isoformat()


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    # ── host_observations ────────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS host_observations (
            observation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            host_id TEXT NOT NULL REFERENCES hosts(host_id) ON DELETE CASCADE,
            session_id TEXT NOT NULL,
            inventory_generation BIGINT NOT NULL DEFAULT 0,
            agent_version TEXT,
            capabilities JSONB NOT NULL DEFAULT '{}'::jsonb,
            conditions JSONB NOT NULL DEFAULT '{}'::jsonb,
            gpu_inventory JSONB NOT NULL DEFAULT '[]'::jsonb,
            observed_workload_count INTEGER NOT NULL DEFAULT 0,
            command_journal_watermark BIGINT,
            worker_reported_at TIMESTAMPTZ,
            received_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            CONSTRAINT uq_host_observation_generation
                UNIQUE (host_id, session_id, inventory_generation)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_host_observations_latest
        ON host_observations (host_id, received_at DESC)
        """
    )

    # ── observed_workloads ───────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS observed_workloads (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            observation_id UUID NOT NULL
                REFERENCES host_observations(observation_id) ON DELETE CASCADE,
            host_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            job_id TEXT,
            attempt_id UUID,
            fencing_token BIGINT,
            container_id TEXT,
            container_name TEXT,
            spec_hash TEXT,
            state TEXT NOT NULL DEFAULT 'unknown'
                CHECK (state IN ('preparing', 'running', 'paused', 'exited',
                                 'removing', 'unmanaged', 'unknown')),
            details JSONB NOT NULL DEFAULT '{}'::jsonb,
            observed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_observed_workloads_host
        ON observed_workloads (host_id, observed_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_observed_workloads_attempt
        ON observed_workloads (attempt_id)
        WHERE attempt_id IS NOT NULL
        """
    )

    # ── telemetry_latest ─────────────────────────────────────────────
    # gpu_uuid = '' is the host-level row; per-GPU rows use the stable
    # device UUID. Writers upsert; readers get fleet-wide current state
    # from any API worker.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS telemetry_latest (
            host_id TEXT NOT NULL,
            gpu_uuid TEXT NOT NULL DEFAULT '',
            sample JSONB NOT NULL,
            received_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (host_id, gpu_uuid)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_telemetry_latest_freshness
        ON telemetry_latest (received_at)
        """
    )

    # ── telemetry_samples (partitioned history) ──────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS telemetry_samples (
            id BIGINT GENERATED ALWAYS AS IDENTITY,
            host_id TEXT NOT NULL,
            gpu_uuid TEXT NOT NULL DEFAULT '',
            sample JSONB NOT NULL,
            received_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (id, received_at)
        ) PARTITION BY RANGE (received_at)
        """
    )
    today = _dt.date.today().replace(day=1)
    for offset in range(_INITIAL_PARTITION_MONTHS):
        suffix, frm, to = _month_bounds(today, offset)
        op.execute(
            f"""
            CREATE TABLE IF NOT EXISTS telemetry_samples_{suffix}
            PARTITION OF telemetry_samples
            FOR VALUES FROM ('{frm}') TO ('{to}')
            """
        )
    # Safety net: writes beyond the pre-created range land here (and page
    # via the partition-lag metric) instead of failing ingestion.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS telemetry_samples_default
        PARTITION OF telemetry_samples DEFAULT
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_telemetry_samples_host
        ON telemetry_samples (host_id, received_at DESC)
        """
    )

    # ── service_heartbeats ───────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS service_heartbeats (
            service TEXT NOT NULL
                CHECK (service IN ('scheduler', 'reconciler', 'outbox',
                                   'maintenance', 'api')),
            replica_id TEXT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            last_heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            schema_revision TEXT,
            service_version TEXT,
            details JSONB NOT NULL DEFAULT '{}'::jsonb,
            PRIMARY KEY (service, replica_id)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_service_heartbeats_freshness
        ON service_heartbeats (service, last_heartbeat_at DESC)
        """
    )

    # ── Durable partition maintenance task (§13.4) ───────────────────
    op.execute(
        """
        INSERT INTO scheduled_tasks (task_name, interval_seconds, payload)
        VALUES (
            'telemetry_partition_maintenance',
            86400,
            '{"table": "telemetry_samples", "months_ahead": 2,
              "retention_months": 6}'::jsonb
        )
        ON CONFLICT (task_name) DO NOTHING
        """
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute(
        "DELETE FROM scheduled_tasks WHERE task_name = 'telemetry_partition_maintenance'"
    )
    op.execute("DROP TABLE IF EXISTS service_heartbeats")
    op.execute("DROP TABLE IF EXISTS telemetry_samples")  # drops partitions too
    op.execute("DROP TABLE IF EXISTS telemetry_latest")
    op.execute("DROP TABLE IF EXISTS observed_workloads")
    op.execute("DROP TABLE IF EXISTS host_observations")
