"""Control-plane core columns for jobs and hosts (Track A, expand phase).

Blueprint §13.1 (migration 054): normalize the coordination fields the
transactional scheduler needs onto ``jobs`` and ``hosts`` without changing
any existing behavior. Everything here is *expand-only*:

- every new column is nullable or has a safe default;
- the legacy ``status`` column remains authoritative until cutover;
- ``phase`` / ``desired_state`` are backfilled as projections of ``status``
  in bounded batches, then verified — NOT NULL is deferred to the contract
  migration (060) once the verification metric holds at zero in production;
- indexes are created with plain ``CREATE INDEX IF NOT EXISTS`` inside the
  migration transaction. Current deployments have small ``jobs``/``hosts``
  tables (single-host production); if a deployment ever has millions of
  rows, re-run these as ``CREATE INDEX CONCURRENTLY`` via the documented
  runbook instead of holding this migration open.

Revision ID: 054
Revises: 053
Create Date: 2026-07-17
"""

from alembic import op
import sqlalchemy as sa

revision = "054"
down_revision = "053"
branch_labels = None
depends_on = None

# Bounded backfill batch size — keeps row locks and WAL bursts small on
# production-sized tables while remaining a single pass on dev/test.
BACKFILL_BATCH_SIZE = 1000

# Legacy job ``status`` → control-plane ``phase`` projection. Must cover
# every member of scheduler.VALID_STATUSES; verified below after backfill.
_PHASE_CASE_SQL = """
    CASE status
        WHEN 'queued'      THEN 'pending'
        WHEN 'preempted'   THEN 'pending'
        WHEN 'assigned'    THEN 'scheduled'
        WHEN 'leased'      THEN 'scheduled'
        WHEN 'starting'    THEN 'starting'
        WHEN 'restarting'  THEN 'starting'
        WHEN 'running'     THEN 'running'
        WHEN 'stopping'    THEN 'running'
        WHEN 'completed'   THEN 'succeeded'
        WHEN 'failed'      THEN 'failed'
        WHEN 'cancelled'   THEN 'stopped'
        WHEN 'stopped'     THEN 'stopped'
        WHEN 'paused'      THEN 'stopped'
        WHEN 'terminated'  THEN 'stopped'
        ELSE NULL
    END
"""

_DESIRED_STATE_CASE_SQL = """
    CASE
        WHEN status IN ('stopping', 'stopped', 'paused', 'cancelled', 'terminated')
            THEN 'stopped'
        ELSE 'running'
    END
"""


def upgrade() -> None:
    # Fail fast instead of queueing behind long-held locks: if another
    # writer holds a conflicting lock, abort and let the operator retry.
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    # ── jobs: identity / tenancy ─────────────────────────────────────
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS tenant_id TEXT")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS team_id TEXT")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS owner_id TEXT")

    # ── jobs: desired state / phase / reasons ────────────────────────
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS desired_state TEXT")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS phase TEXT")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS reason_code TEXT")
    op.execute(
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS reason_details JSONB"
    )

    # ── jobs: optimistic concurrency / reconciliation generations ────
    op.execute(
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS generation BIGINT NOT NULL DEFAULT 1"
    )
    op.execute(
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS observed_generation BIGINT NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS version BIGINT NOT NULL DEFAULT 1"
    )

    # ── jobs: attempt linkage + canonical spec ───────────────────────
    # FK to job_attempts is added in migration 055 after that table exists.
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS active_attempt_id UUID")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS spec JSONB")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS spec_hash TEXT")

    # ── jobs: queue ordering (PostgreSQL time, not host clocks) ──────
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS queued_at TIMESTAMPTZ")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS next_schedule_at TIMESTAMPTZ")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ")
    op.execute(
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS effective_priority BIGINT NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS fair_share_finish NUMERIC(20, 6) "
        "NOT NULL DEFAULT 0"
    )

    # ── jobs: short scheduling claim (Stage B of placement protocol) ─
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS schedule_claim_owner TEXT")
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS schedule_claim_token UUID")
    op.execute(
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS schedule_claim_expires_at TIMESTAMPTZ"
    )
    op.execute(
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS schedule_attempt_count INTEGER "
        "NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS last_schedule_conflict_at TIMESTAMPTZ"
    )

    # ── jobs: billing linkage (FK added with wallet_holds in 058) ────
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS wallet_hold_id UUID")

    # ── hosts: tenancy / location ────────────────────────────────────
    op.execute("ALTER TABLE hosts ADD COLUMN IF NOT EXISTS tenant_id TEXT")
    op.execute("ALTER TABLE hosts ADD COLUMN IF NOT EXISTS provider_id TEXT")
    op.execute("ALTER TABLE hosts ADD COLUMN IF NOT EXISTS owner_id TEXT")
    op.execute("ALTER TABLE hosts ADD COLUMN IF NOT EXISTS region TEXT")
    op.execute("ALTER TABLE hosts ADD COLUMN IF NOT EXISTS country TEXT")
    op.execute("ALTER TABLE hosts ADD COLUMN IF NOT EXISTS province TEXT")

    # ── hosts: administrative vs observed state split ────────────────
    op.execute(
        "ALTER TABLE hosts ADD COLUMN IF NOT EXISTS administrative_state TEXT "
        "NOT NULL DEFAULT 'admitted'"
    )
    op.execute(
        "ALTER TABLE hosts ADD COLUMN IF NOT EXISTS availability_state TEXT "
        "NOT NULL DEFAULT 'unknown'"
    )
    op.execute(
        "ALTER TABLE hosts ADD COLUMN IF NOT EXISTS generation BIGINT NOT NULL DEFAULT 1"
    )
    op.execute(
        "ALTER TABLE hosts ADD COLUMN IF NOT EXISTS observed_generation BIGINT "
        "NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE hosts ADD COLUMN IF NOT EXISTS version BIGINT NOT NULL DEFAULT 1"
    )
    op.execute(
        "ALTER TABLE hosts ADD COLUMN IF NOT EXISTS inventory_generation BIGINT "
        "NOT NULL DEFAULT 0"
    )
    op.execute("ALTER TABLE hosts ADD COLUMN IF NOT EXISTS last_observed_at TIMESTAMPTZ")
    op.execute("ALTER TABLE hosts ADD COLUMN IF NOT EXISTS observation_session_id TEXT")
    op.execute("ALTER TABLE hosts ADD COLUMN IF NOT EXISTS drain_deadline TIMESTAMPTZ")
    op.execute("ALTER TABLE hosts ADD COLUMN IF NOT EXISTS drain_reason TEXT")
    op.execute(
        "ALTER TABLE hosts ADD COLUMN IF NOT EXISTS capabilities JSONB "
        "NOT NULL DEFAULT '{}'::jsonb"
    )
    op.execute(
        "ALTER TABLE hosts ADD COLUMN IF NOT EXISTS conditions JSONB "
        "NOT NULL DEFAULT '{}'::jsonb"
    )

    # ── CHECK constraints: added NOT VALID, then validated ───────────
    # NOT VALID keeps the ACCESS EXCLUSIVE window to a catalog update;
    # VALIDATE takes only SHARE UPDATE EXCLUSIVE and scans without
    # blocking writes. NULL passes CHECK, so pre-backfill rows are fine.
    for name, table, expr in (
        (
            "ck_jobs_desired_state",
            "jobs",
            "desired_state IS NULL OR desired_state IN ('running', 'stopped')",
        ),
        (
            "ck_jobs_phase",
            "jobs",
            "phase IS NULL OR phase IN "
            "('pending', 'scheduled', 'starting', 'running', "
            "'succeeded', 'failed', 'stopped')",
        ),
        (
            "ck_hosts_administrative_state",
            "hosts",
            "administrative_state IN ('admitted', 'draining', 'disabled')",
        ),
        (
            "ck_hosts_availability_state",
            "hosts",
            "availability_state IN ('ready', 'not_ready', 'unknown')",
        ),
    ):
        op.execute(
            f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                     WHERE conname = '{name}'
                       AND conrelid = '{table}'::regclass
                ) THEN
                    ALTER TABLE {table}
                        ADD CONSTRAINT {name} CHECK ({expr}) NOT VALID;
                END IF;
            END$$;
            """
        )
        op.execute(f"ALTER TABLE {table} VALIDATE CONSTRAINT {name}")

    # ── Backfill: bounded batches, resumable, verified ───────────────
    _backfill_jobs()
    _backfill_hosts()
    _verify_backfill()

    # ── Indexes ──────────────────────────────────────────────────────
    # Stage-B queue claim scan: pending work ordered exactly as the
    # claim query orders it, so the scan is index-only until the lock.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_jobs_schedule_queue
        ON jobs (effective_priority DESC, fair_share_finish ASC, queued_at ASC)
        WHERE phase = 'pending' AND desired_state = 'running'
        """
    )
    # Expired-claim sweep by the maintenance/reconcile loop.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_jobs_schedule_claim_expiry
        ON jobs (schedule_claim_expires_at)
        WHERE schedule_claim_expires_at IS NOT NULL
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_tenant_phase ON jobs (tenant_id, phase)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_owner_phase ON jobs (owner_id, phase)"
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_hosts_admission
        ON hosts (administrative_state, availability_state, last_observed_at)
        """
    )


def _backfill_jobs() -> None:
    """Project legacy job rows into the new control-plane columns.

    Batched by primary key with SKIP LOCKED so a concurrent writer (the
    live scheduler during an expand deploy) never blocks the migration
    and vice versa; rows updated underneath us are re-selected on the
    next pass because the predicate re-checks ``phase IS NULL``.

    Termination invariant: the batch predicate selects only rows the
    status mapping can actually project (``phase CASE`` is not NULL), so
    every claimed row leaves the candidate set and the loop drains. A row
    with an unknown legacy status is deliberately *never* claimed — it
    falls through to ``_verify_backfill``, which aborts the migration and
    names the offending status.

    Without that ``IS NOT NULL`` guard this loop does not terminate: the
    ``ELSE NULL`` branch writes NULL back into the column the predicate
    filters on, so an unmappable row is re-selected forever and
    ``alembic upgrade`` hangs holding its locks instead of failing
    cleanly. Found by ``tests/test_migration_from_production_snapshot.py``
    (Track B B1.2); a from-empty upgrade cannot reach this path.
    """
    bind = op.get_bind()
    # Defence in depth: the predicate above makes a stall impossible, but a
    # migration that hangs is a far worse failure than one that raises.
    max_passes = 1_000_000
    for _ in range(max_passes):
        result = bind.execute(
            sa.text(
                f"""
                WITH batch AS (
                    SELECT job_id
                      FROM jobs
                     WHERE phase IS NULL
                       AND ({_PHASE_CASE_SQL}) IS NOT NULL
                     LIMIT :batch_size
                       FOR UPDATE SKIP LOCKED
                )
                UPDATE jobs j
                   SET phase = {_PHASE_CASE_SQL},
                       desired_state = {_DESIRED_STATE_CASE_SQL},
                       -- Transitional single-user tenancy projection:
                       -- billing already treats the owner as the paying
                       -- customer (billing_cycles.customer_id), so the
                       -- owner is the tenant until real workspaces land.
                       owner_id = COALESCE(NULLIF(j.payload->>'owner', ''), j.owner_id),
                       tenant_id = COALESCE(
                           NULLIF(j.payload->>'tenant_id', ''),
                           NULLIF(j.payload->>'owner', ''),
                           j.tenant_id
                       ),
                       team_id = COALESCE(NULLIF(j.payload->>'team_id', ''), j.team_id),
                       effective_priority = j.priority,
                       queued_at = to_timestamp(j.submitted_at),
                       updated_at = clock_timestamp()
                  FROM batch
                 WHERE j.job_id = batch.job_id
                """
            ),
            {"batch_size": BACKFILL_BATCH_SIZE},
        )
        if result.rowcount == 0:
            return
    raise RuntimeError(
        f"migration 054 jobs backfill did not converge after {max_passes} "
        f"passes; the batch predicate is no longer draining the candidate "
        f"set. Refusing to loop indefinitely."
    )


def _backfill_hosts() -> None:
    """Project legacy host payload fields into normalized columns."""
    bind = op.get_bind()
    while True:
        result = bind.execute(
            sa.text(
                """
                WITH batch AS (
                    SELECT host_id
                      FROM hosts
                     WHERE last_observed_at IS NULL
                     LIMIT :batch_size
                       FOR UPDATE SKIP LOCKED
                )
                UPDATE hosts h
                   SET owner_id = COALESCE(NULLIF(h.payload->>'owner', ''), h.owner_id),
                       provider_id = COALESCE(
                           NULLIF(h.payload->>'provider_id', ''),
                           NULLIF(h.payload->>'owner', ''),
                           h.provider_id
                       ),
                       region = COALESCE(NULLIF(h.payload->>'region', ''), h.region),
                       country = COALESCE(NULLIF(h.payload->>'country', ''), h.country),
                       province = COALESCE(NULLIF(h.payload->>'province', ''), h.province),
                       administrative_state = CASE h.status
                           WHEN 'disabled' THEN 'disabled'
                           WHEN 'draining' THEN 'draining'
                           ELSE 'admitted'
                       END,
                       availability_state = CASE h.status
                           WHEN 'active' THEN 'ready'
                           WHEN 'dead'   THEN 'not_ready'
                           ELSE 'unknown'
                       END,
                       -- Sentinel marking the row as backfilled; the real
                       -- observation pipeline overwrites it on first
                       -- heartbeat after cutover.
                       last_observed_at = to_timestamp(h.registered_at)
                  FROM batch
                 WHERE h.host_id = batch.host_id
                """
            ),
            {"batch_size": BACKFILL_BATCH_SIZE},
        )
        if result.rowcount == 0:
            break


def _verify_backfill() -> None:
    """Abort the migration if any row remains unprojected.

    Blueprint §13.1: verify counts and unmappable records before the
    schema is considered expanded. An unknown legacy status maps phase to
    NULL — that is a data bug we must surface now, not at NOT NULL time.
    """
    bind = op.get_bind()
    unmapped_jobs = bind.execute(
        sa.text(
            "SELECT count(*) FROM jobs WHERE phase IS NULL OR desired_state IS NULL"
        )
    ).scalar_one()
    if unmapped_jobs:
        statuses = bind.execute(
            sa.text(
                "SELECT DISTINCT status FROM jobs "
                "WHERE phase IS NULL OR desired_state IS NULL LIMIT 20"
            )
        ).scalars().all()
        raise RuntimeError(
            f"migration 054 backfill left {unmapped_jobs} jobs rows without a "
            f"phase/desired_state projection; unmapped legacy statuses: {statuses}. "
            "Extend the status mapping and re-run."
        )
    unmapped_hosts = bind.execute(
        sa.text("SELECT count(*) FROM hosts WHERE last_observed_at IS NULL")
    ).scalar_one()
    if unmapped_hosts:
        raise RuntimeError(
            f"migration 054 backfill left {unmapped_hosts} hosts rows unprojected"
        )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")

    op.execute("DROP INDEX IF EXISTS idx_hosts_admission")
    op.execute("DROP INDEX IF EXISTS idx_jobs_owner_phase")
    op.execute("DROP INDEX IF EXISTS idx_jobs_tenant_phase")
    op.execute("DROP INDEX IF EXISTS idx_jobs_schedule_claim_expiry")
    op.execute("DROP INDEX IF EXISTS idx_jobs_schedule_queue")

    op.execute("ALTER TABLE hosts DROP CONSTRAINT IF EXISTS ck_hosts_availability_state")
    op.execute(
        "ALTER TABLE hosts DROP CONSTRAINT IF EXISTS ck_hosts_administrative_state"
    )
    op.execute("ALTER TABLE jobs DROP CONSTRAINT IF EXISTS ck_jobs_phase")
    op.execute("ALTER TABLE jobs DROP CONSTRAINT IF EXISTS ck_jobs_desired_state")

    for col in (
        "conditions",
        "capabilities",
        "drain_reason",
        "drain_deadline",
        "observation_session_id",
        "last_observed_at",
        "inventory_generation",
        "version",
        "observed_generation",
        "generation",
        "availability_state",
        "administrative_state",
        "province",
        "country",
        "region",
        "owner_id",
        "provider_id",
        "tenant_id",
    ):
        op.execute(f"ALTER TABLE hosts DROP COLUMN IF EXISTS {col}")

    for col in (
        "wallet_hold_id",
        "last_schedule_conflict_at",
        "schedule_attempt_count",
        "schedule_claim_expires_at",
        "schedule_claim_token",
        "schedule_claim_owner",
        "fair_share_finish",
        "effective_priority",
        "updated_at",
        "next_schedule_at",
        "queued_at",
        "spec_hash",
        "spec",
        "active_attempt_id",
        "version",
        "observed_generation",
        "generation",
        "reason_details",
        "reason_code",
        "phase",
        "desired_state",
        "owner_id",
        "team_id",
        "tenant_id",
    ):
        op.execute(f"ALTER TABLE jobs DROP COLUMN IF EXISTS {col}")
