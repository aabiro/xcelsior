"""Job attempts, physical GPU inventory, device allocations, fenced leases.

Blueprint §13.2 (migration 055): create the authority-boundary tables for
the transactional placement protocol.

- ``job_attempts``: one row per execution attempt; ``fencing_token`` is
  drawn from ``placement_fencing_token_seq`` and is the monotonic authority
  boundary (ADR-005). Partial unique index enforces at most one active
  attempt per job (§10.6).
- ``host_gpu_devices``: normalized physical GPU inventory keyed by stable
  device UUID, with MIG children via ``parent_gpu_device_id``. This is the
  concrete row-lock target for reservation transactions.
- ``gpu_device_allocations``: physical device occupancy per attempt.
  Deliberately distinct from the legacy marketplace ``gpu_allocations``
  table (migration 005), which records commercial offer/price allocation —
  do not overload or rename that table here.
- ``placement_leases``: fenced execution leases (offered → active →
  released/expired/fenced). Legacy ``leases`` stays read-write for the v1
  worker protocol until cutover; only currently *active* legacy leases are
  backfilled as transitional attempt/lease records so reconciliation has a
  fence to reason about from day one.

Expand-only: nothing here is read or written by production code yet.

Revision ID: 055
Revises: 054
Create Date: 2026-07-17
"""

from alembic import op
import sqlalchemy as sa

revision = "055"
down_revision = "054"
branch_labels = None
depends_on = None

# Attempt statuses considered "active" — must match the partial unique
# index predicate and the future scheduler reservation code.
ACTIVE_ATTEMPT_STATUSES = (
    "reserved",
    "command_pending",
    "lease_offered",
    "lease_claimed",
    "starting",
    "running",
)

_ATTEMPT_STATUSES = ACTIVE_ATTEMPT_STATUSES + (
    "succeeded",
    "failed",
    "cancelled",
    "preempted",
    "lost",
    "fenced",
)


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    # ── Fencing token authority ──────────────────────────────────────
    # One global monotonic sequence: a higher token always means newer
    # authority, across jobs and hosts. BIGINT gives effectively
    # unlimited headroom.
    op.execute("CREATE SEQUENCE IF NOT EXISTS placement_fencing_token_seq AS BIGINT")

    # ── job_attempts ─────────────────────────────────────────────────
    attempt_status_list = ", ".join(f"'{s}'" for s in _ATTEMPT_STATUSES)
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS job_attempts (
            attempt_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
            attempt_number INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'reserved'
                CHECK (status IN ({attempt_status_list})),
            host_id TEXT,
            fencing_token BIGINT NOT NULL,
            job_generation BIGINT NOT NULL DEFAULT 1,
            spec_hash TEXT,
            policy_version TEXT,
            placement_score BIGINT,
            placement_explanation JSONB,
            failure_code TEXT,
            failure_details JSONB,
            reserved_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            command_created_at TIMESTAMPTZ,
            lease_claimed_at TIMESTAMPTZ,
            started_at TIMESTAMPTZ,
            ended_at TIMESTAMPTZ,
            created_by TEXT,
            trace_id TEXT,
            CONSTRAINT uq_job_attempt_number UNIQUE (job_id, attempt_number),
            CONSTRAINT uq_job_attempt_fence UNIQUE (fencing_token)
        )
        """
    )
    active_list = ", ".join(f"'{s}'" for s in ACTIVE_ATTEMPT_STATUSES)
    # §10.6 last line of defense: two schedulers can never both commit an
    # active attempt for the same job, whatever bug reaches this point.
    op.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_job_one_active_attempt
        ON job_attempts (job_id)
        WHERE status IN ({active_list})
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_attempts_host "
        "ON job_attempts (host_id) WHERE host_id IS NOT NULL"
    )

    # ── host_gpu_devices ─────────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS host_gpu_devices (
            gpu_device_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            host_id TEXT NOT NULL REFERENCES hosts(host_id) ON DELETE CASCADE,
            gpu_uuid TEXT NOT NULL,
            parent_gpu_device_id UUID
                REFERENCES host_gpu_devices(gpu_device_id) ON DELETE CASCADE,
            device_index INTEGER,
            pci_bus_id TEXT,
            model TEXT NOT NULL DEFAULT '',
            vendor TEXT NOT NULL DEFAULT 'nvidia',
            architecture TEXT,
            total_vram_mb INTEGER NOT NULL DEFAULT 0
                CHECK (total_vram_mb >= 0),
            allocatable_vram_mb INTEGER NOT NULL DEFAULT 0
                CHECK (allocatable_vram_mb >= 0),
            allocation_mode TEXT NOT NULL DEFAULT 'exclusive'
                CHECK (allocation_mode IN ('exclusive', 'shared', 'mig')),
            max_shares INTEGER NOT NULL DEFAULT 1 CHECK (max_shares >= 1),
            topology_group TEXT,
            health TEXT NOT NULL DEFAULT 'unknown'
                CHECK (health IN ('healthy', 'degraded', 'unhealthy', 'unknown')),
            condition_details JSONB NOT NULL DEFAULT '{}'::jsonb,
            inventory_generation BIGINT NOT NULL DEFAULT 0,
            last_observed_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            retired_at TIMESTAMPTZ,
            CONSTRAINT uq_host_gpu_uuid UNIQUE (host_id, gpu_uuid)
        )
        """
    )
    # Reservation transactions lock devices in stable (host_id, gpu_uuid)
    # order (§2.4) — the unique constraint above provides that index.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_host_gpu_devices_parent "
        "ON host_gpu_devices (parent_gpu_device_id) "
        "WHERE parent_gpu_device_id IS NOT NULL"
    )

    # ── gpu_device_allocations ───────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS gpu_device_allocations (
            allocation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            attempt_id UUID NOT NULL
                REFERENCES job_attempts(attempt_id) ON DELETE CASCADE,
            job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
            host_id TEXT NOT NULL,
            gpu_device_id UUID NOT NULL
                REFERENCES host_gpu_devices(gpu_device_id) ON DELETE CASCADE,
            allocation_mode TEXT NOT NULL DEFAULT 'exclusive'
                CHECK (allocation_mode IN ('exclusive', 'shared', 'mig')),
            requested_vram_mb INTEGER NOT NULL DEFAULT 0
                CHECK (requested_vram_mb >= 0),
            requested_shares INTEGER NOT NULL DEFAULT 1
                CHECK (requested_shares >= 1),
            status TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active', 'released', 'fenced')),
            allocated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            released_at TIMESTAMPTZ,
            release_reason TEXT,
            -- Released/fenced rows must record when; active rows must not.
            CONSTRAINT ck_allocation_release_consistent CHECK (
                (status = 'active' AND released_at IS NULL)
                OR (status <> 'active' AND released_at IS NOT NULL)
            )
        )
        """
    )
    # §10.6: one exclusive owner per physical device, enforced by the
    # database even if every application-level check fails.
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_gpu_one_exclusive_allocation
        ON gpu_device_allocations (gpu_device_id)
        WHERE status = 'active' AND allocation_mode = 'exclusive'
        """
    )
    # One attempt cannot hold two active allocations on the same device.
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_attempt_device_active_allocation
        ON gpu_device_allocations (attempt_id, gpu_device_id)
        WHERE status = 'active'
        """
    )
    # Fractional-capacity sum recalculation under the device row lock.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_gpu_device_allocations_device_active
        ON gpu_device_allocations (gpu_device_id)
        WHERE status = 'active'
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_gpu_device_allocations_job "
        "ON gpu_device_allocations (job_id)"
    )

    # ── placement_leases ─────────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS placement_leases (
            lease_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
            attempt_id UUID NOT NULL
                REFERENCES job_attempts(attempt_id) ON DELETE CASCADE,
            host_id TEXT NOT NULL,
            fencing_token BIGINT NOT NULL,
            status TEXT NOT NULL DEFAULT 'offered'
                CHECK (status IN ('offered', 'active', 'released', 'expired', 'fenced')),
            offered_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            claim_deadline TIMESTAMPTZ NOT NULL,
            claimed_at TIMESTAMPTZ,
            last_renewed_at TIMESTAMPTZ,
            expires_at TIMESTAMPTZ,
            released_at TIMESTAMPTZ,
            claim_ttl_sec INTEGER NOT NULL DEFAULT 60 CHECK (claim_ttl_sec > 0),
            renewal_ttl_sec INTEGER NOT NULL DEFAULT 300 CHECK (renewal_ttl_sec > 0),
            last_worker_session_id TEXT,
            -- An active lease always has a claim time and an expiry.
            CONSTRAINT ck_lease_active_shape CHECK (
                status <> 'active'
                OR (claimed_at IS NOT NULL AND expires_at IS NOT NULL)
            )
        )
        """
    )
    # §10.6: at most one live (offered or active) lease per attempt.
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_attempt_one_active_lease
        ON placement_leases (attempt_id)
        WHERE status IN ('offered', 'active')
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_placement_leases_host_live
        ON placement_leases (host_id)
        WHERE status IN ('offered', 'active')
        """
    )
    # Lease-expiry sweeps scan by deadline.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_placement_leases_expiry
        ON placement_leases (expires_at)
        WHERE status = 'active'
        """
    )

    # ── jobs.active_attempt_id FK (deferred from 054) ────────────────
    # NOT VALID first so the ACCESS EXCLUSIVE window is a catalog-only
    # update; VALIDATE then scans without blocking writes. The column is
    # all-NULL at this point, so validation is trivially clean.
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                 WHERE conname = 'fk_jobs_active_attempt'
                   AND conrelid = 'jobs'::regclass
            ) THEN
                ALTER TABLE jobs
                    ADD CONSTRAINT fk_jobs_active_attempt
                    FOREIGN KEY (active_attempt_id)
                    REFERENCES job_attempts(attempt_id)
                    ON DELETE SET NULL
                    NOT VALID;
            END IF;
        END$$;
        """
    )
    op.execute("ALTER TABLE jobs VALIDATE CONSTRAINT fk_jobs_active_attempt")

    _backfill_active_legacy_leases()


def _backfill_active_legacy_leases() -> None:
    """Create transitional attempt + fenced lease records for live work.

    Only leases that are currently ``active`` in the legacy table are
    projected: each gets a v1 attempt row (attempt_number 1, status mapped
    from the job's current status) with a fresh fencing token, an active
    ``placement_leases`` row carrying the legacy expiry, and the job's
    ``active_attempt_id`` pointer. Terminal/expired legacy leases carry no
    authority and are deliberately left behind.

    Idempotent: a job that already has an attempt row is skipped, so
    re-running the migration (or a resumed partial apply) cannot create a
    second attempt — the partial unique index would refuse it anyway.
    """
    bind = op.get_bind()
    bind.execute(
        sa.text(
            """
            WITH live AS (
                SELECT l.lease_id,
                       l.job_id,
                       l.host_id,
                       l.granted_at,
                       l.expires_at,
                       l.last_renewed,
                       l.duration_sec,
                       j.status AS job_status,
                       j.generation AS job_generation
                  FROM leases l
                  JOIN jobs j ON j.job_id = l.job_id
                 WHERE l.status = 'active'
                   AND j.status IN ('assigned', 'leased', 'starting',
                                    'running', 'stopping', 'restarting')
                   AND NOT EXISTS (
                       SELECT 1 FROM job_attempts a WHERE a.job_id = l.job_id
                   )
            ),
            new_attempts AS (
                INSERT INTO job_attempts (
                    job_id, attempt_number, status, host_id, fencing_token,
                    job_generation, reserved_at, lease_claimed_at, started_at,
                    created_by
                )
                SELECT job_id,
                       1,
                       CASE job_status
                           WHEN 'assigned'   THEN 'lease_claimed'
                           WHEN 'leased'     THEN 'lease_claimed'
                           WHEN 'starting'   THEN 'starting'
                           WHEN 'restarting' THEN 'starting'
                           ELSE 'running'
                       END,
                       host_id,
                       nextval('placement_fencing_token_seq'),
                       job_generation,
                       to_timestamp(granted_at),
                       to_timestamp(granted_at),
                       CASE WHEN job_status IN ('running', 'stopping')
                            THEN to_timestamp(last_renewed) END,
                       'migration:055'
                  FROM live
                RETURNING attempt_id, job_id, host_id, fencing_token
            ),
            new_leases AS (
                INSERT INTO placement_leases (
                    job_id, attempt_id, host_id, fencing_token, status,
                    offered_at, claim_deadline, claimed_at, last_renewed_at,
                    expires_at, renewal_ttl_sec, last_worker_session_id
                )
                SELECT a.job_id,
                       a.attempt_id,
                       a.host_id,
                       a.fencing_token,
                       'active',
                       to_timestamp(live.granted_at),
                       to_timestamp(live.granted_at),
                       to_timestamp(live.granted_at),
                       to_timestamp(live.last_renewed),
                       to_timestamp(live.expires_at),
                       GREATEST(COALESCE(live.duration_sec, 300), 1),
                       'legacy:' || live.lease_id
                  FROM new_attempts a
                  JOIN live ON live.job_id = a.job_id
                RETURNING attempt_id, job_id
            )
            UPDATE jobs j
               SET active_attempt_id = a.attempt_id,
                   version = j.version + 1,
                   updated_at = clock_timestamp()
              FROM new_attempts a
             WHERE j.job_id = a.job_id
            """
        )
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")

    op.execute("ALTER TABLE jobs DROP CONSTRAINT IF EXISTS fk_jobs_active_attempt")
    op.execute("UPDATE jobs SET active_attempt_id = NULL WHERE active_attempt_id IS NOT NULL")

    op.execute("DROP TABLE IF EXISTS placement_leases")
    op.execute("DROP TABLE IF EXISTS gpu_device_allocations")
    op.execute("DROP TABLE IF EXISTS host_gpu_devices")
    op.execute("DROP TABLE IF EXISTS job_attempts")
    op.execute("DROP SEQUENCE IF EXISTS placement_fencing_token_seq")
