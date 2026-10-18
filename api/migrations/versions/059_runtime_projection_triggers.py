"""Runtime projection triggers for jobs/hosts control-plane columns.

Track A Phase 4 prerequisite (A2.5): migration 054 backfilled ``phase`` /
``desired_state`` / ``effective_priority`` / ``queued_at`` on jobs and
``administrative_state`` / ``availability_state`` on hosts **once**, but
runtime writers still only touch the legacy columns — and job status is
written from ~15 call sites (scheduler upsert, billing, reaper, agent
routes) with raw SQL. Until every writer is projection-native, the only
place that can keep the projection true on *every* write is the database
itself.

Two ``BEFORE INSERT OR UPDATE`` triggers derive the control-plane columns
from the legacy truth (status / priority / submitted_at / payload):

- jobs: ``phase`` (054's exact status→phase CASE), ``desired_state``,
  ``effective_priority`` (mirrors ``priority``), ``queued_at``
  (``to_timestamp(submitted_at)`` — matches legacy FIFO ordering, so the
  §10.2 claim order equals the legacy queue order during transition).
- hosts: ``administrative_state`` now honors the payload ``admitted``
  flag that legacy allocation actually gates on (054's status-only rule
  marked *every* non-disabled host admitted — too loose; the new
  scheduler would place onto hosts legacy refuses). Never-admitted hosts
  become ``pending``, added to the CHECK constraint here.

The triggers are transitional: they are dropped in the contract phase
once all writers speak the new schema natively (and ``phase`` becomes
authoritative rather than derived).

Also re-runs the drift backfill for rows written since 054 ran.

Revision ID: 059
Revises: 058
Create Date: 2026-07-17
"""

from alembic import op

revision = "059"
down_revision = "058"
branch_labels = None
depends_on = None

# Must stay identical to migration 054's _PHASE_CASE_SQL.
_PHASE_CASE = """
    CASE {src}
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

_DESIRED_CASE = """
    CASE
        WHEN {src} IN ('stopping', 'stopped', 'paused', 'cancelled', 'terminated')
            THEN 'stopped'
        ELSE 'running'
    END
"""

_HOST_ADMIN_CASE = """
    CASE
        WHEN {status} = 'disabled' THEN 'disabled'
        WHEN {status} = 'draining'
             OR COALESCE(({payload}->>'draining')::boolean, false) THEN 'draining'
        WHEN COALESCE(({payload}->>'admitted')::boolean, false) THEN 'admitted'
        ELSE 'pending'
    END
"""

_HOST_AVAIL_CASE = """
    CASE {status}
        WHEN 'active' THEN 'ready'
        WHEN 'dead'   THEN 'not_ready'
        ELSE 'unknown'
    END
"""


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    # ── CHECK expansion: hosts may now be pending admission ──────────
    op.execute(
        "ALTER TABLE hosts DROP CONSTRAINT IF EXISTS ck_hosts_administrative_state"
    )
    op.execute(
        """
        ALTER TABLE hosts ADD CONSTRAINT ck_hosts_administrative_state
        CHECK (administrative_state IN
               ('pending', 'admitted', 'draining', 'disabled'))
        """
    )

    # ── jobs projection trigger ──────────────────────────────────────
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION control_plane_project_job()
        RETURNS trigger AS $$
        BEGIN
            NEW.phase := {_PHASE_CASE.format(src="NEW.status")};
            NEW.desired_state := {_DESIRED_CASE.format(src="NEW.status")};
            NEW.effective_priority := COALESCE(NEW.priority, 0);
            IF NEW.status = 'queued' THEN
                NEW.queued_at := COALESCE(
                    to_timestamp(NEW.submitted_at),
                    NEW.queued_at,
                    clock_timestamp()
                );
            END IF;
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql
        """
    )
    op.execute("DROP TRIGGER IF EXISTS trg_jobs_control_plane_projection ON jobs")
    op.execute(
        """
        CREATE TRIGGER trg_jobs_control_plane_projection
        BEFORE INSERT OR UPDATE ON jobs
        FOR EACH ROW EXECUTE FUNCTION control_plane_project_job()
        """
    )

    # ── hosts projection trigger ─────────────────────────────────────
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION control_plane_project_host()
        RETURNS trigger AS $$
        BEGIN
            NEW.administrative_state :=
                {_HOST_ADMIN_CASE.format(status="NEW.status", payload="NEW.payload")};
            NEW.availability_state :=
                {_HOST_AVAIL_CASE.format(status="NEW.status")};
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql
        """
    )
    op.execute("DROP TRIGGER IF EXISTS trg_hosts_control_plane_projection ON hosts")
    op.execute(
        """
        CREATE TRIGGER trg_hosts_control_plane_projection
        BEFORE INSERT OR UPDATE ON hosts
        FOR EACH ROW EXECUTE FUNCTION control_plane_project_host()
        """
    )

    # ── drift backfill: rows written since 054's one-time backfill ───
    op.execute(
        f"""
        UPDATE jobs SET
            phase = {_PHASE_CASE.format(src="status")},
            desired_state = {_DESIRED_CASE.format(src="status")},
            effective_priority = COALESCE(priority, 0),
            queued_at = CASE
                WHEN status = 'queued'
                    THEN COALESCE(queued_at, to_timestamp(submitted_at), now())
                ELSE queued_at
            END
        WHERE phase IS NULL
           OR desired_state IS NULL
           OR effective_priority IS DISTINCT FROM COALESCE(priority, 0)
           OR (status = 'queued' AND queued_at IS NULL)
        """
    )
    # Hosts: re-derive ALL rows — 054's status-only rule wrongly admitted
    # hosts whose payload admitted flag is false; the fleet is small.
    op.execute(
        f"""
        UPDATE hosts SET
            administrative_state =
                {_HOST_ADMIN_CASE.format(status="status", payload="payload")},
            availability_state = {_HOST_AVAIL_CASE.format(status="status")}
        """
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")
    op.execute("DROP TRIGGER IF EXISTS trg_jobs_control_plane_projection ON jobs")
    op.execute("DROP FUNCTION IF EXISTS control_plane_project_job()")
    op.execute("DROP TRIGGER IF EXISTS trg_hosts_control_plane_projection ON hosts")
    op.execute("DROP FUNCTION IF EXISTS control_plane_project_host()")
    # Restore 054's loose admitted rule so its CHECK can be reinstated.
    op.execute(
        """
        UPDATE hosts SET administrative_state = CASE status
            WHEN 'disabled' THEN 'disabled'
            WHEN 'draining' THEN 'draining'
            ELSE 'admitted'
        END
        WHERE administrative_state = 'pending'
        """
    )
    op.execute(
        "ALTER TABLE hosts DROP CONSTRAINT IF EXISTS ck_hosts_administrative_state"
    )
    op.execute(
        """
        ALTER TABLE hosts ADD CONSTRAINT ck_hosts_administrative_state
        CHECK (administrative_state IN ('admitted', 'draining', 'disabled'))
        """
    )
