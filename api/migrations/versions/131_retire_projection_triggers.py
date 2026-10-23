"""Retire lifecycle projection triggers after the native writer handoff.

Per-table transactions avoid accumulating locks across the two hot tables.
Each unit is idempotent: an interrupted migration may leave the host unit
committed at ledger 130, and retry safely finishes the job unit. Deploy the
qualified native API/scheduler/maintenance writers before this contraction.
"""

from sqlalchemy import text

from migrations.lock_safe import apply_in_own_transactions

revision = "131"
down_revision = "130"
branch_labels = None
depends_on = None


def _contract_hosts(conn):
    conn.execute(text("SET LOCAL statement_timeout='5min'"))
    conn.execute(text("ALTER TABLE hosts ALTER COLUMN administrative_state SET DEFAULT 'pending'"))
    conn.execute(text("ALTER TABLE hosts DROP CONSTRAINT IF EXISTS ck_hosts_admission_authority"))
    conn.execute(text("""ALTER TABLE hosts ADD CONSTRAINT ck_hosts_admission_authority
        CHECK (administrative_state NOT IN ('admitted','draining')
               OR admission_state='admitted') NOT VALID"""))
    conn.execute(text("ALTER TABLE hosts VALIDATE CONSTRAINT ck_hosts_admission_authority"))
    conn.execute(text("DROP TRIGGER IF EXISTS trg_hosts_control_plane_projection ON hosts"))
    conn.execute(text("DROP FUNCTION IF EXISTS control_plane_project_host()"))


def _contract_jobs(conn):
    conn.execute(text("SET LOCAL statement_timeout='5min'"))
    conn.execute(text("DROP TRIGGER IF EXISTS trg_jobs_control_plane_projection ON jobs"))
    conn.execute(text("DROP FUNCTION IF EXISTS control_plane_project_job()"))


def _restore_hosts(conn):
    conn.execute(text("SET LOCAL statement_timeout='5min'"))
    conn.execute(text("ALTER TABLE hosts DROP CONSTRAINT IF EXISTS ck_hosts_admission_authority"))
    conn.execute(text("ALTER TABLE hosts ALTER COLUMN administrative_state SET DEFAULT 'admitted'"))
    conn.execute(text(_RESTORE_HOST_SQL))
    conn.execute(text("DROP TRIGGER IF EXISTS trg_hosts_control_plane_projection ON hosts"))
    conn.execute(text("""CREATE TRIGGER trg_hosts_control_plane_projection
        BEFORE INSERT OR UPDATE ON hosts FOR EACH ROW
        EXECUTE FUNCTION control_plane_project_host()"""))


def _restore_jobs(conn):
    conn.execute(text("SET LOCAL statement_timeout='5min'"))
    conn.execute(text(_RESTORE_JOB_SQL))
    conn.execute(text("DROP TRIGGER IF EXISTS trg_jobs_control_plane_projection ON jobs"))
    conn.execute(text("""CREATE TRIGGER trg_jobs_control_plane_projection
        BEFORE INSERT OR UPDATE OF status,priority,submitted_at ON jobs
        FOR EACH ROW EXECUTE FUNCTION control_plane_project_job()"""))


def upgrade():
    apply_in_own_transactions(
        [("host admission contract", _contract_hosts), ("retire job projection", _contract_jobs)],
        tables=["hosts", "jobs"],
    )


def downgrade():
    apply_in_own_transactions(
        [("restore job projection", _restore_jobs), ("restore host projection", _restore_hosts)],
        tables=["jobs", "hosts"],
    )


# Frozen 130/082 definitions: downgrade restores their exact semantics. Never
# import live application code or replay an older migration's data backfill.

_RESTORE_JOB_SQL = """CREATE OR REPLACE FUNCTION control_plane_project_job()
        RETURNS trigger AS $$ BEGIN
            IF TG_OP='INSERT' THEN
                NEW.phase :=\x20
    CASE NEW.status
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
;
                NEW.desired_state :=\x20
    CASE
        WHEN NEW.status IN ('stopping', 'stopped', 'paused', 'cancelled', 'terminated')
            THEN 'stopped'
        ELSE 'running'
    END
;
                NEW.effective_priority := COALESCE(NEW.priority,0);
            ELSE
                IF NEW.status IS DISTINCT FROM OLD.status THEN
                    NEW.phase :=\x20
    CASE NEW.status
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
;
                    NEW.desired_state :=\x20
    CASE
        WHEN NEW.status IN ('stopping', 'stopped', 'paused', 'cancelled', 'terminated')
            THEN 'stopped'
        ELSE 'running'
    END
;
                END IF;
                IF NEW.priority IS DISTINCT FROM OLD.priority THEN
                    NEW.effective_priority := COALESCE(NEW.priority,0);
                END IF;
            END IF;
            IF NEW.status IN ('queued','preempted') THEN
                IF TG_OP='INSERT' THEN
                    NEW.queued_at := COALESCE(to_timestamp(NEW.submitted_at),NEW.queued_at);
                ELSIF NEW.status IS DISTINCT FROM OLD.status
                   OR NEW.submitted_at IS DISTINCT FROM OLD.submitted_at THEN
                    NEW.queued_at := COALESCE(to_timestamp(NEW.submitted_at),NEW.queued_at);
                END IF;
            END IF;
            RETURN NEW;
        END $$ LANGUAGE plpgsql"""

_RESTORE_HOST_SQL = """
        CREATE OR REPLACE FUNCTION control_plane_project_host()
        RETURNS trigger AS $$
        DECLARE
            base_payload jsonb;
        BEGIN
            base_payload := CASE
                WHEN jsonb_typeof(NEW.payload) = 'object' THEN NEW.payload
                ELSE '{}'::jsonb
            END;

            base_payload := jsonb_set(
                base_payload,
                '{admitted}',
                to_jsonb(NEW.admission_state = 'admitted'),
                true
            );
            base_payload := jsonb_set(
                base_payload,
                '{admission_state}',
                to_jsonb(NEW.admission_state),
                true
            );
            base_payload := jsonb_set(
                base_payload,
                '{admission_version}',
                to_jsonb(NEW.admission_version),
                true
            );
            IF NEW.admission_decision_id IS NOT NULL THEN
                base_payload := jsonb_set(
                    base_payload,
                    '{admission_decision_id}',
                    to_jsonb(NEW.admission_decision_id::text),
                    true
                );
            END IF;
            NEW.payload := base_payload;

            NEW.administrative_state := CASE
                WHEN NEW.status = 'disabled' THEN 'disabled'
                WHEN NEW.admission_state = 'admitted'
                     AND NEW.status = 'draining' THEN 'draining'
                WHEN NEW.admission_state = 'admitted' THEN 'admitted'
                ELSE 'pending'
            END;
            NEW.availability_state := CASE NEW.status
                WHEN 'active' THEN 'ready'
                WHEN 'dead' THEN 'not_ready'
                ELSE 'unknown'
            END;
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql
        """
