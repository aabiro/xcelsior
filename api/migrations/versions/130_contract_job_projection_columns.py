"""Require native job projections and stop unrelated writes resetting queue state.

This contracts required columns and narrows the legacy projection bridge. Full
trigger removal still requires every status writer to maintain native columns.
"""

from alembic import op
from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "130"
down_revision = "129"
branch_labels = None
depends_on = None

_PHASE_CASE = "\n    CASE {src}\n        WHEN 'queued'      THEN 'pending'\n        WHEN 'preempted'   THEN 'pending'\n        WHEN 'assigned'    THEN 'scheduled'\n        WHEN 'leased'      THEN 'scheduled'\n        WHEN 'starting'    THEN 'starting'\n        WHEN 'restarting'  THEN 'starting'\n        WHEN 'running'     THEN 'running'\n        WHEN 'stopping'    THEN 'running'\n        WHEN 'completed'   THEN 'succeeded'\n        WHEN 'failed'      THEN 'failed'\n        WHEN 'cancelled'   THEN 'stopped'\n        WHEN 'stopped'     THEN 'stopped'\n        WHEN 'paused'      THEN 'stopped'\n        WHEN 'terminated'  THEN 'stopped'\n        ELSE NULL\n    END\n"
_DESIRED_CASE = "\n    CASE\n        WHEN {src} IN ('stopping', 'stopped', 'paused', 'cancelled', 'terminated')\n            THEN 'stopped'\n        ELSE 'running'\n    END\n"


def _contract_jobs(conn):
    conn.execute(text("SET LOCAL statement_timeout='5min'"))
    conn.execute(text(f"""CREATE OR REPLACE FUNCTION control_plane_project_job()
        RETURNS trigger AS $$ BEGIN
            IF TG_OP='INSERT' THEN
                NEW.phase := {_PHASE_CASE.format(src='NEW.status')};
                NEW.desired_state := {_DESIRED_CASE.format(src='NEW.status')};
                NEW.effective_priority := COALESCE(NEW.priority,0);
            ELSE
                IF NEW.status IS DISTINCT FROM OLD.status THEN
                    NEW.phase := {_PHASE_CASE.format(src='NEW.status')};
                    NEW.desired_state := {_DESIRED_CASE.format(src='NEW.status')};
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
        END $$ LANGUAGE plpgsql"""))
    conn.execute(text("DROP TRIGGER IF EXISTS trg_jobs_control_plane_projection ON jobs"))
    conn.execute(text("""CREATE TRIGGER trg_jobs_control_plane_projection
        BEFORE INSERT OR UPDATE OF status,priority,submitted_at ON jobs
        FOR EACH ROW EXECUTE FUNCTION control_plane_project_job()"""))
    # Existing projections win. Missing values can only come from recorded
    # status/submission facts; there is no made-up queue time or phase.
    conn.execute(text(f"""UPDATE jobs SET
        phase=COALESCE(phase,{_PHASE_CASE.format(src='status')}),
        desired_state=COALESCE(desired_state,{_DESIRED_CASE.format(src='status')}),
        queued_at=CASE WHEN COALESCE(phase,{_PHASE_CASE.format(src='status')})='pending'
            THEN COALESCE(queued_at,to_timestamp(submitted_at)) ELSE queued_at END
        WHERE phase IS NULL OR desired_state IS NULL OR (phase='pending' AND queued_at IS NULL)"""))
    conn.execute(text("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM jobs WHERE phase IS NULL OR desired_state IS NULL
                   OR (phase='pending' AND queued_at IS NULL)) THEN
            RAISE EXCEPTION 'job column contraction requires recorded status and submission evidence';
        END IF;
    END $$"""))
    for name, expression in [
        ("ck_jobs_phase_present", "phase IS NOT NULL"),
        ("ck_jobs_desired_present", "desired_state IS NOT NULL"),
        ("ck_jobs_pending_queued_at", "phase<>'pending' OR queued_at IS NOT NULL"),
    ]:
        # Per-table transaction makes a partial constraint unit impossible;
        # idempotence also covers an interrupted Alembic stamp update.
        conn.execute(text(f"ALTER TABLE jobs DROP CONSTRAINT IF EXISTS {name}"))
        conn.execute(text(f"ALTER TABLE jobs ADD CONSTRAINT {name} CHECK ({expression}) NOT VALID"))
        conn.execute(text(f"ALTER TABLE jobs VALIDATE CONSTRAINT {name}"))
    conn.execute(text("ALTER TABLE jobs ALTER COLUMN phase SET NOT NULL"))
    conn.execute(text("ALTER TABLE jobs ALTER COLUMN desired_state SET NOT NULL"))
    conn.execute(text("ALTER TABLE jobs DROP CONSTRAINT ck_jobs_phase_present"))
    conn.execute(text("ALTER TABLE jobs DROP CONSTRAINT ck_jobs_desired_present"))


def upgrade():
    apply_in_own_transactions([("job projection column contract", _contract_jobs)], tables=["jobs"])


def downgrade():
    op.execute("SET LOCAL lock_timeout='5s'")
    op.execute("SET LOCAL statement_timeout='5min'")
    op.execute("ALTER TABLE jobs ALTER COLUMN phase DROP NOT NULL")
    op.execute("ALTER TABLE jobs ALTER COLUMN desired_state DROP NOT NULL")
    op.execute("ALTER TABLE jobs DROP CONSTRAINT IF EXISTS ck_jobs_pending_queued_at")
    op.execute(f"""CREATE OR REPLACE FUNCTION control_plane_project_job()
        RETURNS trigger AS $$ BEGIN
            NEW.phase := {_PHASE_CASE.format(src='NEW.status')};
            NEW.desired_state := {_DESIRED_CASE.format(src='NEW.status')};
            NEW.effective_priority := COALESCE(NEW.priority,0);
            IF NEW.status='queued' THEN
                NEW.queued_at := COALESCE(to_timestamp(NEW.submitted_at),NEW.queued_at,clock_timestamp());
            END IF;
            RETURN NEW;
        END $$ LANGUAGE plpgsql""")
    op.execute("DROP TRIGGER IF EXISTS trg_jobs_control_plane_projection ON jobs")
    op.execute("""CREATE TRIGGER trg_jobs_control_plane_projection BEFORE INSERT OR UPDATE ON jobs
        FOR EACH ROW EXECUTE FUNCTION control_plane_project_job()""")
