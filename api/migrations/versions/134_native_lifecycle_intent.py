"""Move recorded controller intent out of observation JSONB authority.

Apply with lifecycle writers quiesced, then hand every writer to the native
column runtime before reopening. No intent is inferred from job status.
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "134"
down_revision = "133"
branch_labels = None
depends_on = None

REQUEST_FUNCTION = """CREATE OR REPLACE FUNCTION request_job_reconciliation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP='DELETE' THEN
        DELETE FROM reconciliation_queue WHERE resource_type='job' AND resource_id=OLD.job_id;
        RETURN OLD;
    END IF;
    IF TG_OP='UPDATE' THEN
        IF ROW(NEW.status,NEW.desired_state,NEW.phase,NEW.host_id,
               NEW.active_attempt_id,NEW.generation,NEW.lifecycle_intent) IS NOT DISTINCT FROM
           ROW(OLD.status,OLD.desired_state,OLD.phase,OLD.host_id,
               OLD.active_attempt_id,OLD.generation,OLD.lifecycle_intent) THEN
            RETURN NEW;
        END IF;
    END IF;
    INSERT INTO reconciliation_queue(resource_type,resource_id,reason,requested_by)
    VALUES('job',NEW.job_id,'job_changed','jobs_trigger')
    ON CONFLICT(resource_type,resource_id) DO UPDATE
    SET due_at=LEAST(reconciliation_queue.due_at,clock_timestamp()),
        reason=EXCLUDED.reason,updated_at=clock_timestamp();
    RETURN NEW;
END;
$$"""


def _expand(conn):
    # Lock freezes the source evidence and queue trigger while authority moves.
    conn.execute(text("LOCK TABLE jobs IN ACCESS EXCLUSIVE MODE"))
    conn.execute(text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS lifecycle_intent TEXT"))
    conn.execute(text("""DO $$ BEGIN
        IF EXISTS(SELECT 1 FROM jobs WHERE payload?'lifecycle_intent'
            AND payload->'lifecycle_intent'<>'null'::jsonb
            AND (jsonb_typeof(payload->'lifecycle_intent')<>'string'
                 OR payload->>'lifecycle_intent' NOT IN ('stop','cancel','terminate','restart')))
        THEN RAISE EXCEPTION 'unqualified recorded lifecycle intent; review required'; END IF;
        IF EXISTS(SELECT 1 FROM jobs WHERE lifecycle_intent IS NOT NULL
            AND lifecycle_intent IS DISTINCT FROM payload->>'lifecycle_intent')
        THEN RAISE EXCEPTION 'native and recorded lifecycle intent disagree'; END IF;
    END $$"""))
    conn.execute(text("""UPDATE jobs SET lifecycle_intent=payload->>'lifecycle_intent'
        WHERE lifecycle_intent IS NULL AND payload->>'lifecycle_intent' IS NOT NULL"""))
    conn.execute(text("ALTER TABLE jobs DROP CONSTRAINT IF EXISTS ck_jobs_lifecycle_intent"))
    conn.execute(text("""ALTER TABLE jobs ADD CONSTRAINT ck_jobs_lifecycle_intent CHECK
        (lifecycle_intent IN ('stop','cancel','terminate','restart')) NOT VALID"""))
    conn.execute(text("ALTER TABLE jobs VALIDATE CONSTRAINT ck_jobs_lifecycle_intent"))
    conn.execute(text(REQUEST_FUNCTION))


def upgrade():
    apply_in_own_transactions(
        [("native lifecycle intent", _expand)], tables=["jobs", "reconciliation_queue"]
    )


def _restore(conn):
    conn.execute(text("LOCK TABLE jobs IN ACCESS EXCLUSIVE MODE"))
    # A rollback to old readers retains the latest controller intent, including
    # consumed NULL. It never revives a stale JSONB key.
    conn.execute(text("""UPDATE jobs SET payload=(payload-'lifecycle_intent') || CASE
        WHEN lifecycle_intent IS NULL THEN '{}'::jsonb
        ELSE jsonb_build_object('lifecycle_intent',lifecycle_intent) END"""))
    conn.execute(
        text(
            REQUEST_FUNCTION.replace(
                "NEW.lifecycle_intent", "NEW.payload->>'lifecycle_intent'"
            ).replace("OLD.lifecycle_intent", "OLD.payload->>'lifecycle_intent'")
        )
    )
    conn.execute(text("ALTER TABLE jobs DROP COLUMN lifecycle_intent"))


def downgrade():
    apply_in_own_transactions(
        [("restore recorded lifecycle intent", _restore)], tables=["jobs", "reconciliation_queue"]
    )
