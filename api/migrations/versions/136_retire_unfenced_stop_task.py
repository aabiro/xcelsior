"""Archive the obsolete unfenced stop task after its maintenance writer handoff."""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "136"
down_revision = "135"
branch_labels = None
depends_on = None

TASK = "reconcile_paused_stopped"


def _retire(conn):
    conn.execute(text("LOCK TABLE scheduled_tasks IN SHARE ROW EXCLUSIVE MODE"))
    active = conn.execute(
        text("""SELECT 1 FROM scheduled_tasks
        WHERE task_name=:task AND claim_owner IS NOT NULL
          AND claim_expires_at>clock_timestamp()"""),
        {"task": TASK},
    ).first()
    if active:
        raise RuntimeError("obsolete stop task still claimed; quiesce its maintenance writer")
    conn.execute(text("""CREATE TABLE IF NOT EXISTS retired_control_task_archive (
        task_name TEXT PRIMARY KEY CHECK(task_name='reconcile_paused_stopped'),
        record JSONB NOT NULL,
        record_sha256 TEXT NOT NULL CHECK
            (record_sha256=encode(sha256(convert_to(record::text,'UTF8')),'hex')),
        archived_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        source_revision TEXT NOT NULL DEFAULT '136' CHECK(source_revision='136')
    )"""))
    conn.execute(
        text("""INSERT INTO retired_control_task_archive(task_name,record,record_sha256)
        SELECT task_name,to_jsonb(t),encode(sha256(convert_to(to_jsonb(t)::text,'UTF8')),'hex')
          FROM scheduled_tasks t WHERE task_name=:task
        ON CONFLICT(task_name) DO NOTHING"""),
        {"task": TASK},
    )
    mismatch = conn.execute(
        text("""SELECT 1 FROM scheduled_tasks t
        LEFT JOIN retired_control_task_archive a USING(task_name)
        WHERE t.task_name=:task AND a.record IS DISTINCT FROM to_jsonb(t)"""),
        {"task": TASK},
    ).first()
    if mismatch:
        raise RuntimeError("obsolete stop task archive mismatch; retirement refused")
    conn.execute(text("DELETE FROM scheduled_tasks WHERE task_name=:task"), {"task": TASK})
    conn.execute(text("""DO $$ BEGIN
        IF NOT EXISTS(SELECT 1 FROM pg_constraint WHERE conrelid='scheduled_tasks'::regclass
            AND conname='scheduled_tasks_no_unfenced_stop') THEN
            ALTER TABLE scheduled_tasks ADD CONSTRAINT scheduled_tasks_no_unfenced_stop
                CHECK(task_name<>'reconcile_paused_stopped');
        END IF;
    END $$"""))


def upgrade():
    apply_in_own_transactions(
        [("retire unfenced stop task", _retire)],
        tables=["scheduled_tasks", "retired_control_task_archive"],
    )


def _restore(conn):
    conn.execute(text("LOCK TABLE scheduled_tasks IN SHARE ROW EXCLUSIVE MODE"))
    conn.execute(
        text(
            "ALTER TABLE scheduled_tasks DROP CONSTRAINT IF EXISTS scheduled_tasks_no_unfenced_stop"
        )
    )
    conn.execute(text("""INSERT INTO scheduled_tasks
        SELECT (jsonb_populate_record(NULL::scheduled_tasks,record)).*
          FROM retired_control_task_archive ON CONFLICT(task_name) DO NOTHING"""))


def downgrade():
    apply_in_own_transactions(
        [("restore retired stop task evidence", _restore)],
        tables=["scheduled_tasks", "retired_control_task_archive"],
    )
