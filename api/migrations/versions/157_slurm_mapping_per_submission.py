"""One Slurm mapping per submission, so each attempt of a job keeps its own.

Phase 2.1 of the Slurm blueprint (hpc/IMPLEMENTATION_PLAN.md §5a, decision
D7). `slurm_job_mappings` was keyed by `xcelsior_job_id`, so a job could map
to one Slurm job, ever. When the scheduler retries a job, the retry is a new
attempt and a new Slurm job; with one row per job, registering it could only
overwrite the first, and `register_slurm_job` did exactly that (`ON CONFLICT
(xcelsior_job_id) DO UPDATE SET slurm_job_id`), losing the first job's
record and the run it would bill.

So a row is now one submission:

- `mapping_id` is the primary key.
- `submission_name` is what the submission is called in Slurm
  (`xcl-<submission_name>`) and in its output file names, and what the
  reconciler matches a job's name against. Existing rows keep their job id
  as their name, so every name already submitted still matches; a scheduler
  attempt is named `<job id>-a<attempt number>`.
- `xcelsior_job_id` is the job, no longer unique. A job has at most one
  *open* mapping (`uq_slurm_open_job`), as it has at most one active attempt
  (055's `uq_job_one_active_attempt`).

`slurm_job_outputs` follows its submission: keyed by `(mapping_id, stream)`
and cascading from it, so two attempts' output never share a row.

A `BEFORE INSERT` trigger fills `submission_name` from `xcelsior_job_id`
when a writer leaves it out, as 156's code does, so an insert from the
release still serving during a blue-green deploy is not refused. That
release's `ON CONFLICT (xcelsior_job_id)` has no unique index to name any
more and fails for the length of the deploy; no deployment submits Slurm
jobs yet (no cluster is registered in production), so nothing is lost.

Each unit is idempotent and runs in its own short-lock transaction
(migrations/lock_safe.py); the key swap checks what it finds before acting.

`downgrade` refuses while any job has more than one mapping, since the old
key can't hold them, and otherwise restores 156's keys.
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "157"
down_revision = "156"
branch_labels = None
depends_on = None


def _primary_key_columns(conn, table: str) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            text(
                "SELECT a.attname FROM pg_index i "
                "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
                "WHERE i.indrelid = CAST(:t AS regclass) AND i.indisprimary ORDER BY a.attnum"
            ),
            {"t": table},
        ).fetchall()
    ]


def _mapping_columns(conn):
    conn.execute(text("ALTER TABLE slurm_job_mappings ADD COLUMN IF NOT EXISTS mapping_id UUID"))
    conn.execute(
        text("ALTER TABLE slurm_job_mappings ADD COLUMN IF NOT EXISTS submission_name TEXT")
    )
    conn.execute(
        text(
            "UPDATE slurm_job_mappings SET mapping_id = gen_random_uuid() WHERE mapping_id IS NULL"
        )
    )
    conn.execute(
        text(
            "UPDATE slurm_job_mappings SET submission_name = xcelsior_job_id "
            "WHERE submission_name IS NULL"
        )
    )
    conn.execute(
        text("ALTER TABLE slurm_job_mappings ALTER COLUMN mapping_id SET DEFAULT gen_random_uuid()")
    )
    conn.execute(text("ALTER TABLE slurm_job_mappings ALTER COLUMN mapping_id SET NOT NULL"))
    conn.execute(text("ALTER TABLE slurm_job_mappings ALTER COLUMN submission_name SET NOT NULL"))


def _submission_name_default(conn):
    conn.execute(text("""
            CREATE OR REPLACE FUNCTION slurm_mapping_submission_name() RETURNS trigger AS $$
            BEGIN
                NEW.submission_name := COALESCE(NEW.submission_name, NEW.xcelsior_job_id);
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
            """))
    conn.execute(
        text("DROP TRIGGER IF EXISTS trg_slurm_mapping_submission_name ON slurm_job_mappings")
    )
    conn.execute(
        text(
            "CREATE TRIGGER trg_slurm_mapping_submission_name BEFORE INSERT ON slurm_job_mappings "
            "FOR EACH ROW EXECUTE FUNCTION slurm_mapping_submission_name()"
        )
    )


def _mapping_indexes(conn):
    conn.execute(
        text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_slurm_mapping_id "
            "ON slurm_job_mappings (mapping_id)"
        )
    )
    conn.execute(
        text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_slurm_submission_name "
            "ON slurm_job_mappings (submission_name)"
        )
    )
    conn.execute(
        text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_slurm_open_job "
            "ON slurm_job_mappings (xcelsior_job_id) WHERE terminal_at IS NULL"
        )
    )
    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_slurm_mappings_job "
            "ON slurm_job_mappings (xcelsior_job_id)"
        )
    )


def _output_column(conn):
    conn.execute(text("ALTER TABLE slurm_job_outputs ADD COLUMN IF NOT EXISTS mapping_id UUID"))
    columns = {
        r[0]
        for r in conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'slurm_job_outputs'"
            )
        ).fetchall()
    }
    if "xcelsior_job_id" in columns:
        conn.execute(
            text(
                "UPDATE slurm_job_outputs o SET mapping_id = m.mapping_id "
                "FROM slurm_job_mappings m "
                "WHERE m.xcelsior_job_id = o.xcelsior_job_id AND o.mapping_id IS NULL"
            )
        )
    conn.execute(text("ALTER TABLE slurm_job_outputs ALTER COLUMN mapping_id SET NOT NULL"))


def _swap_keys(conn):
    if _primary_key_columns(conn, "slurm_job_mappings") != ["mapping_id"]:
        conn.execute(
            text(
                "ALTER TABLE slurm_job_outputs "
                "DROP CONSTRAINT IF EXISTS slurm_job_outputs_xcelsior_job_id_fkey"
            )
        )
        conn.execute(
            text("ALTER TABLE slurm_job_outputs DROP CONSTRAINT IF EXISTS slurm_job_outputs_pkey")
        )
        conn.execute(
            text("ALTER TABLE slurm_job_mappings DROP CONSTRAINT IF EXISTS slurm_job_mappings_pkey")
        )
        conn.execute(
            text(
                "ALTER TABLE slurm_job_mappings ADD CONSTRAINT slurm_job_mappings_pkey "
                "PRIMARY KEY USING INDEX uq_slurm_mapping_id"
            )
        )
    if _primary_key_columns(conn, "slurm_job_outputs") != ["mapping_id", "stream"]:
        conn.execute(
            text("ALTER TABLE slurm_job_outputs DROP CONSTRAINT IF EXISTS slurm_job_outputs_pkey")
        )
        conn.execute(
            text(
                "ALTER TABLE slurm_job_outputs ADD CONSTRAINT slurm_job_outputs_pkey "
                "PRIMARY KEY (mapping_id, stream)"
            )
        )
    conn.execute(
        text(
            "ALTER TABLE slurm_job_outputs "
            "DROP CONSTRAINT IF EXISTS slurm_job_outputs_mapping_id_fkey"
        )
    )
    conn.execute(
        text(
            "ALTER TABLE slurm_job_outputs ADD CONSTRAINT slurm_job_outputs_mapping_id_fkey "
            "FOREIGN KEY (mapping_id) REFERENCES slurm_job_mappings (mapping_id) "
            "ON DELETE CASCADE"
        )
    )
    conn.execute(text("ALTER TABLE slurm_job_outputs DROP COLUMN IF EXISTS xcelsior_job_id"))


def _refuse_jobs_with_several(conn):
    several = conn.execute(
        text(
            "SELECT xcelsior_job_id, count(*) FROM slurm_job_mappings "
            "GROUP BY xcelsior_job_id HAVING count(*) > 1 LIMIT 10"
        )
    ).fetchall()
    if several:
        raise RuntimeError(
            "migration 157 downgrade: these jobs have more than one Slurm mapping, which "
            f"156's key on xcelsior_job_id can't hold: {[tuple(r) for r in several]}"
        )


def _restore_keys(conn):
    conn.execute(
        text("ALTER TABLE slurm_job_outputs ADD COLUMN IF NOT EXISTS xcelsior_job_id TEXT")
    )
    conn.execute(
        text(
            "UPDATE slurm_job_outputs o SET xcelsior_job_id = m.xcelsior_job_id "
            "FROM slurm_job_mappings m WHERE m.mapping_id = o.mapping_id"
        )
    )
    conn.execute(
        text(
            "ALTER TABLE slurm_job_outputs "
            "DROP CONSTRAINT IF EXISTS slurm_job_outputs_mapping_id_fkey"
        )
    )
    conn.execute(
        text("ALTER TABLE slurm_job_outputs DROP CONSTRAINT IF EXISTS slurm_job_outputs_pkey")
    )
    if _primary_key_columns(conn, "slurm_job_mappings") != ["xcelsior_job_id"]:
        conn.execute(
            text("ALTER TABLE slurm_job_mappings DROP CONSTRAINT IF EXISTS slurm_job_mappings_pkey")
        )
        conn.execute(
            text(
                "ALTER TABLE slurm_job_mappings ADD CONSTRAINT slurm_job_mappings_pkey "
                "PRIMARY KEY (xcelsior_job_id)"
            )
        )
    conn.execute(text("ALTER TABLE slurm_job_outputs ALTER COLUMN xcelsior_job_id SET NOT NULL"))
    conn.execute(
        text(
            "ALTER TABLE slurm_job_outputs ADD CONSTRAINT slurm_job_outputs_pkey "
            "PRIMARY KEY (xcelsior_job_id, stream)"
        )
    )
    conn.execute(
        text(
            "ALTER TABLE slurm_job_outputs ADD CONSTRAINT slurm_job_outputs_xcelsior_job_id_fkey "
            "FOREIGN KEY (xcelsior_job_id) REFERENCES slurm_job_mappings (xcelsior_job_id) "
            "ON DELETE CASCADE"
        )
    )
    conn.execute(text("ALTER TABLE slurm_job_outputs DROP COLUMN IF EXISTS mapping_id"))


def _drop_mapping_additions(conn):
    conn.execute(
        text("DROP TRIGGER IF EXISTS trg_slurm_mapping_submission_name ON slurm_job_mappings")
    )
    conn.execute(text("DROP FUNCTION IF EXISTS slurm_mapping_submission_name()"))
    conn.execute(text("DROP INDEX IF EXISTS uq_slurm_open_job"))
    conn.execute(text("DROP INDEX IF EXISTS idx_slurm_mappings_job"))
    conn.execute(text("DROP INDEX IF EXISTS uq_slurm_submission_name"))
    conn.execute(text("ALTER TABLE slurm_job_mappings DROP COLUMN IF EXISTS submission_name"))
    conn.execute(text("ALTER TABLE slurm_job_mappings DROP COLUMN IF EXISTS mapping_id"))


def upgrade():
    apply_in_own_transactions(
        [
            ("slurm_job_mappings mapping_id and submission_name", _mapping_columns),
            ("slurm_job_mappings submission_name default", _submission_name_default),
            ("slurm_job_mappings per-submission indexes", _mapping_indexes),
            ("slurm_job_outputs mapping_id", _output_column),
            ("slurm keys move to the submission", _swap_keys),
        ],
        tables=("slurm_job_mappings", "slurm_job_outputs"),
    )


def downgrade():
    apply_in_own_transactions(
        [
            ("slurm_job_mappings one per job", _refuse_jobs_with_several),
            ("slurm keys back to the job", _restore_keys),
            ("slurm_job_mappings per-submission additions", _drop_mapping_additions),
        ],
        tables=("slurm_job_mappings", "slurm_job_outputs"),
    )
