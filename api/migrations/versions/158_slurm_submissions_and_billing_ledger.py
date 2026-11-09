"""Slurm: the submit log, the per-run billing ledger and its drift report.

Phase 2 of the Slurm blueprint (hpc/IMPLEMENTATION_PLAN.md §5a, rows 2.2 to
2.4; decision D8).

**`slurm_submissions`: one row per submission intent, written before any
`sbatch`.** The scheduler creates it (``pending``) in the transaction that
creates the attempt; the submitter moves it to ``submitting`` right before
``sbatch`` and to ``submitted`` once the mapping is registered. A timeout or
a dropped connection leaves ``unknown``: Slurm may or may not hold the job,
so the submission is resolved by name (``xcl-<submission_name>`` in squeue,
then a bounded sacct) on later passes, twice at least one poll interval
apart, before it is given up as ``lost``. A request sbatch refuses is
``rejected`` and never retried. Without this row a crash between ``sbatch``
and the mapping insert left a job on the cluster that nothing in Xcelsior
knew about; with it, the job is found by its name and adopted.

**`slurm_billing_segments`: one row per accounting record of a settled
job.** ``sacct -D -X`` returns one record per run; a requeued or preempted
job has several under one Slurm id. Each is kept with its raw fields
(``ElapsedRaw``, ``Suspended``, ``TimelimitRaw``, ``AllocTRES``) and the
seconds it bills under the written rules, with the rule's name, so a bill
can be explained record by record. ``shadow_cost_micros`` is what the
cluster's rate card would charge; it is charged only once ``charged`` is set
by settlement through a usage meter (behind the `slurm_placement` switch).

**`slurm_billing_drift`: one row per cluster per day.** The ledger's GPU
seconds against what the cluster's own accounting says for that day, so a
week of shadow mode can show the two agree.

**`slurm_clusters`** gains ``rate_card`` (what a GPU hour on the cluster
costs, in micros, and the GPU model it bills as) and ``suspend_holds_gres``
(D8: whether a suspended job keeps its GPUs, so whether its suspended time
bills). Both default to what an owned cluster does.

Each unit is idempotent and runs in its own short-lock transaction
(migrations/lock_safe.py). ``downgrade`` drops the three tables and the two
columns; the ledger is derived from accounting and can be rebuilt.
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "158"
down_revision = "157"
branch_labels = None
depends_on = None

SUBMISSION_STATES = ("pending", "submitting", "submitted", "unknown", "rejected", "lost")

# Written out in full, not built from a list of names: no identifier is
# interpolated into DDL (tests/test_sql_injection_guard.py).
ADD_CLUSTER_COLUMNS = (
    "ALTER TABLE slurm_clusters ADD COLUMN IF NOT EXISTS rate_card JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE slurm_clusters ADD COLUMN IF NOT EXISTS suspend_holds_gres BOOLEAN NOT NULL DEFAULT TRUE",
)
DROP_CLUSTER_COLUMNS = (
    "ALTER TABLE slurm_clusters DROP COLUMN IF EXISTS rate_card",
    "ALTER TABLE slurm_clusters DROP COLUMN IF EXISTS suspend_holds_gres",
)


def _cluster_columns(conn):
    for statement in ADD_CLUSTER_COLUMNS:
        conn.execute(text(statement))


def _submissions(conn):
    states = ", ".join(f"'{s}'" for s in SUBMISSION_STATES)
    conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS slurm_submissions (
                submission_name    TEXT PRIMARY KEY,
                cluster_id         TEXT NOT NULL REFERENCES slurm_clusters (cluster_id),
                xcelsior_job_id    TEXT NOT NULL,
                attempt_id         UUID NOT NULL UNIQUE
                                   REFERENCES job_attempts (attempt_id) ON DELETE CASCADE,
                tenant_id          TEXT,
                state              TEXT NOT NULL DEFAULT 'pending',
                slurm_job_id       TEXT,
                script_sha256      TEXT,
                sbatch_calls       INTEGER NOT NULL DEFAULT 0,
                checks             INTEGER NOT NULL DEFAULT 0,
                error_code         TEXT,
                error              TEXT,
                created_at         TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                submit_started_at  TIMESTAMPTZ,
                submitted_at       TIMESTAMPTZ,
                last_check_at      TIMESTAMPTZ,
                resolved_at        TIMESTAMPTZ,
                CONSTRAINT ck_slurm_submission_state CHECK (state IN ({states})),
                CONSTRAINT ck_slurm_submission_name
                    CHECK (submission_name ~ '^[A-Za-z0-9][A-Za-z0-9_.-]{{0,63}}$')
            )
            """))
    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_slurm_submissions_due "
            "ON slurm_submissions (cluster_id, state) "
            "WHERE state IN ('pending', 'submitting', 'unknown')"
        )
    )
    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_slurm_submissions_job "
            "ON slurm_submissions (xcelsior_job_id)"
        )
    )


def _segments(conn):
    conn.execute(text("""
            CREATE TABLE IF NOT EXISTS slurm_billing_segments (
                segment_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                mapping_id       UUID NOT NULL
                                 REFERENCES slurm_job_mappings (mapping_id) ON DELETE CASCADE,
                run_index        INTEGER NOT NULL,
                cluster_id       TEXT NOT NULL,
                slurm_job_id     TEXT NOT NULL,
                xcelsior_job_id  TEXT NOT NULL,
                attempt_id       UUID,
                tenant_id        TEXT,
                slurm_state      TEXT NOT NULL,
                submitted_at     TIMESTAMPTZ,
                started_at       TIMESTAMPTZ,
                ended_at         TIMESTAMPTZ,
                elapsed_sec      INTEGER NOT NULL,
                suspended_sec    INTEGER NOT NULL DEFAULT 0,
                time_limit_sec   INTEGER,
                gpu_count        INTEGER NOT NULL DEFAULT 0,
                cpu_count        INTEGER NOT NULL DEFAULT 0,
                node_count       INTEGER NOT NULL DEFAULT 0,
                billing_tres     INTEGER,
                alloc_tres       TEXT,
                billable_sec     INTEGER NOT NULL,
                billing_rule     TEXT NOT NULL,
                gpu_seconds      BIGINT NOT NULL,
                rate_micros_per_gpu_hour BIGINT,
                shadow_cost_micros       BIGINT,
                charged          BOOLEAN NOT NULL DEFAULT FALSE,
                meter_id         TEXT,
                raw              JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at       TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                CONSTRAINT uq_slurm_segment_run UNIQUE (mapping_id, run_index),
                CONSTRAINT ck_slurm_segment_seconds
                    CHECK (elapsed_sec >= 0 AND suspended_sec >= 0 AND billable_sec >= 0
                           AND gpu_seconds >= 0)
            )
            """))
    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_slurm_segments_cluster_day "
            "ON slurm_billing_segments (cluster_id, ended_at)"
        )
    )
    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_slurm_segments_attempt "
            "ON slurm_billing_segments (attempt_id) WHERE attempt_id IS NOT NULL"
        )
    )


def _drift(conn):
    conn.execute(text("""
            CREATE TABLE IF NOT EXISTS slurm_billing_drift (
                cluster_id          TEXT NOT NULL REFERENCES slurm_clusters (cluster_id)
                                    ON DELETE CASCADE,
                day                 DATE NOT NULL,
                sacct_gpu_seconds   BIGINT NOT NULL,
                ledger_gpu_seconds  BIGINT NOT NULL,
                sacct_records       INTEGER NOT NULL,
                ledger_records      INTEGER NOT NULL,
                drift_pct           NUMERIC(9, 4),
                unmatched           JSONB NOT NULL DEFAULT '{}'::jsonb,
                checked_at          TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                PRIMARY KEY (cluster_id, day)
            )
            """))


def _drop_tables(conn):
    conn.execute(text("DROP TABLE IF EXISTS slurm_billing_drift"))
    conn.execute(text("DROP TABLE IF EXISTS slurm_billing_segments"))
    conn.execute(text("DROP TABLE IF EXISTS slurm_submissions"))


def _drop_cluster_columns(conn):
    for statement in DROP_CLUSTER_COLUMNS:
        conn.execute(text(statement))


def upgrade():
    apply_in_own_transactions(
        [
            ("slurm_clusters rate card and suspend policy", _cluster_columns),
            ("slurm_submissions", _submissions),
            ("slurm_billing_segments", _segments),
            ("slurm_billing_drift", _drift),
        ],
        tables=(
            "slurm_clusters",
            "slurm_submissions",
            "slurm_billing_segments",
            "slurm_billing_drift",
        ),
    )


def downgrade():
    apply_in_own_transactions(
        [
            ("slurm Phase 2 tables", _drop_tables),
            ("slurm_clusters rate card and suspend policy", _drop_cluster_columns),
        ],
        tables=(
            "slurm_clusters",
            "slurm_submissions",
            "slurm_billing_segments",
            "slurm_billing_drift",
        ),
    )
