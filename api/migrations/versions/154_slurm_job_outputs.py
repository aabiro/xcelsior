"""Cached tails of Slurm job output, so reading output never reaches a cluster.

Plan item 1.7 (hpc/IMPLEMENTATION_PLAN.md §5a): output is fetched at most once
per poll interval per job, however many readers there are. Readers (the v1
API, agents through MCP) read only this table. A read that finds the copy
stale records `requested_at`; the reconciler, the only code that contacts a
cluster, fetches requested tails on its next pass over that cluster.

One row per job and stream (`out`, `err`):

- `content` is the tail of the file, at most the adapter's read size;
  `truncated` says earlier output was cut.
- `requested_at`: a reader wanted a fresher copy than the last fetch.
- `attempted_at` / `fetched_at`: the last fetch attempt, and the last one
  that succeeded. `error_code` and `error` describe a failed attempt.
- `final`: fetched after the job settled. The file can't grow any more, so
  the row is never fetched again.

Rows follow their mapping (`ON DELETE CASCADE`). Mappings are kept, so in
practice the tail stays as long as the job's record does.

`downgrade` drops the table.
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "154"
down_revision = "153"
branch_labels = None
depends_on = None


def _create(conn):
    conn.execute(text("""
            CREATE TABLE IF NOT EXISTS slurm_job_outputs (
                xcelsior_job_id TEXT NOT NULL
                    REFERENCES slurm_job_mappings (xcelsior_job_id) ON DELETE CASCADE,
                stream          TEXT NOT NULL,
                slurm_job_id    TEXT NOT NULL,
                content         TEXT NOT NULL DEFAULT '',
                truncated       BOOLEAN NOT NULL DEFAULT FALSE,
                requested_at    TIMESTAMPTZ,
                attempted_at    TIMESTAMPTZ,
                fetched_at      TIMESTAMPTZ,
                final           BOOLEAN NOT NULL DEFAULT FALSE,
                error_code      TEXT,
                error           TEXT,
                PRIMARY KEY (xcelsior_job_id, stream),
                CONSTRAINT ck_slurm_output_stream CHECK (stream IN ('out', 'err'))
            )
            """))


def upgrade():
    apply_in_own_transactions([("slurm_job_outputs", _create)], tables=("slurm_job_outputs",))


def downgrade():
    apply_in_own_transactions(
        [
            (
                "slurm_job_outputs",
                lambda conn: conn.execute(text("DROP TABLE IF EXISTS slurm_job_outputs")),
            )
        ],
        tables=("slurm_job_outputs",),
    )
