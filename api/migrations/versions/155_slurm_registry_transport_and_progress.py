"""Slurm: connect from the registry, judge success, and track progress.

Plan items 1.2 and 1.7 (hpc/IMPLEMENTATION_PLAN.md §5a).

**`slurm_clusters`: where to connect, and how both ends are proven.**
`ssh_host`, `ssh_port` and `remote_user` say where; `known_hosts_line` is
the cluster's host key, pinned when the cluster is registered, so a changed
key is a refused connection, never a prompt; `credential_ref` names where the
private key lives (`env:NAME` or `file:/path`), never the key itself. A
cluster given an `ssh_host` must carry both, by CHECK. Rows without one keep
using their adapter profile's transport. `failure_counts` counts failed
passes, accounting calls and output fetches by stage and error code, so the
metrics read durable numbers rather than one process's counters.

**`slurm_job_mappings`:**

- `derived_exit_code`: Slurm's DerivedExitCode. A job succeeded only if it
  COMPLETED with ExitCode 0:0 *and* DerivedExitCode 0:0; the second catches
  a script that exits 0 after a failed step.
- `time_limit_seconds`: from squeue, for scheduler progress (elapsed time
  against the limit).
- `left_controller_at`: the first pass that found the job gone from the
  controller. While it is set and the mapping is open, the job is waiting on
  accounting; that is what the `ending_awaiting_accounting` alert watches.

**`slurm_job_outputs`** accepts a third stream, `progress`: the small JSON
file a job may rewrite to report its own progress, cached like its output.

Each unit is idempotent and runs in its own short-lock transaction
(migrations/lock_safe.py). `downgrade` drops the columns and restores 154's
stream constraint, deleting cached progress rows first.
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "155"
down_revision = "154"
branch_labels = None
depends_on = None

# Written out in full, not built from a list of names: no identifier is
# interpolated into DDL (tests/test_sql_injection_guard.py).
ADD_CLUSTER_COLUMNS = (
    "ALTER TABLE slurm_clusters ADD COLUMN IF NOT EXISTS ssh_host TEXT",
    "ALTER TABLE slurm_clusters ADD COLUMN IF NOT EXISTS ssh_port INTEGER",
    "ALTER TABLE slurm_clusters ADD COLUMN IF NOT EXISTS remote_user TEXT",
    "ALTER TABLE slurm_clusters ADD COLUMN IF NOT EXISTS known_hosts_line TEXT",
    "ALTER TABLE slurm_clusters ADD COLUMN IF NOT EXISTS credential_ref TEXT",
    "ALTER TABLE slurm_clusters ADD COLUMN IF NOT EXISTS failure_counts JSONB "
    "NOT NULL DEFAULT '{}'::jsonb",
)
DROP_CLUSTER_COLUMNS = (
    "ALTER TABLE slurm_clusters DROP COLUMN IF EXISTS ssh_host",
    "ALTER TABLE slurm_clusters DROP COLUMN IF EXISTS ssh_port",
    "ALTER TABLE slurm_clusters DROP COLUMN IF EXISTS remote_user",
    "ALTER TABLE slurm_clusters DROP COLUMN IF EXISTS known_hosts_line",
    "ALTER TABLE slurm_clusters DROP COLUMN IF EXISTS credential_ref",
    "ALTER TABLE slurm_clusters DROP COLUMN IF EXISTS failure_counts",
)
ADD_MAPPING_COLUMNS = (
    "ALTER TABLE slurm_job_mappings ADD COLUMN IF NOT EXISTS derived_exit_code TEXT",
    "ALTER TABLE slurm_job_mappings ADD COLUMN IF NOT EXISTS time_limit_seconds INTEGER",
    "ALTER TABLE slurm_job_mappings ADD COLUMN IF NOT EXISTS left_controller_at TIMESTAMPTZ",
)
DROP_MAPPING_COLUMNS = (
    "ALTER TABLE slurm_job_mappings DROP COLUMN IF EXISTS derived_exit_code",
    "ALTER TABLE slurm_job_mappings DROP COLUMN IF EXISTS time_limit_seconds",
    "ALTER TABLE slurm_job_mappings DROP COLUMN IF EXISTS left_controller_at",
)


def _cluster_columns(conn):
    for statement in ADD_CLUSTER_COLUMNS:
        conn.execute(text(statement))
    conn.execute(
        text("ALTER TABLE slurm_clusters DROP CONSTRAINT IF EXISTS ck_slurm_cluster_transport")
    )
    conn.execute(
        text(
            "ALTER TABLE slurm_clusters ADD CONSTRAINT ck_slurm_cluster_transport CHECK ("
            " (ssh_port IS NULL OR ssh_port BETWEEN 1 AND 65535)"
            " AND (credential_ref IS NULL OR credential_ref ~ '^(env:[A-Z][A-Z0-9_]*|file:/.+)$')"
            " AND (ssh_host IS NULL OR (known_hosts_line IS NOT NULL AND credential_ref IS NOT NULL))"
            ")"
        )
    )


def _mapping_columns(conn):
    for statement in ADD_MAPPING_COLUMNS:
        conn.execute(text(statement))


def _output_streams(streams):
    def run(conn):
        conn.execute(
            text("ALTER TABLE slurm_job_outputs DROP CONSTRAINT IF EXISTS ck_slurm_output_stream")
        )
        conn.execute(
            text(
                "ALTER TABLE slurm_job_outputs ADD CONSTRAINT ck_slurm_output_stream "
                f"CHECK (stream IN ({streams}))"
            )
        )

    return run


def _drop_cluster_columns(conn):
    conn.execute(
        text("ALTER TABLE slurm_clusters DROP CONSTRAINT IF EXISTS ck_slurm_cluster_transport")
    )
    for statement in DROP_CLUSTER_COLUMNS:
        conn.execute(text(statement))


def _drop_mapping_columns(conn):
    for statement in DROP_MAPPING_COLUMNS:
        conn.execute(text(statement))


def _drop_progress_rows(conn):
    conn.execute(text("DELETE FROM slurm_job_outputs WHERE stream = 'progress'"))


def upgrade():
    apply_in_own_transactions(
        [
            ("slurm_clusters transport columns", _cluster_columns),
            ("slurm_job_mappings verdict and progress columns", _mapping_columns),
            ("slurm_job_outputs progress stream", _output_streams("'out', 'err', 'progress'")),
        ],
        tables=("slurm_clusters", "slurm_job_mappings", "slurm_job_outputs"),
    )


def downgrade():
    apply_in_own_transactions(
        [
            ("slurm_job_outputs progress rows", _drop_progress_rows),
            ("slurm_job_outputs progress stream", _output_streams("'out', 'err'")),
            ("slurm_job_mappings verdict and progress columns", _drop_mapping_columns),
            ("slurm_clusters transport columns", _drop_cluster_columns),
        ],
        tables=("slurm_clusters", "slurm_job_mappings", "slurm_job_outputs"),
    )
