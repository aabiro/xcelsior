"""A Slurm job id may come back: unique among open mappings, not for all time.

Migration 067 made `(cluster_id, slurm_job_id)` unique across every mapping
ever written, settled ones included. Slurm does not promise that. A controller
whose state is lost (a rebuild, a wiped StateSaveLocation) numbers jobs from
FirstJobId again, and ids wrap at MaxJobId. The next job to draw a number an
old, settled mapping already holds could not be registered: `sbatch` had
already succeeded, so the job ran on the cluster with no mapping, and the
submit returned a 500.

What 067's index was for still holds: two Xcelsior jobs must never act on
one live Slurm job at the same time. So uniqueness now covers open mappings
only (`terminal_at IS NULL`). Lookups by Slurm id across settled mappings
keep `idx_slurm_mappings_slurm_job`. Telling two jobs under one number
apart is the reconciler's work, by the ``xcl-<xcelsior job id>`` name the
adapter gives every job.

Each unit is idempotent and runs in its own short-lock transaction
(migrations/lock_safe.py). The new index is built before the old one is
dropped, so uniqueness among open mappings holds throughout.

`downgrade` restores 067's index, and refuses while any Slurm id is mapped
more than once on one cluster, since that index could not be built.
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "153"
down_revision = "152"
branch_labels = None
depends_on = None


def _open_unique(conn):
    conn.execute(
        text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_slurm_open_cluster_job "
            "ON slurm_job_mappings (cluster_id, slurm_job_id) WHERE terminal_at IS NULL"
        )
    )


def _drop_all_time_unique(conn):
    conn.execute(text("DROP INDEX IF EXISTS uq_slurm_cluster_job"))


def _refuse_reused_ids(conn):
    reused = (
        conn.execute(
            text(
                "SELECT cluster_id, slurm_job_id, count(*) AS n FROM slurm_job_mappings "
                "GROUP BY cluster_id, slurm_job_id HAVING count(*) > 1 LIMIT 10"
            )
        )
        .mappings()
        .all()
    )
    if reused:
        raise RuntimeError(
            "migration 153 downgrade: these Slurm ids are mapped more than once on one "
            f"cluster, so 067's index cannot be rebuilt: {[dict(r) for r in reused]}"
        )


def _all_time_unique(conn):
    conn.execute(
        text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_slurm_cluster_job "
            "ON slurm_job_mappings (cluster_id, slurm_job_id)"
        )
    )


def _drop_open_unique(conn):
    conn.execute(text("DROP INDEX IF EXISTS uq_slurm_open_cluster_job"))


def upgrade():
    apply_in_own_transactions(
        [
            ("slurm_job_mappings open-job uniqueness", _open_unique),
            ("slurm_job_mappings all-time uniqueness", _drop_all_time_unique),
        ],
        tables=("slurm_job_mappings",),
    )


def downgrade():
    apply_in_own_transactions(
        [
            ("slurm_job_mappings reused ids", _refuse_reused_ids),
            ("slurm_job_mappings all-time uniqueness", _all_time_unique),
            ("slurm_job_mappings open-job uniqueness", _drop_open_unique),
        ],
        tables=("slurm_job_mappings",),
    )
