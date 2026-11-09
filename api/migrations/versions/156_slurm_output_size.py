"""The size a job's output file had when its tail was last read.

For clusters whose wrapper allows `cat` but not `tail` (Alliance's
slurm_commands.sh), the whole file crosses the wire on every read. So the
reconciler first asks `ls -l` for the size (the wrapper allows `ls`), and
reads only when the file has changed and is under a cap
(hpc/IMPLEMENTATION_PLAN.md §5a). `size_bytes` is what it compares against.
NULL where no size check runs.

One idempotent unit through `lock_safe`; `downgrade` drops the column.
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "156"
down_revision = "155"
branch_labels = None
depends_on = None


def upgrade():
    apply_in_own_transactions(
        [
            (
                "slurm_job_outputs size_bytes",
                lambda conn: conn.execute(
                    text("ALTER TABLE slurm_job_outputs ADD COLUMN IF NOT EXISTS size_bytes BIGINT")
                ),
            )
        ],
        tables=("slurm_job_outputs",),
    )


def downgrade():
    apply_in_own_transactions(
        [
            (
                "slurm_job_outputs size_bytes",
                lambda conn: conn.execute(
                    text("ALTER TABLE slurm_job_outputs DROP COLUMN IF EXISTS size_bytes")
                ),
            )
        ],
        tables=("slurm_job_outputs",),
    )
