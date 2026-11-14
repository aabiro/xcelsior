"""Whether a Slurm placement may ever be charged, fixed when it is placed.

Blueprint 2.4 and P2.9's rollout: shadow, then admin-only, then allowlisted
tenants, then everyone (`XCELSIOR_SLURM_PLACEMENT`). A job placed in shadow
mode is computed and never charged, and that must stay true after the switch
moves on: otherwise every shadow job still running, or finished but not yet
closed, would be charged the moment an operator turned charging on, for work
nobody agreed to pay for.

So the decision is made once, at placement, and kept on the submission:
`charge` is true only for a placement made in a charging mode. The lifecycle
task (workers/slurm_lifecycle.py) closes every finished attempt, and charges
one only when its submission says so. Existing rows were all placed in shadow
mode and default to false.

One idempotent unit through `lock_safe`; `downgrade` drops the column.
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "163"
down_revision = "162"
branch_labels = None
depends_on = None


def upgrade():
    apply_in_own_transactions(
        [
            (
                "slurm_submissions charge",
                lambda conn: conn.execute(
                    text(
                        "ALTER TABLE slurm_submissions "
                        "ADD COLUMN IF NOT EXISTS charge BOOLEAN NOT NULL DEFAULT FALSE"
                    )
                ),
            )
        ],
        tables=("slurm_submissions",),
    )


def downgrade():
    apply_in_own_transactions(
        [
            (
                "slurm_submissions charge",
                lambda conn: conn.execute(
                    text("ALTER TABLE slurm_submissions DROP COLUMN IF EXISTS charge")
                ),
            )
        ],
        tables=("slurm_submissions",),
    )
