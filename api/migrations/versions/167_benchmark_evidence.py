"""The benchmarks table becomes the history of xcelsior-bench runs.

`benchmarks` (002) was created for compute scores and never written. Each
verification report from an agent that runs xcelsior-bench now adds a row, so a
host's benchmark history is evidence a renter and an agent can read:

* `gpu_uuid` — the GPU the run measured (blank when the agent sent none);
* `bench_version` — the xcelsior-bench version, so results are compared only
  within a version;
* `verification_id` — the verification the run belonged to;
* `verdict` — `pass` / `fail` / `unknown`, judged on the server with the
  benchmark's own rule against the VRAM the listing advertises.

The full report stays in `details` (JSONB). Rows are kept: they are history.
One idempotent unit through `lock_safe`; `downgrade` drops the columns.
"""

from sqlalchemy import text

from migrations.lock_safe import apply_in_own_transactions

revision = "167"
down_revision = "166"
branch_labels = None
depends_on = None


def upgrade():
    apply_in_own_transactions(
        [
            (
                "benchmarks evidence columns",
                lambda conn: conn.execute(
                    text(
                        "ALTER TABLE benchmarks "
                        "ADD COLUMN IF NOT EXISTS gpu_uuid TEXT NOT NULL DEFAULT '', "
                        "ADD COLUMN IF NOT EXISTS bench_version TEXT NOT NULL DEFAULT '', "
                        "ADD COLUMN IF NOT EXISTS verification_id TEXT NOT NULL DEFAULT '', "
                        "ADD COLUMN IF NOT EXISTS verdict TEXT NOT NULL DEFAULT ''"
                    )
                ),
            )
        ],
        tables=("benchmarks",),
    )


def downgrade():
    apply_in_own_transactions(
        [
            (
                "benchmarks evidence columns",
                lambda conn: conn.execute(
                    text(
                        "ALTER TABLE benchmarks "
                        "DROP COLUMN IF EXISTS gpu_uuid, "
                        "DROP COLUMN IF EXISTS bench_version, "
                        "DROP COLUMN IF EXISTS verification_id, "
                        "DROP COLUMN IF EXISTS verdict"
                    )
                ),
            )
        ],
        tables=("benchmarks",),
    )
