"""Receipt shares: a finished run its owner chose to count on public workload pages.

* `receipt_shares` — `job_id` is the key: one share per run. The owner turns it
  on and off; turning it off sets `revoked_at` rather than deleting, so a page
  can be traced to the shares behind it. `source` is `operator` when a
  platform admin shared it (an operator run on the real fleet) and `customer`
  otherwise; the public page labels the two.

The page itself stores nothing: it is computed from the shared runs' receipts,
which the ledger already holds. Billing domain. One idempotent unit through
`lock_safe`; `downgrade` drops the table.
"""

from sqlalchemy import text

from migrations.lock_safe import apply_in_own_transactions

revision = "169"
down_revision = "168"
branch_labels = None
depends_on = None


def _create(conn):
    conn.execute(
        text(
            "CREATE TABLE IF NOT EXISTS receipt_shares ("
            " job_id TEXT PRIMARY KEY,"
            " owner_id TEXT NOT NULL,"
            " shared_by TEXT NOT NULL,"
            " source TEXT NOT NULL CHECK (source IN ('customer', 'operator')),"
            " created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            " revoked_at TIMESTAMPTZ)"
        )
    )


def upgrade():
    apply_in_own_transactions(
        [("receipt shares table", _create)],
        tables=("receipt_shares",),
    )


def downgrade():
    apply_in_own_transactions(
        [
            (
                "receipt shares table",
                lambda conn: conn.execute(text("DROP TABLE IF EXISTS receipt_shares")),
            )
        ],
        tables=("receipt_shares",),
    )
