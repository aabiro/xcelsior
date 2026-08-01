"""Finish `payout_splits` against the data-architecture companion §4.4.

Migration `080` hardened this table substantially — it added
`claim_expires_at`, `next_attempt_at`, `updated_at` and `settled_at` as
`TIMESTAMPTZ` and moved the money to integer micros — but left two things
from the original definition:

- `created_at` is still `DOUBLE PRECISION` epoch seconds, the one float time
  column among five (§4.4.5).
- There is no `tenant_id` (§4.4.10). The table carries `customer_id` and
  `provider_id`, so ownership is *derivable*, but every tenant-scoped query
  has to know which of the two is the tenant, and cross-tenant denial cannot
  be proven at the storage layer. For a settlement row — the record of who
  gets paid what — that is the wrong place to be imprecise.

`customer_id` is the paying tenant, so it is the correct backfill source. The
provider is a counterparty on the row, not its owner.
"""

from alembic import op

revision = "089"
down_revision = "088"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    # §4.4.5 — the last float time column on this table.
    op.execute(
        """
        ALTER TABLE payout_splits
            ALTER COLUMN created_at TYPE TIMESTAMPTZ
                USING to_timestamp(created_at),
            ALTER COLUMN created_at SET DEFAULT now()
        """
    )

    # §4.4.10 — explicit tenant ownership.
    op.execute("ALTER TABLE payout_splits ADD COLUMN IF NOT EXISTS tenant_id TEXT")
    op.execute(
        """
        UPDATE payout_splits
           SET tenant_id = COALESCE(NULLIF(customer_id, ''), provider_id, job_id)
         WHERE tenant_id IS NULL
        """
    )
    op.execute("ALTER TABLE payout_splits ALTER COLUMN tenant_id SET NOT NULL")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_payout_splits_tenant "
        "ON payout_splits (tenant_id, created_at DESC)"
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    op.execute("DROP INDEX IF EXISTS idx_payout_splits_tenant")
    op.execute("ALTER TABLE payout_splits DROP COLUMN IF EXISTS tenant_id")
    op.execute(
        """
        ALTER TABLE payout_splits
            ALTER COLUMN created_at DROP DEFAULT,
            ALTER COLUMN created_at TYPE DOUBLE PRECISION
                USING extract(epoch FROM created_at)
        """
    )
