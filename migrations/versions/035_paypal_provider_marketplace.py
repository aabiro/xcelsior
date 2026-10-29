"""PayPal provider onboarding + marketplace payout columns.

Revision ID: 035
Revises: 034
"""

from alembic import op

revision = "035"
down_revision = "034"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE provider_accounts "
        "ADD COLUMN IF NOT EXISTS paypal_tracking_id TEXT NOT NULL DEFAULT '';"
    )
    op.execute(
        "ALTER TABLE provider_accounts "
        "ADD COLUMN IF NOT EXISTS paypal_merchant_id TEXT NOT NULL DEFAULT '';"
    )
    op.execute(
        "ALTER TABLE provider_accounts "
        "ADD COLUMN IF NOT EXISTS paypal_payer_id TEXT NOT NULL DEFAULT '';"
    )
    op.execute(
        "ALTER TABLE provider_accounts "
        "ADD COLUMN IF NOT EXISTS paypal_status TEXT NOT NULL DEFAULT '';"
    )
    op.execute(
        "ALTER TABLE provider_accounts "
        "ADD COLUMN IF NOT EXISTS paypal_onboarded_at DOUBLE PRECISION NOT NULL DEFAULT 0;"
    )
    op.execute(
        "ALTER TABLE payout_splits "
        "ADD COLUMN IF NOT EXISTS paypal_capture_id TEXT NOT NULL DEFAULT '';"
    )
    op.execute(
        "ALTER TABLE payout_splits "
        "ADD COLUMN IF NOT EXISTS payment_rail TEXT NOT NULL DEFAULT 'stripe';"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE payout_splits DROP COLUMN IF EXISTS payment_rail;")
    op.execute("ALTER TABLE payout_splits DROP COLUMN IF EXISTS paypal_capture_id;")
    op.execute("ALTER TABLE provider_accounts DROP COLUMN IF EXISTS paypal_onboarded_at;")
    op.execute("ALTER TABLE provider_accounts DROP COLUMN IF EXISTS paypal_status;")
    op.execute("ALTER TABLE provider_accounts DROP COLUMN IF EXISTS paypal_payer_id;")
    op.execute("ALTER TABLE provider_accounts DROP COLUMN IF EXISTS paypal_merchant_id;")
    op.execute("ALTER TABLE provider_accounts DROP COLUMN IF EXISTS paypal_tracking_id;")