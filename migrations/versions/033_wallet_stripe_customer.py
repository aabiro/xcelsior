"""add stripe_customer_id to wallets

Saved-card auto-reload requires a Stripe Customer that off-session
PaymentIntents can be charged against. Previously the wallet only stored a
``stripe_payment_method_id`` and the auto-top-up loop passed the app-level
``customer_id`` as the Stripe ``customer=`` argument — which is not a Stripe
customer id, so the charge could never succeed. This column stores the real
``cus_…`` id created via ``BillingEngine.ensure_stripe_customer``.

Revision ID: 033
Revises: 032
Create Date: 2026-06-02
"""

from alembic import op

revision = "033"
down_revision = "032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE wallets "
        "ADD COLUMN IF NOT EXISTS stripe_customer_id TEXT NOT NULL DEFAULT '';"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE wallets DROP COLUMN IF EXISTS stripe_customer_id;")
