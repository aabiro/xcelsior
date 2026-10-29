"""Marketplace Connect Checkout sales ledger.

Records completed destination-charge Checkout Sessions (one row per
checkout.session.completed event) so the storefront can show fulfilment and
we have a per-provider ledger of marketplace revenue.

Revision ID: 039
Revises: 038
"""

from alembic import op

revision = "039"
down_revision = "038"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS marketplace_sales (
            session_id           TEXT PRIMARY KEY,
            payment_intent_id    TEXT NOT NULL DEFAULT '',
            destination_account  TEXT NOT NULL DEFAULT '',
            product_id           TEXT NOT NULL DEFAULT '',
            amount_total_cents   BIGINT NOT NULL DEFAULT 0,
            currency             TEXT NOT NULL DEFAULT '',
            customer_email       TEXT NOT NULL DEFAULT '',
            event_id             TEXT NOT NULL DEFAULT '',
            created_at           DOUBLE PRECISION NOT NULL DEFAULT 0
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_marketplace_sales_dest "
        "ON marketplace_sales (destination_account);"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS marketplace_sales;")
