"""Add pricing_mode to usage_meters and billing_cycles for spot billing.

Revision ID: 041
Revises: 040
Create Date: 2026-06-09
"""

from alembic import op

revision = "041"
down_revision = "040"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE usage_meters "
        "ADD COLUMN IF NOT EXISTS pricing_mode TEXT NOT NULL DEFAULT 'on_demand';"
    )
    op.execute(
        "ALTER TABLE billing_cycles "
        "ADD COLUMN IF NOT EXISTS pricing_mode TEXT NOT NULL DEFAULT 'on_demand';"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_usage_meters_pricing_mode "
        "ON usage_meters (pricing_mode, created_at DESC);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_usage_meters_pricing_mode;")
    op.execute("ALTER TABLE billing_cycles DROP COLUMN IF EXISTS pricing_mode;")
    op.execute("ALTER TABLE usage_meters DROP COLUMN IF EXISTS pricing_mode;")