"""Add per-endpoint unbilled token-cost accrual for blended serverless billing.

Feeds the blended meter (XCELSIOR_SERVERLESS_BLENDED_BILLING): token cost accrues
here as requests complete and is consumed when a worker uptime slice is billed,
so the slice can be charged the higher of GPU-seconds vs. token cost.

Revision ID: 045
"""

from alembic import op

revision = "045"
down_revision = "044"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE serverless_endpoints "
        "ADD COLUMN IF NOT EXISTS unbilled_token_cost_cad DOUBLE PRECISION NOT NULL DEFAULT 0"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE serverless_endpoints DROP COLUMN IF EXISTS unbilled_token_cost_cad")
