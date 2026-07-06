"""Add TTFT and latency columns to serverless_token_ledger for proxy SLO metrics.

Revision ID: 048
"""

from alembic import op

revision = "048"
down_revision = "047"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE serverless_token_ledger "
        "ADD COLUMN IF NOT EXISTS ttft_ms INTEGER NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE serverless_token_ledger "
        "ADD COLUMN IF NOT EXISTS latency_ms INTEGER NOT NULL DEFAULT 0"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE serverless_token_ledger DROP COLUMN IF EXISTS latency_ms")
    op.execute("ALTER TABLE serverless_token_ledger DROP COLUMN IF EXISTS ttft_ms")