"""Add serverless execution mode and queue timeout.

Revision ID: 052
Revises: 051
Create Date: 2026-07-11
"""

from alembic import op

revision = "052"
down_revision = "051"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE serverless_endpoints
        ADD COLUMN IF NOT EXISTS execution_mode TEXT NOT NULL DEFAULT 'sync'
        """
    )
    op.execute(
        """
        ALTER TABLE serverless_endpoints
        ADD COLUMN IF NOT EXISTS queue_timeout_sec INTEGER NOT NULL DEFAULT 120
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE serverless_endpoints DROP COLUMN IF EXISTS queue_timeout_sec")
    op.execute("ALTER TABLE serverless_endpoints DROP COLUMN IF EXISTS execution_mode")
