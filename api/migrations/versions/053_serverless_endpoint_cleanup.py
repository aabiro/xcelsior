"""Serverless endpoint display cleanup and dashboard test execution.

Revision ID: 053
Revises: 052
Create Date: 2026-07-12
"""

from alembic import op

revision = "053"
down_revision = "052"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE serverless_workers
        ADD COLUMN IF NOT EXISTS billing_exempt BOOLEAN NOT NULL DEFAULT FALSE
        """
    )
    op.execute(
        """
        ALTER TABLE serverless_workers
        ADD COLUMN IF NOT EXISTS warm_expires_at DOUBLE PRECISION NOT NULL DEFAULT 0
        """
    )
    op.execute(
        """
        ALTER TABLE serverless_jobs
        ADD COLUMN IF NOT EXISTS billing_exempt BOOLEAN NOT NULL DEFAULT FALSE
        """
    )
    op.execute(
        """
        UPDATE serverless_endpoints
           SET name = regexp_replace(name, '^.*/', '')
         WHERE name LIKE '%/%'
           AND deleted_at = 0
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE serverless_jobs DROP COLUMN IF EXISTS billing_exempt")
    op.execute("ALTER TABLE serverless_workers DROP COLUMN IF EXISTS warm_expires_at")
    op.execute("ALTER TABLE serverless_workers DROP COLUMN IF EXISTS billing_exempt")
