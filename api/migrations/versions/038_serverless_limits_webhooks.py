"""Serverless Phase 8 — max_queue_size + webhook delivery status.

Revision ID: 038
Revises: 037
Create Date: 2026-06-08
"""

from typing import Sequence, Union

from alembic import op

revision: str = "038"
down_revision: Union[str, None] = "037"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE serverless_endpoints
        ADD COLUMN IF NOT EXISTS max_queue_size INTEGER NOT NULL DEFAULT 100
        """
    )
    op.execute(
        """
        ALTER TABLE serverless_jobs
        ADD COLUMN IF NOT EXISTS webhook_status TEXT NOT NULL DEFAULT '',
        ADD COLUMN IF NOT EXISTS webhook_attempts INTEGER NOT NULL DEFAULT 0,
        ADD COLUMN IF NOT EXISTS webhook_next_retry_at DOUBLE PRECISION NOT NULL DEFAULT 0,
        ADD COLUMN IF NOT EXISTS webhook_last_error TEXT NOT NULL DEFAULT ''
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_serverless_jobs_webhook_retry
        ON serverless_jobs (webhook_next_retry_at)
        WHERE webhook_url IS NOT NULL
          AND webhook_url != ''
          AND webhook_status IN ('pending', '')
          AND status IN ('COMPLETED', 'FAILED', 'CANCELLED')
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_serverless_jobs_webhook_retry")
    op.execute(
        """
        ALTER TABLE serverless_jobs
        DROP COLUMN IF EXISTS webhook_last_error,
        DROP COLUMN IF EXISTS webhook_next_retry_at,
        DROP COLUMN IF EXISTS webhook_attempts,
        DROP COLUMN IF EXISTS webhook_status
        """
    )
    op.execute(
        "ALTER TABLE serverless_endpoints DROP COLUMN IF EXISTS max_queue_size"
    )