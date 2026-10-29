"""Expand-only: link usage_meters to placement attempts.

Adds nullable ``attempt_id`` on ``usage_meters`` and a partial unique
index so attempt-owned terminal metering is idempotent (one billable
meter row per attempt). Legacy meters without an attempt keep the
existing job/meter_id identity.

Revision ID: 062
Revises: 061
Create Date: 2026-07-20
"""

from typing import Sequence, Union

from alembic import op

revision: str = "062"
down_revision: Union[str, None] = "061"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE usage_meters "
        "ADD COLUMN IF NOT EXISTS attempt_id TEXT NULL"
    )
    # One terminal usage meter per placement attempt (fenced work).
    # Legacy rows leave attempt_id NULL and are outside this uniqueness.
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_usage_meters_one_per_attempt
            ON usage_meters (attempt_id)
         WHERE attempt_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_usage_meters_attempt
            ON usage_meters (attempt_id)
         WHERE attempt_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_usage_meters_attempt")
    op.execute("DROP INDEX IF EXISTS uq_usage_meters_one_per_attempt")
    op.execute("ALTER TABLE usage_meters DROP COLUMN IF EXISTS attempt_id")
