"""Bind OAuth refresh-token families to their protected resource audience.

Revision ID: 077
Revises: 076
"""

from alembic import op


revision = "077"
down_revision = "076"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE oauth_refresh_tokens
        ADD COLUMN IF NOT EXISTS audience TEXT NOT NULL
        DEFAULT 'xcelsior-api'
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE oauth_refresh_tokens DROP COLUMN IF EXISTS audience")
