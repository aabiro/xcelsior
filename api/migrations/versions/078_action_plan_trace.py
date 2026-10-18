"""Carry MCP W3C trace identity from action plan to scheduler attempt.

Revision ID: 078
Revises: 077
"""

from alembic import op


revision = "078"
down_revision = "077"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE action_plans ADD COLUMN IF NOT EXISTS trace_id TEXT")
    op.execute(
        """
        ALTER TABLE action_plans
        ADD CONSTRAINT ck_action_plans_trace_id
        CHECK (trace_id IS NULL OR trace_id ~ '^[0-9a-f]{32}$')
        """
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE action_plans DROP CONSTRAINT IF EXISTS ck_action_plans_trace_id"
    )
    op.execute("ALTER TABLE action_plans DROP COLUMN IF EXISTS trace_id")
