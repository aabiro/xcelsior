"""Add source column to ai_conversations for admin AI Insights.

Tracks which AI surface each conversation originated from:
xcel, analytics, wizard, or support.

Revision ID: 017
Revises: 016
Create Date: 2026-06-15
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "017"
down_revision: Union[str, None] = "016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "ai_conversations",
        sa.Column("source", sa.Text(), nullable=False, server_default="xcel"),
    )
    op.create_index("ix_ai_conversations_source", "ai_conversations", ["source"])


def downgrade() -> None:
    op.drop_index("ix_ai_conversations_source", table_name="ai_conversations")
    op.drop_column("ai_conversations", "source")
