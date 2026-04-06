"""Add uptime_pct column and final_score index to reputation_scores.

Revision ID: 018
Revises: 017
Create Date: 2026-04-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "018"
down_revision: Union[str, None] = "017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "reputation_scores",
        sa.Column("uptime_pct", sa.Float(), nullable=False, server_default="0.0"),
    )
    op.create_index(
        "ix_reputation_scores_final_score",
        "reputation_scores",
        ["final_score"],
    )


def downgrade() -> None:
    op.drop_index("ix_reputation_scores_final_score", table_name="reputation_scores")
    op.drop_column("reputation_scores", "uptime_pct")
