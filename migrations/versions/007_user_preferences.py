"""Add preferences JSONB column to users table

Stores user preferences server-side (onboarding state, UI settings)
instead of relying on localStorage which is per-browser only.

Revision ID: 007
Revises: 006
Create Date: 2026-03-31
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "preferences",
            sa.dialects.postgresql.JSONB(),
            server_default="{}",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "preferences")
