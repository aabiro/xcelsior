"""Add web push subscriptions for desktop and installed PWA notifications.

Revision ID: 020
Revises: 019
Create Date: 2026-04-08
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "020"
down_revision: Union[str, None] = "019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "web_push_subscriptions",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("user_email", sa.Text(), nullable=False),
        sa.Column("endpoint", sa.Text(), nullable=False),
        sa.Column("p256dh", sa.Text(), nullable=False),
        sa.Column("auth", sa.Text(), nullable=False),
        sa.Column("user_agent", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("last_used_at", sa.Float(), nullable=False),
        sa.Column("revoked_at", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("endpoint", name="uq_web_push_subscriptions_endpoint"),
    )

    op.create_index(
        "idx_web_push_subscriptions_user_active",
        "web_push_subscriptions",
        ["user_email", "revoked_at", "last_used_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_web_push_subscriptions_user_active", table_name="web_push_subscriptions")
    op.drop_table("web_push_subscriptions")
