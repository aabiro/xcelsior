"""Add enhanced metadata columns to notifications table.

Currently notifications have minimal fields. These additions let the AI
answer targeted questions like "show my critical billing alerts" or
"what system events require my action?" by filtering on priority,
entity_type, and action_url.

Revision ID: 016
Revises: 015
Create Date: 2026-04-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "016"
down_revision: Union[str, None] = "015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "notifications",
        sa.Column("action_url", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "notifications",
        sa.Column("entity_type", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "notifications",
        sa.Column("entity_id", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "notifications",
        # 0=normal, 1=high, 2=critical
        sa.Column("priority", sa.Integer(), nullable=False, server_default="0"),
    )

    # Back-fill priority from notification type where possible
    op.execute(
        """
        UPDATE notifications
        SET priority = CASE
            WHEN type IN ('billing_alert', 'balance_critical', 'payment_failed', 'wallet_empty') THEN 2
            WHEN type IN ('job_failed', 'host_offline', 'sla_violation', 'balance_low') THEN 1
            ELSE 0
        END
        """
    )

    # Index for "show my critical unread notifications"
    op.create_index(
        "idx_notifications_priority_unread",
        "notifications",
        ["user_email", "priority", "read", "created_at"],
        postgresql_ops={"created_at": "DESC"},
    )

    # Index for entity lookups ("notifications about job X")
    op.create_index(
        "idx_notifications_entity",
        "notifications",
        ["entity_type", "entity_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_notifications_entity", table_name="notifications")
    op.drop_index("idx_notifications_priority_unread", table_name="notifications")
    op.drop_column("notifications", "priority")
    op.drop_column("notifications", "entity_id")
    op.drop_column("notifications", "entity_type")
    op.drop_column("notifications", "action_url")
