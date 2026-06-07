"""Add shared team billing wallet column.

Revision ID: 036
Revises: 035
"""

from alembic import op

revision = "036"
down_revision = "035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE teams "
        "ADD COLUMN IF NOT EXISTS billing_customer_id TEXT NOT NULL DEFAULT '';"
    )
    op.execute(
        """
        UPDATE teams t
        SET billing_customer_id = COALESCE(
            NULLIF(u.customer_id, ''),
            NULLIF(u.user_id, ''),
            t.owner_email
        )
        FROM users u
        WHERE u.email = t.owner_email
          AND (t.billing_customer_id = '' OR t.billing_customer_id IS NULL)
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE teams DROP COLUMN IF EXISTS billing_customer_id;")