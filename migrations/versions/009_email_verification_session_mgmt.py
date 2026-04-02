"""Add email verification columns and session tracking columns

- users: email_verified, email_verification_token, email_verification_expires
- sessions: ip_address, user_agent, last_active

Revision ID: 009
Revises: 008
Create Date: 2025-06-08
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Email verification on users
    op.add_column(
        "users",
        sa.Column("email_verified", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "users",
        sa.Column("email_verification_token", sa.Text(), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("email_verification_expires", sa.Float(), nullable=True),
    )

    # Session tracking columns
    op.add_column(
        "sessions",
        sa.Column("ip_address", sa.Text(), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column("user_agent", sa.Text(), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column("last_active", sa.Float(), nullable=True),
    )

    # Mark all existing users as verified (they signed up before verification was required)
    op.execute("UPDATE users SET email_verified = 1")


def downgrade() -> None:
    op.drop_column("sessions", "last_active")
    op.drop_column("sessions", "user_agent")
    op.drop_column("sessions", "ip_address")
    op.drop_column("users", "email_verification_expires")
    op.drop_column("users", "email_verification_token")
    op.drop_column("users", "email_verified")
