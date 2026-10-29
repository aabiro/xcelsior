"""Add platform admin flag to auth tables

Separates platform admin privilege from the account role so users can remain
submitters/providers while also holding admin access.

Revision ID: 011
Revises: 010
Create Date: 2026-04-02
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    user_columns = {column["name"] for column in inspector.get_columns("users")}
    session_columns = {column["name"] for column in inspector.get_columns("sessions")}
    api_key_columns = {column["name"] for column in inspector.get_columns("api_keys")}

    if "is_admin" not in user_columns:
        op.add_column(
            "users",
            sa.Column("is_admin", sa.Integer(), nullable=False, server_default="0"),
        )
    if "is_admin" not in session_columns:
        op.add_column(
            "sessions",
            sa.Column("is_admin", sa.Integer(), nullable=False, server_default="0"),
        )
    if "is_admin" not in api_key_columns:
        op.add_column(
            "api_keys",
            sa.Column("is_admin", sa.Integer(), nullable=False, server_default="0"),
        )

    # Preserve any legacy accounts that used role='admin' for platform access.
    op.execute("UPDATE users SET is_admin = 1 WHERE LOWER(role) = 'admin'")
    op.execute(
        """
        UPDATE sessions AS s
        SET is_admin = COALESCE(u.is_admin, 0)
        FROM users AS u
        WHERE s.email = u.email
        """
    )
    op.execute(
        """
        UPDATE api_keys AS k
        SET is_admin = COALESCE(u.is_admin, 0)
        FROM users AS u
        WHERE k.email = u.email
        """
    )


def downgrade() -> None:
    op.drop_column("api_keys", "is_admin")
    op.drop_column("sessions", "is_admin")
    op.drop_column("users", "is_admin")
