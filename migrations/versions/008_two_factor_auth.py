"""Add two-factor authentication tables

- mfa_methods: stores per-user MFA methods (TOTP, SMS, passkey)
- mfa_backup_codes: one-time recovery codes
- Adds mfa_enabled flag on users table

Revision ID: 008
Revises: 007
Create Date: 2025-06-08
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    user_columns = {column["name"] for column in inspector.get_columns("users")}

    # Add mfa_enabled flag to users
    if "mfa_enabled" not in user_columns:
        op.add_column(
            "users",
            sa.Column("mfa_enabled", sa.Integer(), nullable=False, server_default="0"),
        )

    # MFA methods — one row per method per user
    if not inspector.has_table("mfa_methods"):
        op.create_table(
            "mfa_methods",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("email", sa.Text(), nullable=False),
            sa.Column("method_type", sa.Text(), nullable=False),       # totp | sms | passkey
            sa.Column("secret", sa.Text()),                              # TOTP base32 secret (encrypted)
            sa.Column("phone_number", sa.Text()),                        # SMS phone (E.164)
            sa.Column("credential_id", sa.Text()),                       # passkey credential ID (base64url)
            sa.Column("public_key", sa.Text()),                          # passkey public key (base64url)
            sa.Column("sign_count", sa.Integer(), server_default="0"),   # passkey sign counter
            sa.Column("device_name", sa.Text()),                         # passkey friendly name
            sa.Column("enabled", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("created_at", sa.Float(), nullable=False),
        )
    op.execute("CREATE INDEX IF NOT EXISTS idx_mfa_methods_email ON mfa_methods (email)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_mfa_methods_credential ON mfa_methods (credential_id)")

    # Backup recovery codes
    if not inspector.has_table("mfa_backup_codes"):
        op.create_table(
            "mfa_backup_codes",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("email", sa.Text(), nullable=False),
            sa.Column("code_hash", sa.Text(), nullable=False),
            sa.Column("used", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("created_at", sa.Float(), nullable=False),
        )
    op.execute("CREATE INDEX IF NOT EXISTS idx_mfa_backup_email ON mfa_backup_codes (email)")

    # Pending MFA challenges (short-lived, for login flow)
    if not inspector.has_table("mfa_challenges"):
        op.create_table(
            "mfa_challenges",
            sa.Column("challenge_id", sa.Text(), primary_key=True),
            sa.Column("email", sa.Text(), nullable=False),
            sa.Column("session_token", sa.Text()),                       # partial session token
            sa.Column("challenge_data", sa.Text()),                      # JSON for passkey challenge
            sa.Column("created_at", sa.Float(), nullable=False),
            sa.Column("expires_at", sa.Float(), nullable=False),
        )
    op.execute("CREATE INDEX IF NOT EXISTS idx_mfa_challenge_email ON mfa_challenges (email)")


def downgrade() -> None:
    op.drop_table("mfa_challenges")
    op.drop_table("mfa_backup_codes")
    op.drop_table("mfa_methods")
    op.drop_column("users", "mfa_enabled")
