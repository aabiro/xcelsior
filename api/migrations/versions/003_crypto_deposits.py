"""Add Bitcoin deposit support

Revision ID: 003
Revises: 002
Create Date: 2026-03-28
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "crypto_deposits",
        sa.Column("deposit_id", sa.Text(), primary_key=True),
        sa.Column("customer_id", sa.Text(), nullable=False),
        sa.Column("btc_address", sa.Text(), nullable=False),
        sa.Column("amount_btc", sa.Float(), nullable=False),
        sa.Column("amount_cad", sa.Float(), nullable=False),
        sa.Column("btc_cad_rate", sa.Float(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("confirmations", sa.Integer(), server_default="0"),
        sa.Column("txid", sa.Text(), server_default=""),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("expires_at", sa.Float(), nullable=False),
        sa.Column("confirmed_at", sa.Float(), server_default="0"),
        sa.Column("credited_at", sa.Float(), server_default="0"),
    )
    op.create_index("idx_crypto_deposits_status", "crypto_deposits", ["status"])
    op.create_index("idx_crypto_deposits_customer", "crypto_deposits", ["customer_id"])
    op.create_index("idx_crypto_deposits_address", "crypto_deposits", ["btc_address"])


def downgrade() -> None:
    op.drop_index("idx_crypto_deposits_address")
    op.drop_index("idx_crypto_deposits_customer")
    op.drop_index("idx_crypto_deposits_status")
    op.drop_table("crypto_deposits")
