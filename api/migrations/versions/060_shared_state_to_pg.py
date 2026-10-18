"""shared_state_to_pg

Revision ID: a0985327493e
Revises: 059
Create Date: 2026-07-19 07:44:28.675427
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision: str = 'a0985327493e'
down_revision: Union[str, None] = '059'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Lightning deposits
    op.create_table(
        "ln_deposits",
        sa.Column("deposit_id", sa.String(), primary_key=True),
        sa.Column("customer_id", sa.String(), nullable=False),
        sa.Column("label", sa.String(), nullable=False, unique=True),
        sa.Column("bolt11", sa.String(), nullable=False),
        sa.Column("payment_hash", sa.String(), nullable=False),
        sa.Column("amount_msat", sa.BigInteger(), nullable=False),
        sa.Column("amount_sats", sa.BigInteger(), nullable=False),
        sa.Column("amount_btc", sa.Float(), nullable=False),
        sa.Column("amount_cad", sa.Float(), nullable=False),
        sa.Column("btc_cad_rate", sa.Float(), nullable=False),
        sa.Column("status", sa.String(), server_default="pending"),
        sa.Column("payment_preimage", sa.String(), server_default=""),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("expires_at", sa.Float(), nullable=False),
        sa.Column("paid_at", sa.Float(), server_default="0"),
        sa.Column("credited_at", sa.Float(), server_default="0"),
    )
    op.create_index("idx_ln_deposits_customer", "ln_deposits", ["customer_id"])
    op.create_index("idx_ln_deposits_label", "ln_deposits", ["label"])
    op.create_index("idx_ln_deposits_status", "ln_deposits", ["status"])

    # Slurm job mappings
    op.create_table(
        "slurm_job_mappings",
        sa.Column("xcelsior_job_id", sa.String(), primary_key=True),
        sa.Column("slurm_job_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("idx_slurm_mappings_slurm_job", "slurm_job_mappings", ["slurm_job_id"])


def downgrade() -> None:
    op.drop_index("idx_slurm_mappings_slurm_job", table_name="slurm_job_mappings")
    op.drop_table("slurm_job_mappings")

    op.drop_index("idx_ln_deposits_status", table_name="ln_deposits")
    op.drop_index("idx_ln_deposits_label", table_name="ln_deposits")
    op.drop_index("idx_ln_deposits_customer", table_name="ln_deposits")
    op.drop_table("ln_deposits")
