"""Add denormalized provider stats columns to provider_accounts.

Without this, every "how much have I earned?" query requires a full
aggregation scan over payout_ledger. These columns are updated
incrementally on each payout and enable instant AI responses.

Revision ID: 014
Revises: 013
Create Date: 2026-04-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "014"
down_revision: Union[str, None] = "013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "provider_accounts",
        sa.Column("total_earned_cad", sa.Numeric(12, 2), nullable=False, server_default="0"),
    )
    op.add_column(
        "provider_accounts",
        sa.Column("total_paid_out_cad", sa.Numeric(12, 2), nullable=False, server_default="0"),
    )
    op.add_column(
        "provider_accounts",
        sa.Column("jobs_hosted", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "provider_accounts",
        sa.Column("last_payout_at", sa.Float(), nullable=False, server_default="0"),
    )

    # Back-fill from existing payout_ledger data
    op.execute(
        """
        UPDATE provider_accounts pa
        SET
            total_earned_cad = COALESCE(agg.total_earned, 0),
            total_paid_out_cad = COALESCE(agg.total_paid, 0),
            jobs_hosted = COALESCE(agg.job_count, 0),
            last_payout_at = COALESCE(agg.last_at, 0)
        FROM (
            SELECT
                provider_id,
                SUM(provider_payout_cad) AS total_earned,
                SUM(CASE WHEN status = 'paid' THEN provider_payout_cad ELSE 0 END) AS total_paid,
                COUNT(DISTINCT job_id) AS job_count,
                MAX(created_at) AS last_at
            FROM payout_ledger
            GROUP BY provider_id
        ) agg
        WHERE pa.provider_id = agg.provider_id
        """
    )


def downgrade() -> None:
    op.drop_column("provider_accounts", "last_payout_at")
    op.drop_column("provider_accounts", "jobs_hosted")
    op.drop_column("provider_accounts", "total_paid_out_cad")
    op.drop_column("provider_accounts", "total_earned_cad")
