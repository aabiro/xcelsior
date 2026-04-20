"""Add resource_type column to billing_cycles for clean resource classification.

Replaces the fragile convention of encoding resource type in gpu_model/tier columns
(e.g. gpu_model='storage', tier='volume') with an explicit discriminator.
Values: 'gpu' (default), 'volume', 'inference'.

Revision ID: 021
Revises: 020
Create Date: 2026-04-12
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "021"
down_revision: Union[str, None] = "020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Idempotent: db.py's CREATE TABLE IF NOT EXISTS may have already added
    # this column on fresh installs before alembic runs. Use raw SQL with
    # IF NOT EXISTS to be safe across both upgrade paths.
    op.execute(
        "ALTER TABLE billing_cycles "
        "ADD COLUMN IF NOT EXISTS resource_type TEXT NOT NULL DEFAULT 'gpu'"
    )
    # Backfill existing rows based on the old convention
    op.execute("UPDATE billing_cycles SET resource_type = 'volume' WHERE tier = 'volume'")
    op.execute("UPDATE billing_cycles SET resource_type = 'inference' WHERE tier = 'inference'")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_billing_cycles_resource_type "
        "ON billing_cycles (resource_type)"
    )


def downgrade() -> None:
    op.drop_index("idx_billing_cycles_resource_type", table_name="billing_cycles")
    op.drop_column("billing_cycles", "resource_type")
