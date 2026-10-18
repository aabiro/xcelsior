"""Instance lifecycle: storage billing rates + storage fields on jobs.

Adds:
  - storage_billing_rates table: per-type-per-GB-per-hour rates
  - Default rates for nvme, ssd, hdd storage types
  - storage_gb, storage_type, storage_rate_cad_per_gb_hr fields stored in
    the jobs JSONB payload (no schema changes to the jobs table itself since
    all metadata lives in payload)

The billing engine reads storage_billing_rates at charge time and caches
the rate in the job payload as storage_rate_cad_per_gb_hr for audit trails.

Revision ID: 019
Revises: 018
Create Date: 2025-05-01
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "019"
down_revision: Union[str, None] = "018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── storage_billing_rates ─────────────────────────────────────────
    op.create_table(
        "storage_billing_rates",
        sa.Column("storage_type", sa.Text(), nullable=False),
        sa.Column("rate_cad_per_gb_hr", sa.Float(), nullable=False),
        sa.Column("description", sa.Text(), server_default=""),
        sa.Column("updated_at", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("storage_type"),
    )

    # Seed default rates (competitive with Vast.ai / RunPod)
    # NVMe: ~$0.00035/GB/hr  ≈ $0.25/GB/month
    # SSD:  ~$0.00014/GB/hr  ≈ $0.10/GB/month
    # HDD:  ~$0.00007/GB/hr  ≈ $0.05/GB/month
    import time
    now = time.time()
    op.execute(
        sa.text(
            """INSERT INTO storage_billing_rates (storage_type, rate_cad_per_gb_hr, description, updated_at)
               VALUES
                 ('nvme', 0.00035, 'NVMe SSD — high-speed local storage', :now),
                 ('ssd',  0.00014, 'SATA SSD — standard local storage', :now),
                 ('hdd',  0.00007, 'Hard disk — archival/bulk storage', :now)
               ON CONFLICT (storage_type) DO NOTHING"""
        ).bindparams(now=now)
    )


def downgrade() -> None:
    op.drop_table("storage_billing_rates")
