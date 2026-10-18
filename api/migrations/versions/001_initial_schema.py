"""Initial PostgreSQL schema with JSONB

Revision ID: 001
Revises: None
Create Date: 2026-02-12
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # State table (generic namespace store)
    op.create_table(
        "state",
        sa.Column("namespace", sa.Text(), primary_key=True),
        sa.Column("payload", JSONB(), nullable=False),
    )

    # Jobs table with JSONB payload
    op.create_table(
        "jobs",
        sa.Column("job_id", sa.Text(), primary_key=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("submitted_at", sa.Float(), nullable=False),
        sa.Column("host_id", sa.Text(), nullable=True),
        sa.Column("payload", JSONB(), nullable=False),
    )

    # Hosts table with JSONB payload
    op.create_table(
        "hosts",
        sa.Column("host_id", sa.Text(), primary_key=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("registered_at", sa.Float(), nullable=False),
        sa.Column("payload", JSONB(), nullable=False),
    )

    # Performance indexes
    op.create_index(
        "idx_jobs_queue",
        "jobs",
        [sa.text("status"), sa.text("priority DESC"), sa.text("submitted_at ASC")],
    )
    op.create_index(
        "idx_hosts_status",
        "hosts",
        [sa.text("status"), sa.text("registered_at ASC")],
    )

    # GIN indexes on JSONB for flexible querying of GPU capabilities
    op.execute(
        "CREATE INDEX idx_hosts_payload_gin ON hosts USING GIN (payload)"
    )
    op.execute(
        "CREATE INDEX idx_jobs_payload_gin ON jobs USING GIN (payload)"
    )

    # Expression indexes for common queries
    op.execute(
        "CREATE INDEX idx_hosts_gpu_model ON hosts ((payload->>'gpu_model'))"
    )
    op.execute(
        "CREATE INDEX idx_hosts_free_vram ON hosts "
        "(((payload->>'free_vram_gb')::float))"
    )
    op.execute(
        "CREATE INDEX idx_jobs_tier ON jobs ((payload->>'tier'))"
    )


def downgrade() -> None:
    op.drop_index("idx_jobs_tier")
    op.drop_index("idx_hosts_free_vram")
    op.drop_index("idx_hosts_gpu_model")
    op.drop_index("idx_jobs_payload_gin")
    op.drop_index("idx_hosts_payload_gin")
    op.drop_index("idx_hosts_status")
    op.drop_index("idx_jobs_queue")
    op.drop_table("hosts")
    op.drop_table("jobs")
    op.drop_table("state")
