"""Add spot pricing and preemption support

Revision ID: 002
Revises: 001
Create Date: 2026-02-12
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Spot pricing history
    op.create_table(
        "spot_prices",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("gpu_model", sa.Text(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("supply", sa.Integer(), nullable=False),
        sa.Column("demand", sa.Integer(), nullable=False),
        sa.Column("computed_at", sa.Float(), nullable=False),
    )

    op.create_index(
        "idx_spot_prices_model_time",
        "spot_prices",
        [sa.text("gpu_model"), sa.text("computed_at DESC")],
    )

    # Node security versions (for version gating)
    op.create_table(
        "node_versions",
        sa.Column("host_id", sa.Text(), primary_key=True),
        sa.Column("runc_version", sa.Text()),
        sa.Column("nvidia_toolkit_version", sa.Text()),
        sa.Column("nvidia_driver_version", sa.Text()),
        sa.Column("docker_version", sa.Text()),
        sa.Column("agent_version", sa.Text()),
        sa.Column("compute_score", sa.Float()),
        sa.Column("last_benchmark_at", sa.Float()),
        sa.Column("admitted", sa.Boolean(), default=False),
        sa.Column("rejection_reason", sa.Text()),
        sa.Column("updated_at", sa.Float(), nullable=False),
    )

    # Compute score benchmarks
    op.create_table(
        "benchmarks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("host_id", sa.Text(), nullable=False),
        sa.Column("gpu_model", sa.Text(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("tflops", sa.Float()),
        sa.Column("memory_bandwidth_gbps", sa.Float()),
        sa.Column("benchmark_type", sa.Text(), default="standard"),
        sa.Column("details", JSONB()),
        sa.Column("run_at", sa.Float(), nullable=False),
    )

    op.create_index("idx_benchmarks_host", "benchmarks", ["host_id"])


def downgrade() -> None:
    op.drop_table("benchmarks")
    op.drop_table("node_versions")
    op.drop_table("spot_prices")
