"""Add telemetry_snapshots table for persistent GPU telemetry history.

Enables AI queries like "what was my GPU utilization last week?" or
"show temperature trends for host X over the past month".
The nvml_telemetry agent currently writes live data that is discarded —
this table persists snapshots for historical analysis.

Revision ID: 013
Revises: 012
Create Date: 2026-04-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "telemetry_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("host_id", sa.Text(), nullable=False),
        sa.Column("job_id", sa.Text(), server_default=""),
        sa.Column("gpu_index", sa.Integer(), server_default="0"),
        sa.Column("gpu_model", sa.Text(), server_default=""),
        sa.Column("gpu_util_pct", sa.Float(), server_default="0"),
        sa.Column("memory_used_gb", sa.Float(), server_default="0"),
        sa.Column("memory_total_gb", sa.Float(), server_default="0"),
        sa.Column("temp_c", sa.Float(), server_default="0"),
        sa.Column("power_draw_w", sa.Float(), server_default="0"),
        sa.Column("fan_speed_pct", sa.Float(), server_default="0"),
        sa.Column("recorded_at", sa.Float(), nullable=False),
    )

    # Index for "give me host X telemetry for the last N days"
    op.create_index(
        "idx_telemetry_host_recorded",
        "telemetry_snapshots",
        ["host_id", "recorded_at"],
        postgresql_ops={"recorded_at": "DESC"},
    )

    # Index for "all telemetry related to job Y"
    op.create_index(
        "idx_telemetry_job_recorded",
        "telemetry_snapshots",
        ["job_id", "recorded_at"],
        postgresql_ops={"recorded_at": "DESC"},
    )

    # Partial index for recent data (last 30 days) — most AI queries are recency-oriented
    op.execute(
        "CREATE INDEX idx_telemetry_recent ON telemetry_snapshots (host_id, recorded_at DESC) "
        "WHERE recorded_at > EXTRACT(EPOCH FROM NOW()) - 2592000"
    )


def downgrade() -> None:
    op.drop_index("idx_telemetry_recent", table_name="telemetry_snapshots")
    op.drop_index("idx_telemetry_job_recorded", table_name="telemetry_snapshots")
    op.drop_index("idx_telemetry_host_recorded", table_name="telemetry_snapshots")
    op.drop_table("telemetry_snapshots")
