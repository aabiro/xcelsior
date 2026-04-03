"""Add serverless inference columns and inference_results token tracking.

- inference_endpoints: docker_image, mode, health_endpoint, api_format, region,
  worker_job_id, total_cost_cad
- inference_results: input_tokens, output_tokens, created_at

Revision ID: 010
Revises: 009
Create Date: 2025-06-09
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Serverless inference endpoint enhancements
    op.add_column(
        "inference_endpoints",
        sa.Column("docker_image", sa.Text(), server_default="xcelsior/vllm:latest"),
    )
    op.add_column(
        "inference_endpoints",
        sa.Column("mode", sa.Text(), server_default="sync"),
    )
    op.add_column(
        "inference_endpoints",
        sa.Column("health_endpoint", sa.Text(), server_default="/health"),
    )
    op.add_column(
        "inference_endpoints",
        sa.Column("api_format", sa.Text(), server_default="openai"),
    )
    op.add_column(
        "inference_endpoints",
        sa.Column("region", sa.Text(), server_default="ca-east"),
    )
    op.add_column(
        "inference_endpoints",
        sa.Column("worker_job_id", sa.Text(), nullable=True),
    )
    op.add_column(
        "inference_endpoints",
        sa.Column("total_cost_cad", sa.Float(), server_default="0"),
    )

    # Token tracking on inference results
    op.add_column(
        "inference_results",
        sa.Column("input_tokens", sa.Integer(), server_default="0"),
    )
    op.add_column(
        "inference_results",
        sa.Column("output_tokens", sa.Integer(), server_default="0"),
    )
    op.add_column(
        "inference_results",
        sa.Column("created_at", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("inference_results", "created_at")
    op.drop_column("inference_results", "output_tokens")
    op.drop_column("inference_results", "input_tokens")
    op.drop_column("inference_endpoints", "total_cost_cad")
    op.drop_column("inference_endpoints", "worker_job_id")
    op.drop_column("inference_endpoints", "region")
    op.drop_column("inference_endpoints", "api_format")
    op.drop_column("inference_endpoints", "health_endpoint")
    op.drop_column("inference_endpoints", "mode")
    op.drop_column("inference_endpoints", "docker_image")
