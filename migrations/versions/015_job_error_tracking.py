"""Add error_message and exit_code columns to jobs table.

Currently job failure details are buried in the JSONB payload, making
it impossible for the AI to directly answer "why did my job fail?".
These columns expose error info as first-class queryable fields that
the AI's get_job_details tool can return directly.

Revision ID: 015
Revises: 014
Create Date: 2026-04-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column("error_message", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "jobs",
        sa.Column("exit_code", sa.Integer(), nullable=True),
    )

    # Back-fill error_message from existing payload JSONB where available
    op.execute(
        """
        UPDATE jobs
        SET error_message = COALESCE(
            payload->>'error_message',
            payload->>'error',
            payload->>'failure_reason',
            ''
        )
        WHERE payload IS NOT NULL
          AND (
            payload ? 'error_message'
            OR payload ? 'error'
            OR payload ? 'failure_reason'
          )
        """
    )
    op.execute(
        """
        UPDATE jobs
        SET exit_code = (payload->>'exit_code')::integer
        WHERE payload IS NOT NULL
          AND payload ? 'exit_code'
          AND (payload->>'exit_code') ~ '^-?[0-9]+$'
        """
    )

    # Index to quickly find all failed jobs with actual error text
    op.create_index(
        "idx_jobs_error_message",
        "jobs",
        ["error_message"],
        postgresql_where=sa.text("error_message != ''"),
    )


def downgrade() -> None:
    op.drop_index("idx_jobs_error_message", table_name="jobs")
    op.drop_column("jobs", "exit_code")
    op.drop_column("jobs", "error_message")
