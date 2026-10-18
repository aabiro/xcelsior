"""P3/Phase-E/E7 — registry_health_cache table for cross-process probe results.

Revision ID: 028
Revises: 027
Create Date: 2026-04-22
"""

from typing import Sequence, Union

from alembic import op


revision: str = "028"
down_revision: Union[str, None] = "027"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS registry_health_cache (
            registry      TEXT PRIMARY KEY,
            reachable     BOOLEAN NOT NULL DEFAULT FALSE,
            last_probe_at DOUBLE PRECISION NOT NULL DEFAULT 0,
            latency_ms    DOUBLE PRECISION NOT NULL DEFAULT 0,
            status_code   INTEGER NULL,
            error         TEXT NULL
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS registry_health_cache")
