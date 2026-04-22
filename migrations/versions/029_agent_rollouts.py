"""P1.2 — agent_rollouts table for auto-rollback watchdog.

Revision ID: 029
Revises: 028
Create Date: 2026-04-22
"""

from typing import Sequence, Union

from alembic import op


revision: str = "029"
down_revision: Union[str, None] = "028"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_rollouts (
            id             BIGSERIAL PRIMARY KEY,
            host_id        TEXT NOT NULL,
            from_sha       TEXT,
            target_sha     TEXT NOT NULL,
            enqueued_at    DOUBLE PRECISION NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW()),
            completed_at   DOUBLE PRECISION NULL,
            status         TEXT NOT NULL DEFAULT 'pending',
            last_check_at  DOUBLE PRECISION NULL,
            error          TEXT NULL
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_rollouts_pending "
        "ON agent_rollouts (enqueued_at) WHERE status = 'pending'"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_rollouts_host "
        "ON agent_rollouts (host_id, enqueued_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_agent_rollouts_host")
    op.execute("DROP INDEX IF EXISTS idx_agent_rollouts_pending")
    op.execute("DROP TABLE IF EXISTS agent_rollouts")
