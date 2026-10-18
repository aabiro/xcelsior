"""drop pause/resume state; collapse into stop/start

Phase 2 of the RunPod-style lifecycle unification. The legacy
``user_paused`` and ``paused_low_balance`` job statuses are replaced
with a single ``stopped`` state + a ``payload.stop_reason`` string
("user" or "low_balance"). Stopped containers are still preserved by
the internal pause_container docker primitive, so no container state
is lost during migration.

Revision ID: 031
Revises: 030
Create Date: 2025-01-21
"""
from alembic import op

revision = "031"
down_revision = "030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE jobs
        SET status = 'stopped',
            payload = jsonb_set(
                payload,
                '{stop_reason}',
                CASE
                    WHEN status = 'paused_low_balance' THEN '"low_balance"'::jsonb
                    ELSE '"user"'::jsonb
                END,
                true
            )
        WHERE status IN ('user_paused', 'paused_low_balance');
        """
    )


def downgrade() -> None:
    # Best-effort reverse: instances that carry stop_reason get mapped back.
    # Stopped rows without a reason stay as 'stopped'.
    op.execute(
        """
        UPDATE jobs
        SET status = CASE
            WHEN payload->>'stop_reason' = 'low_balance' THEN 'paused_low_balance'
            WHEN payload->>'stop_reason' = 'user'        THEN 'user_paused'
            ELSE status
        END
        WHERE status = 'stopped' AND payload ? 'stop_reason';
        """
    )
