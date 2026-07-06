"""Add cached-token and TTFT columns for serverless token SKU SLOs.

Revision ID: 047
"""

from alembic import op

revision = "047"
down_revision = "046"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE serverless_jobs "
        "ADD COLUMN IF NOT EXISTS cached_tokens INTEGER NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE serverless_jobs "
        "ADD COLUMN IF NOT EXISTS ttft_ms INTEGER NOT NULL DEFAULT 0"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS serverless_token_ledger (
            ledger_id TEXT PRIMARY KEY,
            endpoint_id TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cached_tokens INTEGER NOT NULL DEFAULT 0,
            cost_cad DOUBLE PRECISION NOT NULL DEFAULT 0,
            created_at DOUBLE PRECISION NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_serverless_token_ledger_idem
        ON serverless_token_ledger (endpoint_id, idempotency_key)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_serverless_token_ledger_idem")
    op.execute("DROP TABLE IF EXISTS serverless_token_ledger")
    op.execute("ALTER TABLE serverless_jobs DROP COLUMN IF EXISTS ttft_ms")
    op.execute("ALTER TABLE serverless_jobs DROP COLUMN IF EXISTS cached_tokens")