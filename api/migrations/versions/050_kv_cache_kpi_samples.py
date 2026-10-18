"""KV-cache hit KPI samples for token SKU GA tracking.

Revision ID: 050
"""

from alembic import op

revision = "050"
down_revision = "049"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS serverless_kv_cache_samples (
            sample_id BIGSERIAL PRIMARY KEY,
            sample_ts DOUBLE PRECISION NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            cached_tokens INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_kv_cache_samples_ts "
        "ON serverless_kv_cache_samples (sample_ts DESC)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS serverless_kv_cache_samples")