"""Semantic cache, batch jobs, and prefix affinity tables.

Revision ID: 049
"""

from alembic import op

revision = "049"
down_revision = "048"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS serverless_semantic_cache (
            cache_id TEXT PRIMARY KEY,
            endpoint_id TEXT NOT NULL,
            prompt_norm TEXT NOT NULL,
            prompt_fingerprint TEXT NOT NULL,
            response_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            usage_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at DOUBLE PRECISION NOT NULL,
            UNIQUE (endpoint_id, prompt_fingerprint)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_semantic_cache_endpoint "
        "ON serverless_semantic_cache (endpoint_id, created_at DESC)"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS serverless_batches (
            batch_id TEXT PRIMARY KEY,
            endpoint_id TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'validating',
            requests_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            results_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            input_count INTEGER NOT NULL DEFAULT 0,
            completed_count INTEGER NOT NULL DEFAULT 0,
            failed_count INTEGER NOT NULL DEFAULT 0,
            discount_rate DOUBLE PRECISION NOT NULL DEFAULT 0.5,
            completion_window TEXT NOT NULL DEFAULT '24h',
            created_at DOUBLE PRECISION NOT NULL,
            completed_at DOUBLE PRECISION
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS serverless_prefix_affinity (
            endpoint_id TEXT NOT NULL,
            prefix_hash TEXT NOT NULL,
            worker_id TEXT NOT NULL,
            last_seen_at DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (endpoint_id, prefix_hash)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS serverless_cache_savings (
            savings_id TEXT PRIMARY KEY,
            endpoint_id TEXT NOT NULL,
            saved_cost_cad DOUBLE PRECISION NOT NULL DEFAULT 0,
            similarity DOUBLE PRECISION NOT NULL DEFAULT 1.0,
            created_at DOUBLE PRECISION NOT NULL
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS serverless_cache_savings")
    op.execute("DROP TABLE IF EXISTS serverless_prefix_affinity")
    op.execute("DROP TABLE IF EXISTS serverless_batches")
    op.execute("DROP INDEX IF EXISTS idx_semantic_cache_endpoint")
    op.execute("DROP TABLE IF EXISTS serverless_semantic_cache")