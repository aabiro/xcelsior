"""Serverless inference endpoints — data model.

Creates serverless_endpoints, serverless_workers, serverless_jobs,
serverless_job_stream_events, serverless_api_keys with queue indexes.

Revision ID: 037
Revises: 036
Create Date: 2026-06-08
"""

from typing import Sequence, Union

from alembic import op

revision: str = "037"
down_revision: Union[str, None] = "036"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS serverless_endpoints (
            endpoint_id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            name TEXT NOT NULL DEFAULT '',
            mode TEXT NOT NULL DEFAULT 'preset',
            managed_engine TEXT NOT NULL DEFAULT 'vllm',
            model_ref TEXT NOT NULL DEFAULT '',
            model_revision TEXT NOT NULL DEFAULT 'main',
            image_ref TEXT NOT NULL DEFAULT '',
            startup_command TEXT NOT NULL DEFAULT '',
            http_port INTEGER NOT NULL DEFAULT 8080,
            health_check_path TEXT NOT NULL DEFAULT '/health',
            cuda_version TEXT NOT NULL DEFAULT '12.4',
            registry_auth_ref TEXT NOT NULL DEFAULT '',
            gpu_tier TEXT NOT NULL DEFAULT '',
            gpu_count INTEGER NOT NULL DEFAULT 1,
            vram_required_gb DOUBLE PRECISION NOT NULL DEFAULT 0,
            min_workers INTEGER NOT NULL DEFAULT 0,
            max_workers INTEGER NOT NULL DEFAULT 4,
            max_concurrency INTEGER NOT NULL DEFAULT 4,
            idle_timeout_sec INTEGER NOT NULL DEFAULT 300,
            scaling_policy_type TEXT NOT NULL DEFAULT 'queue_request_count',
            scaling_policy_value INTEGER NOT NULL DEFAULT 1,
            request_timeout_sec INTEGER NOT NULL DEFAULT 120,
            max_request_bytes BIGINT NOT NULL DEFAULT 10485760,
            keep_warm BOOLEAN NOT NULL DEFAULT FALSE,
            cache_volume_id TEXT,
            region TEXT NOT NULL DEFAULT 'ca-east',
            env JSONB NOT NULL DEFAULT '{}'::jsonb,
            status TEXT NOT NULL DEFAULT 'provisioning',
            total_requests BIGINT NOT NULL DEFAULT 0,
            total_gpu_seconds BIGINT NOT NULL DEFAULT 0,
            total_cost_cad DOUBLE PRECISION NOT NULL DEFAULT 0,
            created_at DOUBLE PRECISION NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL,
            deleted_at DOUBLE PRECISION NOT NULL DEFAULT 0
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_serverless_endpoints_owner "
        "ON serverless_endpoints (owner_id, deleted_at, created_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_serverless_endpoints_status "
        "ON serverless_endpoints (status) WHERE deleted_at = 0"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS serverless_workers (
            worker_id TEXT PRIMARY KEY,
            endpoint_id TEXT NOT NULL REFERENCES serverless_endpoints(endpoint_id),
            scheduler_job_id TEXT,
            state TEXT NOT NULL DEFAULT 'booting',
            host_id TEXT NOT NULL DEFAULT '',
            gpu_count INTEGER NOT NULL DEFAULT 1,
            current_concurrency INTEGER NOT NULL DEFAULT 0,
            allocated_at DOUBLE PRECISION,
            released_at DOUBLE PRECISION,
            last_heartbeat_at DOUBLE PRECISION NOT NULL DEFAULT 0,
            error_message TEXT NOT NULL DEFAULT '',
            created_at DOUBLE PRECISION NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_serverless_workers_endpoint_state "
        "ON serverless_workers (endpoint_id, state)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_serverless_workers_scheduler_job "
        "ON serverless_workers (scheduler_job_id) WHERE scheduler_job_id IS NOT NULL"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS serverless_jobs (
            job_id TEXT PRIMARY KEY,
            endpoint_id TEXT NOT NULL REFERENCES serverless_endpoints(endpoint_id),
            worker_id TEXT REFERENCES serverless_workers(worker_id),
            owner_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'IN_QUEUE',
            payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            output JSONB,
            error TEXT,
            idempotency_key TEXT,
            webhook_url TEXT,
            queued_at DOUBLE PRECISION NOT NULL,
            started_at DOUBLE PRECISION,
            finished_at DOUBLE PRECISION,
            cold_start_seconds INTEGER NOT NULL DEFAULT 0,
            gpu_seconds INTEGER NOT NULL DEFAULT 0,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cost_cad DOUBLE PRECISION NOT NULL DEFAULT 0,
            created_at DOUBLE PRECISION NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_serverless_jobs_queue "
        "ON serverless_jobs (endpoint_id, status, queued_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_serverless_jobs_metrics "
        "ON serverless_jobs (endpoint_id, finished_at) "
        "WHERE finished_at IS NOT NULL"
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_serverless_jobs_idempotency
        ON serverless_jobs (endpoint_id, idempotency_key)
        WHERE idempotency_key IS NOT NULL
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS serverless_job_stream_events (
            job_id TEXT NOT NULL REFERENCES serverless_jobs(job_id) ON DELETE CASCADE,
            seq_no INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (job_id, seq_no)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS serverless_api_keys (
            key_id TEXT PRIMARY KEY,
            endpoint_id TEXT REFERENCES serverless_endpoints(endpoint_id),
            owner_id TEXT NOT NULL,
            name TEXT NOT NULL DEFAULT 'default',
            key_prefix TEXT NOT NULL,
            key_hash TEXT NOT NULL,
            scopes TEXT NOT NULL DEFAULT 'inference:write',
            rate_limit_rpm INTEGER NOT NULL DEFAULT 60,
            last_used_at DOUBLE PRECISION NOT NULL DEFAULT 0,
            revoked_at DOUBLE PRECISION NOT NULL DEFAULT 0,
            created_at DOUBLE PRECISION NOT NULL
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_serverless_api_keys_hash "
        "ON serverless_api_keys (key_hash) WHERE revoked_at = 0"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_serverless_api_keys_endpoint "
        "ON serverless_api_keys (endpoint_id) WHERE revoked_at = 0"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_serverless_api_keys_endpoint")
    op.execute("DROP INDEX IF EXISTS idx_serverless_api_keys_hash")
    op.execute("DROP TABLE IF EXISTS serverless_api_keys")

    op.execute("DROP TABLE IF EXISTS serverless_job_stream_events")

    op.execute("DROP INDEX IF EXISTS idx_serverless_jobs_idempotency")
    op.execute("DROP INDEX IF EXISTS idx_serverless_jobs_metrics")
    op.execute("DROP INDEX IF EXISTS idx_serverless_jobs_queue")
    op.execute("DROP TABLE IF EXISTS serverless_jobs")

    op.execute("DROP INDEX IF EXISTS idx_serverless_workers_scheduler_job")
    op.execute("DROP INDEX IF EXISTS idx_serverless_workers_endpoint_state")
    op.execute("DROP TABLE IF EXISTS serverless_workers")

    op.execute("DROP INDEX IF EXISTS idx_serverless_endpoints_status")
    op.execute("DROP INDEX IF EXISTS idx_serverless_endpoints_owner")
    op.execute("DROP TABLE IF EXISTS serverless_endpoints")