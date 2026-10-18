"""Create schema storage and artifact catalog tables (Phase 8).

Adds:
- ``storage.artifacts``: authoritative catalog metadata.
- ``storage.artifact_upload_sessions``: upload state and idempotency.
- ``storage.artifact_replicas``: multi-provider replicas (B2/R2).
- ``storage.artifact_deletion_jobs``: asynchronous lifecycle deletion queue.

Revision ID: 064
Revises: 063
Create Date: 2026-07-20
"""

from typing import Sequence, Union
from alembic import op

revision: str = "064"
down_revision: Union[str, None] = "063"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Fail fast on lock contention
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    # Create storage schema
    op.execute("CREATE SCHEMA IF NOT EXISTS storage")

    # 1. artifacts table
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS storage.artifacts (
            artifact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id TEXT NOT NULL,
            owner_user_id TEXT,
            job_id TEXT REFERENCES jobs(job_id) ON DELETE SET NULL,
            attempt_id UUID REFERENCES job_attempts(attempt_id) ON DELETE SET NULL,
            artifact_type TEXT NOT NULL,
            logical_name TEXT NOT NULL,
            state TEXT NOT NULL CHECK (
                state IN (
                    'requested', 'upload_authorized', 'uploading',
                    'uploaded_unverified', 'available', 'expiring',
                    'delete_pending', 'deleted', 'corrupt', 'quarantined',
                    'abandoned', 'delete_failed'
                )
            ),
            primary_provider TEXT NOT NULL,
            primary_bucket TEXT NOT NULL,
            object_key TEXT NOT NULL,
            object_generation TEXT,
            content_type TEXT NOT NULL,
            size_bytes BIGINT,
            crc32c TEXT,
            sha256 TEXT,
            encryption_key_version TEXT,
            residency_region TEXT NOT NULL,
            retention_class TEXT NOT NULL,
            retain_until TIMESTAMPTZ,
            legal_hold BOOLEAN NOT NULL DEFAULT false,
            created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            available_at TIMESTAMPTZ,
            deleted_at TIMESTAMPTZ,
            version BIGINT NOT NULL DEFAULT 1,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            UNIQUE (primary_provider, primary_bucket, object_key),
            CHECK (size_bytes IS NULL OR size_bytes >= 0)
        )
        """
    )

    # Indexes on artifacts
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_artifacts_tenant_created
            ON storage.artifacts (tenant_id, created_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_artifacts_job
            ON storage.artifacts (tenant_id, job_id, artifact_type, created_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_artifacts_retention
            ON storage.artifacts (retain_until)
         WHERE state IN ('available', 'expiring') AND legal_hold = false
        """
    )

    # 2. artifact_upload_sessions table
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS storage.artifact_upload_sessions (
            upload_session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            artifact_id UUID NOT NULL REFERENCES storage.artifacts(artifact_id) ON DELETE CASCADE,
            tenant_id TEXT NOT NULL,
            principal_id TEXT NOT NULL,
            provider_upload_id TEXT,
            expected_size_bytes BIGINT,
            expected_sha256 TEXT,
            expires_at TIMESTAMPTZ NOT NULL,
            completed_at TIMESTAMPTZ,
            idempotency_key TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            UNIQUE (tenant_id, idempotency_key)
        )
        """
    )

    # 3. artifact_replicas table
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS storage.artifact_replicas (
            artifact_id UUID NOT NULL REFERENCES storage.artifacts(artifact_id) ON DELETE CASCADE,
            provider TEXT NOT NULL,
            bucket TEXT NOT NULL,
            object_key TEXT NOT NULL,
            generation TEXT,
            state TEXT NOT NULL,
            verified_at TIMESTAMPTZ,
            last_error TEXT,
            PRIMARY KEY (artifact_id, provider, bucket)
        )
        """
    )

    # 4. artifact_deletion_jobs table
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS storage.artifact_deletion_jobs (
            deletion_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            artifact_id UUID NOT NULL REFERENCES storage.artifacts(artifact_id) ON DELETE CASCADE,
            reason TEXT NOT NULL,
            requested_by TEXT NOT NULL,
            state TEXT NOT NULL CHECK (
                state IN ('requested', 'claimed', 'delete_pending', 'delete_failed', 'completed')
            ),
            attempt_count INTEGER NOT NULL DEFAULT 0,
            next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            last_error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            completed_at TIMESTAMPTZ
        )
        """
    )

    # Unique index for active deletion job per artifact
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_artifact_one_active_deletion
            ON storage.artifact_deletion_jobs (artifact_id)
         WHERE state IN ('requested', 'claimed', 'delete_pending', 'delete_failed')
        """
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")
    op.execute("DROP INDEX IF EXISTS storage.idx_artifact_one_active_deletion")
    op.execute("DROP TABLE IF EXISTS storage.artifact_deletion_jobs")
    op.execute("DROP TABLE IF EXISTS storage.artifact_replicas")
    op.execute("DROP TABLE IF EXISTS storage.artifact_upload_sessions")
    op.execute("DROP INDEX IF EXISTS storage.idx_artifacts_retention")
    op.execute("DROP INDEX IF EXISTS storage.idx_artifacts_job")
    op.execute("DROP INDEX IF EXISTS storage.idx_artifacts_tenant_created")
    op.execute("DROP TABLE IF EXISTS storage.artifacts")
    op.execute("DROP SCHEMA IF EXISTS storage CASCADE")
