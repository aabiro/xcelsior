"""Durable, per-sink privacy deletion workflow.

Account erasure is an asynchronous cross-store operation.  These tables make
the request, its deadline, worker lease, and every sink outcome authoritative
instead of allowing an HTTP request to claim that deletion completed while a
dependency was unavailable.

Revision ID: 081
Revises: 080
Create Date: 2026-07-30
"""

from alembic import op

revision = "081"
down_revision = "080"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    # This table used to be created from an application request.  Alembic is
    # the only production DDL authority; preserve any live rows while bringing
    # the object under the migration ledger.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS user_encryption_keys (
            user_id TEXT PRIMARY KEY,
            fernet_key TEXT NOT NULL,
            created_at DOUBLE PRECISION NOT NULL
                DEFAULT extract(epoch FROM clock_timestamp()),
            destroyed_at DOUBLE PRECISION NOT NULL DEFAULT 0,
            active BOOLEAN NOT NULL DEFAULT TRUE
        )
        """
    )

    op.execute(
        """
        CREATE TABLE privacy_deletion_requests (
            request_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            subject_reference_hash TEXT NOT NULL,
            subject_user_id TEXT,
            subject_email TEXT,
            subject_customer_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
            requested_by TEXT NOT NULL,
            request_source TEXT NOT NULL DEFAULT 'self_service',
            legal_basis TEXT NOT NULL DEFAULT 'user_request',
            idempotency_key TEXT NOT NULL,
            status_token_hash TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'requested'
                CHECK (state IN (
                    'requested', 'validated', 'processing', 'verifying',
                    'completed', 'failed', 'cancelled'
                )),
            deadline_at TIMESTAMPTZ NOT NULL,
            claim_owner TEXT,
            claim_token UUID,
            claim_expires_at TIMESTAMPTZ,
            attempt_count INTEGER NOT NULL DEFAULT 0
                CHECK (attempt_count >= 0),
            next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            last_error TEXT,
            evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
            request_event_id UUID,
            created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            validated_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            UNIQUE (subject_reference_hash, idempotency_key),
            CHECK (jsonb_typeof(subject_customer_ids) = 'array')
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX privacy_one_active_request_per_subject
            ON privacy_deletion_requests (subject_reference_hash)
         WHERE state NOT IN ('completed', 'cancelled')
        """
    )
    op.execute(
        """
        CREATE INDEX privacy_deletion_requests_due
            ON privacy_deletion_requests (next_attempt_at, created_at)
         WHERE state NOT IN ('completed', 'cancelled', 'failed')
        """
    )
    op.execute(
        """
        CREATE INDEX privacy_deletion_requests_deadline
            ON privacy_deletion_requests (deadline_at)
         WHERE state NOT IN ('completed', 'cancelled')
        """
    )

    op.execute(
        """
        CREATE TABLE privacy_deletion_sink_status (
            request_id UUID NOT NULL
                REFERENCES privacy_deletion_requests(request_id)
                ON DELETE RESTRICT,
            sink TEXT NOT NULL
                CHECK (sink IN (
                    'authority', 'redis', 'artifacts', 'retrieval',
                    'analytics', 'posthog', 'verification'
                )),
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN (
                    'pending', 'in_progress', 'completed',
                    'not_applicable', 'legal_hold', 'failed'
                )),
            attempt_count INTEGER NOT NULL DEFAULT 0
                CHECK (attempt_count >= 0),
            deadline_at TIMESTAMPTZ NOT NULL,
            next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            last_error TEXT,
            external_reference TEXT,
            evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (request_id, sink)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX privacy_deletion_sink_due
            ON privacy_deletion_sink_status (next_attempt_at, request_id)
         WHERE status IN ('pending', 'failed')
        """
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")
    op.execute("DROP TABLE IF EXISTS privacy_deletion_sink_status")
    op.execute("DROP TABLE IF EXISTS privacy_deletion_requests")
    # Do not drop user_encryption_keys: it may predate this revision and
    # contains the only keys capable of decrypting retained audit payloads.
