"""Hash-only callback credentials bound to one native worker attempt."""

from alembic import op

revision = "126"
down_revision = "125"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout='5s'")
    op.execute("""CREATE TABLE serverless_attempt_credentials (
        attempt_id UUID PRIMARY KEY REFERENCES job_attempts(attempt_id) ON DELETE CASCADE,
        worker_id TEXT NOT NULL REFERENCES serverless_workers(worker_id) ON DELETE CASCADE,
        tenant_id TEXT NOT NULL CHECK (tenant_id<>''),
        token_hash TEXT NOT NULL UNIQUE CHECK (length(token_hash)=64),
        issued_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        expires_at TIMESTAMPTZ NOT NULL,
        revoked_at TIMESTAMPTZ
    )""")
    op.execute(
        "CREATE INDEX serverless_attempt_credentials_expiry ON serverless_attempt_credentials(expires_at)"
    )
    op.execute(
        "CREATE INDEX serverless_attempt_credentials_tenant ON serverless_attempt_credentials(tenant_id,attempt_id)"
    )


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout='5s'")
    op.execute("DROP TABLE serverless_attempt_credentials")
