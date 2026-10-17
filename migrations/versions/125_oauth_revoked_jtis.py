"""Remember a revoked machine-token id until that token would have expired.

A client-credentials JWT is signed for up to 90 days. Disabling the client
or rotating its secret invalidates every token. This table lets one stolen
token be rejected without waiting for expiry and without replacing the others.
"""

from alembic import op

revision = "125"
down_revision = "124"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("""
        CREATE TABLE IF NOT EXISTS oauth_revoked_jtis (
            jti TEXT PRIMARY KEY,
            client_id TEXT NOT NULL,
            expires_at DOUBLE PRECISION NOT NULL,
            revoked_at DOUBLE PRECISION NOT NULL
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS oauth_revoked_jtis_expires_at "
        "ON oauth_revoked_jtis (expires_at)"
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("DROP TABLE IF EXISTS oauth_revoked_jtis")
