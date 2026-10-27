"""Remember which Facebook account signs in, and the public deletion receipt.

Facebook's data-deletion callback names a Meta user id. The account row has to
store that id or the callback cannot find the person. The confirmation table
lets https://xcelsior.ca/delete-data show the result of a code without a session.
"""

from alembic import op

revision = "145"
down_revision = "144"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS oauth_subject TEXT NOT NULL DEFAULT ''"
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS users_oauth_subject_unique
            ON users (oauth_subject)
         WHERE oauth_subject <> ''
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS facebook_deletion_confirmations (
            code_hash TEXT PRIMARY KEY,
            request_id UUID,
            outcome TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
        )
        """
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("DROP TABLE IF EXISTS facebook_deletion_confirmations")
    op.execute("DROP INDEX IF EXISTS users_oauth_subject_unique")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS oauth_subject")
