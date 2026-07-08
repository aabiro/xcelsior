"""store profile avatars in Postgres (durable across redeploys)

Avatars were written through the artifact manager, which — with
XCELSIOR_STORAGE_BACKEND=local and no XCELSIOR_STORAGE_LOCAL_DIR set — landed
in the repo/container ``artifacts/`` directory. That path is ephemeral, so the
``preferences.avatar_key`` survived in Postgres but the image file was wiped on
every redeploy and the UI fell back to the default avatar. Store the bytes in
Postgres instead; images are capped at 2 MB.

Revision ID: 051
Revises: 050
Create Date: 2026-07-07
"""

from alembic import op

revision = "051"
down_revision = "050"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS user_avatars (
            user_id TEXT PRIMARY KEY,
            content_type TEXT NOT NULL,
            data BYTEA NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL DEFAULT 0
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS user_avatars;")
