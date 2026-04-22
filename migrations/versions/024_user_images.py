"""P3.1 — user_images table for pod save-as-template snapshots.

Records metadata for `docker commit`-style container snapshots that users
save to turn a running instance into a reusable image. Actual image data
is stored locally on the GPU host (v1) or pushed to a per-user registry
namespace once ``XCELSIOR_REGISTRY_URL`` is configured.

Idempotent — db.py also CREATEs this table at startup for fresh installs.

Revision ID: 024
Revises: 023
Create Date: 2026-04-22
"""

from typing import Sequence, Union

from alembic import op


revision: str = "024"
down_revision: Union[str, None] = "023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS user_images (
            image_id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            name TEXT NOT NULL,
            tag TEXT NOT NULL DEFAULT 'latest',
            description TEXT DEFAULT '',
            source_job_id TEXT,
            host_id TEXT,
            image_ref TEXT NOT NULL,
            size_bytes BIGINT DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at DOUBLE PRECISION NOT NULL,
            deleted_at DOUBLE PRECISION DEFAULT 0,
            UNIQUE (owner_id, name, tag)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_images_owner "
        "ON user_images (owner_id, deleted_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_images_source_job "
        "ON user_images (source_job_id) WHERE source_job_id IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_user_images_source_job")
    op.execute("DROP INDEX IF EXISTS idx_user_images_owner")
    op.execute("DROP TABLE IF EXISTS user_images")
