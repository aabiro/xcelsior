"""Add volume_snapshots table for P2.5 instant CoW snapshots.

Stores metadata for reflink-based snapshots of LUKS-encrypted (or plain)
persistent volumes. The actual snapshot data lives on NFS at
``{NFS_EXPORT_BASE}/_snapshots/{volume_id}/{snapshot_id}.img`` (encrypted)
or ``…/{snapshot_id}/`` (unencrypted).

Idempotent — db.py also runs CREATE TABLE IF NOT EXISTS at startup, so
fresh installs may already have the table when alembic runs.

Revision ID: 023
Revises: 022
Create Date: 2026-04-22
"""

from typing import Sequence, Union

from alembic import op


revision: str = "023"
down_revision: Union[str, None] = "022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS volume_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            volume_id TEXT NOT NULL REFERENCES volumes(volume_id),
            owner_id TEXT NOT NULL,
            label TEXT DEFAULT '',
            size_bytes BIGINT DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'ready',
            created_at DOUBLE PRECISION NOT NULL,
            deleted_at DOUBLE PRECISION DEFAULT 0
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_volume_snapshots_volume "
        "ON volume_snapshots (volume_id, deleted_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_volume_snapshots_owner "
        "ON volume_snapshots (owner_id, deleted_at)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_volume_snapshots_owner")
    op.execute("DROP INDEX IF EXISTS idx_volume_snapshots_volume")
    op.execute("DROP TABLE IF EXISTS volume_snapshots")
