"""Add key_ciphertext column to volumes table for LUKS encryption key storage.

Stores the Fernet-encrypted LUKS key material per volume.
Empty string = unencrypted volume (backward compatible).

Revision ID: 022
Revises: 021
Create Date: 2026-04-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "022"
down_revision: Union[str, None] = "021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "volumes",
        sa.Column("key_ciphertext", sa.Text(), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("volumes", "key_ciphertext")
