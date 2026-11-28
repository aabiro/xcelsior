"""Record independently verified off-host copies of NFS volumes."""

import sqlalchemy as sa
from alembic import op

revision = "172"
down_revision = "171"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "volume_object_checkpoints",
        sa.Column(
            "volume_id",
            sa.Text(),
            sa.ForeignKey("volumes.volume_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("object_key", sa.Text(), primary_key=True),
        sa.Column("bucket", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("encrypted", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("size_bytes > 0", name="volume_object_checkpoint_positive_size"),
        sa.CheckConstraint("sha256 ~ '^[a-f0-9]{64}$'", name="volume_object_checkpoint_sha256"),
    )
    op.create_index(
        "volume_object_checkpoints_volume_time",
        "volume_object_checkpoints",
        ["volume_id", "created_at"],
    )


def downgrade():
    op.drop_table("volume_object_checkpoints")
