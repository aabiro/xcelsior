"""add spot_enabled + spot_min_cents to gpu_offers

The marketplace API and dashboard moved to an enable + floor spot model
(spot_enabled bool, spot_min_cents int), but gpu_offers only had the older
spot_multiplier column — so POST /api/v2/marketplace/offers passed unknown
kwargs to upsert_offer and never worked. These columns persist the newer
model; spot_multiplier is kept as the discount factor, floored by
spot_min_cents at allocation time.

Revision ID: 034
Revises: 033
Create Date: 2026-06-02
"""

from alembic import op

revision = "034"
down_revision = "033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE gpu_offers "
        "ADD COLUMN IF NOT EXISTS spot_enabled BOOLEAN NOT NULL DEFAULT true;"
    )
    op.execute(
        "ALTER TABLE gpu_offers "
        "ADD COLUMN IF NOT EXISTS spot_min_cents INTEGER NOT NULL DEFAULT 0;"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE gpu_offers DROP COLUMN IF EXISTS spot_min_cents;")
    op.execute("ALTER TABLE gpu_offers DROP COLUMN IF EXISTS spot_enabled;")
