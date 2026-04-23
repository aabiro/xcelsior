"""Canonicalize GPU names: add variant columns & rebuild unique index.

Adds ``form_factor`` and ``high_frequency`` columns to ``gpu_pricing`` so
variants (SXM vs PCIe, high-frequency CPU pairings) can be priced
independently without encoding them in the display name. Clears legacy
``NVIDIA GeForce …`` rows so the startup seed re-populates with canonical
short titles (e.g. ``RTX 4090``, ``A100``).

Revision ID: 030
Revises: 029
Create Date: 2026-04-23
"""

from typing import Sequence, Union

from alembic import op


revision: str = "030"
down_revision: Union[str, None] = "029"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # New variant columns — default values make the migration safe on
    # existing rows but those rows are deleted below anyway.
    op.execute(
        "ALTER TABLE gpu_pricing "
        "ADD COLUMN IF NOT EXISTS form_factor TEXT NOT NULL DEFAULT 'PCIe'"
    )
    op.execute(
        "ALTER TABLE gpu_pricing "
        "ADD COLUMN IF NOT EXISTS high_frequency BOOLEAN NOT NULL DEFAULT FALSE"
    )

    # Drop the old unique constraint/index so we can rebuild with the
    # variant columns included. The constraint name follows Postgres'
    # default naming for UNIQUE table constraints.
    op.execute(
        "ALTER TABLE gpu_pricing "
        "DROP CONSTRAINT IF EXISTS gpu_pricing_gpu_model_tier_pricing_mode_key"
    )
    # Some environments created it as a plain index rather than a named
    # constraint; drop that too if present.
    op.execute(
        "DROP INDEX IF EXISTS gpu_pricing_gpu_model_tier_pricing_mode_key"
    )

    # Wipe legacy rows (``NVIDIA GeForce RTX …`` etc.) — startup seed in
    # db.py will repopulate with canonical short names.
    op.execute("DELETE FROM gpu_pricing")

    # Rebuild the unique constraint including the new variant columns.
    op.execute(
        "ALTER TABLE gpu_pricing "
        "ADD CONSTRAINT gpu_pricing_variant_unique "
        "UNIQUE (gpu_model, vram_gb, form_factor, high_frequency, tier, pricing_mode)"
    )

    # Replace the active-filter index to include variant columns so
    # catalog lookups hit it cleanly.
    op.execute("DROP INDEX IF EXISTS idx_gpu_pricing_model")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_gpu_pricing_model "
        "ON gpu_pricing (gpu_model, vram_gb, form_factor, high_frequency, tier, pricing_mode) "
        "WHERE active = TRUE"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_gpu_pricing_model")
    op.execute(
        "ALTER TABLE gpu_pricing "
        "DROP CONSTRAINT IF EXISTS gpu_pricing_variant_unique"
    )
    op.execute(
        "ALTER TABLE gpu_pricing "
        "DROP COLUMN IF EXISTS high_frequency"
    )
    op.execute(
        "ALTER TABLE gpu_pricing "
        "DROP COLUMN IF EXISTS form_factor"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_gpu_pricing_model "
        "ON gpu_pricing (gpu_model, tier, pricing_mode) WHERE active = TRUE"
    )
    op.execute(
        "ALTER TABLE gpu_pricing "
        "ADD CONSTRAINT gpu_pricing_gpu_model_tier_pricing_mode_key "
        "UNIQUE (gpu_model, tier, pricing_mode)"
    )
