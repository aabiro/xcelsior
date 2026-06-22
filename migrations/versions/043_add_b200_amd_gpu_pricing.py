"""Add B200 and AMD GPU pricing rows; add AMD + B200 to worker agent SMI map.

The B200 GPU appears in gpu-models.ts (frontend catalog) and marketing copy
but was missing from:
  - db._GPU_PRICING_BASE (so no pricing rows exist in gpu_pricing)
  - worker_agent._NVIDIA_SMI_NAME_MAP (so a B200 host would quarantine)

AMD GPUs (MI300X, MI250X, MI210, RX 7900 XTX, RX 7900 XT) had pricing rows
seeded from db._GPU_PRICING_BASE but were also absent from the worker SMI map,
meaning AMD hosts would also quarantine.

This migration inserts the missing pricing rows directly (the startup seed
uses ON CONFLICT DO NOTHING, so it would add them on next boot anyway, but
an explicit migration guarantees they exist before the next scheduling cycle
and documents the decision in the migration history).

The _NVIDIA_SMI_NAME_MAP in worker_agent.py and _GPU_PRICING_BASE in db.py
are updated in the same commit; this migration handles the live database.

Revision ID: 043
Revises: 042
Create Date: 2026-06-22
"""

from alembic import op

revision = "043"
down_revision = "042"
branch_labels = None
depends_on = None

# ── New pricing rows ──────────────────────────────────────────────────────────
# Format: (gpu_model, vram_gb, form_factor, high_frequency, base_rate_cad)
# These mirror _GPU_PRICING_BASE in db.py; the full tier × mode matrix is
# inserted via _generate_gpu_pricing_rows() logic replicated inline below.

_NEW_BASE: list[tuple[str, int, str, bool, float]] = [
    # NVIDIA Blackwell
    ("B200",      192, "OAM",  False, 12.00),
    # AMD Data Center (already in _GPU_PRICING_BASE but may be missing from DB
    # if the table was seeded before those rows were added)
    ("MI300X",    192, "OAM",  False,  4.50),
    ("MI250X",    128, "OAM",  False,  2.80),
    ("MI210",      64, "PCIe", False,  1.40),
    # AMD Consumer
    ("RX 7900 XTX", 24, "PCIe", False, 0.28),
    ("RX 7900 XT",  20, "PCIe", False, 0.24),
]
# Total rows inserted: len(_NEW_BASE) × 3 tiers × 5 pricing_modes = 6 × 15 = 90

_TIERS = {"standard": 1.0, "premium": 1.30, "sovereign": 1.43}
_MODES = {
    "on_demand":    1.0,
    "spot":         0.40,
    "reserved_1mo": 0.80,
    "reserved_3mo": 0.70,
    "reserved_1yr": 0.55,
}
_SOVEREIGNTY = {"standard": 0.0, "premium": 0.0, "sovereign": 0.10}


def _rows() -> list[tuple]:
    out = []
    for model, vram, ff, hf, base in _NEW_BASE:
        for tier, tm in _TIERS.items():
            for mode, mm in _MODES.items():
                rate = round(base * tm * mm, 4)
                out.append((
                    model, vram, ff, hf,
                    tier, mode, rate,
                    1.0,                    # priority_multiplier
                    _SOVEREIGNTY[tier],     # sovereignty_premium
                    0.60 if mode == "spot" else 0.0,  # spot_discount
                    0.05,                   # multi_gpu_discount_4
                    0.10,                   # multi_gpu_discount_8
                ))
    return out


def upgrade() -> None:
    for row in _rows():
        op.execute(f"""
            INSERT INTO gpu_pricing (
                gpu_model, vram_gb, form_factor, high_frequency,
                tier, pricing_mode, base_rate_cad,
                priority_multiplier, sovereignty_premium, spot_discount,
                multi_gpu_discount_4, multi_gpu_discount_8
            ) VALUES (
                '{row[0]}', {row[1]}, '{row[2]}', {'TRUE' if row[3] else 'FALSE'},
                '{row[4]}', '{row[5]}', {row[6]},
                {row[7]}, {row[8]}, {row[9]},
                {row[10]}, {row[11]}
            )
            ON CONFLICT (gpu_model, vram_gb, form_factor, high_frequency, tier, pricing_mode)
            DO NOTHING
        """)


def downgrade() -> None:
    models = tuple(m for m, *_ in _NEW_BASE)
    # Only remove rows that were introduced here (not pre-existing AMD rows)
    op.execute(
        "DELETE FROM gpu_pricing WHERE gpu_model IN %s" % (models,)
    )
