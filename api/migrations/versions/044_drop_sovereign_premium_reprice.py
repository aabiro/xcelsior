"""Drop the sovereign tier price premium and reprice high cards toward market.

Phase 1 pricing: we compete on price/DX, not a sovereignty surcharge.
  1. Sovereign tier no longer costs more than standard (was base × 1.43).
  2. Reprice A100 (40/80, SXM/PCIe) and RTX A6000 toward Vast/RunPod levels.

Mirrors _GPU_PRICING_BASE / _TIER_MULT in db.py; updates existing rows in the
live ``gpu_pricing`` table (the seed uses ON CONFLICT DO NOTHING, so code-only
changes don't touch already-seeded rows).

Revision ID: 044
"""

from alembic import op

revision = "044"
down_revision = "043"
branch_labels = None
depends_on = None

# Cards repriced toward market: (gpu_model, vram_gb, form_factor, high_frequency, new_base_cad)
_REPRICED: list[tuple[str, int, str, bool, float]] = [
    ("A100",       80, "SXM",  False, 2.00),  # was 2.60
    ("A100",       80, "PCIe", False, 1.70),  # was 2.20
    ("A100",       40, "SXM",  False, 1.40),  # was 1.80
    ("A100",       40, "PCIe", False, 1.20),  # was 1.50
    ("RTX A6000",  48, "PCIe", False, 1.10),  # was 1.60
]
# Priced tiers (sovereign handled separately — it inherits the standard rate).
_PRICED_TIERS = {"standard": 1.0, "premium": 1.30}
_MODES = {
    "on_demand":    1.0,
    "spot":         0.40,
    "reserved_1mo": 0.80,
    "reserved_3mo": 0.70,
    "reserved_1yr": 0.55,
}

# Old base rates, for downgrade.
_OLD_BASE: dict[tuple[str, int, str, bool], float] = {
    ("A100", 80, "SXM", False): 2.60,
    ("A100", 80, "PCIe", False): 2.20,
    ("A100", 40, "SXM", False): 1.80,
    ("A100", 40, "PCIe", False): 1.50,
    ("RTX A6000", 48, "PCIe", False): 1.60,
}


def _reprice(base_for: dict[tuple[str, int, str, bool], float]) -> None:
    """Recompute standard/premium rows for the repriced cards from a base map."""
    for (model, vram, ff, hf), base in base_for.items():
        for tier, tm in _PRICED_TIERS.items():
            for mode, mm in _MODES.items():
                rate = round(base * tm * mm, 4)
                op.execute(
                    f"""
                    UPDATE gpu_pricing
                       SET base_rate_cad = {rate}
                     WHERE gpu_model = '{model}' AND vram_gb = {vram}
                       AND form_factor = '{ff}' AND high_frequency = {'TRUE' if hf else 'FALSE'}
                       AND tier = '{tier}' AND pricing_mode = '{mode}'
                    """
                )


# Sovereign rows inherit the standard rate (premium dropped); sovereignty_premium → 0.
_SOVEREIGN_TO_STANDARD = """
    UPDATE gpu_pricing s
       SET base_rate_cad = std.base_rate_cad,
           sovereignty_premium = 0.0
      FROM gpu_pricing std
     WHERE s.tier = 'sovereign' AND std.tier = 'standard'
       AND s.gpu_model = std.gpu_model AND s.vram_gb = std.vram_gb
       AND s.form_factor = std.form_factor AND s.high_frequency = std.high_frequency
       AND s.pricing_mode = std.pricing_mode
"""


def upgrade() -> None:
    # 1) Reprice standard/premium rows for the changed cards.
    _reprice({(m, v, f, h): b for (m, v, f, h, b) in _REPRICED})
    # 2) Drop the sovereign premium for ALL cards (sovereign := standard rate).
    op.execute(_SOVEREIGN_TO_STANDARD)


def downgrade() -> None:
    # Restore old card base rates for standard/premium.
    _reprice(_OLD_BASE)
    # Restore the sovereign premium (1.43 = base × 1.43 × mode) from standard rows
    # (standard = base × 1.0 × mode, so sovereign = standard × 1.43), and the +0.10
    # sovereignty_premium flag.
    op.execute(
        """
        UPDATE gpu_pricing s
           SET base_rate_cad = round((std.base_rate_cad * 1.43)::numeric, 4),
               sovereignty_premium = 0.10
          FROM gpu_pricing std
         WHERE s.tier = 'sovereign' AND std.tier = 'standard'
           AND s.gpu_model = std.gpu_model AND s.vram_gb = std.vram_gb
           AND s.form_factor = std.form_factor AND s.high_frequency = std.high_frequency
           AND s.pricing_mode = std.pricing_mode
        """
    )
