"""Settlement status, Stripe meter outbox, competitive GPU reprice.

- payout_splits.settlement_status / settlement_error for daily settlement queue
- stripe_meter_event_outbox for Billing Meters dual-write
- Reprice key GPUs toward global sweet spot (below RunPod, above rock-bottom Vast)

Revision ID: 079
"""

from alembic import op

revision = "079"
down_revision = "078"
branch_labels = None
depends_on = None

# Competitive CAD on-demand base rates (standard tier).
# Anchors: undercut RunPod community cloud list, stay profitable vs Vast floor.
# USD→CAD ~1.37; platform cut separate. Values are customer-facing list rates.
_REPRICED: list[tuple[str, int, str, bool, float]] = [
    ("B200", 192, "OAM", False, 9.50),
    ("H200", 141, "SXM", False, 4.20),
    ("H100", 80, "SXM", False, 3.10),
    ("H100", 80, "PCIe", False, 2.60),
    ("H100 NVL", 94, "PCIe", False, 2.90),
    ("A100", 80, "SXM", False, 1.85),
    ("A100", 80, "PCIe", False, 1.55),
    ("L40S", 48, "PCIe", False, 1.55),
    ("RTX 4090", 24, "PCIe", False, 0.49),
    ("RTX 4090", 24, "PCIe", True, 0.55),
    ("RTX 3090", 24, "PCIe", False, 0.28),
    ("RTX 3060", 12, "PCIe", False, 0.12),
    ("RTX 3060 Ti", 8, "PCIe", False, 0.14),
    ("RTX 2060", 6, "PCIe", False, 0.08),
    ("RTX 2060 Super", 8, "PCIe", False, 0.10),
]

_PRICED_TIERS = {"standard": 1.0, "premium": 1.30, "sovereign": 1.0}
_MODES = {
    "on_demand": 1.0,
    "spot": 0.40,
    "reserved_1mo": 0.80,
    "reserved_3mo": 0.70,
    "reserved_1yr": 0.55,
}


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE payout_splits
          ADD COLUMN IF NOT EXISTS settlement_status TEXT DEFAULT 'pending',
          ADD COLUMN IF NOT EXISTS settlement_error TEXT DEFAULT '',
          ADD COLUMN IF NOT EXISTS payment_rail TEXT DEFAULT 'stripe'
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_payout_splits_settlement
          ON payout_splits (settlement_status, created_at)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS stripe_meter_event_outbox (
            event_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            event_name TEXT NOT NULL,
            value DOUBLE PRECISION NOT NULL DEFAULT 1,
            payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            idempotency_key TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT NOT NULL DEFAULT '',
            created_at DOUBLE PRECISION NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL,
            sent_at DOUBLE PRECISION
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_meter_outbox_pending
          ON stripe_meter_event_outbox (status, created_at)
          WHERE status = 'pending'
        """
    )

    for model, vram, ff, hf, base in _REPRICED:
        for tier, tm in _PRICED_TIERS.items():
            for mode, mm in _MODES.items():
                rate = round(base * tm * mm, 4)
                op.execute(
                    f"""
                    UPDATE gpu_pricing
                       SET base_rate_cad = {rate}
                     WHERE gpu_model = '{model}' AND vram_gb = {vram}
                       AND form_factor = '{ff}'
                       AND high_frequency = {'TRUE' if hf else 'FALSE'}
                       AND tier = '{tier}' AND pricing_mode = '{mode}'
                    """
                )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS stripe_meter_event_outbox")
    op.execute("DROP INDEX IF EXISTS idx_payout_splits_settlement")
    # Leave settlement columns in place (non-destructive).
