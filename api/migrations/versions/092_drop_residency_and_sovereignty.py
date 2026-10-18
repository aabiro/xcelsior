"""Drop residency, sovereignty, and AI Compute Access Fund columns.

Xcelsior is a global marketplace. Capacity is selected on price, availability,
hardware, and host reputation — never on geography — so the schema should not
carry columns that only exist to record or price a jurisdiction.

What goes, and why each is safe:

* ``gpu_pricing.sovereignty_premium`` — every row is ``0.0``. The premium was
  already dropped in code (``_TIER_SOVEREIGNTY`` mapped all tiers to zero), so
  the column had no effect on any price. Removing it removes the temptation.
* ``usage_meters.is_canadian_compute`` — recorded whether compute ran in Canada,
  read only by the AI Compute Access Fund split and a "sovereignty summary"
  analytics block. The fund has closed; both readers are gone.
* ``invoices.canadian_compute_total_cad``, ``non_canadian_compute_total_cad``,
  ``fund_eligible_reimbursement_cad``, ``effective_cost_after_fund_cad`` — the
  fund reimbursement was hardcoded to zero once the program ended, so
  ``effective_cost_after_fund_cad`` always equalled ``total_cad``. These are
  float money columns as well, which the money-in-micros discipline forbids.

The ``sovereign`` pricing tier is renamed to ``dedicated``. The ladder is
``community`` (consumer hardware, best effort) → ``secure`` (data-center grade)
→ ``dedicated`` (dedicated hardware, highest availability). Every rung names
what the capacity *is*. The frontend previously labelled this rung "Enterprise",
which named the buyer rather than the capacity and read like a rank alongside
the reputation tiers (bronze/silver/gold/platinum/diamond); that label is
corrected here too.

Downgrade restores the columns but not their values: the data encoded a
distinction the platform no longer makes, and re-deriving it would mean
re-introducing the geographic classification this migration exists to remove.
"""

from alembic import op

revision = "092"
down_revision = "091"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A database seeded after the code change already carries `dedicated` rows,
    # so a blind rename collides with the (gpu_model, vram_gb, form_factor,
    # high_frequency, tier, pricing_mode) unique key. Drop the superseded
    # `sovereign` duplicates first — they priced identically once the premium
    # was zeroed, so nothing is lost — then rename whatever remains.
    op.execute(
        """
        DELETE FROM gpu_pricing s
         WHERE s.tier = 'sovereign'
           AND EXISTS (
               SELECT 1 FROM gpu_pricing e
                WHERE e.tier = 'dedicated'
                  AND e.gpu_model = s.gpu_model
                  AND e.vram_gb = s.vram_gb
                  AND e.form_factor = s.form_factor
                  AND e.high_frequency = s.high_frequency
                  AND e.pricing_mode = s.pricing_mode
           )
        """
    )
    op.execute("UPDATE gpu_pricing SET tier = 'dedicated' WHERE tier = 'sovereign'")
    op.execute(
        "UPDATE jobs SET payload = jsonb_set(payload, '{tier}', '\"dedicated\"') "
        "WHERE payload->>'tier' = 'sovereign'"
    )
    op.execute("ALTER TABLE gpu_pricing DROP COLUMN IF EXISTS sovereignty_premium")
    op.execute("ALTER TABLE usage_meters DROP COLUMN IF EXISTS is_canadian_compute")
    for column in (
        "canadian_compute_total_cad",
        "non_canadian_compute_total_cad",
        "fund_eligible_reimbursement_cad",
        "effective_cost_after_fund_cad",
    ):
        op.execute(f"ALTER TABLE invoices DROP COLUMN IF EXISTS {column}")


def downgrade() -> None:
    op.execute(
        "ALTER TABLE gpu_pricing ADD COLUMN IF NOT EXISTS "
        "sovereignty_premium DOUBLE PRECISION DEFAULT 0.0"
    )
    op.execute(
        "ALTER TABLE usage_meters ADD COLUMN IF NOT EXISTS "
        "is_canadian_compute INTEGER DEFAULT 0"
    )
    for column in (
        "canadian_compute_total_cad",
        "non_canadian_compute_total_cad",
        "fund_eligible_reimbursement_cad",
        "effective_cost_after_fund_cad",
    ):
        op.execute(
            f"ALTER TABLE invoices ADD COLUMN IF NOT EXISTS {column} DOUBLE PRECISION DEFAULT 0.0"
        )
    op.execute("UPDATE gpu_pricing SET tier = 'sovereign' WHERE tier = 'dedicated'")
    op.execute(
        "UPDATE jobs SET payload = jsonb_set(payload, '{tier}', '\"sovereign\"') "
        "WHERE payload->>'tier' = 'dedicated'"
    )
