"""Authoritative, exact provider settlement.

This is an expand-only settlement cutover:

* provider earnings are represented in integer micro-CAD;
* one canonical settlement key spans Stripe and PayPal;
* external rail identifiers and legacy float projections are preserved;
* settlement workers use durable PostgreSQL claims with expiries; and
* usage meter totals gain an exact projection for eligibility checks.

Legacy money columns stay in place for readers that have not yet moved to the
integer representation.  Their removal belongs to the later contract phase.

Revision ID: 080
Revises: 079
Create Date: 2026-07-30
"""

from alembic import op

revision = "080"
down_revision = "079"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    # The terminal meter is supporting evidence for a provider earning.  The
    # wallet charge ledger remains the paid-amount authority, but both sides
    # must reconcile before a new payout can be prepared.
    op.execute(
        """
        ALTER TABLE usage_meters
          ADD COLUMN IF NOT EXISTS total_cost_micros BIGINT
        """
    )
    op.execute(
        """
        UPDATE usage_meters
           SET total_cost_micros =
               round(COALESCE(total_cost_cad, 0)::numeric * 1000000)::bigint
         WHERE total_cost_micros IS NULL
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION usage_meters_project_total_cost_money()
        RETURNS trigger AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                IF NEW.total_cost_micros IS NULL THEN
                    NEW.total_cost_micros :=
                        round(COALESCE(NEW.total_cost_cad, 0)::numeric * 1000000)::bigint;
                ELSE
                    NEW.total_cost_cad :=
                        (NEW.total_cost_micros::numeric / 1000000)::double precision;
                END IF;
            ELSIF NEW.total_cost_micros IS DISTINCT FROM OLD.total_cost_micros THEN
                NEW.total_cost_cad :=
                    (NEW.total_cost_micros::numeric / 1000000)::double precision;
            ELSIF NEW.total_cost_cad IS DISTINCT FROM OLD.total_cost_cad THEN
                NEW.total_cost_micros :=
                    round(COALESCE(NEW.total_cost_cad, 0)::numeric * 1000000)::bigint;
            END IF;
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql
        """
    )
    op.execute("DROP TRIGGER IF EXISTS trg_usage_meters_project_total_cost_money ON usage_meters")
    op.execute(
        """
        CREATE TRIGGER trg_usage_meters_project_total_cost_money
        BEFORE INSERT OR UPDATE ON usage_meters
        FOR EACH ROW EXECUTE FUNCTION usage_meters_project_total_cost_money()
        """
    )
    op.execute("ALTER TABLE usage_meters ALTER COLUMN total_cost_micros SET DEFAULT 0")
    op.execute("ALTER TABLE usage_meters ALTER COLUMN total_cost_micros SET NOT NULL")
    op.execute(
        """
        ALTER TABLE usage_meters
          ADD CONSTRAINT ck_usage_meters_total_cost_micros_non_negative
          CHECK (total_cost_micros >= 0) NOT VALID
        """
    )
    op.execute(
        "ALTER TABLE usage_meters "
        "VALIDATE CONSTRAINT ck_usage_meters_total_cost_micros_non_negative"
    )

    op.execute(
        """
        ALTER TABLE payout_splits
          ADD COLUMN IF NOT EXISTS customer_id TEXT,
          ADD COLUMN IF NOT EXISTS currency TEXT,
          ADD COLUMN IF NOT EXISTS source_total_micros BIGINT,
          ADD COLUMN IF NOT EXISTS total_micros BIGINT,
          ADD COLUMN IF NOT EXISTS provider_share_micros BIGINT,
          ADD COLUMN IF NOT EXISTS platform_share_micros BIGINT,
          ADD COLUMN IF NOT EXISTS gst_hst_micros BIGINT,
          ADD COLUMN IF NOT EXISTS rounding_adjustment_micros BIGINT,
          ADD COLUMN IF NOT EXISTS platform_cut_bps INTEGER,
          ADD COLUMN IF NOT EXISTS tax_rate_bps INTEGER,
          ADD COLUMN IF NOT EXISTS settlement_key TEXT,
          ADD COLUMN IF NOT EXISTS rail_idempotency_key TEXT,
          ADD COLUMN IF NOT EXISTS paypal_order_id TEXT,
          ADD COLUMN IF NOT EXISTS claim_owner TEXT,
          ADD COLUMN IF NOT EXISTS claim_token TEXT,
          ADD COLUMN IF NOT EXISTS claim_expires_at TIMESTAMPTZ,
          ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ,
          ADD COLUMN IF NOT EXISTS attempt_count INTEGER,
          ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ,
          ADD COLUMN IF NOT EXISTS settled_at TIMESTAMPTZ,
          ADD COLUMN IF NOT EXISTS legacy_imported BOOLEAN
        """
    )

    # Preserve every old row and external identifier.  Exact projections are
    # derived from the stored values; provider gets the residual so that the
    # canonical split has no unallocated micro-CAD.
    op.execute(
        """
        UPDATE payout_splits ps
           SET customer_id = COALESCE(
                   NULLIF(ps.customer_id, ''),
                   (
                       SELECT NULLIF(um.owner, '')
                         FROM usage_meters um
                        WHERE um.job_id = ps.job_id
                        ORDER BY um.completed_at DESC, um.meter_id
                        LIMIT 1
                   ),
                   (
                       SELECT NULLIF(wt.customer_id, '')
                         FROM wallet_transactions wt
                        WHERE wt.job_id = ps.job_id
                        ORDER BY wt.created_at DESC, wt.tx_id
                        LIMIT 1
                   ),
                   (
                       SELECT COALESCE(
                                  NULLIF(j.owner_id, ''),
                                  NULLIF(j.payload->>'owner', '')
                              )
                         FROM jobs j
                        WHERE j.job_id = ps.job_id
                   ),
                   ''
               ),
               currency = upper(COALESCE(
                   NULLIF(ps.currency, ''),
                   (
                       SELECT NULLIF(pa.default_currency, '')
                         FROM provider_accounts pa
                        WHERE pa.provider_id = ps.provider_id
                   ),
                   'CAD'
               )),
               source_total_micros = COALESCE(
                   ps.source_total_micros,
                   round(COALESCE(ps.total_cad, 0)::numeric * 1000000)::bigint
               ),
               total_micros = COALESCE(
                   ps.total_micros,
                   round(COALESCE(ps.total_cad, 0)::numeric * 1000000)::bigint
               ),
               platform_share_micros = COALESCE(
                   ps.platform_share_micros,
                   round(COALESCE(ps.platform_share_cad, 0)::numeric * 1000000)::bigint
               ),
               gst_hst_micros = COALESCE(
                   ps.gst_hst_micros,
                   round(COALESCE(ps.gst_hst_cad, 0)::numeric * 1000000)::bigint
               ),
               rounding_adjustment_micros =
                   COALESCE(ps.rounding_adjustment_micros, 0),
               platform_cut_bps = COALESCE(
                   ps.platform_cut_bps,
                   CASE
                       WHEN COALESCE(ps.total_cad, 0) > 0 THEN
                           round(
                               COALESCE(ps.platform_share_cad, 0)::numeric
                               / ps.total_cad::numeric * 10000
                           )::integer
                       ELSE 1500
                   END
               ),
               tax_rate_bps = COALESCE(
                   ps.tax_rate_bps,
                   CASE
                       WHEN COALESCE(ps.total_cad, 0) > 0 THEN
                           round(
                               COALESCE(ps.gst_hst_cad, 0)::numeric
                               / ps.total_cad::numeric * 10000
                           )::integer
                       ELSE 0
                   END
               ),
               rail_idempotency_key = COALESCE(
                   NULLIF(ps.rail_idempotency_key, ''),
                   'provider-settlement:' || ps.job_id
               ),
               paypal_order_id = COALESCE(ps.paypal_order_id, ''),
               attempt_count = COALESCE(ps.attempt_count, 0),
               updated_at = COALESCE(
                   ps.updated_at,
                   to_timestamp(COALESCE(ps.created_at, extract(epoch FROM clock_timestamp())))
               ),
               legacy_imported = TRUE,
               settlement_status = CASE
                   WHEN COALESCE(ps.stripe_transfer_id, '') <> ''
                     OR COALESCE(ps.paypal_capture_id, '') <> ''
                       THEN 'paid'
                   WHEN COALESCE(NULLIF(ps.settlement_status, ''), 'pending')
                        IN ('pending', 'queued', 'processing', 'awaiting_capture',
                            'paid', 'failed', 'manual_review',
                            'legacy_conflict', 'superseded')
                       THEN COALESCE(NULLIF(ps.settlement_status, ''), 'pending')
                   ELSE 'manual_review'
               END
        """
    )
    op.execute(
        """
        UPDATE payout_splits
           SET provider_share_micros =
                   total_micros - platform_share_micros,
               rounding_adjustment_micros =
                   total_micros - source_total_micros
         WHERE provider_share_micros IS NULL
            OR provider_share_micros + platform_share_micros <> total_micros
        """
    )
    op.execute(
        """
        UPDATE payout_splits
           SET settlement_status = COALESCE(NULLIF(settlement_status, ''), 'pending'),
               settlement_error = COALESCE(settlement_error, ''),
               payment_rail = COALESCE(NULLIF(payment_rail, ''), 'stripe')
        """
    )

    # Legacy databases may contain more than one payout_splits row per job.
    # Keep all of them and all rail IDs, but grant the canonical cross-rail key
    # to exactly one deterministic row.  Conflicts remain available for audit
    # instead of being deleted or rewritten.
    op.execute(
        """
        WITH ranked AS (
            SELECT id,
                   row_number() OVER (
                       PARTITION BY job_id
                       ORDER BY
                           CASE
                               WHEN COALESCE(stripe_transfer_id, '') <> ''
                                 OR COALESCE(paypal_capture_id, '') <> ''
                                   THEN 0
                               ELSE 1
                           END,
                           created_at ASC NULLS LAST,
                           id ASC
                   ) AS position
              FROM payout_splits
             WHERE customer_id <> ''
        )
        UPDATE payout_splits ps
           SET settlement_key = 'provider-job:' || ps.job_id
          FROM ranked
         WHERE ranked.id = ps.id
           AND ranked.position = 1
           AND ps.settlement_key IS NULL
        """
    )
    op.execute(
        """
        UPDATE payout_splits
           SET settlement_status = CASE
                   WHEN COALESCE(stripe_transfer_id, '') <> ''
                     OR COALESCE(paypal_capture_id, '') <> ''
                       THEN 'legacy_conflict'
                   ELSE 'superseded'
               END,
               settlement_error = concat_ws(
                   ';',
                   NULLIF(settlement_error, ''),
                   'duplicate legacy job settlement retained by migration 080'
               )
         WHERE settlement_key IS NULL
        """
    )

    for column, default in (
        ("currency", "'CAD'"),
        ("source_total_micros", "0"),
        ("total_micros", "0"),
        ("provider_share_micros", "0"),
        ("platform_share_micros", "0"),
        ("gst_hst_micros", "0"),
        ("rounding_adjustment_micros", "0"),
        ("platform_cut_bps", "1500"),
        ("tax_rate_bps", "0"),
        ("rail_idempotency_key", "''"),
        ("paypal_order_id", "''"),
        ("attempt_count", "0"),
        ("updated_at", "clock_timestamp()"),
        ("legacy_imported", "FALSE"),
    ):
        op.execute(f"ALTER TABLE payout_splits ALTER COLUMN {column} SET DEFAULT {default}")
        op.execute(f"ALTER TABLE payout_splits ALTER COLUMN {column} SET NOT NULL")
    for column, default in (
        ("settlement_status", "'pending'"),
        ("settlement_error", "''"),
        ("payment_rail", "'stripe'"),
    ):
        op.execute(f"ALTER TABLE payout_splits ALTER COLUMN {column} SET DEFAULT {default}")
        op.execute(f"ALTER TABLE payout_splits ALTER COLUMN {column} SET NOT NULL")

    op.execute(
        """
        ALTER TABLE payout_splits
          ADD CONSTRAINT ck_payout_splits_exact_money
          CHECK (
              source_total_micros >= 0
              AND total_micros >= 0
              AND provider_share_micros >= 0
              AND platform_share_micros >= 0
              AND gst_hst_micros >= 0
              AND source_total_micros + rounding_adjustment_micros = total_micros
              AND provider_share_micros + platform_share_micros = total_micros
          ) NOT VALID
        """
    )
    op.execute(
        """
        ALTER TABLE payout_splits
          ADD CONSTRAINT ck_payout_splits_settlement_terms
          CHECK (
              payment_rail IN ('stripe', 'paypal')
              AND currency ~ '^[A-Z]{3}$'
              AND platform_cut_bps BETWEEN 0 AND 10000
              AND tax_rate_bps BETWEEN 0 AND 10000
              AND settlement_status IN (
                  'pending', 'queued', 'processing', 'awaiting_capture',
                  'paid', 'failed', 'manual_review',
                  'legacy_conflict', 'superseded'
              )
              AND (
                  settlement_key IS NULL
                  OR (
                      customer_id IS NOT NULL
                      AND customer_id <> ''
                      AND rail_idempotency_key <> ''
                  )
              )
          ) NOT VALID
        """
    )
    op.execute(
        """
        ALTER TABLE payout_splits
          ADD CONSTRAINT ck_payout_splits_claim
          CHECK (
              (claim_owner IS NULL AND claim_token IS NULL AND claim_expires_at IS NULL)
              OR
              (claim_owner IS NOT NULL AND claim_token IS NOT NULL AND claim_expires_at IS NOT NULL)
          ) NOT VALID
        """
    )
    op.execute("ALTER TABLE payout_splits VALIDATE CONSTRAINT ck_payout_splits_exact_money")
    op.execute("ALTER TABLE payout_splits VALIDATE CONSTRAINT ck_payout_splits_settlement_terms")
    op.execute("ALTER TABLE payout_splits VALIDATE CONSTRAINT ck_payout_splits_claim")

    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_payout_splits_settlement_key
            ON payout_splits (settlement_key)
         WHERE settlement_key IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_payout_splits_rail_idempotency
            ON payout_splits (rail_idempotency_key)
         WHERE settlement_key IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_payout_splits_stripe_transfer
            ON payout_splits (stripe_transfer_id)
         WHERE COALESCE(stripe_transfer_id, '') <> ''
           AND settlement_key IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_payout_splits_paypal_order
            ON payout_splits (paypal_order_id)
         WHERE paypal_order_id <> ''
           AND settlement_key IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_payout_splits_paypal_capture
            ON payout_splits (paypal_capture_id)
         WHERE COALESCE(paypal_capture_id, '') <> ''
           AND settlement_key IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_payout_splits_claimable
            ON payout_splits (payment_rail, next_attempt_at, created_at)
         WHERE settlement_key IS NOT NULL
           AND settlement_status IN (
               'pending', 'queued', 'failed', 'awaiting_capture', 'processing'
           )
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_payout_splits_claimable")
    op.execute("DROP INDEX IF EXISTS uq_payout_splits_paypal_capture")
    op.execute("DROP INDEX IF EXISTS uq_payout_splits_paypal_order")
    op.execute("DROP INDEX IF EXISTS uq_payout_splits_stripe_transfer")
    op.execute("DROP INDEX IF EXISTS uq_payout_splits_rail_idempotency")
    op.execute("DROP INDEX IF EXISTS uq_payout_splits_settlement_key")

    op.execute("ALTER TABLE payout_splits DROP CONSTRAINT IF EXISTS ck_payout_splits_claim")
    op.execute(
        "ALTER TABLE payout_splits DROP CONSTRAINT IF EXISTS ck_payout_splits_settlement_terms"
    )
    op.execute("ALTER TABLE payout_splits DROP CONSTRAINT IF EXISTS ck_payout_splits_exact_money")

    for column in (
        "legacy_imported",
        "settled_at",
        "updated_at",
        "attempt_count",
        "next_attempt_at",
        "claim_expires_at",
        "claim_token",
        "claim_owner",
        "paypal_order_id",
        "rail_idempotency_key",
        "settlement_key",
        "tax_rate_bps",
        "platform_cut_bps",
        "rounding_adjustment_micros",
        "gst_hst_micros",
        "platform_share_micros",
        "provider_share_micros",
        "total_micros",
        "source_total_micros",
        "currency",
        "customer_id",
    ):
        op.execute(f"ALTER TABLE payout_splits DROP COLUMN IF EXISTS {column}")

    op.execute(
        "ALTER TABLE usage_meters "
        "DROP CONSTRAINT IF EXISTS ck_usage_meters_total_cost_micros_non_negative"
    )
    op.execute("DROP TRIGGER IF EXISTS trg_usage_meters_project_total_cost_money ON usage_meters")
    op.execute("DROP FUNCTION IF EXISTS usage_meters_project_total_cost_money()")
    op.execute("ALTER TABLE usage_meters DROP COLUMN IF EXISTS total_cost_micros")
