"""Bind live compute prices and paid queue policy to immutable Stripe Prices."""

from alembic import op

from migrations.lock_safe import apply_in_own_transactions
from sqlalchemy import text

revision = "147"
down_revision = "146"
branch_labels = None
depends_on = None


def per_table(units):
    def apply(statements):
        def run(conn):
            for statement in statements:
                conn.execute(text(statement))

        return run

    apply_in_own_transactions(
        [(table, apply(statements)) for table, statements in units],
        tables=[table for table, _ in units],
    )


def upgrade() -> None:
    per_table(
        [
            (
                "stripe_compute_prices",
                [
                    """CREATE TABLE IF NOT EXISTS stripe_compute_prices (
                stripe_price_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL DEFAULT 'xcelsior-platform' CHECK (tenant_id='xcelsior-platform'),
                stripe_product_id TEXT NOT NULL,
                lookup_key TEXT NOT NULL,
                unit_amount_micros BIGINT NOT NULL CHECK (unit_amount_micros > 0),
                currency TEXT NOT NULL DEFAULT 'CAD' CHECK (currency='CAD'),
                livemode BOOLEAN NOT NULL,
                catalog_version TEXT NOT NULL,
                active BOOLEAN NOT NULL DEFAULT TRUE,
                imported_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
            )""",
                    "CREATE UNIQUE INDEX IF NOT EXISTS stripe_compute_prices_active_lookup ON stripe_compute_prices(lookup_key) WHERE active",
                    "CREATE INDEX IF NOT EXISTS stripe_compute_prices_tenant ON stripe_compute_prices(tenant_id,lookup_key)",
                    """DO $price_roles$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='xcelsior_api') THEN GRANT SELECT ON stripe_compute_prices TO xcelsior_api; END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='xcelsior_scheduler') THEN GRANT SELECT ON stripe_compute_prices TO xcelsior_scheduler; END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='xcelsior_reconciler') THEN GRANT SELECT ON stripe_compute_prices TO xcelsior_reconciler; END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='xcelsior_billing') THEN GRANT SELECT ON stripe_compute_prices TO xcelsior_billing; END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='xcelsior_projector') THEN GRANT SELECT ON stripe_compute_prices TO xcelsior_projector; END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='xcelsior_readonly') THEN GRANT SELECT ON stripe_compute_prices TO xcelsior_readonly; END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='xcelsior_migrator') THEN GRANT ALL PRIVILEGES ON stripe_compute_prices TO xcelsior_migrator; END IF;
END $price_roles$;""",
                    """CREATE OR REPLACE FUNCTION stripe_compute_price_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN
                    IF TG_OP='DELETE' THEN
                        RAISE EXCEPTION 'Stripe price snapshots cannot be deleted' USING ERRCODE='restrict_violation';
                    END IF;
                    IF ROW(NEW.stripe_price_id,NEW.stripe_product_id,NEW.lookup_key,NEW.unit_amount_micros,NEW.currency,NEW.livemode,NEW.catalog_version)
                       IS DISTINCT FROM ROW(OLD.stripe_price_id,OLD.stripe_product_id,OLD.lookup_key,OLD.unit_amount_micros,OLD.currency,OLD.livemode,OLD.catalog_version) THEN
                        RAISE EXCEPTION 'Stripe price snapshots are immutable' USING ERRCODE='restrict_violation';
                    END IF;
                    RETURN NEW;
                END $$""",
                    "DROP TRIGGER IF EXISTS stripe_compute_price_immutable ON stripe_compute_prices",
                    "CREATE TRIGGER stripe_compute_price_immutable BEFORE UPDATE OR DELETE ON stripe_compute_prices FOR EACH ROW EXECUTE FUNCTION stripe_compute_price_immutable()",
                ],
            ),
            (
                "gpu_pricing",
                [
                    "ALTER TABLE gpu_pricing ADD COLUMN IF NOT EXISTS stripe_price_id TEXT REFERENCES stripe_compute_prices(stripe_price_id)",
                ],
            ),
            (
                "job_attempts",
                [
                    "ALTER TABLE job_attempts ADD COLUMN IF NOT EXISTS host_exclusive BOOLEAN NOT NULL DEFAULT FALSE",
                ],
            ),
            (
                "usage_meters",
                [
                    "ALTER TABLE usage_meters ADD COLUMN IF NOT EXISTS pricing_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb",
                ],
            ),
            (
                "billing_cycles",
                [
                    "ALTER TABLE billing_cycles ADD COLUMN IF NOT EXISTS pricing_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb",
                ],
            ),
        ]
    )


def downgrade() -> None:
    per_table(
        [
            (
                "billing_cycles",
                ["ALTER TABLE billing_cycles DROP COLUMN IF EXISTS pricing_snapshot"],
            ),
            ("usage_meters", ["ALTER TABLE usage_meters DROP COLUMN IF EXISTS pricing_snapshot"]),
            ("job_attempts", ["ALTER TABLE job_attempts DROP COLUMN IF EXISTS host_exclusive"]),
            ("gpu_pricing", ["ALTER TABLE gpu_pricing DROP COLUMN IF EXISTS stripe_price_id"]),
            (
                "stripe_compute_prices",
                [
                    "DROP TABLE IF EXISTS stripe_compute_prices",
                    "DROP FUNCTION IF EXISTS stripe_compute_price_immutable()",
                ],
            ),
        ]
    )
