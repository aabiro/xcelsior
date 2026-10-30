"""Typed funded queue bids and durable evidence for ordering experiments."""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "148"
down_revision = "147"
branch_labels = None
depends_on = None


def _run(statements):
    def run(conn):
        for statement in statements:
            conn.execute(text(statement))

    return run


def upgrade():
    apply_in_own_transactions(
        [
            (
                "jobs",
                _run(
                    [
                        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS queue_entered_at TIMESTAMPTZ",
                        """CREATE OR REPLACE FUNCTION queue_bid_track_entry() RETURNS trigger LANGUAGE plpgsql AS $$
                    BEGIN
                        IF NEW.spec->'service_tier_quote'->>'policy_version'='service-tiers-v1' AND NEW.phase='pending' THEN
                            IF TG_OP='INSERT' THEN
                                NEW.queue_entered_at := clock_timestamp();
                            ELSIF OLD.phase IS DISTINCT FROM 'pending' THEN
                                NEW.queue_entered_at := clock_timestamp();
                            ELSE
                                NEW.queue_entered_at := COALESCE(NEW.queue_entered_at,OLD.queue_entered_at,OLD.queued_at,clock_timestamp());
                            END IF;
                        END IF;
                        RETURN NEW;
                    END $$""",
                        "DROP TRIGGER IF EXISTS queue_bid_track_entry ON jobs",
                        "CREATE TRIGGER queue_bid_track_entry BEFORE INSERT OR UPDATE ON jobs FOR EACH ROW EXECUTE FUNCTION queue_bid_track_entry()",
                    ]
                ),
            ),
            (
                "queue_bid_events",
                _run(
                    [
                        """CREATE TABLE IF NOT EXISTS queue_bid_events (
                event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                event_type TEXT NOT NULL CHECK(event_type IN ('submitted','claimed')),
                claim_number BIGINT NOT NULL DEFAULT 0,
                service_tier TEXT NOT NULL,
                effective_tier INTEGER NOT NULL CHECK(effective_tier BETWEEN 1 AND 3),
                bid_price_id TEXT REFERENCES stripe_compute_prices(stripe_price_id),
                bid_units BIGINT NOT NULL CHECK(bid_units>=0),
                bid_micros BIGINT NOT NULL CHECK(bid_micros>=0),
                aging_steps BIGINT NOT NULL DEFAULT 0,
                aging_credit_micros NUMERIC(30,0) NOT NULL DEFAULT 0,
                queue_seconds NUMERIC(30,6) NOT NULL DEFAULT 0,
                policy_version TEXT NOT NULL,
                pricing_snapshot JSONB NOT NULL,
                ordering_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
                recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                UNIQUE(job_id,event_type,claim_number)
            )""",
                        "CREATE INDEX IF NOT EXISTS queue_bid_events_tenant_time ON queue_bid_events(tenant_id,recorded_at)",
                        "CREATE INDEX IF NOT EXISTS queue_bid_events_policy_time ON queue_bid_events(policy_version,recorded_at)",
                    """DO $price_roles$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='xcelsior_api') THEN GRANT SELECT, INSERT ON queue_bid_events TO xcelsior_api; END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='xcelsior_scheduler') THEN GRANT SELECT, INSERT ON queue_bid_events TO xcelsior_scheduler; END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='xcelsior_reconciler') THEN GRANT SELECT, INSERT ON queue_bid_events TO xcelsior_reconciler; END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='xcelsior_billing') THEN GRANT SELECT ON queue_bid_events TO xcelsior_billing; END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='xcelsior_projector') THEN GRANT SELECT ON queue_bid_events TO xcelsior_projector; END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='xcelsior_readonly') THEN GRANT SELECT ON queue_bid_events TO xcelsior_readonly; END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname='xcelsior_migrator') THEN GRANT ALL PRIVILEGES ON queue_bid_events TO xcelsior_migrator; END IF;
END $price_roles$;""",
                    ]
                ),
            ),
        ],
        tables=["jobs", "queue_bid_events"],
    )


def downgrade():
    apply_in_own_transactions(
        [
            ("queue_bid_events", _run(["DROP TABLE IF EXISTS queue_bid_events"])),
            (
                "jobs",
                _run(
                    [
                        "DROP TRIGGER IF EXISTS queue_bid_track_entry ON jobs",
                        "DROP FUNCTION IF EXISTS queue_bid_track_entry()",
                        "ALTER TABLE jobs DROP COLUMN IF EXISTS queue_entered_at",
                    ]
                ),
            ),
        ],
        tables=["queue_bid_events", "jobs"],
    )
