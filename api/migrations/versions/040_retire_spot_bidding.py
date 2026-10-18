"""Retire spot bidding; add pricing_mode and spot_rate_cad to jobs.

Removes max_bid from job payloads and introduces explicit pricing_mode
(on_demand | spot | reserved) plus optional spot_rate_cad snapshot column.

Revision ID: 040
Revises: 039
Create Date: 2026-06-09
"""

from alembic import op

revision = "040"
down_revision = "039"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE jobs "
        "ADD COLUMN IF NOT EXISTS pricing_mode TEXT NOT NULL DEFAULT 'on_demand';"
    )
    op.execute(
        "ALTER TABLE jobs "
        "ADD COLUMN IF NOT EXISTS spot_rate_cad DOUBLE PRECISION;"
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'jobs_pricing_mode_check'
            ) THEN
                ALTER TABLE jobs
                ADD CONSTRAINT jobs_pricing_mode_check
                CHECK (pricing_mode IN ('on_demand', 'spot', 'reserved'));
            END IF;
        END $$;
        """
    )

    # Strip max_bid and normalize spot jobs in payload.
    op.execute(
        """
        UPDATE jobs
        SET payload = payload - 'max_bid'
        WHERE payload ? 'max_bid';
        """
    )
    op.execute(
        """
        UPDATE jobs
        SET payload = jsonb_set(
                jsonb_set(
                    jsonb_set(payload, '{pricing_mode}', '"spot"', true),
                    '{preemptible}', 'true', true
                ),
                '{spot}', 'true', true
            ),
            pricing_mode = 'spot'
        WHERE payload->>'spot' = 'true'
           OR (payload->>'tier') = 'spot'
           OR (payload->>'preemptible') = 'true';
        """
    )
    op.execute(
        """
        UPDATE jobs
        SET pricing_mode = COALESCE(payload->>'pricing_mode', 'on_demand')
        WHERE pricing_mode IS DISTINCT FROM COALESCE(payload->>'pricing_mode', 'on_demand');
        """
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_pricing_mode_status "
        "ON jobs (pricing_mode, status);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_jobs_pricing_mode_status;")
    op.execute("ALTER TABLE jobs DROP CONSTRAINT IF EXISTS jobs_pricing_mode_check;")
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS spot_rate_cad;")
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS pricing_mode;")