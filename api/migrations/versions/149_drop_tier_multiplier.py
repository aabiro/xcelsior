"""Drop ``tier_multiplier`` from ``billing_cycles`` and ``usage_meters``.

Service-tier prices are Stripe Prices: the purchased tier is already in the
accepted quote's amount, so no charge has multiplied a rate since the legacy
priority tiers were retired. Every writer stores 1.0 and no reader uses the
value, so the column carries no information.

Contract step (rule 5): ship it in the release *after* the code that stopped
reading and writing the column is live. Deploys are blue-green; a live older
release still names the column in its INSERTs while this runs.

A row whose multiplier is not 1 is billing history whose rate times duration
does not reproduce its amount without that factor. Dropping it would lose
evidence, so the upgrade refuses instead and names the table; capture those
rows deliberately before retrying.

``downgrade`` restores the column with its original type and default. It cannot
restore values, which were all 1 by the upgrade's own check.
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "149"
down_revision = "148"
branch_labels = None
depends_on = None

TABLES = ("billing_cycles", "usage_meters")


def _drop(table):
    def run(conn):
        present = conn.execute(
            text(
                "SELECT 1 FROM information_schema.columns WHERE table_schema='public'"
                " AND table_name=:table AND column_name='tier_multiplier'"
            ),
            {"table": table},
        ).fetchone()
        if present is None:
            return
        scaled = conn.execute(
            text(f"SELECT count(*) FROM {table} WHERE tier_multiplier IS DISTINCT FROM 1")
        ).scalar()
        if scaled:
            raise RuntimeError(
                f"{table} has {scaled} rows with tier_multiplier other than 1; "
                "their amounts depend on it. Capture them before dropping the column."
            )
        conn.execute(text(f"ALTER TABLE {table} DROP COLUMN tier_multiplier"))

    return run


def _restore(table):
    def run(conn):
        conn.execute(
            text(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS tier_multiplier "
                "DOUBLE PRECISION DEFAULT 1"
            )
        )

    return run


def upgrade():
    apply_in_own_transactions([(table, _drop(table)) for table in TABLES], tables=TABLES)


def downgrade():
    apply_in_own_transactions([(table, _restore(table)) for table in TABLES], tables=TABLES)
