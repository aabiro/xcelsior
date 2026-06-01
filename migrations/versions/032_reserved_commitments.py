"""create reserved_commitments table

Persists reserved pricing commitments created via
``POST /api/pricing/reserve``. Previously the reserve endpoint computed a
commitment object and discarded it, so customers had no way to see their
active commitments or how much they had saved. This table backs the new
``GET /api/pricing/reservations`` endpoint (active commitments + realized
savings vs on-demand).

Revision ID: 032
Revises: 031
Create Date: 2026-06-01
"""
from alembic import op

revision = "032"
down_revision = "031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS reserved_commitments (
            commitment_id        TEXT PRIMARY KEY,
            customer_id          TEXT NOT NULL,
            commitment_type      TEXT NOT NULL,
            gpu_model            TEXT NOT NULL,
            quantity             INTEGER NOT NULL DEFAULT 1,
            province             TEXT,
            base_rate_cad        DOUBLE PRECISION NOT NULL,
            discounted_rate_cad  DOUBLE PRECISION NOT NULL,
            discount_pct         DOUBLE PRECISION NOT NULL,
            min_hours_per_day    DOUBLE PRECISION NOT NULL DEFAULT 0,
            status               TEXT NOT NULL DEFAULT 'active',
            created_at           DOUBLE PRECISION NOT NULL,
            start_at             DOUBLE PRECISION NOT NULL,
            end_at               DOUBLE PRECISION NOT NULL
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_reserved_commitments_customer "
        "ON reserved_commitments (customer_id, status);"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS reserved_commitments;")
