"""Store optional customer GST/HST details and provider seller identity.

Customers may buy as a business and leave a business name and GST/HST number.
Providers are independent suppliers. The platform keeps the legal name, mailing
address, and tax identifier needed to report their earnings. The identifier
itself is stored only as ciphertext.
"""

from alembic import op

revision = "144"
down_revision = "143"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS business_name TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS gst_hst_number TEXT NOT NULL DEFAULT ''")
    op.execute(
        "ALTER TABLE provider_accounts ADD COLUMN IF NOT EXISTS address_line1 TEXT NOT NULL DEFAULT ''"
    )
    op.execute(
        "ALTER TABLE provider_accounts ADD COLUMN IF NOT EXISTS address_line2 TEXT NOT NULL DEFAULT ''"
    )
    op.execute("ALTER TABLE provider_accounts ADD COLUMN IF NOT EXISTS city TEXT NOT NULL DEFAULT ''")
    op.execute(
        "ALTER TABLE provider_accounts ADD COLUMN IF NOT EXISTS postal_code TEXT NOT NULL DEFAULT ''"
    )
    op.execute(
        "ALTER TABLE provider_accounts ADD COLUMN IF NOT EXISTS tax_id_kind TEXT NOT NULL DEFAULT ''"
    )
    op.execute(
        "ALTER TABLE provider_accounts "
        "ADD COLUMN IF NOT EXISTS tax_id_ciphertext TEXT NOT NULL DEFAULT ''"
    )
    op.execute(
        "ALTER TABLE provider_accounts ADD COLUMN IF NOT EXISTS tax_id_last4 TEXT NOT NULL DEFAULT ''"
    )
    op.execute(
        "ALTER TABLE provider_accounts "
        "ADD COLUMN IF NOT EXISTS seller_terms_accepted_at DOUBLE PRECISION"
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("ALTER TABLE provider_accounts DROP COLUMN IF EXISTS seller_terms_accepted_at")
    op.execute("ALTER TABLE provider_accounts DROP COLUMN IF EXISTS tax_id_last4")
    op.execute("ALTER TABLE provider_accounts DROP COLUMN IF EXISTS tax_id_ciphertext")
    op.execute("ALTER TABLE provider_accounts DROP COLUMN IF EXISTS tax_id_kind")
    op.execute("ALTER TABLE provider_accounts DROP COLUMN IF EXISTS postal_code")
    op.execute("ALTER TABLE provider_accounts DROP COLUMN IF EXISTS city")
    op.execute("ALTER TABLE provider_accounts DROP COLUMN IF EXISTS address_line2")
    op.execute("ALTER TABLE provider_accounts DROP COLUMN IF EXISTS address_line1")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS gst_hst_number")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS business_name")
