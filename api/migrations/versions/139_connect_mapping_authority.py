"""Move Connect mapping bootstrap from API import to the migration login.

Existing mappings are retained verbatim. Fresh databases receive the same
tables; downgrade retains financial/provider mappings for older readers.
"""

from alembic import op

revision = "139"
down_revision = "138"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("SET LOCAL lock_timeout='5s'")
    op.execute("""CREATE TABLE IF NOT EXISTS connect_accounts (
        id SERIAL PRIMARY KEY,
        display_name TEXT NOT NULL,
        contact_email TEXT NOT NULL,
        stripe_account_id TEXT UNIQUE NOT NULL,
        created_at TIMESTAMPTZ DEFAULT now()
    )""")
    op.execute("""CREATE TABLE IF NOT EXISTS connect_products (
        id SERIAL PRIMARY KEY,
        stripe_product_id TEXT UNIQUE NOT NULL,
        stripe_price_id TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT DEFAULT '',
        price_cents INTEGER NOT NULL,
        currency TEXT DEFAULT 'usd',
        account_id TEXT NOT NULL,
        created_at TIMESTAMPTZ DEFAULT now()
    )""")


def downgrade():
    # Older API imports already expected these tables. Removing them would
    # discard existing provider/product history and is not a schema rollback.
    pass
