"""Move Connect mapping bootstrap from API import to the migration login.

Original mapping fields are retained verbatim, with explicit platform catalog
scope and tenant-leading indexes. Fresh databases receive the same tables; downgrade retains financial/provider mappings for older readers.
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
    # These are the platform administrator's shared Connect catalog, never
    # recovered personal workspace ownership inferred from a contact email.
    for table in ("connect_accounts", "connect_products"):
        op.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS tenant_id TEXT "
            "NOT NULL DEFAULT 'system:stripe-connect'"
        )
    op.execute(
        "CREATE INDEX IF NOT EXISTS connect_accounts_tenant_account "
        "ON connect_accounts(tenant_id,stripe_account_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS connect_products_tenant_account "
        "ON connect_products(tenant_id,account_id,stripe_product_id)"
    )


def downgrade():
    # Older API imports already expected these tables. Removing them would
    # discard existing provider/product history and is not a schema rollback.
    pass
