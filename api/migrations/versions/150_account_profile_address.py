"""Store account onboarding choices and the address entered by the account owner.

Existing accounts remain incomplete until their owner supplies the missing data.
No address or account choice is inferred from historic defaults or network IPs.
"""

from alembic import op

revision = "150"
down_revision = "149"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS address_line1 TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS address_line2 TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS city TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS postal_code TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS account_type TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS onboarding_intent TEXT NOT NULL DEFAULT ''")


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS onboarding_intent")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS account_type")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS postal_code")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS city")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS address_line2")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS address_line1")
