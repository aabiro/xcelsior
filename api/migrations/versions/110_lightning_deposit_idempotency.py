"""Idempotency for the Lightning rail. The other half of Gate P1 clause 2's crypto rails.

The clause says *"the crypto **rails**"*, plural, and there are two: on-chain
BTC (`crypto_deposits`, migration 109) and Lightning (`ln_deposits`). Fixing
only the first would have left the clause half met while reading as done — the
same shape as asserting a refusal against a route that does not exist.

Lightning's failure is the sharper of the two. `create_deposit` took no key, so
a retried request called `create_invoice` again and minted a **second bolt11
with a second payment hash**. On-chain, two addresses at least both belong to
the same wallet and both credit if paid. A second invoice is a distinct payment
request: a wallet that pays the first sees nothing settle against the second,
and the caller holding the second is waiting on a payment that already happened.

The index and column mirror 109 exactly — scoped to `customer_id` because a
caller-chosen key is only meaningful inside the account that chose it, and
partial so that every existing keyless row does not collide with every other.
"""

from alembic import op

revision = "110"
down_revision = "109"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("ALTER TABLE ln_deposits ADD COLUMN IF NOT EXISTS idempotency_key TEXT")
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_ln_deposits_idempotency
            ON ln_deposits (customer_id, idempotency_key)
         WHERE idempotency_key IS NOT NULL AND idempotency_key <> ''
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_ln_deposits_idempotency")
    op.execute("ALTER TABLE ln_deposits DROP COLUMN IF EXISTS idempotency_key")
