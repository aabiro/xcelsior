"""Idempotency for the crypto funding rail. Gate P1 clause 2's third named rail.

Gate P1: *"Replaying any funding call with the same idempotency key produces
exactly one charge. Asserted for manual top-up, auto-top-up, and the crypto
rails."* Two of the three were asserted. This is the third, and it was not
merely unasserted — **the mechanism did not exist**.

`POST /api/billing/crypto/deposit` accepted no key and deduplicated nothing.
Every call ran `get_new_address`, inserted a fresh row and locked a fresh
BTC/CAD rate. So a client that retried a timed-out request — an agent, a
double-clicked button, a proxy replay — received a *second* Bitcoin address for
one intended deposit, with a different locked rate and a different expiry.

That is a worse shape than a duplicated card charge, not a milder one. A
duplicate charge is visible and refundable. Two live addresses for one intent
are not obviously duplicates to the person looking at them: whichever they pay,
the other stays open, and the wallet has burned an address that now has to be
watched. The failure is silent on the server and confusing on the client.

## Why the unique index is scoped to the customer

`(customer_id, idempotency_key)` rather than a global unique on the key. A
global one would make one tenant's key collide with another's — turning a
replay guard into a way to probe whether a key exists, and letting an unlucky
collision hand a caller someone else's deposit row. Keys are caller-chosen
strings; they are only meaningful inside the account that chose them.

## Why partial

`WHERE idempotency_key IS NOT NULL AND idempotency_key <> ''` — every row that
exists today has no key and must not collide with every other keyless row. The
column is nullable with no default for the same reason 108 gave: "no key was
supplied" is the honest state for the existing corpus, and a default would
manufacture one for all of them.
"""

from alembic import op

revision = "109"
down_revision = "108"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("ALTER TABLE crypto_deposits ADD COLUMN IF NOT EXISTS idempotency_key TEXT")
    # CONCURRENTLY is not available inside a migration transaction, and this
    # table is small (one row per deposit intent, not per event), so an ordinary
    # index under the lock timeout above is the right trade.
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_crypto_deposits_idempotency
            ON crypto_deposits (customer_id, idempotency_key)
         WHERE idempotency_key IS NOT NULL AND idempotency_key <> ''
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_crypto_deposits_idempotency")
    op.execute("ALTER TABLE crypto_deposits DROP COLUMN IF EXISTS idempotency_key")
