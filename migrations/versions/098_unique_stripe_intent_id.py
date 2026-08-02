"""One row per Stripe intent in `payment_intents`.

`StripeConnectManager._handle_payment_succeeded` decides who to credit by
looking a confirmation event up against this column:

    SELECT customer_id, amount_cents FROM payment_intents
     WHERE stripe_intent_id = %s

with a bare `fetchone()`. Nothing has ever stopped two rows sharing an intent
id, and if two did, the credited customer and amount would be whichever row
Postgres returned first — a coin flip that moves money.

The auto-top-up path now writes this table too, and it needs `ON CONFLICT
(stripe_intent_id) DO NOTHING` so that a retried sweep re-registering an intent
Stripe deduplicated is a no-op rather than a second row. `ON CONFLICT` requires
a matching unique index; this is that index.

Partial, because a row may legitimately have no intent id yet: some paths insert
before Stripe is called, and a total unique index would collapse every one of
those onto a single empty string. `NULL` and `''` are both excluded, and the
`ON CONFLICT` clause in `billing.check_low_balance_and_topup` repeats this
predicate verbatim so Postgres can infer this index as the arbiter.

Verified before writing: zero duplicate intent ids and zero empty ones across
the table, so this builds without a cleanup step. The de-duplication below is
kept anyway for environments whose data was not inspected.

The SQL is written as literals rather than composed from a shared constant.
Interpolating even a trusted constant into DDL trips
`tests/test_sql_injection_guard.py`, and the guard is right to be unconditional
— a migration is exactly where a formatted identifier would go unnoticed.
"""

from alembic import op

revision = "098"
down_revision = "097"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Defensive: if duplicates exist, keep the earliest row for each intent.
    # Later rows are re-registrations of a charge Stripe already deduplicated,
    # so they describe no additional money.
    op.execute(
        """
        DELETE FROM payment_intents a
         USING payment_intents b
         WHERE a.stripe_intent_id = b.stripe_intent_id
           AND a.stripe_intent_id IS NOT NULL
           AND a.stripe_intent_id <> ''
           AND (a.created_at, a.intent_id) > (b.created_at, b.intent_id)
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_payment_intents_stripe_intent_id
            ON payment_intents (stripe_intent_id)
         WHERE stripe_intent_id IS NOT NULL AND stripe_intent_id <> ''
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_payment_intents_stripe_intent_id")
