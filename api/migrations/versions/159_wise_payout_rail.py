"""Wise as a third provider payout rail, paid in approved batches.

Stripe and PayPal settle one job at a time. Wise charges per transfer, so a
provider's queued jobs are added up and paid as one transfer. That needs three
things the per-job rails did not:

**`provider_wise_recipients`: where a provider is paid.** One row per provider.
It holds Wise's recipient id, the target currency, a masked hint for display
and a fingerprint of the destination, never bank details: those are entered
once and live at Wise. ``payouts_enabled_after`` is the cool-off: a destination
that was just set or changed cannot be paid until it passes, so a stolen
session cannot redirect the next payout and collect it the same day.

**`wise_payouts`: one row per transfer, written before Wise is called.** The
row carries the idempotency id (``customer_transaction_id``) from the moment it
exists, so a timeout leaves ``unknown`` rather than a second transfer: the same
id is sent again and Wise answers with the transfer it already has. The
destination is copied onto the row at batching time, and the transfer is
created against that copy, so a later change of destination cannot move a
payout that was already approved. A provider has at most one open payout.

**`wise_webhook_inbox`: each delivery once.** Keyed on Wise's delivery id.

``payout_splits`` gains ``wise_payout_id`` and the rail ``wise``; a split that
is in a payout has the new status ``in_transit``, which no claim query selects,
so the per-job claimers never touch it.

Each unit is idempotent and runs in its own short-lock transaction
(migrations/lock_safe.py). ``downgrade`` refuses while any Wise payout exists:
dropping the tables would lose the record of money that moved.
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "159"
down_revision = "158"
branch_labels = None
depends_on = None

# Written out in full, not built from lists: no identifier is interpolated
# into DDL (tests/test_sql_injection_guard.py).
SETTLEMENT_TERMS_WITH_WISE = """
    ALTER TABLE payout_splits
      ADD CONSTRAINT ck_payout_splits_settlement_terms
      CHECK (
          payment_rail IN ('stripe', 'paypal', 'wise')
          AND currency ~ '^[A-Z]{3}$'
          AND platform_cut_bps BETWEEN 0 AND 10000
          AND tax_rate_bps BETWEEN 0 AND 10000
          AND settlement_status IN (
              'pending', 'queued', 'processing', 'awaiting_capture',
              'in_transit', 'paid', 'failed', 'manual_review',
              'legacy_conflict', 'superseded'
          )
          AND (
              settlement_key IS NULL
              OR (
                  customer_id IS NOT NULL
                  AND customer_id <> ''
                  AND rail_idempotency_key <> ''
              )
          )
          AND (settlement_status <> 'in_transit' OR wise_payout_id IS NOT NULL)
          AND (wise_payout_id IS NULL OR payment_rail = 'wise')
      ) NOT VALID
"""

SETTLEMENT_TERMS_WITHOUT_WISE = """
    ALTER TABLE payout_splits
      ADD CONSTRAINT ck_payout_splits_settlement_terms
      CHECK (
          payment_rail IN ('stripe', 'paypal')
          AND currency ~ '^[A-Z]{3}$'
          AND platform_cut_bps BETWEEN 0 AND 10000
          AND tax_rate_bps BETWEEN 0 AND 10000
          AND settlement_status IN (
              'pending', 'queued', 'processing', 'awaiting_capture',
              'paid', 'failed', 'manual_review',
              'legacy_conflict', 'superseded'
          )
          AND (
              settlement_key IS NULL
              OR (
                  customer_id IS NOT NULL
                  AND customer_id <> ''
                  AND rail_idempotency_key <> ''
              )
          )
      ) NOT VALID
"""

CREATE_RECIPIENTS = """
    CREATE TABLE IF NOT EXISTS provider_wise_recipients (
        provider_id            TEXT PRIMARY KEY,
        wise_recipient_id      BIGINT NOT NULL,
        target_currency        TEXT NOT NULL,
        recipient_type         TEXT NOT NULL DEFAULT 'email',
        display_hint           TEXT NOT NULL DEFAULT '',
        fingerprint            TEXT NOT NULL,
        status                 TEXT NOT NULL DEFAULT 'active',
        payouts_enabled_after  TIMESTAMPTZ NOT NULL,
        created_at             TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        updated_at             TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        updated_by             TEXT NOT NULL DEFAULT '',
        CONSTRAINT ck_provider_wise_recipients_terms CHECK (
            target_currency ~ '^[A-Z]{3}$'
            AND status IN ('active', 'disabled')
            AND fingerprint ~ '^[0-9a-f]{64}$'
            AND wise_recipient_id > 0
        )
    )
"""

CREATE_PAYOUTS = """
    CREATE TABLE IF NOT EXISTS wise_payouts (
        id                       BIGSERIAL PRIMARY KEY,
        provider_id              TEXT NOT NULL,
        source_currency          TEXT NOT NULL DEFAULT 'CAD',
        amount_micros            BIGINT NOT NULL,
        split_count              INTEGER NOT NULL,
        target_currency          TEXT NOT NULL,
        wise_recipient_id        BIGINT NOT NULL,
        recipient_fingerprint    TEXT NOT NULL,
        customer_transaction_id  UUID NOT NULL,
        wise_quote_id            TEXT NOT NULL DEFAULT '',
        wise_transfer_id         BIGINT,
        status                   TEXT NOT NULL DEFAULT 'pending_approval',
        wise_state               TEXT NOT NULL DEFAULT '',
        fee_micros               BIGINT,
        target_amount            TEXT NOT NULL DEFAULT '',
        approved_by              TEXT NOT NULL DEFAULT '',
        approved_at              TIMESTAMPTZ,
        attempt_count            INTEGER NOT NULL DEFAULT 0,
        error                    TEXT NOT NULL DEFAULT '',
        created_at               TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        updated_at               TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        sent_at                  TIMESTAMPTZ,
        CONSTRAINT ck_wise_payouts_terms CHECK (
            amount_micros > 0
            AND amount_micros % 10000 = 0
            AND split_count > 0
            AND source_currency ~ '^[A-Z]{3}$'
            AND target_currency ~ '^[A-Z]{3}$'
            AND recipient_fingerprint ~ '^[0-9a-f]{64}$'
            AND status IN (
                'pending_approval', 'approved', 'creating', 'unknown',
                'awaiting_funding', 'funded', 'sent',
                'rejected', 'needs_review', 'released'
            )
            AND (status NOT IN ('approved', 'creating', 'unknown', 'awaiting_funding',
                                'funded', 'sent')
                 OR (approved_by <> '' AND approved_at IS NOT NULL))
            AND (status NOT IN ('awaiting_funding', 'funded', 'sent')
                 OR wise_transfer_id IS NOT NULL)
        )
    )
"""

CREATE_INBOX = """
    CREATE TABLE IF NOT EXISTS wise_webhook_inbox (
        delivery_id   TEXT PRIMARY KEY,
        event_type    TEXT NOT NULL DEFAULT '',
        transfer_id   BIGINT,
        payload       JSONB NOT NULL DEFAULT '{}'::jsonb,
        outcome       TEXT NOT NULL DEFAULT '',
        received_at   TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        processed_at  TIMESTAMPTZ
    )
"""


def _exec(sql: str):
    return lambda conn: conn.execute(text(sql))


def upgrade():
    apply_in_own_transactions(
        [
            ("provider_wise_recipients", _exec(CREATE_RECIPIENTS)),
            ("wise_payouts", _exec(CREATE_PAYOUTS)),
            (
                "wise_payouts idempotency id",
                _exec(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_wise_payouts_customer_transaction "
                    "ON wise_payouts (customer_transaction_id)"
                ),
            ),
            (
                "wise_payouts transfer id",
                _exec(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_wise_payouts_transfer "
                    "ON wise_payouts (wise_transfer_id) WHERE wise_transfer_id IS NOT NULL"
                ),
            ),
            (
                "wise_payouts one open per provider",
                _exec(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_wise_payouts_one_open "
                    "ON wise_payouts (provider_id) WHERE status IN ("
                    "'pending_approval', 'approved', 'creating', 'unknown', "
                    "'awaiting_funding', 'funded', 'needs_review')"
                ),
            ),
            (
                "wise_payouts by status",
                _exec(
                    "CREATE INDEX IF NOT EXISTS idx_wise_payouts_status "
                    "ON wise_payouts (status, updated_at)"
                ),
            ),
            ("wise_webhook_inbox", _exec(CREATE_INBOX)),
        ],
        tables=("provider_wise_recipients", "wise_payouts", "wise_webhook_inbox"),
    )
    apply_in_own_transactions(
        [
            (
                "payout_splits wise_payout_id",
                _exec("ALTER TABLE payout_splits ADD COLUMN IF NOT EXISTS wise_payout_id BIGINT"),
            ),
            (
                "payout_splits drop old settlement terms",
                _exec(
                    "ALTER TABLE payout_splits "
                    "DROP CONSTRAINT IF EXISTS ck_payout_splits_settlement_terms"
                ),
            ),
            ("payout_splits settlement terms with wise", _exec(SETTLEMENT_TERMS_WITH_WISE)),
            (
                "payout_splits validate settlement terms",
                _exec(
                    "ALTER TABLE payout_splits "
                    "VALIDATE CONSTRAINT ck_payout_splits_settlement_terms"
                ),
            ),
            (
                "payout_splits by wise payout",
                _exec(
                    "CREATE INDEX IF NOT EXISTS idx_payout_splits_wise_payout "
                    "ON payout_splits (wise_payout_id) WHERE wise_payout_id IS NOT NULL"
                ),
            ),
        ],
        tables=("payout_splits",),
    )


def _refuse_if_money_moved(conn):
    exists = conn.execute(text("SELECT to_regclass('public.wise_payouts') IS NOT NULL")).scalar()
    if not exists:
        return
    count = conn.execute(text("SELECT count(*) FROM wise_payouts")).scalar()
    if count:
        raise RuntimeError(
            f"refusing to downgrade 159: wise_payouts holds {count} payout record(s); "
            "export and reconcile them first"
        )


def downgrade():
    apply_in_own_transactions(
        [("refuse when wise payouts exist", _refuse_if_money_moved)],
        tables=("wise_payouts",),
    )
    apply_in_own_transactions(
        [
            (
                "payout_splits drop wise settlement terms",
                _exec(
                    "ALTER TABLE payout_splits "
                    "DROP CONSTRAINT IF EXISTS ck_payout_splits_settlement_terms"
                ),
            ),
            (
                "payout_splits return wise rows to review",
                _exec(
                    "UPDATE payout_splits SET payment_rail = 'stripe', "
                    "settlement_status = 'manual_review', "
                    "settlement_error = 'wise rail removed by downgrade of 159', "
                    "wise_payout_id = NULL "
                    "WHERE payment_rail = 'wise'"
                ),
            ),
            ("payout_splits settlement terms without wise", _exec(SETTLEMENT_TERMS_WITHOUT_WISE)),
            (
                "payout_splits validate settlement terms",
                _exec(
                    "ALTER TABLE payout_splits "
                    "VALIDATE CONSTRAINT ck_payout_splits_settlement_terms"
                ),
            ),
            (
                "payout_splits drop wise index",
                _exec("DROP INDEX IF EXISTS idx_payout_splits_wise_payout"),
            ),
            (
                "payout_splits drop wise_payout_id",
                _exec("ALTER TABLE payout_splits DROP COLUMN IF EXISTS wise_payout_id"),
            ),
        ],
        tables=("payout_splits",),
    )
    apply_in_own_transactions(
        [
            ("drop wise_webhook_inbox", _exec("DROP TABLE IF EXISTS wise_webhook_inbox")),
            ("drop wise_payouts", _exec("DROP TABLE IF EXISTS wise_payouts")),
            (
                "drop provider_wise_recipients",
                _exec("DROP TABLE IF EXISTS provider_wise_recipients"),
            ),
        ],
        tables=("provider_wise_recipients", "wise_payouts", "wise_webhook_inbox"),
    )
