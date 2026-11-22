"""Receipt emails: one row per finished run that was emailed (or skipped).

* `receipt_emails` — `job_id` is the key, so two sweeps cannot both email one
  run: a sweep claims the row (`sending`) before it sends and records `sent`,
  `failed` (retried up to a limit) or `skipped` (opted out, no account, not an
  instance).
* `receipt_email_settings` — one row holding `enabled_since`, the moment this
  migration ran. Runs that ended before it are never emailed, so turning the
  feature on does not mail every customer a receipt for every old run.

Both live in the billing domain. One idempotent unit through `lock_safe`;
`downgrade` drops both tables.
"""

from sqlalchemy import text

from migrations.lock_safe import apply_in_own_transactions

revision = "168"
down_revision = "167"
branch_labels = None
depends_on = None


def _create(conn):
    conn.execute(
        text(
            "CREATE TABLE IF NOT EXISTS receipt_emails ("
            " job_id TEXT PRIMARY KEY,"
            " owner_id TEXT NOT NULL,"
            " status TEXT NOT NULL CHECK (status IN ('sending', 'sent', 'failed', 'skipped')),"
            " detail TEXT NOT NULL DEFAULT '',"
            " attempts INTEGER NOT NULL DEFAULT 1,"
            " created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            " updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
    )
    conn.execute(
        text(
            "CREATE TABLE IF NOT EXISTS receipt_email_settings ("
            " id BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),"
            " enabled_since TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
    )
    conn.execute(
        text("INSERT INTO receipt_email_settings (id) VALUES (TRUE) ON CONFLICT (id) DO NOTHING")
    )


def upgrade():
    apply_in_own_transactions(
        [("receipt email tables", _create)],
        tables=("receipt_emails", "receipt_email_settings"),
    )


def downgrade():
    apply_in_own_transactions(
        [
            (
                "receipt email tables",
                lambda conn: conn.execute(
                    text("DROP TABLE IF EXISTS receipt_emails, receipt_email_settings")
                ),
            )
        ],
        tables=("receipt_emails", "receipt_email_settings"),
    )
