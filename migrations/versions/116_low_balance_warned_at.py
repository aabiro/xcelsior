"""Give the low-balance warning watermark its own column.

`wallets.grace_until` carries two incompatible meanings on one column, and the
overload is a trap for whoever implements the feature it is named after.

## What it means today

`serverless/service.py` refuses work at zero balance *unless* the customer is
inside a grace window:

    if wallet["balance_cad"] <= 0 and float(wallet.get("grace_until") or 0) < time.time():
        raise WalletPreflightError("Insufficient wallet balance …", 402)

And `BillingEngine._warn_low_balance` writes `now` into the same column as a
rate-limit watermark, with a comment saying so: *"Reuse grace_until as last-warn
watermark when still positive balance."*

Nothing anywhere sets it to a **future** time. Every write is either `0` (on
top-up, reactivation, reset) or `now` (a warning stamp, already in the past by
the time anything reads it). So the grace clause is a condition that cannot be
false for the reason it is written — it reads as "unless they are in a grace
period", and there is no grace period.

That alone is only misleading. The trap is what happens when someone fixes it.

## The trap

Grant a real grace window — `grace_until = now + 3 days` — and the rate-limiter
reads that future value as *"last warned at now + 3 days"*:

    last = float(wallet.get("grace_until") or 0)
    if last and now - last < 6 * 3600 and balance_cad > 0:
        return {"warned": False, …, "reason": "rate_limited"}

`now - last` is **negative**, which is less than six hours, so the warning is
suppressed — for the entire grace period, and beyond it until the value falls
six hours into the past. A customer would stop being told their balance is low
at precisely the moment that warning matters most: while a grace window they did
not ask for is quietly running out.

Nobody would find that by reading either function alone. Both are correct in
isolation; the defect lives in the column they share.

## Why a column and not a convention

A comment saying "only ever store a past timestamp here" is the mechanism that
already failed — it is present, it is accurate, and it describes a landmine
rather than removing it. Two meanings need two columns.

`low_balance_warned_at` is nullable with no default and **no backfill**. The
watermark is ephemeral: it exists to suppress a duplicate notification within
six hours, and starting empty means at worst one extra low-balance email to
customers who were warned just before this ran. Copying `grace_until` forward
would import exactly the wrong values — every `0` and every past stamp — for a
saving of one notification.

`grace_until` is left alone. It keeps its name and its only real meaning, and
becomes safe to set to a future time, which is what the serverless check has
been asking for.
"""

from alembic import op

revision = "116"
down_revision = "115"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    # Nullable, no default: on PostgreSQL 11+ this is a catalog-only change and
    # does not rewrite the table.
    op.execute(
        """
        ALTER TABLE wallets
          ADD COLUMN IF NOT EXISTS low_balance_warned_at DOUBLE PRECISION
        """
    )

    # A watermark is a moment, not a duration or a sentinel. `0` was meaningful
    # on `grace_until` because that column also carried "no grace"; here NULL
    # says "never warned" and any stored value is a real instant.
    op.execute(
        """
        ALTER TABLE wallets
          DROP CONSTRAINT IF EXISTS ck_wallets_low_balance_warned_at_positive
        """
    )
    op.execute(
        """
        ALTER TABLE wallets
          ADD CONSTRAINT ck_wallets_low_balance_warned_at_positive
          CHECK (low_balance_warned_at IS NULL OR low_balance_warned_at > 0)
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE wallets
          DROP CONSTRAINT IF EXISTS ck_wallets_low_balance_warned_at_positive
        """
    )
    op.execute("ALTER TABLE wallets DROP COLUMN IF EXISTS low_balance_warned_at")
