"""A host's GPU identity baseline, and holds that stop automatic re-verification.

Verification claimed to cross-check the GPU serial, but the agent never sent a
serial or UUID, so the check compared nothing and passed. The agent now sends
every GPU's UUID and serial, and the server needs somewhere to keep the first
identity it accepted:

* `identity_baseline` holds the GPU UUIDs (and serials where the card exposes
  one) recorded at the host's first fully passing verification. Every later
  verification is compared with it, and a mismatch de-verifies the host. An
  empty object means no baseline has been recorded yet.
* `hold_reason` / `hold_until` keep a de-verified host de-verified even when a
  later benchmark passes. A swapped card or an admin rejection holds until an
  admin clears it (`hold_until` NULL); repeated host-fault job failures hold
  until `hold_until`. Empty `hold_reason` means no hold. The reverification
  sweep skips held hosts, so it cannot undo them.

One idempotent unit through `lock_safe`; `downgrade` drops the columns.
"""

from sqlalchemy import text

from migrations.lock_safe import apply_in_own_transactions

revision = "166"
down_revision = "165"
branch_labels = None
depends_on = None


def upgrade():
    apply_in_own_transactions(
        [
            (
                "host_verifications identity baseline and hold",
                lambda conn: conn.execute(
                    text(
                        "ALTER TABLE host_verifications "
                        "ADD COLUMN IF NOT EXISTS identity_baseline JSONB NOT NULL "
                        "DEFAULT '{}'::jsonb, "
                        "ADD COLUMN IF NOT EXISTS hold_reason TEXT NOT NULL DEFAULT '', "
                        "ADD COLUMN IF NOT EXISTS hold_until DOUBLE PRECISION"
                    )
                ),
            )
        ],
        tables=("host_verifications",),
    )


def downgrade():
    apply_in_own_transactions(
        [
            (
                "host_verifications identity baseline and hold",
                lambda conn: conn.execute(
                    text(
                        "ALTER TABLE host_verifications "
                        "DROP COLUMN IF EXISTS identity_baseline, "
                        "DROP COLUMN IF EXISTS hold_reason, "
                        "DROP COLUMN IF EXISTS hold_until"
                    )
                ),
            )
        ],
        tables=("host_verifications",),
    )
