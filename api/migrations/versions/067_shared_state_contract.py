"""Lightning and Slurm shared-state contracts (Track B B9.3b-2 / B9.3c).

Migration `060_shared_state_to_pg` moved these two tables out of SQLite and
a JSON file, but carried over their file-era shapes. The data-architecture
companion §10.1 lists what each actually needs; this migration supplies the
structural half.

**`ln_deposits`** (companion §10.1 "Lightning requirements"):

- `tenant_id`, projected from `customer_id` under the same transitional
  single-user tenancy rule migration 054 used for jobs — billing already
  treats the paying customer as the tenant;
- an explicit `currency`, because an amount without one is not money;
- `UNIQUE NULLS NOT DISTINCT (payment_hash)` — a provider payment hash
  identifies one payment, so two deposit rows claiming the same hash is a
  double-credit waiting to happen. `NULLS NOT DISTINCT` requires
  PostgreSQL 15+, inside the §23.1 PostgreSQL 16 baseline. Note that
  migration 060 already declares `payment_hash NOT NULL`, so the null
  clause is currently inert; it is stated anyway so the invariant survives
  a future migration that relaxes that column;
- `wallet_ledger_entry_id`, so a credited deposit points at the exact
  ledger row that credited it and "credited but no ledger entry" becomes a
  query rather than an audit exercise;
- a `status` CHECK, so a typo cannot invent a lifecycle state;
- an **immutability trigger**: once written, the amount, currency, payment
  hash, customer, and expiry of a deposit cannot change. The companion
  calls for "immutable expected amount/currency and expiry" because those
  are the terms the customer paid against — a later UPDATE that edits them
  rewrites history under a settled payment.

**`slurm_job_mappings`** (companion §10.1 "Slurm requirements"):

- `tenant_id`, `cluster_id`, `xcelsior_attempt_id`, `desired_state`,
  `observed_state`, `submit_idempotency_key`, `version`, and typed
  timestamps;
- `UNIQUE (cluster_id, slurm_job_id)` — without it the *same* Slurm job can
  be mapped to two Xcelsior jobs, and both will act on its status;
- `UNIQUE (xcelsior_attempt_id)` and `UNIQUE (tenant_id,
  submit_idempotency_key)`, so a retried submit cannot create a second
  external job.

Both tables are empty in every environment checked (dev, pytest), so the
backfills are no-ops there; they are written to be correct anyway because
production was not verifiable.

Expand-only. The legacy `created_at` floats stay until contract phase
(Track B §B16.2).

Revision ID: 067
Revises: 066
Create Date: 2026-07-22
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "067"
down_revision: Union[str, None] = "066"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

LN_STATUSES = ("pending", "paid", "credited", "expired", "failed")
SLURM_DESIRED = ("running", "cancelled")
SLURM_OBSERVED = (
    "pending", "running", "completed", "failed", "cancelled", "unknown",
)


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    _upgrade_ln_deposits()
    _upgrade_slurm_mappings()


# ────────────────────────── ln_deposits ──────────────────────────


def _upgrade_ln_deposits() -> None:
    op.execute("ALTER TABLE ln_deposits ADD COLUMN IF NOT EXISTS tenant_id TEXT")
    op.execute(
        "ALTER TABLE ln_deposits ADD COLUMN IF NOT EXISTS currency CHAR(3) "
        "NOT NULL DEFAULT 'CAD'"
    )
    op.execute(
        "ALTER TABLE ln_deposits ADD COLUMN IF NOT EXISTS wallet_ledger_entry_id TEXT"
    )

    # Transitional tenancy projection, same rule as migration 054 for jobs.
    op.execute(
        "UPDATE ln_deposits SET tenant_id = customer_id WHERE tenant_id IS NULL"
    )

    bind = op.get_bind()
    unprojected = bind.execute(
        sa.text("SELECT count(*) FROM ln_deposits WHERE tenant_id IS NULL")
    ).scalar_one()
    if unprojected:
        raise RuntimeError(
            f"migration 067 left {unprojected} ln_deposits rows without a "
            "tenant_id; every deposit must have a paying customer."
        )

    # A payment hash identifies one provider payment. Duplicates would let
    # two deposit rows credit one payment, so surface them rather than
    # letting the index creation fail with an opaque error.
    dupes = (
        bind.execute(
            sa.text(
                """
                SELECT payment_hash, count(*) AS n
                  FROM ln_deposits
                 GROUP BY payment_hash
                HAVING count(*) > 1
                 LIMIT 10
                """
            )
        )
        .mappings()
        .all()
    )
    if dupes:
        raise RuntimeError(
            f"migration 067: ln_deposits has duplicate payment_hash values "
            f"{[dict(d) for d in dupes]}. Each is a potential double credit — "
            "resolve them before enforcing uniqueness."
        )

    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_ln_deposits_payment_hash "
        "ON ln_deposits (payment_hash) NULLS NOT DISTINCT"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_ln_deposits_tenant_created "
        "ON ln_deposits (tenant_id, created_at_ts DESC)"
    )

    statuses = ", ".join(f"'{s}'" for s in LN_STATUSES)
    op.execute(
        f"""
        ALTER TABLE ln_deposits
        ADD CONSTRAINT ck_ln_deposits_status
        CHECK (status IN ({statuses})) NOT VALID
        """
    )
    op.execute("ALTER TABLE ln_deposits VALIDATE CONSTRAINT ck_ln_deposits_status")

    # Immutable payment terms. A settled deposit's amount, currency, payer,
    # payment hash, and expiry are what the customer paid against; an
    # UPDATE that edits them rewrites history under a completed payment.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION ln_deposits_guard_immutable()
        RETURNS trigger AS $$
        BEGIN
            IF NEW.customer_id IS DISTINCT FROM OLD.customer_id
               OR NEW.currency IS DISTINCT FROM OLD.currency
               OR NEW.amount_msat IS DISTINCT FROM OLD.amount_msat
               OR NEW.payment_hash IS DISTINCT FROM OLD.payment_hash
               OR NEW.expires_at_ts IS DISTINCT FROM OLD.expires_at_ts
            THEN
                RAISE EXCEPTION
                    'ln_deposits: payment terms are immutable (deposit_id=%)',
                    OLD.deposit_id
                    USING ERRCODE = 'integrity_constraint_violation';
            END IF;
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql
        """
    )
    op.execute("DROP TRIGGER IF EXISTS trg_ln_deposits_immutable ON ln_deposits")
    # AFTER the projection trigger (066) alphabetically, so the typed
    # columns are already populated when this compares them.
    op.execute(
        """
        CREATE TRIGGER trg_ln_deposits_immutable
        BEFORE UPDATE ON ln_deposits
        FOR EACH ROW EXECUTE FUNCTION ln_deposits_guard_immutable()
        """
    )


# ─────────────────────── slurm_job_mappings ───────────────────────


def _upgrade_slurm_mappings() -> None:
    for ddl in (
        "ADD COLUMN IF NOT EXISTS tenant_id TEXT",
        "ADD COLUMN IF NOT EXISTS cluster_id TEXT",
        "ADD COLUMN IF NOT EXISTS xcelsior_attempt_id UUID",
        "ADD COLUMN IF NOT EXISTS desired_state TEXT",
        "ADD COLUMN IF NOT EXISTS observed_state TEXT",
        "ADD COLUMN IF NOT EXISTS submit_idempotency_key TEXT",
        "ADD COLUMN IF NOT EXISTS version BIGINT NOT NULL DEFAULT 1",
        "ADD COLUMN IF NOT EXISTS submitted_at TIMESTAMPTZ",
        "ADD COLUMN IF NOT EXISTS last_observed_at TIMESTAMPTZ",
        "ADD COLUMN IF NOT EXISTS terminal_at TIMESTAMPTZ",
        "ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb",
    ):
        op.execute(f"ALTER TABLE slurm_job_mappings {ddl}")

    # Backfill legacy rows into the contract shape. `cluster_id` has no
    # legacy source — 'unknown' is honest and keeps the composite unique
    # index meaningful for rows written from now on.
    op.execute(
        """
        UPDATE slurm_job_mappings
           SET cluster_id = COALESCE(cluster_id, 'unknown'),
               desired_state = COALESCE(desired_state, 'running'),
               observed_state = COALESCE(observed_state, 'unknown'),
               submitted_at = COALESCE(
                   submitted_at,
                   CASE WHEN created_at IS NULL OR created_at <= 0
                        THEN clock_timestamp()
                        ELSE to_timestamp(created_at) END
               ),
               submit_idempotency_key = COALESCE(
                   submit_idempotency_key, 'legacy:' || xcelsior_job_id
               )
        """
    )

    bind = op.get_bind()
    dupes = (
        bind.execute(
            sa.text(
                """
                SELECT cluster_id, slurm_job_id, count(*) AS n
                  FROM slurm_job_mappings
                 GROUP BY cluster_id, slurm_job_id
                HAVING count(*) > 1
                 LIMIT 10
                """
            )
        )
        .mappings()
        .all()
    )
    if dupes:
        raise RuntimeError(
            f"migration 067: slurm_job_mappings maps one external job to "
            f"several Xcelsior jobs {[dict(d) for d in dupes]}. Both would "
            "act on its status; resolve before enforcing uniqueness."
        )

    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_slurm_cluster_job "
        "ON slurm_job_mappings (cluster_id, slurm_job_id)"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_slurm_attempt "
        "ON slurm_job_mappings (xcelsior_attempt_id) "
        "WHERE xcelsior_attempt_id IS NOT NULL"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_slurm_submit_idempotency "
        "ON slurm_job_mappings (tenant_id, submit_idempotency_key) "
        "WHERE tenant_id IS NOT NULL AND submit_idempotency_key IS NOT NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_slurm_open "
        "ON slurm_job_mappings (observed_state, last_observed_at) "
        "WHERE terminal_at IS NULL"
    )

    desired = ", ".join(f"'{s}'" for s in SLURM_DESIRED)
    observed = ", ".join(f"'{s}'" for s in SLURM_OBSERVED)
    op.execute(
        f"""
        ALTER TABLE slurm_job_mappings
        ADD CONSTRAINT ck_slurm_desired_state
        CHECK (desired_state IS NULL OR desired_state IN ({desired})) NOT VALID
        """
    )
    op.execute(
        "ALTER TABLE slurm_job_mappings VALIDATE CONSTRAINT ck_slurm_desired_state"
    )
    op.execute(
        f"""
        ALTER TABLE slurm_job_mappings
        ADD CONSTRAINT ck_slurm_observed_state
        CHECK (observed_state IS NULL OR observed_state IN ({observed})) NOT VALID
        """
    )
    op.execute(
        "ALTER TABLE slurm_job_mappings VALIDATE CONSTRAINT ck_slurm_observed_state"
    )


def downgrade() -> None:
    """Reverse the expand.

    **Lossy, by nature.** Dropping `cluster_id` discards the only thing
    that distinguishes two mappings to the same external `slurm_job_id` on
    different clusters. A subsequent re-upgrade re-defaults both to
    'unknown', they collide, and `_upgrade_slurm_mappings` refuses with a
    named conflict rather than silently dropping one — which is the
    intended behaviour, and a worked example of blueprint §13.8: "do not
    assume destructive `downgrade` can safely restore data". Roll forward
    with a fix migration instead of round-tripping populated data.
    """
    op.execute("SET lock_timeout = '5s'")

    op.execute("DROP TRIGGER IF EXISTS trg_ln_deposits_immutable ON ln_deposits")
    op.execute("DROP FUNCTION IF EXISTS ln_deposits_guard_immutable()")
    op.execute("DROP INDEX IF EXISTS uq_ln_deposits_payment_hash")
    op.execute("DROP INDEX IF EXISTS idx_ln_deposits_tenant_created")
    op.execute("ALTER TABLE ln_deposits DROP CONSTRAINT IF EXISTS ck_ln_deposits_status")
    for col in ("tenant_id", "currency", "wallet_ledger_entry_id"):
        op.execute(f"ALTER TABLE ln_deposits DROP COLUMN IF EXISTS {col}")

    op.execute("DROP INDEX IF EXISTS uq_slurm_cluster_job")
    op.execute("DROP INDEX IF EXISTS uq_slurm_attempt")
    op.execute("DROP INDEX IF EXISTS uq_slurm_submit_idempotency")
    op.execute("DROP INDEX IF EXISTS idx_slurm_open")
    op.execute(
        "ALTER TABLE slurm_job_mappings DROP CONSTRAINT IF EXISTS ck_slurm_desired_state"
    )
    op.execute(
        "ALTER TABLE slurm_job_mappings DROP CONSTRAINT IF EXISTS ck_slurm_observed_state"
    )
    for col in (
        "tenant_id", "cluster_id", "xcelsior_attempt_id", "desired_state",
        "observed_state", "submit_idempotency_key", "version", "submitted_at",
        "last_observed_at", "terminal_at", "metadata",
    ):
        op.execute(f"ALTER TABLE slurm_job_mappings DROP COLUMN IF EXISTS {col}")
