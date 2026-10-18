"""Finish the auto-top-up money cutover to integer micros.

``wallets`` carried both representations: ``auto_topup_amount_cad`` /
``auto_topup_threshold_cad`` as double precision, and ``*_micros`` as integers
that a previous migration backfilled but no code ever read. Money in binary
floating point cannot represent most decimal amounts exactly, so every read of
the float pair reintroduced rounding — and the auto-top-up charge was computed
as ``int(amount_cad * 100)``, which truncates.

The application now reads and writes the integer columns (``billing.py``,
``routes/billing.py``); the external contract is unchanged and still speaks
CAD, converted at the boundary by ``money.cad_to_micros`` /
``money.micros_to_cad``.

This migration reconciles the two before retiring the float pair: any row whose
micros value disagrees with its CAD value is rewritten from the CAD figure,
which is what the application has actually been honouring. Only then are the
float columns dropped.

Deliberately not touching ``balance_cad`` / ``balance_micros``: balance has 66
live readers against 17 for the micros column, so that cutover is its own piece
of work and its own migration.
"""

from alembic import op

revision = "086"
down_revision = "085"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    # Reconcile before dropping. In practice the trigger below has kept the two
    # representations identical, so this is a no-op safety net rather than a
    # repair — but it must run before the source column disappears.
    # ROUND before casting: a plain cast truncates, so 10.999999 would land a
    # micro short.
    op.execute(
        """
        UPDATE wallets
           SET auto_topup_amount_micros =
                   ROUND(COALESCE(auto_topup_amount_cad, 0)::numeric * 1000000)::bigint,
               auto_topup_threshold_micros =
                   ROUND(COALESCE(auto_topup_threshold_cad, 0)::numeric * 1000000)::bigint
         WHERE auto_topup_amount_micros IS DISTINCT FROM
                   ROUND(COALESCE(auto_topup_amount_cad, 0)::numeric * 1000000)::bigint
            OR auto_topup_threshold_micros IS DISTINCT FROM
                   ROUND(COALESCE(auto_topup_threshold_cad, 0)::numeric * 1000000)::bigint
        """
    )

    # The projection trigger mirrors every _cad <-> _micros pair, including the
    # auto-top-up pair. PL/pgSQL resolves NEW.<field> at execution time, so
    # dropping the columns without rewriting this function makes *every* wallet
    # INSERT and UPDATE fail with "record \"new\" has no field
    # auto_topup_amount_cad" — that is, it takes billing down. Replace the
    # function first, in the same transaction as the drop.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION wallets_project_money() RETURNS trigger AS $BODY$

        BEGIN
            IF TG_OP = 'INSERT' THEN
                
            IF NEW.balance_micros IS NULL THEN
                NEW.balance_micros := CASE WHEN NEW.balance_cad IS NULL THEN NULL ELSE round(NEW.balance_cad::numeric * 1000000)::bigint END;
            ELSE
                NEW.balance_cad := CASE WHEN NEW.balance_micros IS NULL THEN NULL ELSE (NEW.balance_micros::numeric / 1000000)::double precision END;
            END IF;
            
            IF NEW.total_deposited_micros IS NULL THEN
                NEW.total_deposited_micros := CASE WHEN NEW.total_deposited_cad IS NULL THEN NULL ELSE round(NEW.total_deposited_cad::numeric * 1000000)::bigint END;
            ELSE
                NEW.total_deposited_cad := CASE WHEN NEW.total_deposited_micros IS NULL THEN NULL ELSE (NEW.total_deposited_micros::numeric / 1000000)::double precision END;
            END IF;
            
            IF NEW.total_spent_micros IS NULL THEN
                NEW.total_spent_micros := CASE WHEN NEW.total_spent_cad IS NULL THEN NULL ELSE round(NEW.total_spent_cad::numeric * 1000000)::bigint END;
            ELSE
                NEW.total_spent_cad := CASE WHEN NEW.total_spent_micros IS NULL THEN NULL ELSE (NEW.total_spent_micros::numeric / 1000000)::double precision END;
            END IF;
            
            IF NEW.total_refunded_micros IS NULL THEN
                NEW.total_refunded_micros := CASE WHEN NEW.total_refunded_cad IS NULL THEN NULL ELSE round(NEW.total_refunded_cad::numeric * 1000000)::bigint END;
            ELSE
                NEW.total_refunded_cad := CASE WHEN NEW.total_refunded_micros IS NULL THEN NULL ELSE (NEW.total_refunded_micros::numeric / 1000000)::double precision END;
            END IF;
            
            
            
            ELSE
                
            IF NEW.balance_micros IS DISTINCT FROM OLD.balance_micros THEN
                NEW.balance_cad := CASE WHEN NEW.balance_micros IS NULL THEN NULL ELSE (NEW.balance_micros::numeric / 1000000)::double precision END;
            ELSIF NEW.balance_cad IS DISTINCT FROM OLD.balance_cad THEN
                NEW.balance_micros := CASE WHEN NEW.balance_cad IS NULL THEN NULL ELSE round(NEW.balance_cad::numeric * 1000000)::bigint END;
            END IF;
            
            IF NEW.total_deposited_micros IS DISTINCT FROM OLD.total_deposited_micros THEN
                NEW.total_deposited_cad := CASE WHEN NEW.total_deposited_micros IS NULL THEN NULL ELSE (NEW.total_deposited_micros::numeric / 1000000)::double precision END;
            ELSIF NEW.total_deposited_cad IS DISTINCT FROM OLD.total_deposited_cad THEN
                NEW.total_deposited_micros := CASE WHEN NEW.total_deposited_cad IS NULL THEN NULL ELSE round(NEW.total_deposited_cad::numeric * 1000000)::bigint END;
            END IF;
            
            IF NEW.total_spent_micros IS DISTINCT FROM OLD.total_spent_micros THEN
                NEW.total_spent_cad := CASE WHEN NEW.total_spent_micros IS NULL THEN NULL ELSE (NEW.total_spent_micros::numeric / 1000000)::double precision END;
            ELSIF NEW.total_spent_cad IS DISTINCT FROM OLD.total_spent_cad THEN
                NEW.total_spent_micros := CASE WHEN NEW.total_spent_cad IS NULL THEN NULL ELSE round(NEW.total_spent_cad::numeric * 1000000)::bigint END;
            END IF;
            
            IF NEW.total_refunded_micros IS DISTINCT FROM OLD.total_refunded_micros THEN
                NEW.total_refunded_cad := CASE WHEN NEW.total_refunded_micros IS NULL THEN NULL ELSE (NEW.total_refunded_micros::numeric / 1000000)::double precision END;
            ELSIF NEW.total_refunded_cad IS DISTINCT FROM OLD.total_refunded_cad THEN
                NEW.total_refunded_micros := CASE WHEN NEW.total_refunded_cad IS NULL THEN NULL ELSE round(NEW.total_refunded_cad::numeric * 1000000)::bigint END;
            END IF;
            
            
            
            END IF;
            RETURN NEW;
        END
        $BODY$ LANGUAGE plpgsql
        """
    )

    op.execute("ALTER TABLE wallets DROP COLUMN IF EXISTS auto_topup_amount_cad")
    op.execute("ALTER TABLE wallets DROP COLUMN IF EXISTS auto_topup_threshold_cad")


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    op.execute(
        "ALTER TABLE wallets ADD COLUMN IF NOT EXISTS auto_topup_amount_cad "
        "DOUBLE PRECISION DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE wallets ADD COLUMN IF NOT EXISTS auto_topup_threshold_cad "
        "DOUBLE PRECISION DEFAULT 0"
    )
    # Reconstructed from micros, which is authoritative after this migration.
    op.execute(
        """
        UPDATE wallets
           SET auto_topup_amount_cad = COALESCE(auto_topup_amount_micros, 0) / 1000000.0,
               auto_topup_threshold_cad = COALESCE(auto_topup_threshold_micros, 0) / 1000000.0
        """
    )
    # Restore the projection for the reinstated columns.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION wallets_project_money() RETURNS trigger AS $BODY$

        BEGIN
            IF TG_OP = 'INSERT' THEN
                
            IF NEW.balance_micros IS NULL THEN
                NEW.balance_micros := CASE WHEN NEW.balance_cad IS NULL THEN NULL ELSE round(NEW.balance_cad::numeric * 1000000)::bigint END;
            ELSE
                NEW.balance_cad := CASE WHEN NEW.balance_micros IS NULL THEN NULL ELSE (NEW.balance_micros::numeric / 1000000)::double precision END;
            END IF;
            
            IF NEW.total_deposited_micros IS NULL THEN
                NEW.total_deposited_micros := CASE WHEN NEW.total_deposited_cad IS NULL THEN NULL ELSE round(NEW.total_deposited_cad::numeric * 1000000)::bigint END;
            ELSE
                NEW.total_deposited_cad := CASE WHEN NEW.total_deposited_micros IS NULL THEN NULL ELSE (NEW.total_deposited_micros::numeric / 1000000)::double precision END;
            END IF;
            
            IF NEW.total_spent_micros IS NULL THEN
                NEW.total_spent_micros := CASE WHEN NEW.total_spent_cad IS NULL THEN NULL ELSE round(NEW.total_spent_cad::numeric * 1000000)::bigint END;
            ELSE
                NEW.total_spent_cad := CASE WHEN NEW.total_spent_micros IS NULL THEN NULL ELSE (NEW.total_spent_micros::numeric / 1000000)::double precision END;
            END IF;
            
            IF NEW.total_refunded_micros IS NULL THEN
                NEW.total_refunded_micros := CASE WHEN NEW.total_refunded_cad IS NULL THEN NULL ELSE round(NEW.total_refunded_cad::numeric * 1000000)::bigint END;
            ELSE
                NEW.total_refunded_cad := CASE WHEN NEW.total_refunded_micros IS NULL THEN NULL ELSE (NEW.total_refunded_micros::numeric / 1000000)::double precision END;
            END IF;
            
            IF NEW.auto_topup_amount_micros IS NULL THEN
                NEW.auto_topup_amount_micros := CASE WHEN NEW.auto_topup_amount_cad IS NULL THEN NULL ELSE round(NEW.auto_topup_amount_cad::numeric * 1000000)::bigint END;
            ELSE
                NEW.auto_topup_amount_cad := CASE WHEN NEW.auto_topup_amount_micros IS NULL THEN NULL ELSE (NEW.auto_topup_amount_micros::numeric / 1000000)::double precision END;
            END IF;
            
            IF NEW.auto_topup_threshold_micros IS NULL THEN
                NEW.auto_topup_threshold_micros := CASE WHEN NEW.auto_topup_threshold_cad IS NULL THEN NULL ELSE round(NEW.auto_topup_threshold_cad::numeric * 1000000)::bigint END;
            ELSE
                NEW.auto_topup_threshold_cad := CASE WHEN NEW.auto_topup_threshold_micros IS NULL THEN NULL ELSE (NEW.auto_topup_threshold_micros::numeric / 1000000)::double precision END;
            END IF;
            
            ELSE
                
            IF NEW.balance_micros IS DISTINCT FROM OLD.balance_micros THEN
                NEW.balance_cad := CASE WHEN NEW.balance_micros IS NULL THEN NULL ELSE (NEW.balance_micros::numeric / 1000000)::double precision END;
            ELSIF NEW.balance_cad IS DISTINCT FROM OLD.balance_cad THEN
                NEW.balance_micros := CASE WHEN NEW.balance_cad IS NULL THEN NULL ELSE round(NEW.balance_cad::numeric * 1000000)::bigint END;
            END IF;
            
            IF NEW.total_deposited_micros IS DISTINCT FROM OLD.total_deposited_micros THEN
                NEW.total_deposited_cad := CASE WHEN NEW.total_deposited_micros IS NULL THEN NULL ELSE (NEW.total_deposited_micros::numeric / 1000000)::double precision END;
            ELSIF NEW.total_deposited_cad IS DISTINCT FROM OLD.total_deposited_cad THEN
                NEW.total_deposited_micros := CASE WHEN NEW.total_deposited_cad IS NULL THEN NULL ELSE round(NEW.total_deposited_cad::numeric * 1000000)::bigint END;
            END IF;
            
            IF NEW.total_spent_micros IS DISTINCT FROM OLD.total_spent_micros THEN
                NEW.total_spent_cad := CASE WHEN NEW.total_spent_micros IS NULL THEN NULL ELSE (NEW.total_spent_micros::numeric / 1000000)::double precision END;
            ELSIF NEW.total_spent_cad IS DISTINCT FROM OLD.total_spent_cad THEN
                NEW.total_spent_micros := CASE WHEN NEW.total_spent_cad IS NULL THEN NULL ELSE round(NEW.total_spent_cad::numeric * 1000000)::bigint END;
            END IF;
            
            IF NEW.total_refunded_micros IS DISTINCT FROM OLD.total_refunded_micros THEN
                NEW.total_refunded_cad := CASE WHEN NEW.total_refunded_micros IS NULL THEN NULL ELSE (NEW.total_refunded_micros::numeric / 1000000)::double precision END;
            ELSIF NEW.total_refunded_cad IS DISTINCT FROM OLD.total_refunded_cad THEN
                NEW.total_refunded_micros := CASE WHEN NEW.total_refunded_cad IS NULL THEN NULL ELSE round(NEW.total_refunded_cad::numeric * 1000000)::bigint END;
            END IF;
            
            IF NEW.auto_topup_amount_micros IS DISTINCT FROM OLD.auto_topup_amount_micros THEN
                NEW.auto_topup_amount_cad := CASE WHEN NEW.auto_topup_amount_micros IS NULL THEN NULL ELSE (NEW.auto_topup_amount_micros::numeric / 1000000)::double precision END;
            ELSIF NEW.auto_topup_amount_cad IS DISTINCT FROM OLD.auto_topup_amount_cad THEN
                NEW.auto_topup_amount_micros := CASE WHEN NEW.auto_topup_amount_cad IS NULL THEN NULL ELSE round(NEW.auto_topup_amount_cad::numeric * 1000000)::bigint END;
            END IF;
            
            IF NEW.auto_topup_threshold_micros IS DISTINCT FROM OLD.auto_topup_threshold_micros THEN
                NEW.auto_topup_threshold_cad := CASE WHEN NEW.auto_topup_threshold_micros IS NULL THEN NULL ELSE (NEW.auto_topup_threshold_micros::numeric / 1000000)::double precision END;
            ELSIF NEW.auto_topup_threshold_cad IS DISTINCT FROM OLD.auto_topup_threshold_cad THEN
                NEW.auto_topup_threshold_micros := CASE WHEN NEW.auto_topup_threshold_cad IS NULL THEN NULL ELSE round(NEW.auto_topup_threshold_cad::numeric * 1000000)::bigint END;
            END IF;
            
            END IF;
            RETURN NEW;
        END
        $BODY$ LANGUAGE plpgsql
        """
    )
