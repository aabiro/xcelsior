"""Archive Lightning float/epoch projections before removing their columns.

Apply only with all old Lightning writers stopped. Typed payment facts remain
untouched. Unknown payment dates stay NULL; no payment or wallet entry is made.
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "141"
down_revision = "140"
branch_labels = None
depends_on = None

COLUMNS = ("amount_btc", "btc_cad_rate", "created_at", "expires_at", "paid_at", "credited_at")
RECORD = "jsonb_build_object(" + ",".join(f"'{name}',d.{name}" for name in COLUMNS) + ")"

IMMUTABILITY = """CREATE OR REPLACE FUNCTION ln_deposits_guard_immutable()
RETURNS trigger AS $$ BEGIN
    IF ROW(NEW.customer_id,NEW.tenant_id,NEW.currency,NEW.amount_msat,NEW.amount_sats,
           NEW.amount_micros,NEW.amount_cad_minor,NEW.btc_cad_rate_exact,
           NEW.payment_hash,NEW.created_at_ts,NEW.expires_at_ts)
       IS DISTINCT FROM
       ROW(OLD.customer_id,OLD.tenant_id,OLD.currency,OLD.amount_msat,OLD.amount_sats,
           OLD.amount_micros,OLD.amount_cad_minor,OLD.btc_cad_rate_exact,
           OLD.payment_hash,OLD.created_at_ts,OLD.expires_at_ts)
    THEN
        RAISE EXCEPTION 'ln_deposits: payment terms are immutable (deposit_id=%)',
            OLD.deposit_id USING ERRCODE='integrity_constraint_violation';
    END IF;
    RETURN NEW;
END $$ LANGUAGE plpgsql"""


def _contract(conn):
    conn.execute(text("LOCK TABLE ln_deposits IN ACCESS EXCLUSIVE MODE"))
    shape = dict(
        conn.execute(
            text("""SELECT column_name,data_type FROM information_schema.columns
                WHERE table_schema='public' AND table_name='ln_deposits'
                  AND column_name=ANY(:names)"""),
            {"names": list(COLUMNS)},
        ).fetchall()
    )
    archived = conn.execute(
        text("SELECT to_regclass('lightning_retired_projection_archive')")
    ).scalar_one()
    if shape != dict.fromkeys(COLUMNS, "double precision") and not (not shape and archived):
        raise RuntimeError("unexpected Lightning projection shape; contraction refused")
    conn.execute(text("""CREATE TABLE IF NOT EXISTS lightning_retired_projection_archive (
        deposit_id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL CHECK(tenant_id<>''),
        tenant_attributed BOOLEAN NOT NULL,
        record JSONB NOT NULL,
        record_sha256 TEXT NOT NULL CHECK
            (record_sha256=encode(sha256(convert_to(record::text,'UTF8')),'hex')),
        archived_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        source_revision TEXT NOT NULL DEFAULT '141' CHECK(source_revision='141')
    )"""))
    conn.execute(text("""CREATE INDEX IF NOT EXISTS lightning_retired_projection_tenant
        ON lightning_retired_projection_archive(tenant_id,deposit_id)"""))
    if shape:
        conn.execute(text(f"""INSERT INTO lightning_retired_projection_archive
                (deposit_id,tenant_id,tenant_attributed,record,record_sha256)
            SELECT deposit_id,
                COALESCE(NULLIF(tenant_id,''),NULLIF(customer_id,''),
                    'system:unattributed-lightning-archive'),
                COALESCE(NULLIF(tenant_id,''),NULLIF(customer_id,'')) IS NOT NULL,
                record,encode(sha256(convert_to(record::text,'UTF8')),'hex')
            FROM (SELECT d.deposit_id,d.tenant_id,d.customer_id,{RECORD} AS record
                FROM ln_deposits d) evidence
            ON CONFLICT(deposit_id) DO NOTHING"""))
        if conn.execute(text(f"""SELECT 1 FROM ln_deposits d LEFT JOIN
                lightning_retired_projection_archive a USING(deposit_id)
                WHERE a.deposit_id IS NULL OR a.record IS DISTINCT FROM {RECORD}
                LIMIT 1""")).first():
            raise RuntimeError("Lightning projection archive mismatch; contraction refused")
        # Do not discard unexpected dependent views or other schema objects.
        for name in COLUMNS:
            conn.execute(text(f"ALTER TABLE ln_deposits DROP COLUMN {name}"))
    conn.execute(text(IMMUTABILITY))


def upgrade():
    apply_in_own_transactions(
        [("archive Lightning float and epoch projections", _contract)],
        tables=["ln_deposits", "lightning_retired_projection_archive"],
    )


def downgrade():
    # Keep exact historical evidence and the single native writer contract.
    # Returning to an obsolete runtime requires a reviewed compatible release.
    pass
