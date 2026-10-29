"""Scope deletion receipts to the request's workspace without retaining identity."""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "148"
down_revision = "147"
branch_labels = None
depends_on = None


def _expand(conn):
    conn.execute(text("ALTER TABLE facebook_deletion_confirmations ADD COLUMN IF NOT EXISTS tenant_id TEXT"))
    conn.execute(text("""CREATE OR REPLACE FUNCTION facebook_deletion_receipt_tenant()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW.request_id IS NULL THEN
                NEW.tenant_id := 'system:facebook-deletion';
            ELSE
                SELECT tenant_id INTO NEW.tenant_id FROM privacy_deletion_requests
                    WHERE request_id=NEW.request_id;
                IF NEW.tenant_id IS NULL THEN
                    RAISE EXCEPTION 'Deletion receipt requires its durable request tenant'
                        USING ERRCODE='foreign_key_violation';
                END IF;
            END IF;
            RETURN NEW;
        END $$"""))
    conn.execute(text("DROP TRIGGER IF EXISTS facebook_deletion_receipt_tenant ON facebook_deletion_confirmations"))
    conn.execute(text("CREATE TRIGGER facebook_deletion_receipt_tenant BEFORE INSERT OR UPDATE ON facebook_deletion_confirmations FOR EACH ROW EXECUTE FUNCTION facebook_deletion_receipt_tenant()"))


def upgrade():
    tables = ["facebook_deletion_confirmations", "privacy_deletion_requests"]
    apply_in_own_transactions([("expand receipt scope", _expand)], tables=tables)
    state = {"count": 1}

    def batch(conn):
        result = conn.execute(text("""WITH pending AS (
            SELECT code_hash FROM facebook_deletion_confirmations WHERE tenant_id IS NULL
            ORDER BY code_hash LIMIT 500 FOR UPDATE SKIP LOCKED)
            UPDATE facebook_deletion_confirmations f SET request_id=f.request_id
            FROM pending p WHERE f.code_hash=p.code_hash"""))
        state["count"] = result.rowcount

    while state["count"]:
        apply_in_own_transactions([("scope receipt batch", batch)], tables=tables)

    def constrain(conn):
        conn.execute(text("ALTER TABLE facebook_deletion_confirmations ALTER COLUMN tenant_id SET NOT NULL"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS facebook_deletion_confirmations_tenant ON facebook_deletion_confirmations(tenant_id,created_at)"))

    apply_in_own_transactions([("constrain receipt scope", constrain)], tables=tables)


def downgrade():
    def remove(conn):
        conn.execute(text("DROP TRIGGER IF EXISTS facebook_deletion_receipt_tenant ON facebook_deletion_confirmations"))
        conn.execute(text("DROP FUNCTION IF EXISTS facebook_deletion_receipt_tenant()"))
        conn.execute(text("ALTER TABLE facebook_deletion_confirmations DROP COLUMN IF EXISTS tenant_id"))
    apply_in_own_transactions([("remove receipt scope", remove)], tables=["facebook_deletion_confirmations"])
