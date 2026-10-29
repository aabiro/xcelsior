"""Give the public deletion receipt a tenant, like every other privacy table.

145 created `facebook_deletion_confirmations` without the companion's tenant
column (§4.4.10). The tenant follows 090's rule for privacy rows: the deletion
request's tenant when the receipt names one, otherwise the receipt's own code
hash, which is as unlinkable as the subject reference hash it stands beside.

Deploys are blue-green, so the 145 writer keeps inserting without a tenant
after this commits. A BEFORE INSERT trigger derives the tenant by the same rule
for any row that arrives without one; otherwise those inserts would fail, and
`_remember_quietly` would hide it behind a receipt code that never opens. The
trigger can go once no 145 writer is deployed.
"""

from alembic import op

revision = "146"
down_revision = "145"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute(
        "ALTER TABLE facebook_deletion_confirmations ADD COLUMN IF NOT EXISTS tenant_id TEXT"
    )
    op.execute("""
        CREATE OR REPLACE FUNCTION facebook_deletion_confirmations_default_tenant()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW.tenant_id IS NULL THEN
                NEW.tenant_id := COALESCE(
                    (SELECT r.tenant_id FROM privacy_deletion_requests r
                      WHERE r.request_id = NEW.request_id),
                    NEW.code_hash
                );
            END IF;
            RETURN NEW;
        END
        $$
        """)
    op.execute("""
        DROP TRIGGER IF EXISTS facebook_deletion_confirmations_default_tenant
            ON facebook_deletion_confirmations
        """)
    op.execute("""
        CREATE TRIGGER facebook_deletion_confirmations_default_tenant
            BEFORE INSERT ON facebook_deletion_confirmations
            FOR EACH ROW EXECUTE FUNCTION facebook_deletion_confirmations_default_tenant()
        """)
    op.execute("""
        UPDATE facebook_deletion_confirmations c
           SET tenant_id = COALESCE(
                   (SELECT r.tenant_id FROM privacy_deletion_requests r
                     WHERE r.request_id = c.request_id),
                   c.code_hash
               )
         WHERE c.tenant_id IS NULL
        """)
    op.execute("ALTER TABLE facebook_deletion_confirmations ALTER COLUMN tenant_id SET NOT NULL")
    op.execute("""
        CREATE INDEX IF NOT EXISTS facebook_deletion_confirmations_tenant
            ON facebook_deletion_confirmations (tenant_id, created_at)
        """)


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("DROP INDEX IF EXISTS facebook_deletion_confirmations_tenant")
    op.execute("""
        DROP TRIGGER IF EXISTS facebook_deletion_confirmations_default_tenant
            ON facebook_deletion_confirmations
        """)
    op.execute("DROP FUNCTION IF EXISTS facebook_deletion_confirmations_default_tenant()")
    op.execute("ALTER TABLE facebook_deletion_confirmations DROP COLUMN IF EXISTS tenant_id")
