"""Give the public deletion receipt a tenant, like every other privacy table.

145 created `facebook_deletion_confirmations` without the companion's tenant
column (§4.4.10). The tenant follows 090's rule for privacy rows: the deletion
request's tenant when the receipt names one, otherwise the receipt's own code
hash, which is as unlinkable as the subject reference hash it stands beside.
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
    op.execute(
        """
        UPDATE facebook_deletion_confirmations c
           SET tenant_id = COALESCE(
                   (SELECT r.tenant_id FROM privacy_deletion_requests r
                     WHERE r.request_id = c.request_id),
                   c.code_hash
               )
         WHERE c.tenant_id IS NULL
        """
    )
    op.execute("ALTER TABLE facebook_deletion_confirmations ALTER COLUMN tenant_id SET NOT NULL")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS facebook_deletion_confirmations_tenant
            ON facebook_deletion_confirmations (tenant_id, created_at)
        """
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("DROP INDEX IF EXISTS facebook_deletion_confirmations_tenant")
    op.execute("ALTER TABLE facebook_deletion_confirmations DROP COLUMN IF EXISTS tenant_id")
