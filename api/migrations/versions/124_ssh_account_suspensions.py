"""Expose suspended account identifiers to the SSH gateway.

The gateway must drop relays when an account is suspended without receiving
the rest of the state table. The view runs with its owner's rights, so the
restricted gateway role can read identifiers and nothing else.
"""

from alembic import op

revision = "124"
down_revision = "123"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("""
        CREATE OR REPLACE VIEW public.ssh_account_suspensions
        WITH (security_invoker = false) AS
        SELECT substr(namespace, length('security.suspended.') + 1) AS identifier
          FROM state
         WHERE namespace LIKE 'security.suspended.%'
           AND (
                position('"active"\\: true' IN payload::text) > 0
                OR position('"active"\\:true' IN payload::text) > 0
           )
    """)
    op.execute("REVOKE ALL ON public.ssh_account_suspensions FROM PUBLIC")
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'xcelsior_ssh_gateway') THEN
                GRANT SELECT ON public.ssh_account_suspensions TO xcelsior_ssh_gateway;
            END IF;
        END $$
    """)


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("DROP VIEW IF EXISTS public.ssh_account_suspensions")
