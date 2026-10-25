"""Grant the observation-session registry to the runtime roles that ingest it.

Revision 142 creates the table. A deploy that does not re-run role
provisioning otherwise leaves the first service-role read with no privilege.
Missing roles are skipped so a database that has not provisioned them yet
still upgrades. Downgrade revokes only those grants.
"""

from alembic import op

revision = "143"
down_revision = "142"
branch_labels = None
depends_on = None

_GRANT = """
DO $obs_grant$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'xcelsior_api') THEN
    GRANT SELECT, INSERT, UPDATE ON TABLE host_observation_sessions TO xcelsior_api;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'xcelsior_scheduler') THEN
    GRANT SELECT, INSERT, UPDATE ON TABLE host_observation_sessions TO xcelsior_scheduler;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'xcelsior_reconciler') THEN
    GRANT SELECT, INSERT, UPDATE ON TABLE host_observation_sessions TO xcelsior_reconciler;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'xcelsior_billing') THEN
    GRANT SELECT ON TABLE host_observation_sessions TO xcelsior_billing;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'xcelsior_projector') THEN
    GRANT SELECT ON TABLE host_observation_sessions TO xcelsior_projector;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'xcelsior_readonly') THEN
    GRANT SELECT ON TABLE host_observation_sessions TO xcelsior_readonly;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'xcelsior_migrator') THEN
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE host_observation_sessions TO xcelsior_migrator;
  END IF;
END
$obs_grant$;
"""

_REVOKE = """
DO $obs_grant$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'xcelsior_api') THEN
    REVOKE SELECT, INSERT, UPDATE ON TABLE host_observation_sessions FROM xcelsior_api;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'xcelsior_scheduler') THEN
    REVOKE SELECT, INSERT, UPDATE ON TABLE host_observation_sessions FROM xcelsior_scheduler;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'xcelsior_reconciler') THEN
    REVOKE SELECT, INSERT, UPDATE ON TABLE host_observation_sessions FROM xcelsior_reconciler;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'xcelsior_billing') THEN
    REVOKE SELECT ON TABLE host_observation_sessions FROM xcelsior_billing;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'xcelsior_projector') THEN
    REVOKE SELECT ON TABLE host_observation_sessions FROM xcelsior_projector;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'xcelsior_readonly') THEN
    REVOKE SELECT ON TABLE host_observation_sessions FROM xcelsior_readonly;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'xcelsior_migrator') THEN
    REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLE host_observation_sessions FROM xcelsior_migrator;
  END IF;
END
$obs_grant$;
"""


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute(_GRANT)


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute(_REVOKE)
