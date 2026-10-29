"""Remove plaintext root passwords from instances that are not running.

Until instance_credentials.protect_ssh_status, the worker's SSH report was
stored as-is, root password included, in ``jobs.payload -> ssh_status``. New
reports are encrypted. The old ones serve no purpose once an instance is no
longer running: the worker sets a fresh random password every time it sets up
SSH, so a stopped instance gets a new one when it resumes.

Running instances are left alone (their password is still the live one, and
the API already hides it outside the owner's detail view). Production had 34
such rows, all in finished or stopped instances, and no copy anywhere else.
"""

from alembic import op

revision = "120"
down_revision = "119"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")
    op.execute(
        """
        UPDATE jobs
           SET payload = jsonb_set(payload, '{ssh_status}', (payload -> 'ssh_status') - 'root_password')
         WHERE status IN ('completed', 'failed', 'cancelled', 'terminated', 'stopped')
           AND jsonb_typeof(payload -> 'ssh_status') = 'object'
           AND (payload -> 'ssh_status') ? 'root_password'
        """
    )


def downgrade() -> None:
    # Deleted secrets are not restorable, and must not be.
    pass
