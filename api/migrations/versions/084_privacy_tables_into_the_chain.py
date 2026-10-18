"""Bring the two runtime-created privacy tables into the migration chain.

``casl_consent`` and ``user_encryption_keys`` were created lazily by
``privacy.py`` via ``CREATE TABLE IF NOT EXISTS`` on first use, deliberately
outside Alembic. That works on a long-lived database that has already run the
code, and fails everywhere else:

- a database built purely from migrations does not have them, so privacy sink
  deletion raises ``UndefinedTable`` until some unrelated request happens to
  create the table first;
- ``control_plane/db_roles.py`` grants against ``casl_consent``, which cannot
  be granted before it exists;
- the schema is not reviewable from the chain, and no downgrade exists.

The cutover goal is one known checkpoint covering code, database revision and
configuration, so schema ownership moves here. The definitions match the
runtime DDL exactly, and the in-code ``CREATE TABLE IF NOT EXISTS`` calls stay
as harmless no-ops for databases that predate this revision.

Both tables use TIMESTAMPTZ, per companion 4.4.5. An earlier revision of this
migration preserved the runtime DDL's float epoch columns verbatim, on the
reasoning that formalising a table should not change it. That was wrong:
importing a table into the chain unchanged just makes its defect official.
Migration 090 converts databases that already ran the earlier version.
"""

from alembic import op

revision = "084"
down_revision = "083"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS casl_consent (
            consent_id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
            user_id TEXT NOT NULL,
            consent_type TEXT NOT NULL CHECK (consent_type IN ('express', 'implied')),
            purpose TEXT NOT NULL,
            granted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at TIMESTAMPTZ,
            withdrawn_at TIMESTAMPTZ,
            source TEXT DEFAULT '',
            ip_address TEXT DEFAULT '',
            active BOOLEAN DEFAULT TRUE,
            UNIQUE(user_id, purpose)
        )
        """
    )
    # Deletion and consent lookups are both per-user.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_casl_consent_user ON casl_consent (user_id)"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS user_encryption_keys (
            user_id TEXT PRIMARY KEY,
            fernet_key TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            destroyed_at TIMESTAMPTZ,
            active BOOLEAN DEFAULT TRUE
        )
        """
    )


def downgrade() -> None:
    # Deliberately not dropped: these hold consent evidence and the per-user
    # encryption keys that make encrypted personal data recoverable. Dropping
    # them on a downgrade would destroy records that privacy law requires be
    # retained, and would render existing ciphertext permanently unreadable.
    op.execute("DROP INDEX IF EXISTS idx_casl_consent_user")
