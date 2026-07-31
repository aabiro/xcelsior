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

``created_at``/``destroyed_at`` on ``user_encryption_keys`` are kept as REAL to
match what the runtime DDL has already created in existing databases; changing
the type here would rewrite a table this migration is only meant to formalise.
``casl_consent`` uses DOUBLE PRECISION because ``privacy.py`` already widens it
in place — REAL quantises epoch seconds by roughly two minutes, which silently
shifted consent expiries.
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
            granted_at DOUBLE PRECISION NOT NULL DEFAULT (extract(epoch FROM now())),
            expires_at DOUBLE PRECISION DEFAULT 0,
            withdrawn_at DOUBLE PRECISION DEFAULT 0,
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
            created_at REAL NOT NULL DEFAULT (extract(epoch FROM now())),
            destroyed_at REAL DEFAULT 0,
            active BOOLEAN DEFAULT TRUE
        )
        """
    )

    # Widen any database still carrying the REAL columns the runtime DDL
    # originally created. Guarded so the rewrite runs at most once.
    op.execute(
        """
        DO $$ BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'casl_consent'
                  AND column_name = 'expires_at'
                  AND data_type = 'real'
            ) THEN
                ALTER TABLE casl_consent
                    ALTER COLUMN granted_at TYPE DOUBLE PRECISION,
                    ALTER COLUMN expires_at TYPE DOUBLE PRECISION,
                    ALTER COLUMN withdrawn_at TYPE DOUBLE PRECISION;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    # Deliberately not dropped: these hold consent evidence and the per-user
    # encryption keys that make encrypted personal data recoverable. Dropping
    # them on a downgrade would destroy records that privacy law requires be
    # retained, and would render existing ciphertext permanently unreadable.
    op.execute("DROP INDEX IF EXISTS idx_casl_consent_user")
