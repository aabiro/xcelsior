"""Convert the privacy tables' epoch columns that 090 skipped because they were REAL.

090 turned the float-epoch columns on ``casl_consent`` and ``user_encryption_keys``
into TIMESTAMPTZ, but guarded each conversion on ``data_type = 'double precision'``.
Databases whose ``casl_consent`` predates the first 084 (the SQLite-era table was
REAL) passed that guard untouched. They are at head with ``granted_at real``, and
every consent write (``granted_at = now()``) fails with DatatypeMismatch.

Same conversion as 090, but guarded on any numeric epoch type. On a database
that is already correct (production was TIMESTAMPTZ on 2026-10-10) every guard
is false and this is a no-op.
"""

from alembic import op

revision = "119"
down_revision = "118"
branch_labels = None
depends_on = None

_NUMERIC_TYPES = "('real', 'double precision', 'numeric', 'integer', 'bigint', 'smallint')"

_COLUMNS = (
    ("casl_consent", "granted_at", False),
    ("casl_consent", "expires_at", True),
    ("casl_consent", "withdrawn_at", True),
    ("user_encryption_keys", "created_at", False),
    ("user_encryption_keys", "destroyed_at", True),
)


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    for table, column, nullable in _COLUMNS:
        if nullable:
            # 0 meant "never" and must not become 1970 (see 090).
            convert = f"""
                ALTER TABLE {table}
                    ALTER COLUMN {column} DROP DEFAULT,
                    ALTER COLUMN {column} DROP NOT NULL,
                    ALTER COLUMN {column} TYPE TIMESTAMPTZ
                        USING CASE WHEN {column} IS NULL OR {column} = 0 THEN NULL
                                   ELSE to_timestamp({column}) END;
            """
        else:
            convert = f"""
                ALTER TABLE {table}
                    ALTER COLUMN {column} DROP DEFAULT,
                    ALTER COLUMN {column} TYPE TIMESTAMPTZ
                        USING to_timestamp({column}),
                    ALTER COLUMN {column} SET DEFAULT now();
            """
        op.execute(f"""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                     WHERE table_name = '{table}'
                       AND column_name = '{column}'
                       AND data_type IN {_NUMERIC_TYPES}
                ) THEN
                    {convert}
                END IF;
            END $$;
            """)


def downgrade() -> None:
    # The pre-119 state is either already TIMESTAMPTZ (no-op) or a float epoch
    # that every reader treated as broken; 090's downgrade owns the float form.
    pass
