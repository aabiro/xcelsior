"""Complete data-companion 4.4 parity: no exemptions, no float time.

Three things this migration owns:

1. **`casl_consent` and `user_encryption_keys` were created wrong by `084`.**
   That migration lifted them out of `privacy.py`'s runtime DDL and into the
   chain, but preserved the runtime shape verbatim — float epoch seconds and no
   tenant. Moving a table into Alembic without bringing it up to the standard
   just makes the defect official. Both are converted here.

2. **`privacy_deletion_requests` / `privacy_deletion_sink_status` had no
   tenant.** These were previously treated as exempt on the grounds that a
   deletion subject must stay unlinkable. That conflated two different things.
   The companion keeps tenant and pseudonymises *identity*: an artifact row
   "must own identity, tenant, checksum, state, retention, region, and
   deletion status" (2.1), and warehouse keys are pseudonymous with "direct
   identity only in restricted mapping" (11.2). The tenant is the workspace,
   not the person. Erasure blanks `subject_email` and `subject_user_id` and
   keeps `subject_reference_hash`; `tenant_id` says which workspace the
   request belongs to, which is what makes an operator's tenant-scoped view
   and cross-tenant denial provable. It does not re-identify the subject.

3. Time columns on the two consent/key tables become `TIMESTAMPTZ`, with 0 and
   NULL both meaning "not yet", per 4.4.5.

`casl_consent.expires_at`/`withdrawn_at` and `user_encryption_keys.destroyed_at`
use 0 as the sentinel for "never"; that becomes NULL rather than 1970, which
would otherwise sort as "expired long ago".
"""

from alembic import op

revision = "090"
down_revision = "089"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    # ── 4.4.5: typed timestamps on the tables 084 imported as-is ──────
    # 084 now creates both tables with TIMESTAMPTZ directly, so a database
    # built from empty arrives here already converted and `to_timestamp()`
    # would be applied to a timestamp. Only databases that ran the earlier
    # 084 still hold float epochs, so each conversion is guarded on the
    # column's actual type rather than assumed.
    for table, column, nullable in (
        ("casl_consent", "granted_at", False),
        ("casl_consent", "expires_at", True),
        ("casl_consent", "withdrawn_at", True),
        ("user_encryption_keys", "created_at", False),
        ("user_encryption_keys", "destroyed_at", True),
    ):
        if nullable:
            # 0 meant "never" and must not become 1970, which would sort as
            # "expired long ago" instead of "no expiry".
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
        op.execute(
            f"""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                     WHERE table_name = '{table}'
                       AND column_name = '{column}'
                       AND data_type = 'double precision'
                ) THEN
                    {convert}
                END IF;
            END $$;
            """
        )

    # ── 4.4.10: tenant ownership, on every governed table ─────────────
    for table in (
        "casl_consent",
        "user_encryption_keys",
        "privacy_deletion_requests",
        "privacy_deletion_sink_status",
    ):
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS tenant_id TEXT")

    for table in ("casl_consent", "user_encryption_keys"):
        op.execute(
            f"""
            UPDATE {table} t
               SET tenant_id = COALESCE(
                       NULLIF(u.team_id, ''), NULLIF(u.customer_id, ''), t.user_id)
              FROM users u
             WHERE u.user_id = t.user_id AND t.tenant_id IS NULL
            """
        )
        op.execute(f"UPDATE {table} SET tenant_id = user_id WHERE tenant_id IS NULL")

    # The deletion request's tenant is the workspace it was raised in. Prefer a
    # live user row; fall back to the retained pseudonymous reference so a
    # request whose subject is already erased still has a tenant.
    op.execute(
        """
        UPDATE privacy_deletion_requests r
           SET tenant_id = COALESCE(
                   NULLIF(u.team_id, ''), NULLIF(u.customer_id, ''),
                   NULLIF(r.subject_user_id, ''), r.subject_reference_hash)
          FROM users u
         WHERE u.user_id = r.subject_user_id AND r.tenant_id IS NULL
        """
    )
    op.execute(
        """
        UPDATE privacy_deletion_requests
           SET tenant_id = COALESCE(NULLIF(subject_user_id, ''), subject_reference_hash)
         WHERE tenant_id IS NULL
        """
    )
    op.execute(
        """
        UPDATE privacy_deletion_sink_status s
           SET tenant_id = r.tenant_id
          FROM privacy_deletion_requests r
         WHERE r.request_id = s.request_id AND s.tenant_id IS NULL
        """
    )
    # Orphaned sink rows cannot happen (FK), but a NOT NULL add must not fail.
    op.execute(
        "UPDATE privacy_deletion_sink_status SET tenant_id = request_id "
        "WHERE tenant_id IS NULL"
    )

    for table in (
        "casl_consent",
        "user_encryption_keys",
        "privacy_deletion_requests",
        "privacy_deletion_sink_status",
    ):
        op.execute(f"ALTER TABLE {table} ALTER COLUMN tenant_id SET NOT NULL")

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_casl_consent_tenant "
        "ON casl_consent (tenant_id, user_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_encryption_keys_tenant "
        "ON user_encryption_keys (tenant_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_privacy_deletion_requests_tenant "
        "ON privacy_deletion_requests (tenant_id, created_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_privacy_deletion_sink_status_tenant "
        "ON privacy_deletion_sink_status (tenant_id, request_id)"
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    for name in (
        "idx_casl_consent_tenant",
        "idx_user_encryption_keys_tenant",
        "idx_privacy_deletion_requests_tenant",
        "idx_privacy_deletion_sink_status_tenant",
    ):
        op.execute(f"DROP INDEX IF EXISTS {name}")
    for table in (
        "casl_consent",
        "user_encryption_keys",
        "privacy_deletion_requests",
        "privacy_deletion_sink_status",
    ):
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS tenant_id")

    # Mirror of the guarded upgrade: convert only what is currently typed.
    for table, column, nullable in (
        ("casl_consent", "granted_at", False),
        ("casl_consent", "expires_at", True),
        ("casl_consent", "withdrawn_at", True),
        ("user_encryption_keys", "created_at", False),
        ("user_encryption_keys", "destroyed_at", True),
    ):
        default = "0" if nullable else "extract(epoch FROM now())"
        op.execute(
            f"""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                     WHERE table_name = '{table}'
                       AND column_name = '{column}'
                       AND data_type = 'timestamp with time zone'
                ) THEN
                    ALTER TABLE {table}
                        ALTER COLUMN {column} DROP DEFAULT,
                        ALTER COLUMN {column} TYPE DOUBLE PRECISION
                            USING COALESCE(extract(epoch FROM {column}), 0),
                        ALTER COLUMN {column} SET DEFAULT {default};
                END IF;
            END $$;
            """
        )
