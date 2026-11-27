"""Contract step for 164 and 165: a user's address lives only on `users`.

164 and 165 were the expand step (migrations/README.md rule 5). They gave every
table a user owns a `user_id` key to `users`, but kept the email copies, and
the copied `role`/`name`/`is_admin` on `sessions` and `api_keys`, because the
release then still serving read and wrote them. Two `SECURITY DEFINER`
triggers kept the copies true. Every release now serving reads the user
through `users` (prod has run that code since 164 went live), so this step
removes what only the old release needed:

* Rows whose user no longer exists are deleted (a `SET NULL` reference is
  cleared instead), and every key 164 or 165 left `NOT VALID` is validated.
  A `NOT VALID` key exempted exactly those rows, so they are the only ones
  this touches. Each table logs how many it removed.
* Each user id column becomes `NOT NULL`, except where NULL is a real value:
  `chat_conversations.user_id` (anonymous chats), `oauth_refresh_tokens.user_id`
  (client-credential tokens), and `teams.owner_user_id` and
  `oauth_clients.created_by_user_id` (a team and a connector outlive the
  account that created them, `ON DELETE SET NULL`).
* `team_members` is keyed by `(team_id, user_id)`; the email key goes.
* The triggers and their functions are dropped, then the email copies, the
  copied `role`/`name`/`is_admin` (`api_keys.name` stays: it is the key's own
  label, not a copy) and `oauth_clients.created_by_email`. Their indexes go
  with them.

It also adds the key 169 left out. A `receipt_shares` row names the run it
shares, and a share of a run that no longer exists (the privacy pipeline
deletes jobs) means nothing, so `job_id` references `jobs` with
`ON DELETE CASCADE`. The share's `owner_id` copied `jobs.owner_id`, which
never changes once set; readers now take the owner from the job, so the copy
loses `NOT NULL` here and goes in the next contract step, after the release
that still writes it has stopped serving.

Order matters. The users trigger rewrites every child's email column, so it
goes first; each child's trigger fills its email copy, which is `NOT NULL`, so
the trigger and the column go in the same transaction.

`downgrade` brings the columns back and fills them from `users` and `jobs`,
restores the email key on `team_members`, re-creates the triggers through
164's and 165's own functions, and drops the `receipt_shares` key. It does not
restore the rows the upgrade deleted: they named users that no longer exist.
"""

from __future__ import annotations

import logging
from importlib import import_module

from sqlalchemy import text
from sqlalchemy.engine import Connection

from migrations.lock_safe import apply_in_own_transactions

revision = "171"
down_revision = "170"
branch_labels = None
depends_on = None


#: (table, email column, user-id column, ON DELETE, user id may be NULL)
CHILDREN: tuple[tuple[str, str, str, str, bool], ...] = (
    ("sessions", "email", "user_id", "CASCADE", False),
    ("api_keys", "email", "user_id", "CASCADE", False),
    ("oauth_refresh_tokens", "email", "user_id", "CASCADE", True),
    ("oauth_consent_grants", "email", "user_id", "CASCADE", False),
    ("user_ssh_keys", "email", "user_id", "CASCADE", False),
    ("mfa_methods", "email", "user_id", "CASCADE", False),
    ("mfa_backup_codes", "email", "user_id", "CASCADE", False),
    ("mfa_challenges", "email", "user_id", "CASCADE", False),
    ("team_members", "email", "user_id", "CASCADE", False),
    ("teams", "owner_email", "owner_user_id", "SET NULL", True),
    ("notifications", "user_email", "user_id", "CASCADE", False),
    ("web_push_subscriptions", "user_email", "user_id", "CASCADE", False),
    ("chat_conversations", "user_email", "user_id", "CASCADE", True),
    ("oauth_clients", "created_by_email", "created_by_user_id", "SET NULL", True),
)

#: Columns copied from `users` onto a session or key: (table, column, users column).
COPIED: tuple[tuple[str, str, str], ...] = (
    ("sessions", "role", "role"),
    ("sessions", "name", "name"),
    ("sessions", "is_admin", "is_admin"),
    ("api_keys", "role", "role"),
    ("api_keys", "is_admin", "is_admin"),
)

#: How each dropped column is re-created by `downgrade`, as it was before.
COLUMN_TYPES: dict[tuple[str, str], str] = {
    ("sessions", "role"): "TEXT NOT NULL DEFAULT 'submitter'",
    ("sessions", "name"): "TEXT NOT NULL DEFAULT ''",
    ("sessions", "is_admin"): "INTEGER NOT NULL DEFAULT 0",
    ("api_keys", "role"): "TEXT NOT NULL DEFAULT 'submitter'",
    ("api_keys", "is_admin"): "INTEGER NOT NULL DEFAULT 0",
}

#: Email copies that were `NOT NULL` before this step (the rest were nullable).
EMAIL_WAS_NOT_NULL: frozenset[str] = frozenset(
    {
        "sessions",
        "api_keys",
        "oauth_consent_grants",
        "user_ssh_keys",
        "mfa_methods",
        "mfa_backup_codes",
        "mfa_challenges",
        "team_members",
        "teams",
        "notifications",
        "web_push_subscriptions",
    }
)

#: Email indexes `downgrade` re-creates (the upgrade drops them with their column).
EMAIL_INDEXES: dict[str, tuple[str, ...]] = {
    "sessions": (
        "CREATE INDEX IF NOT EXISTS idx_sessions_email_type "
        "ON sessions (email, session_type, last_active DESC)",
    ),
    "api_keys": ("CREATE INDEX IF NOT EXISTS idx_api_keys_email ON api_keys (email)",),
    "oauth_refresh_tokens": (
        "CREATE INDEX IF NOT EXISTS idx_oauth_refresh_tokens_email "
        "ON oauth_refresh_tokens (email)",
    ),
    "user_ssh_keys": (
        "CREATE INDEX IF NOT EXISTS idx_user_ssh_keys_email_created "
        "ON user_ssh_keys (email, created_at DESC)",
    ),
    "mfa_methods": ("CREATE INDEX IF NOT EXISTS idx_mfa_methods_email ON mfa_methods (email)",),
    "mfa_backup_codes": (
        "CREATE INDEX IF NOT EXISTS idx_mfa_backup_email ON mfa_backup_codes (email)",
    ),
    "mfa_challenges": (
        "CREATE INDEX IF NOT EXISTS idx_mfa_challenge_email ON mfa_challenges (email)",
    ),
    "team_members": ("CREATE INDEX IF NOT EXISTS idx_team_members_email ON team_members (email)",),
    "notifications": (
        "CREATE INDEX IF NOT EXISTS idx_notifications_email_read_created "
        "ON notifications (user_email, read, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_notifications_priority_unread "
        "ON notifications (user_email, priority, read, created_at DESC)",
    ),
    "web_push_subscriptions": (
        "CREATE INDEX IF NOT EXISTS idx_web_push_subscriptions_user_active "
        "ON web_push_subscriptions (user_email, revoked_at, last_used_at)",
    ),
    "oauth_clients": (
        "CREATE INDEX IF NOT EXISTS idx_oauth_clients_owner ON oauth_clients (created_by_email)",
    ),
}

#: 164's tables already keyed by a real user id: (table, ON DELETE).
USER_ID_TABLES: tuple[tuple[str, str], ...] = (
    ("agent_api_keys", "CASCADE"),
    ("user_avatars", "CASCADE"),
    ("business_logos", "CASCADE"),
)

#: The key 169 left out: (child, column, parent, parent column, ON DELETE).
RECEIPT_SHARE_JOB = ("receipt_shares", "job_id", "jobs", "job_id", "CASCADE")

log = logging.getLogger("alembic.171")


def _m164():
    return import_module("migrations.versions.164_user_identity_by_user_id")


def _m165():
    return import_module("migrations.versions.165_live_reference_foreign_keys")


def _fk_name(table: str, column: str, parent: str) -> str:
    return f"fk_{table}_{column}_{parent}"


def _validated(conn: Connection, table: str, name: str) -> bool | None:
    """True or False for a key that exists, None for one that does not."""
    row = conn.execute(
        text(
            "SELECT convalidated FROM pg_constraint "
            "WHERE conrelid = to_regclass(:t) AND conname = :n"
        ),
        {"t": f"public.{table}", "n": name},
    ).first()
    return None if row is None else bool(row[0])


def _has_column(conn: Connection, table: str, column: str) -> bool:
    return bool(
        conn.execute(
            text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = :t AND column_name = :c"
            ),
            {"t": table, "c": column},
        ).first()
    )


# ── orphans and keys ───────────────────────────────────────────────────


def _settle_reference(
    table: str,
    column: str,
    parent: str,
    parent_col: str,
    on_delete: str,
    gone_by_email: str | None = None,
):
    """Remove the rows a `NOT VALID` key exempted, then validate it.

    ``gone_by_email`` names the email copy of a 164 child: a row with a NULL
    user id but an address belonged to a user who no longer exists (164 left
    it unmapped), which the key's ON DELETE would have removed.
    """
    name = _fk_name(table, column, parent)

    def apply(conn: Connection) -> None:
        conn.execute(text("SET LOCAL statement_timeout = '60s'"))
        missing = (
            f"{table}.{column} IS NOT NULL AND NOT EXISTS "
            f"(SELECT 1 FROM {parent} p WHERE p.{parent_col} = {table}.{column})"
        )
        if on_delete == "SET NULL":
            result = conn.execute(text(f"UPDATE {table} SET {column} = NULL WHERE {missing}"))
            action = "cleared"
        else:
            unmapped = ""
            if gone_by_email and _has_column(conn, table, gone_by_email):
                unmapped = f" OR ({table}.{column} IS NULL AND {table}.{gone_by_email} IS NOT NULL)"
            result = conn.execute(text(f"DELETE FROM {table} WHERE {missing}{unmapped}"))
            action = "deleted"
        if result.rowcount:
            log.warning(
                "%s.%s: %s %d rows naming a %s row that no longer exists",
                table,
                column,
                action,
                result.rowcount,
                parent,
            )
        if _validated(conn, table, name) is False:
            conn.execute(text(f"ALTER TABLE {table} VALIDATE CONSTRAINT {name}"))

    return apply


def _receipt_share_key(conn: Connection) -> None:
    table, column, parent, parent_col, on_delete = RECEIPT_SHARE_JOB
    name = _fk_name(table, column, parent)
    if _validated(conn, table, name) is None:
        conn.execute(text(f"""
            ALTER TABLE {table} ADD CONSTRAINT {name}
            FOREIGN KEY ({column}) REFERENCES {parent}({parent_col})
            ON DELETE {on_delete} NOT VALID
            """))
    conn.execute(text("ALTER TABLE receipt_shares ALTER COLUMN owner_id DROP NOT NULL"))


# ── contract ───────────────────────────────────────────────────────────


def _drop_users_triggers(conn: Connection) -> None:
    conn.execute(text("DROP TRIGGER IF EXISTS xcl_users_email_sync ON users"))
    conn.execute(text("DROP TRIGGER IF EXISTS xcl_users_email_sync_oauth_clients ON users"))
    conn.execute(text("DROP FUNCTION IF EXISTS xcl_users_email_sync()"))
    conn.execute(text("DROP FUNCTION IF EXISTS xcl_users_email_sync_oauth_clients()"))


def _contract_child(table: str, email_col: str, id_col: str, nullable: bool):
    def apply(conn: Connection) -> None:
        if not nullable:
            conn.execute(text(f"ALTER TABLE {table} ALTER COLUMN {id_col} SET NOT NULL"))
        if table == "team_members" and _has_column(conn, table, email_col):
            # 164 built this unique index to become the key.
            conn.execute(text("""
                ALTER TABLE team_members
                  DROP CONSTRAINT IF EXISTS team_members_pkey,
                  ADD CONSTRAINT team_members_pkey
                      PRIMARY KEY USING INDEX uq_team_members_team_user
                """))
        conn.execute(text(f"DROP TRIGGER IF EXISTS xcl_{table}_user_identity ON {table}"))
        drops = [email_col] + [c for t, c, _u in COPIED if t == table]
        conn.execute(
            text(f"ALTER TABLE {table} " + ", ".join(f"DROP COLUMN IF EXISTS {c}" for c in drops))
        )

    return apply


def _drop_compat_function(conn: Connection) -> None:
    conn.execute(text("DROP FUNCTION IF EXISTS xcl_user_identity_compat()"))


def upgrade() -> None:
    # Every reference 164 and 165 added, settled before any copy is dropped:
    # a row 164 left unmapped is recognised by its email.
    for table, email_col, id_col, on_delete, _n in CHILDREN:
        apply_in_own_transactions(
            [
                (
                    f"{table}.{id_col} settle",
                    _settle_reference(table, id_col, "users", "user_id", on_delete, email_col),
                )
            ],
            tables=(table, "users"),
        )
    for table, on_delete in USER_ID_TABLES:
        apply_in_own_transactions(
            [
                (
                    f"{table}.user_id settle",
                    _settle_reference(table, "user_id", "users", "user_id", on_delete),
                )
            ],
            tables=(table, "users"),
        )
    for table, column, parent, parent_col, on_delete in _m165().REFERENCES:
        if (table, column) == ("oauth_clients", "created_by_user_id"):
            continue  # settled above as a 164-style child
        apply_in_own_transactions(
            [
                (
                    f"{table}.{column} settle",
                    _settle_reference(table, column, parent, parent_col, on_delete),
                )
            ],
            tables=(table, parent),
        )

    apply_in_own_transactions(
        [("receipt_shares job key", _receipt_share_key)], tables=("receipt_shares", "jobs")
    )
    apply_in_own_transactions(
        [("receipt_shares job settle", _settle_reference(*RECEIPT_SHARE_JOB))],
        tables=("receipt_shares", "jobs"),
    )

    apply_in_own_transactions(
        [("users email sync triggers", _drop_users_triggers)], tables=("users",)
    )
    for table, email_col, id_col, _d, nullable in CHILDREN:
        apply_in_own_transactions(
            [(f"{table} contract", _contract_child(table, email_col, id_col, nullable))],
            tables=(table,),
        )
    apply_in_own_transactions(
        [("identity compat function", _drop_compat_function)], tables=("users",)
    )


# ── downgrade ──────────────────────────────────────────────────────────


def _expand_child(table: str, email_col: str, id_col: str, nullable: bool):
    def apply(conn: Connection) -> None:
        conn.execute(text("SET LOCAL statement_timeout = '60s'"))
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {email_col} TEXT"))
        conn.execute(text(f"""
            UPDATE {table} c SET {email_col} = u.email
              FROM users u WHERE u.user_id = c.{id_col} AND c.{email_col} IS NULL
            """))
        for t, column, users_col in COPIED:
            if t != table:
                continue
            conn.execute(
                text(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} "
                    f"{COLUMN_TYPES[(table, column)]}"
                )
            )
            conn.execute(text(f"""
                UPDATE {table} c SET {column} = u.{users_col}
                  FROM users u WHERE u.user_id = c.{id_col}
                """))
        if table in EMAIL_WAS_NOT_NULL:
            conn.execute(text(f"ALTER TABLE {table} ALTER COLUMN {email_col} SET NOT NULL"))
        if table == "team_members":
            conn.execute(text("""
                ALTER TABLE team_members
                  DROP CONSTRAINT IF EXISTS team_members_pkey,
                  ADD CONSTRAINT team_members_pkey PRIMARY KEY (team_id, email)
                """))
            conn.execute(text("""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_team_members_team_user
                ON team_members (team_id, user_id)
                """))
        for stmt in EMAIL_INDEXES.get(table, ()):
            conn.execute(text(stmt))
        if not nullable and table not in {
            "sessions",
            "api_keys",
            "oauth_consent_grants",
            "user_ssh_keys",
        }:
            # 164 added these columns nullable; the four above were NOT NULL before it.
            conn.execute(text(f"ALTER TABLE {table} ALTER COLUMN {id_col} DROP NOT NULL"))
        conn.execute(text(f"DROP TRIGGER IF EXISTS xcl_{table}_user_identity ON {table}"))
        conn.execute(text(f"""
            CREATE TRIGGER xcl_{table}_user_identity
            BEFORE INSERT OR UPDATE OF {email_col}, {id_col} ON {table}
            FOR EACH ROW EXECUTE FUNCTION xcl_user_identity_compat('{email_col}', '{id_col}')
            """))

    return apply


def _restore_functions(conn: Connection) -> None:
    _m164()._compat_functions(conn)


def _restore_users_triggers(conn: Connection) -> None:
    _m164()._users_sync_trigger(conn)
    _m165()._clients_email_sync(conn)


def _receipt_share_key_back(conn: Connection) -> None:
    table, column, parent, _pc, _d = RECEIPT_SHARE_JOB
    conn.execute(
        text(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {_fk_name(table, column, parent)}")
    )
    conn.execute(text("""
        UPDATE receipt_shares s SET owner_id = COALESCE(j.owner_id, '')
          FROM jobs j WHERE j.job_id = s.job_id AND s.owner_id IS NULL
        """))
    conn.execute(text("UPDATE receipt_shares SET owner_id = '' WHERE owner_id IS NULL"))
    conn.execute(text("ALTER TABLE receipt_shares ALTER COLUMN owner_id SET NOT NULL"))


def downgrade() -> None:
    apply_in_own_transactions(
        [("receipt_shares job key", _receipt_share_key_back)],
        tables=("receipt_shares", "jobs"),
    )
    apply_in_own_transactions(
        [("identity compat functions", _restore_functions)], tables=("users",)
    )
    for table, email_col, id_col, _d, nullable in CHILDREN:
        apply_in_own_transactions(
            [(f"{table} expand", _expand_child(table, email_col, id_col, nullable))],
            tables=(table, "users"),
        )
    apply_in_own_transactions(
        [("users email sync triggers", _restore_users_triggers)], tables=("users",)
    )
