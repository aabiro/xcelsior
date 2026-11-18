"""Refer to a user by `users.user_id`, enforced by foreign keys (expand step).

Until now `users` was keyed by `email`, and every table that belonged to a user
copied the email as its reference, with no foreign key. Changing an email
therefore meant re-keying each copy by hand, and `UserStore.change_email`
missed `mfa_methods`: after a verified email change the login MFA check found
no methods under the new address, skipped the second factor and reset
`mfa_enabled` to 0. SSH keys, connector consent grants and team ownership were
stranded under the old address the same way.

This migration makes the user a single fact:

* `users` is keyed by `user_id`; `email` stays unique and becomes an ordinary
  attribute that can change in one row.
* Each table that belongs to a user gains (or already had) a user-id column,
  backfilled from the email it holds, with a foreign key to `users(user_id)`.
  A user's sessions, keys, MFA methods, SSH keys, grants, memberships,
  notifications and push subscriptions go with the user (`ON DELETE CASCADE`);
  a team outlives its owner (`ON DELETE SET NULL`).
* `chat_messages` now cascades with its conversation, so a conversation that
  cascades with its user does not strand (or block on) its messages.

Expand only (migrations/README.md rule 5). The email copies stay, `NOT NULL`
where they were, because the release still serving during a blue-green deploy
reads and writes them. Two compatibility triggers keep them correct until the
contract step drops them:

* `xcl_user_identity_compat` (each child, `BEFORE INSERT OR UPDATE`) fills the
  user id from the email for a writer that only knows the email, and the email
  from the user id for a writer that only knows the id.
* `xcl_users_email_sync` (`users`, `AFTER UPDATE OF email`) rewrites the copies
  when a user's email changes, so an older reader, a rollback, or the privacy
  pipeline's anonymisation never finds a stale or un-erased address.

Both are `SECURITY DEFINER`: they read `users` and write identity tables on
behalf of service roles that may hold neither right.

Verification is hard where a wrong answer is possible: a row whose existing
user id names a different user than its email does aborts the table's unit,
and so does a row a user exists for that is still unmapped after the backfill
(a concurrent writer held it). A row whose email matches no user at all keeps
a NULL user id; that is the truth (its user no longer exists), the foreign key
ignores NULL, and the contract step refuses to set `NOT NULL` while any remain.
A pre-existing user id that names no user (a session whose user was deleted
without it) leaves that table's key `NOT VALID`: enforced for every new and
changed row, exempting only those orphans, which readers no longer resolve
because they join `users`. The contract step removes them and validates.

The tables are small (one row per login, key, method or notification), so
indexes are built inside each unit under the lock timeout rather than
`CONCURRENTLY`, as in 109.

`downgrade` drops the foreign keys, triggers, functions and the columns this
migration added, restores the email primary key and `chat_messages`' original
foreign key. Columns that existed before (`sessions.user_id` and the like) are
kept.
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.engine import Connection

from migrations.lock_safe import apply_in_own_transactions

revision = "164"
down_revision = "163"
branch_labels = None
depends_on = None


#: (table, email column, user-id column, ON DELETE, column existed before 164)
CHILDREN: tuple[tuple[str, str, str, str, bool], ...] = (
    ("sessions", "email", "user_id", "CASCADE", True),
    ("api_keys", "email", "user_id", "CASCADE", True),
    ("oauth_refresh_tokens", "email", "user_id", "CASCADE", True),
    ("oauth_consent_grants", "email", "user_id", "CASCADE", True),
    ("user_ssh_keys", "email", "user_id", "CASCADE", True),
    ("mfa_methods", "email", "user_id", "CASCADE", False),
    ("mfa_backup_codes", "email", "user_id", "CASCADE", False),
    ("mfa_challenges", "email", "user_id", "CASCADE", False),
    ("team_members", "email", "user_id", "CASCADE", False),
    ("teams", "owner_email", "owner_user_id", "SET NULL", False),
    ("notifications", "user_email", "user_id", "CASCADE", False),
    ("web_push_subscriptions", "user_email", "user_id", "CASCADE", False),
    ("chat_conversations", "user_email", "user_id", "CASCADE", False),
)

#: Tables already keyed by a real `users.user_id` that never had the key.
#: (`ai_conversations`/`ai_confirmations` are not here: their owner falls back
#: to an email for principals without a user id, so it is not a user reference
#: yet. `user_encryption_keys` has no production writer.)
USER_ID_TABLES: tuple[tuple[str, str], ...] = (
    ("agent_api_keys", "CASCADE"),
    ("user_avatars", "CASCADE"),
    ("business_logos", "CASCADE"),
)

#: New indexes, named per table. Old email indexes stay for the older release.
INDEXES: dict[str, tuple[str, ...]] = {
    "sessions": (
        "CREATE INDEX IF NOT EXISTS idx_sessions_user_type "
        "ON sessions (user_id, session_type, last_active DESC)",
    ),
    "api_keys": ("CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys (user_id)",),
    "oauth_refresh_tokens": (
        "CREATE INDEX IF NOT EXISTS idx_oauth_refresh_tokens_user "
        "ON oauth_refresh_tokens (user_id)",
    ),
    # oauth_consent_grants already leads uq_oauth_consent_grants_principal with user_id.
    "oauth_consent_grants": (),
    "user_ssh_keys": (
        "CREATE INDEX IF NOT EXISTS idx_user_ssh_keys_user_created "
        "ON user_ssh_keys (user_id, created_at DESC)",
    ),
    "mfa_methods": ("CREATE INDEX IF NOT EXISTS idx_mfa_methods_user ON mfa_methods (user_id)",),
    "mfa_backup_codes": (
        "CREATE INDEX IF NOT EXISTS idx_mfa_backup_user ON mfa_backup_codes (user_id)",
    ),
    "mfa_challenges": (
        "CREATE INDEX IF NOT EXISTS idx_mfa_challenge_user ON mfa_challenges (user_id)",
    ),
    "team_members": (
        # The contract step promotes this to the primary key.
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_team_members_team_user "
        "ON team_members (team_id, user_id)",
        "CREATE INDEX IF NOT EXISTS idx_team_members_user ON team_members (user_id)",
    ),
    "teams": ("CREATE INDEX IF NOT EXISTS idx_teams_owner_user ON teams (owner_user_id)",),
    "notifications": (
        "CREATE INDEX IF NOT EXISTS idx_notifications_user_read_created "
        "ON notifications (user_id, read, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_notifications_user_priority_unread "
        "ON notifications (user_id, priority, read, created_at DESC)",
    ),
    "web_push_subscriptions": (
        "CREATE INDEX IF NOT EXISTS idx_web_push_subscriptions_user_active_v2 "
        "ON web_push_subscriptions (user_id, revoked_at, last_used_at)",
    ),
    "chat_conversations": (
        "CREATE INDEX IF NOT EXISTS idx_chat_conv_user ON chat_conversations (user_id)",
    ),
}

BATCH = 500

log = logging.getLogger("alembic.164")


def _fk_name(table: str, column: str) -> str:
    return f"fk_{table}_{column}_users"


def _constraint_exists(conn: Connection, table: str, name: str) -> bool:
    return bool(
        conn.execute(
            text(
                "SELECT 1 FROM pg_constraint " "WHERE conrelid = to_regclass(:t) AND conname = :n"
            ),
            {"t": f"public.{table}", "n": name},
        ).first()
    )


# ── users ──────────────────────────────────────────────────────────────


def _users_key(conn: Connection) -> None:
    bad = conn.execute(
        text(
            "SELECT count(*) FILTER (WHERE user_id IS NULL OR user_id = ''), "
            "       count(*) - count(DISTINCT user_id), "
            "       count(*) - count(DISTINCT lower(email)) FROM users"
        )
    ).one()
    if bad[0] or bad[1] or bad[2]:
        raise RuntimeError(
            f"users: {bad[0]} rows without a user_id, {bad[1]} duplicate user_ids and "
            f"{bad[2]} emails that differ only by case; the backfill matches emails "
            "case-insensitively and a user_id key needs exactly one user per row"
        )
    pk_cols = (
        conn.execute(
            text(
                "SELECT a.attname FROM pg_index i "
                "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
                "WHERE i.indrelid = 'public.users'::regclass AND i.indisprimary"
            )
        )
        .scalars()
        .all()
    )
    if pk_cols == ["user_id"]:
        return
    conn.execute(
        text(
            "ALTER TABLE users "
            "DROP CONSTRAINT IF EXISTS users_pkey, "
            "DROP CONSTRAINT IF EXISTS users_user_id_key, "
            "ADD CONSTRAINT users_pkey PRIMARY KEY (user_id), "
            "ADD CONSTRAINT users_email_key UNIQUE (email)"
        )
    )


def _compat_functions(conn: Connection) -> None:
    conn.execute(text("""
        CREATE OR REPLACE FUNCTION xcl_user_identity_compat() RETURNS trigger
        LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
        DECLARE
            email_col text := TG_ARGV[0];
            id_col text := TG_ARGV[1];
            r jsonb := to_jsonb(NEW);
            v_email text := r ->> email_col;
            v_id text := r ->> id_col;
            found_id text;
            found_email text;
        BEGIN
            -- A statement that moved the row by its email alone (the previous
            -- release re-keying a child) names the owner by that email.
            IF TG_OP = 'UPDATE'
               AND v_email IS DISTINCT FROM (to_jsonb(OLD) ->> email_col)
               AND v_id IS NOT DISTINCT FROM (to_jsonb(OLD) ->> id_col)
               AND v_email IS NOT NULL THEN
                SELECT u.user_id INTO found_id FROM users u
                 WHERE lower(u.email) = lower(v_email);
                IF found_id IS NOT NULL THEN
                    v_id := found_id;
                END IF;
            END IF;
            -- Otherwise user_id is the owner, and the email copy is always
            -- that user's address, so both releases agree on whose row it is.
            IF v_id IS NOT NULL THEN
                SELECT u.email INTO found_email FROM users u WHERE u.user_id = v_id;
                IF found_email IS NOT NULL THEN
                    NEW := jsonb_populate_record(
                        NEW, jsonb_build_object(id_col, v_id, email_col, found_email));
                END IF;
            ELSIF v_email IS NOT NULL THEN
                SELECT u.user_id, u.email INTO found_id, found_email FROM users u
                 WHERE lower(u.email) = lower(v_email);
                IF found_id IS NOT NULL THEN
                    NEW := jsonb_populate_record(
                        NEW, jsonb_build_object(id_col, found_id, email_col, found_email));
                END IF;
            END IF;
            RETURN NEW;
        END $$
        """))
    sync = "\n".join(
        f"            UPDATE {t} SET {e} = NEW.email "
        f"WHERE {c} = NEW.user_id AND {e} IS DISTINCT FROM NEW.email;"
        for t, e, c, _d, _x in CHILDREN
    )
    conn.execute(text(f"""
        CREATE OR REPLACE FUNCTION xcl_users_email_sync() RETURNS trigger
        LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
        BEGIN
{sync}
            RETURN NULL;
        END $$
        """))


def _users_sync_trigger(conn: Connection) -> None:
    conn.execute(text("DROP TRIGGER IF EXISTS xcl_users_email_sync ON users"))
    conn.execute(text("""
        CREATE TRIGGER xcl_users_email_sync
        AFTER UPDATE OF email ON users FOR EACH ROW
        WHEN (OLD.email IS DISTINCT FROM NEW.email)
        EXECUTE FUNCTION xcl_users_email_sync()
        """))


# ── children ───────────────────────────────────────────────────────────


def _child_ddl(table: str, email_col: str, id_col: str):
    def apply(conn: Connection) -> None:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {id_col} TEXT"))
        for stmt in INDEXES.get(table, ()):
            conn.execute(text(stmt))
        trigger = f"xcl_{table}_user_identity"
        conn.execute(text(f"DROP TRIGGER IF EXISTS {trigger} ON {table}"))
        conn.execute(text(f"""
            CREATE TRIGGER {trigger}
            BEFORE INSERT OR UPDATE OF {email_col}, {id_col} ON {table}
            FOR EACH ROW EXECUTE FUNCTION xcl_user_identity_compat('{email_col}', '{id_col}')
            """))

    return apply


def _unmapped(table: str, email_col: str, id_col: str) -> str:
    return (
        f"c.{id_col} IS NULL AND c.{email_col} IS NOT NULL AND EXISTS "
        f"(SELECT 1 FROM users u WHERE lower(u.email) = lower(c.{email_col}))"
    )


def _backfill_batch(table: str, email_col: str, id_col: str, done: list[int]):
    def apply(conn: Connection) -> None:
        conn.execute(text("SET LOCAL statement_timeout = '30s'"))
        result = conn.execute(text(f"""
            UPDATE {table} t SET {id_col} = u.user_id
              FROM users u
             WHERE lower(u.email) = lower(t.{email_col})
               AND t.ctid IN (
                   SELECT c.ctid FROM {table} c
                    WHERE {_unmapped(table, email_col, id_col)}
                    LIMIT {BATCH} FOR UPDATE OF c SKIP LOCKED)
            """))
        done.append(result.rowcount)

    return apply


def _verify_child(table: str, email_col: str, id_col: str):
    def apply(conn: Connection) -> None:
        left = conn.execute(
            text(f"SELECT count(*) FROM {table} c WHERE {_unmapped(table, email_col, id_col)}")
        ).scalar_one()
        if left:
            raise RuntimeError(
                f"{table}: {left} rows still unmapped to an existing user after the backfill "
                "(held by another transaction); retry the migration"
            )
        conflicting = conn.execute(text(f"""
            SELECT count(*) FROM {table} c JOIN users u ON u.user_id = c.{id_col}
             WHERE c.{email_col} IS NOT NULL AND lower(u.email) <> lower(c.{email_col})
            """)).scalar_one()
        if conflicting:
            raise RuntimeError(
                f"{table}: {conflicting} rows name one user by {id_col} and another by "
                f"{email_col}; resolve them before re-running (no row was guessed)"
            )

    return apply


def _add_fk(table: str, column: str, on_delete: str):
    name = _fk_name(table, column)

    def apply(conn: Connection) -> None:
        if not _constraint_exists(conn, table, name):
            conn.execute(text(f"""
                ALTER TABLE {table} ADD CONSTRAINT {name}
                FOREIGN KEY ({column}) REFERENCES users(user_id)
                ON DELETE {on_delete} NOT VALID
                """))

    return apply


def _validate_fk(table: str, column: str):
    name = _fk_name(table, column)

    def apply(conn: Connection) -> None:
        orphans = conn.execute(text(f"""
            SELECT count(*) FROM {table} c
             WHERE c.{column} IS NOT NULL
               AND NOT EXISTS (SELECT 1 FROM users u WHERE u.user_id = c.{column})
            """)).scalar_one()
        if orphans:
            # The key is enforced for every new and updated row from here on;
            # only the existing orphans are exempt. Readers join `users`, so an
            # orphan (a session or key whose user was deleted) resolves to
            # nothing. The contract step removes them and validates.
            log.warning(
                "%s.%s: %d rows name a user_id no user has; %s stays NOT VALID",
                table,
                column,
                orphans,
                name,
            )
            return
        conn.execute(text(f"ALTER TABLE {table} VALIDATE CONSTRAINT {name}"))

    return apply


def _chat_messages_cascade(conn: Connection) -> None:
    rule = conn.execute(text("""
        SELECT confdeltype FROM pg_constraint
         WHERE conrelid = 'public.chat_messages'::regclass
           AND conname = 'chat_messages_conversation_id_fkey'
        """)).scalar()
    if rule == "c":
        return
    conn.execute(text("""
        ALTER TABLE chat_messages
          DROP CONSTRAINT IF EXISTS chat_messages_conversation_id_fkey,
          ADD CONSTRAINT chat_messages_conversation_id_fkey
              FOREIGN KEY (conversation_id) REFERENCES chat_conversations(conversation_id)
              ON DELETE CASCADE
        """))


def _backfill_until_done(table: str, email_col: str, id_col: str) -> None:
    while True:
        done: list[int] = []
        apply_in_own_transactions(
            [(f"{table} backfill", _backfill_batch(table, email_col, id_col, done))],
            tables=(table,),
        )
        if not done or done[0] < BATCH:
            return


def upgrade() -> None:
    apply_in_own_transactions(
        [("users primary key", _users_key), ("identity compat functions", _compat_functions)],
        tables=("users",),
    )
    for table, email_col, id_col, _on_delete, _existed in CHILDREN:
        apply_in_own_transactions(
            [(f"{table} user id column", _child_ddl(table, email_col, id_col))],
            tables=(table,),
        )
        _backfill_until_done(table, email_col, id_col)
        apply_in_own_transactions(
            [(f"{table} verify", _verify_child(table, email_col, id_col))],
            tables=(table,),
        )
    apply_in_own_transactions(
        [("users email sync trigger", _users_sync_trigger)], tables=("users",)
    )
    apply_in_own_transactions(
        [("chat_messages cascade", _chat_messages_cascade)],
        tables=("chat_messages",),
    )
    keyed = [(t, c, d) for t, _e, c, d, _x in CHILDREN] + [
        (t, "user_id", d) for t, d in USER_ID_TABLES
    ]
    for table, column, on_delete in keyed:
        apply_in_own_transactions(
            [(f"{table} foreign key", _add_fk(table, column, on_delete))],
            tables=(table, "users"),
        )
        apply_in_own_transactions(
            [(f"{table} validate", _validate_fk(table, column))],
            tables=(table, "users"),
        )


# ── downgrade ──────────────────────────────────────────────────────────


def _drop_child(table: str, email_col: str, id_col: str, existed: bool):
    def apply(conn: Connection) -> None:
        conn.execute(
            text(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {_fk_name(table, id_col)}")
        )
        conn.execute(text(f"DROP TRIGGER IF EXISTS xcl_{table}_user_identity ON {table}"))
        for stmt in INDEXES.get(table, ()):
            name = stmt.split(" IF NOT EXISTS ", 1)[1].split()[0]
            conn.execute(text(f"DROP INDEX IF EXISTS {name}"))
        if not existed:
            conn.execute(text(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {id_col}"))

    return apply


def _drop_user_id_fk(table: str):
    def apply(conn: Connection) -> None:
        conn.execute(
            text(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {_fk_name(table, 'user_id')}")
        )

    return apply


def _users_key_back(conn: Connection) -> None:
    conn.execute(text("DROP TRIGGER IF EXISTS xcl_users_email_sync ON users"))
    conn.execute(text("DROP FUNCTION IF EXISTS xcl_users_email_sync()"))
    conn.execute(text("DROP FUNCTION IF EXISTS xcl_user_identity_compat()"))
    pk_cols = (
        conn.execute(
            text(
                "SELECT a.attname FROM pg_index i "
                "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
                "WHERE i.indrelid = 'public.users'::regclass AND i.indisprimary"
            )
        )
        .scalars()
        .all()
    )
    if pk_cols == ["email"]:
        return
    conn.execute(
        text(
            "ALTER TABLE users "
            "DROP CONSTRAINT IF EXISTS users_pkey, "
            "DROP CONSTRAINT IF EXISTS users_email_key, "
            "ADD CONSTRAINT users_pkey PRIMARY KEY (email), "
            "ADD CONSTRAINT users_user_id_key UNIQUE (user_id)"
        )
    )


def _chat_messages_restrict(conn: Connection) -> None:
    conn.execute(text("""
        ALTER TABLE chat_messages
          DROP CONSTRAINT IF EXISTS chat_messages_conversation_id_fkey,
          ADD CONSTRAINT chat_messages_conversation_id_fkey
              FOREIGN KEY (conversation_id) REFERENCES chat_conversations(conversation_id)
        """))


def downgrade() -> None:
    for table, _d in USER_ID_TABLES:
        apply_in_own_transactions(
            [(f"{table} drop foreign key", _drop_user_id_fk(table))], tables=(table,)
        )
    for table, email_col, id_col, _d, existed in CHILDREN:
        apply_in_own_transactions(
            [(f"{table} drop user id", _drop_child(table, email_col, id_col, existed))],
            tables=(table,),
        )
    apply_in_own_transactions(
        [("chat_messages restrict", _chat_messages_restrict)], tables=("chat_messages",)
    )
    apply_in_own_transactions([("users email key", _users_key_back)], tables=("users",))
