"""Foreign keys for live references, and connector ownership by user id.

The second step of the normalization audit (migrations/NORMALIZATION.md §2a).
Each column below names a row that must exist for the referring row to mean
anything, and none was enforced, so a deleted parent left children behind
that kept working. Examples this closes:

* `team_invites` outlived a deleted team, and accepting one added the invitee
  to a team that no longer existed.
* `agent_api_keys` outlived their OAuth client. Key validation never reads
  `oauth_clients`, so a key bound to a deleted client still authenticated;
  118 cleaned up the Quick Connect case after the fact.
* `gpu_offers` outlived their host after `remove_host`, still listed.

`oauth_clients.created_by_email` decided who may edit or delete a connector,
and nothing rewrote it when the creator changed address, so after an email
change the user could no longer see or delete connectors they registered. It
gains `created_by_user_id` (a key to `users`), filled from the email, and the
email copy is kept current the way 164 keeps its copies: the same
`xcl_user_identity_compat` trigger on writes, and an `AFTER UPDATE OF email`
trigger on `users`. A connector outlives its creator (`ON DELETE SET NULL`):
a team's connector keeps working when the account that registered it goes.

Delete behaviour follows the existing convention for each parent: what
belongs to a team, client, host, endpoint, worker, sweep or promotion goes
with it (`ON DELETE CASCADE`).

Left out on purpose, each for a reason recorded in NORMALIZATION.md:
`serverless_workers.host_id` (cascading would hit `serverless_jobs`, which
blocks, so `remove_host` would start failing), `worker_model_cache.worker_id`
(holds a host id in `workers/inference.py`), `pipeline_stages.plan_id` (text
against a uuid key), `gpu_allocations.offer_id` (marketplace record),
`job_logs.job_id` and `telemetry_latest.host_id` (high-volume writes taken under
the job and host locks), `ai_conversations.user_id` (holds `api-admin` for
the master token, which is not a user), and `slurm_job_mappings.cluster_id`
(the adapter registers a mapping under the sentinel cluster `unknown`, which
is never a `slurm_clusters` row).

Each key is added `NOT VALID` and then validated. A table that already holds
rows whose parent is gone keeps its key `NOT VALID`: enforced for every new and
changed row, exempting only those orphans, and logged. The contract step
removes them and validates.

Expand only. `created_by_email` stays for the release still serving.
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.engine import Connection

from migrations.lock_safe import apply_in_own_transactions

revision = "165"
down_revision = "164"
branch_labels = None
depends_on = None


#: (child table, column, parent table, parent column, ON DELETE)
REFERENCES: tuple[tuple[str, str, str, str, str], ...] = (
    ("team_members", "team_id", "teams", "team_id", "CASCADE"),
    ("team_invites", "team_id", "teams", "team_id", "CASCADE"),
    ("users", "team_id", "teams", "team_id", "SET NULL"),
    ("oauth_clients", "created_by_user_id", "users", "user_id", "SET NULL"),
    ("agent_api_keys", "client_id", "oauth_clients", "client_id", "CASCADE"),
    ("ai_confirmations", "conversation_id", "ai_conversations", "conversation_id", "CASCADE"),
    ("gpu_offers", "host_id", "hosts", "host_id", "CASCADE"),
    ("serverless_batches", "endpoint_id", "serverless_endpoints", "endpoint_id", "CASCADE"),
    ("serverless_semantic_cache", "endpoint_id", "serverless_endpoints", "endpoint_id", "CASCADE"),
    ("serverless_prefix_affinity", "endpoint_id", "serverless_endpoints", "endpoint_id", "CASCADE"),
    ("serverless_prefix_affinity", "worker_id", "serverless_workers", "worker_id", "CASCADE"),
    ("image_sweep_members", "sweep_id", "image_sweeps", "sweep_id", "CASCADE"),
    ("volume_promotion_files", "promotion_id", "volume_promotions", "promotion_id", "CASCADE"),
)

#: A cascade or a parent delete scans the child by this column.
INDEXES: dict[str, tuple[str, ...]] = {
    "oauth_clients": (
        "CREATE INDEX IF NOT EXISTS idx_oauth_clients_created_by_user "
        "ON oauth_clients (created_by_user_id)",
    ),
    "agent_api_keys": (
        "CREATE INDEX IF NOT EXISTS idx_agent_api_keys_client ON agent_api_keys (client_id)",
    ),
    "ai_confirmations": (
        "CREATE INDEX IF NOT EXISTS idx_ai_confirmations_conversation "
        "ON ai_confirmations (conversation_id)",
    ),
    "serverless_batches": (
        "CREATE INDEX IF NOT EXISTS idx_serverless_batches_endpoint "
        "ON serverless_batches (endpoint_id)",
    ),
    "serverless_prefix_affinity": (
        "CREATE INDEX IF NOT EXISTS idx_serverless_prefix_affinity_worker "
        "ON serverless_prefix_affinity (worker_id)",
    ),
}

log = logging.getLogger("alembic.165")


def _fk_name(table: str, column: str, parent: str) -> str:
    return f"fk_{table}_{column}_{parent}"


def _constraint_exists(conn: Connection, table: str, name: str) -> bool:
    return bool(
        conn.execute(
            text(
                "SELECT 1 FROM pg_constraint " "WHERE conrelid = to_regclass(:t) AND conname = :n"
            ),
            {"t": f"public.{table}", "n": name},
        ).first()
    )


# ── connector ownership ────────────────────────────────────────────────


def _clients_owner_column(conn: Connection) -> None:
    conn.execute(text("ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS created_by_user_id TEXT"))
    conn.execute(text("DROP TRIGGER IF EXISTS xcl_oauth_clients_user_identity ON oauth_clients"))
    # 164's function: fills whichever of the two a writer left out.
    conn.execute(text("""
        CREATE TRIGGER xcl_oauth_clients_user_identity
        BEFORE INSERT OR UPDATE OF created_by_email, created_by_user_id ON oauth_clients
        FOR EACH ROW
        EXECUTE FUNCTION xcl_user_identity_compat('created_by_email', 'created_by_user_id')
        """))


def _clients_owner_backfill(conn: Connection) -> None:
    # One row per registered connector: small enough for one statement.
    conn.execute(text("SET LOCAL statement_timeout = '30s'"))
    conn.execute(text("""
        UPDATE oauth_clients c SET created_by_user_id = u.user_id
          FROM users u
         WHERE c.created_by_user_id IS NULL
           AND c.created_by_email IS NOT NULL
           AND lower(u.email) = lower(c.created_by_email)
        """))
    conflicting = conn.execute(text("""
        SELECT count(*) FROM oauth_clients c JOIN users u ON u.user_id = c.created_by_user_id
         WHERE c.created_by_email IS NOT NULL AND lower(u.email) <> lower(c.created_by_email)
        """)).scalar_one()
    if conflicting:
        raise RuntimeError(
            f"oauth_clients: {conflicting} rows name one creator by user id and another by "
            "email; resolve them before re-running (no row was guessed)"
        )


def _clients_email_sync(conn: Connection) -> None:
    conn.execute(text("""
        CREATE OR REPLACE FUNCTION xcl_users_email_sync_oauth_clients() RETURNS trigger
        LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
        BEGIN
            UPDATE oauth_clients SET created_by_email = NEW.email
             WHERE created_by_user_id = NEW.user_id
               AND created_by_email IS DISTINCT FROM NEW.email;
            RETURN NULL;
        END $$
        """))
    conn.execute(text("DROP TRIGGER IF EXISTS xcl_users_email_sync_oauth_clients ON users"))
    conn.execute(text("""
        CREATE TRIGGER xcl_users_email_sync_oauth_clients
        AFTER UPDATE OF email ON users FOR EACH ROW
        WHEN (OLD.email IS DISTINCT FROM NEW.email)
        EXECUTE FUNCTION xcl_users_email_sync_oauth_clients()
        """))


# ── foreign keys ───────────────────────────────────────────────────────


def _indexes(table: str):
    def apply(conn: Connection) -> None:
        for stmt in INDEXES[table]:
            conn.execute(text(stmt))

    return apply


def _add_fk(table: str, column: str, parent: str, parent_col: str, on_delete: str):
    name = _fk_name(table, column, parent)

    def apply(conn: Connection) -> None:
        if not _constraint_exists(conn, table, name):
            conn.execute(text(f"""
                ALTER TABLE {table} ADD CONSTRAINT {name}
                FOREIGN KEY ({column}) REFERENCES {parent}({parent_col})
                ON DELETE {on_delete} NOT VALID
                """))

    return apply


def _validate_fk(table: str, column: str, parent: str, parent_col: str):
    name = _fk_name(table, column, parent)

    def apply(conn: Connection) -> None:
        orphans = conn.execute(text(f"""
            SELECT count(*) FROM {table} c
             WHERE c.{column} IS NOT NULL
               AND NOT EXISTS (SELECT 1 FROM {parent} p WHERE p.{parent_col} = c.{column})
            """)).scalar_one()
        if orphans:
            # Enforced for every new and changed row from here on; only these
            # existing rows are exempt. The contract step removes them.
            log.warning(
                "%s.%s: %d rows name a %s that does not exist; %s stays NOT VALID",
                table,
                column,
                orphans,
                parent,
                name,
            )
            return
        conn.execute(text(f"ALTER TABLE {table} VALIDATE CONSTRAINT {name}"))

    return apply


def upgrade() -> None:
    apply_in_own_transactions(
        [
            ("oauth_clients owner column", _clients_owner_column),
            ("oauth_clients owner backfill", _clients_owner_backfill),
        ],
        tables=("oauth_clients",),
    )
    apply_in_own_transactions(
        [("users email sync for oauth_clients", _clients_email_sync)], tables=("users",)
    )
    for table in INDEXES:
        apply_in_own_transactions([(f"{table} indexes", _indexes(table))], tables=(table,))
    for table, column, parent, parent_col, on_delete in REFERENCES:
        apply_in_own_transactions(
            [
                (
                    f"{table}.{column} foreign key",
                    _add_fk(table, column, parent, parent_col, on_delete),
                )
            ],
            tables=(table, parent),
        )
        apply_in_own_transactions(
            [(f"{table}.{column} validate", _validate_fk(table, column, parent, parent_col))],
            tables=(table, parent),
        )


# ── downgrade ──────────────────────────────────────────────────────────


def _drop_fk(table: str, column: str, parent: str):
    def apply(conn: Connection) -> None:
        conn.execute(
            text(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {_fk_name(table, column, parent)}")
        )

    return apply


def _drop_indexes(table: str):
    def apply(conn: Connection) -> None:
        for stmt in INDEXES[table]:
            name = stmt.split(" IF NOT EXISTS ", 1)[1].split()[0]
            conn.execute(text(f"DROP INDEX IF EXISTS {name}"))

    return apply


def _drop_clients_owner(conn: Connection) -> None:
    conn.execute(text("DROP TRIGGER IF EXISTS xcl_oauth_clients_user_identity ON oauth_clients"))
    conn.execute(text("ALTER TABLE oauth_clients DROP COLUMN IF EXISTS created_by_user_id"))


def _drop_clients_email_sync(conn: Connection) -> None:
    conn.execute(text("DROP TRIGGER IF EXISTS xcl_users_email_sync_oauth_clients ON users"))
    conn.execute(text("DROP FUNCTION IF EXISTS xcl_users_email_sync_oauth_clients()"))


def downgrade() -> None:
    for table, column, parent, _pc, _d in REFERENCES:
        apply_in_own_transactions(
            [(f"{table}.{column} drop foreign key", _drop_fk(table, column, parent))],
            tables=(table,),
        )
    for table in INDEXES:
        apply_in_own_transactions(
            [(f"{table} drop indexes", _drop_indexes(table))], tables=(table,)
        )
    apply_in_own_transactions(
        [("users drop oauth_clients email sync", _drop_clients_email_sync)], tables=("users",)
    )
    apply_in_own_transactions(
        [("oauth_clients drop owner column", _drop_clients_owner)], tables=("oauth_clients",)
    )
