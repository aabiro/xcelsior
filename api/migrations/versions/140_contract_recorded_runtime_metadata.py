"""Recover exact command timing and retire the unused Lightning projection.

No identities, traces, outcomes or payment dates are invented. New runtime
writers already persist native fields. Keep unknown historical facts unknown.
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "140"
down_revision = "139"
branch_labels = None
depends_on = None


def _attempts(conn):
    conn.execute(text("""UPDATE job_attempts a SET command_created_at=recorded.created_at
        FROM (
            SELECT a.attempt_id,min(to_timestamp(c.created_at)) AS created_at
            FROM job_attempts a JOIN agent_commands c ON c.attempt_id=a.attempt_id
                AND c.job_id=a.job_id AND c.host_id=a.host_id
                AND c.fencing_token=a.fencing_token AND c.spec_hash=a.spec_hash
            WHERE a.command_created_at IS NULL AND c.command='start_attempt'
                AND c.created_at>=0 AND c.created_at<=EXTRACT(EPOCH FROM clock_timestamp())
            GROUP BY a.attempt_id
        ) recorded WHERE a.attempt_id=recorded.attempt_id AND a.command_created_at IS NULL"""))


def _lightning(conn):
    # New deposit creation writes all four facts directly. Refuse to retire the
    # old fallback while a live monetary input still needs its projection.
    if conn.execute(text("""SELECT 1 FROM ln_deposits
        WHERE amount_micros IS NULL OR amount_cad_minor IS NULL
           OR btc_cad_rate_exact IS NULL OR created_at_ts IS NULL OR expires_at_ts IS NULL
        LIMIT 1""")).first():
        raise RuntimeError("Lightning canonical payment facts are incomplete; contract refused")
    conn.execute(text("DROP TRIGGER IF EXISTS trg_ln_deposits_project_typed ON ln_deposits"))
    conn.execute(text("DROP FUNCTION IF EXISTS ln_deposits_project_typed()"))
    for column in ("amount_cad_minor", "btc_cad_rate_exact", "created_at_ts", "expires_at_ts"):
        conn.execute(text(f"ALTER TABLE ln_deposits ALTER COLUMN {column} SET NOT NULL"))


def _retire_empty_column(table, column):
    def contract(conn):
        exists = conn.execute(
            text(
                "SELECT 1 FROM information_schema.columns WHERE table_schema='public' AND table_name=:table AND column_name=:column"
            ),
            {"table": table, "column": column},
        ).first()
        if not exists:
            return
        # Names are constants in this reviewed migration.
        conn.execute(text(f"LOCK TABLE {table} IN ACCESS EXCLUSIVE MODE"))
        if conn.execute(text(f"SELECT 1 FROM {table} WHERE {column} IS NOT NULL LIMIT 1")).first():
            raise RuntimeError(
                f"retired {table}.{column} contains evidence; archive before contraction"
            )
        conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {column}"))

    return contract


def upgrade():
    apply_in_own_transactions(
        [
            ("recorded native command timing", _attempts),
            ("retire Lightning projection", _lightning),
            ("retire unused command result", _retire_empty_column("agent_commands", "result")),
            (
                "retire unimplemented journal cursor",
                _retire_empty_column("host_observations", "command_journal_watermark"),
            ),
        ],
        tables=["job_attempts", "agent_commands", "ln_deposits", "host_observations"],
    )


def downgrade():
    # Retain recovered facts and canonical requirements. A rollback needs a
    # compatible runtime; it must not revive unqualified compatibility writers.
    pass
