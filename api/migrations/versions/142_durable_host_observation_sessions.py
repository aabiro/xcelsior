"""Keep worker session retirement and generation fences beyond snapshot retention.

Quiesce old observation writers before applying this revision. Retained snapshots
and the recorded current host session seed the registry; already-pruned unknown
sessions cannot be reconstructed. Downgrade retains this fencing evidence.
"""

from alembic import op
from sqlalchemy import text

from migrations.lock_safe import apply_in_own_transactions

revision = "142"
down_revision = "141"
branch_labels = None
depends_on = None

# A missing accepted snapshot is allowed after retention. The durable generation
# still forbids its replay from becoming a fresh observation.
MISSING = """(h.observation_session_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM host_observation_sessions s
    WHERE s.host_id=h.host_id AND s.session_id=h.observation_session_id))
 OR EXISTS (SELECT 1 FROM host_observations o WHERE o.host_id=h.host_id
    AND NOT EXISTS (SELECT 1 FROM host_observation_sessions s
        WHERE s.host_id=o.host_id AND s.session_id=o.session_id))"""


def _create(conn):
    conn.execute(text("""CREATE TABLE IF NOT EXISTS host_observation_sessions (
        host_id TEXT NOT NULL REFERENCES hosts(host_id) ON DELETE CASCADE,
        session_id TEXT NOT NULL CHECK(length(session_id)>0),
        highest_generation BIGINT NOT NULL DEFAULT -1 CHECK(highest_generation>=-1),
        latest_observation_id UUID REFERENCES host_observations(observation_id)
            ON DELETE SET NULL,
        first_seen_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        retired_at TIMESTAMPTZ,
        PRIMARY KEY(host_id,session_id)
    )"""))
    conn.execute(text("""CREATE UNIQUE INDEX IF NOT EXISTS uq_host_observation_live_session
        ON host_observation_sessions(host_id) WHERE retired_at IS NULL"""))
    conn.execute(text("""CREATE OR REPLACE FUNCTION guard_host_observation_session()
        RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
          IF NEW.host_id IS DISTINCT FROM OLD.host_id
             OR NEW.session_id IS DISTINCT FROM OLD.session_id
             OR NEW.first_seen_at IS DISTINCT FROM OLD.first_seen_at
             OR NEW.highest_generation<OLD.highest_generation
             OR (OLD.retired_at IS NOT NULL
                 AND NEW.retired_at IS DISTINCT FROM OLD.retired_at) THEN
            RAISE EXCEPTION 'host observation session fence cannot regress'
                USING ERRCODE='check_violation';
          END IF;
          RETURN NEW;
        END $$"""))
    conn.execute(
        text(
            "DROP TRIGGER IF EXISTS trg_host_observation_session_fence ON host_observation_sessions"
        )
    )
    conn.execute(text("""CREATE TRIGGER trg_host_observation_session_fence
        BEFORE UPDATE ON host_observation_sessions FOR EACH ROW
        EXECUTE FUNCTION guard_host_observation_session()"""))


def _backfill_batch(conn):
    conn.execute(text("SET LOCAL statement_timeout='30s'"))
    hosts = conn.execute(text(f"""SELECT h.host_id,h.observation_session_id FROM hosts h
        WHERE {MISSING} ORDER BY h.host_id LIMIT 200 FOR UPDATE OF h SKIP LOCKED""")).fetchall()
    if not hosts:
        if conn.execute(text(f"SELECT 1 FROM hosts h WHERE {MISSING} LIMIT 1")).first():
            raise RuntimeError("session backfill has locked hosts; retry after quiescing writers")
        return 0
    for host_id, current in hosts:
        conn.execute(
            text("""UPDATE host_observation_sessions SET retired_at=clock_timestamp()
            WHERE host_id=:host AND session_id IS DISTINCT FROM :current
                AND retired_at IS NULL"""),
            {"host": host_id, "current": current},
        )
        conn.execute(
            text("""INSERT INTO host_observation_sessions
                (host_id,session_id,highest_generation,latest_observation_id,
                 first_seen_at,retired_at)
            SELECT DISTINCT ON (o.session_id) o.host_id,o.session_id,
                o.inventory_generation,
                CASE WHEN o.session_id IS NOT DISTINCT FROM :current
                    THEN o.observation_id END,
                min(o.received_at) OVER (PARTITION BY o.session_id),
                CASE WHEN o.session_id IS DISTINCT FROM :current
                    THEN clock_timestamp() END
            FROM host_observations o WHERE o.host_id=:host
            ORDER BY o.session_id,o.inventory_generation DESC,o.received_at DESC
            ON CONFLICT(host_id,session_id) DO NOTHING"""),
            {"host": host_id, "current": current},
        )
        if current is not None:
            conn.execute(
                text("""INSERT INTO host_observation_sessions(host_id,session_id)
                VALUES(:host,:current) ON CONFLICT(host_id,session_id) DO NOTHING"""),
                {"host": host_id, "current": current},
            )
    return len(hosts)


def _verify(conn):
    if conn.execute(text(f"SELECT 1 FROM hosts h WHERE {MISSING} LIMIT 1")).first():
        raise RuntimeError("session backfill is incomplete")
    if conn.execute(text("""SELECT 1 FROM host_observation_sessions s JOIN hosts h USING(host_id)
        WHERE (s.retired_at IS NULL) IS DISTINCT FROM
            (s.session_id IS NOT DISTINCT FROM h.observation_session_id) LIMIT 1""")).first():
        raise RuntimeError("session registry disagrees with current host identity")


def upgrade():
    touched = ["hosts", "host_observations", "host_observation_sessions"]
    apply_in_own_transactions([("durable observation session registry", _create)], tables=touched)
    for batch in range(10000):
        progress = []

        def backfill(conn):
            progress[:] = [_backfill_batch(conn)]

        apply_in_own_transactions(
            [(f"observation session batch {batch}", backfill)], tables=touched
        )
        if progress == [0]:
            apply_in_own_transactions(
                [("verify observation session registry", _verify)], tables=touched
            )
            return
    raise RuntimeError("session backfill did not converge")


def downgrade():
    # Do not erase retirement/generation evidence on rollback. Returning to an
    # observation writer without durable fencing requires a reviewed handoff.
    op.execute("SET LOCAL lock_timeout='5s'")
