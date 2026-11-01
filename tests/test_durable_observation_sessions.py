"""Retention cannot revive superseded sessions or replay accepted generations."""

import uuid
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest
from psycopg.errors import CheckViolation, UniqueViolation
from psycopg.rows import dict_row

from control_plane.observations import ingest_observation, latest_observation, prune_observations
from db import DatabaseOps, _get_pg_pool
from tests.test_db_service_roles import provisioned_roles  # noqa: F401
from tests.test_from_empty_bootstrap import empty_db  # noqa: F401
from tests.test_migration_from_production_snapshot import _alembic, _connect


@pytest.fixture
def host():
    with _get_pg_pool().connection() as conn:
        name = "durable-session-" + uuid.uuid4().hex
        DatabaseOps.upsert_host(conn, {"host_id": name}, backend="postgres")
        try:
            yield conn, name
        finally:
            conn.rollback()


def observe(conn, host_id, session="old", generation=1, **kwargs):
    return ingest_observation(
        conn,
        host_id=host_id,
        session_id=session,
        observation_generation=generation,
        workloads=kwargs.pop("workloads", []),
        **kwargs,
    )


def authority(conn, host_id):
    return conn.execute(
        "SELECT observation_session_id,last_observed_at,capabilities,conditions,availability_state "
        "FROM hosts WHERE host_id=%s",
        (host_id,),
    ).fetchone()


def prune_host_snapshots(conn, host_id):
    conn.execute(
        "UPDATE host_observations SET received_at=clock_timestamp()-interval '10 days' "
        "WHERE host_id=%s",
        (host_id,),
    )
    assert prune_observations(conn, limit=10000) >= 1


def test_retired_session_cannot_return_after_its_snapshots_are_pruned(host):
    conn, host_id = host
    old = observe(conn, host_id, capabilities={"version": "old"})
    current = observe(conn, host_id, "new", capabilities={"version": "new"})
    conn.execute("DELETE FROM host_observations WHERE observation_id=%s", (old.observation_id,))
    conn.execute("DELETE FROM reconciliation_queue WHERE resource_id=%s", (host_id,))
    before = authority(conn, host_id)
    stale = observe(
        conn,
        host_id,
        "old",
        1000000,
        capabilities={"version": "stale"},
        conditions={"gpu_inventory": "unavailable"},
        gpu_inventory=[],
    )
    assert stale.observation_id and authority(conn, host_id) == before
    assert latest_observation(conn, host_id)["observation_id"] == current.observation_id
    assert conn.execute(
        "SELECT count(*) FROM reconciliation_queue WHERE resource_id=%s", (host_id,)
    ).fetchone() == (0,)
    assert conn.execute(
        "SELECT retired_at IS NOT NULL FROM host_observation_sessions "
        "WHERE host_id=%s AND session_id='old'",
        (host_id,),
    ).fetchone() == (True,)


@pytest.mark.parametrize("replayed", [0, 49, 50])
def test_current_generation_and_accepted_pointer_survive_retention(host, replayed):
    conn, host_id = host
    observe(conn, host_id, generation=50)
    before = authority(conn, host_id)
    prune_host_snapshots(conn, host_id)
    assert latest_observation(conn, host_id) is None
    assert conn.execute(
        "SELECT highest_generation,latest_observation_id FROM "
        "host_observation_sessions WHERE host_id=%s",
        (host_id,),
    ).fetchone() == (50, None)
    observe(conn, host_id, generation=replayed)
    assert authority(conn, host_id) == before
    # Merely retaining a replayed snapshot must not restore freshness either.
    assert latest_observation(conn, host_id) is None
    accepted = observe(conn, host_id, generation=51)
    assert latest_observation(conn, host_id)["observation_id"] == accepted.observation_id


def test_duplicate_delivery_does_not_freshen_authority(host):
    conn, host_id = host
    first = observe(conn, host_id)
    before = authority(conn, host_id)
    duplicate = observe(conn, host_id, capabilities={"must_not_apply": True})
    assert duplicate.duplicate and duplicate.observation_id is None
    assert authority(conn, host_id) == before
    assert latest_observation(conn, host_id)["observation_id"] == first.observation_id


def test_all_superseded_sessions_stay_retired_after_pruning(host):
    conn, host_id = host
    for session in ["first", "second", "third"]:
        observe(conn, host_id, session)
    prune_host_snapshots(conn, host_id)
    before = authority(conn, host_id)
    for session in ["first", "second"]:
        observe(conn, host_id, session, 999999)
    assert authority(conn, host_id) == before
    assert latest_observation(conn, host_id) is None
    observe(conn, host_id, "third", 2)
    assert latest_observation(conn, host_id)["session_id"] == "third"


def test_failed_handoff_rolls_back_session_retirement_together_with_snapshot(host):
    conn, host_id = host
    first = observe(conn, host_id)
    with pytest.raises(psycopg.errors.InvalidTextRepresentation), conn.transaction():
        observe(conn, host_id, "new", workloads=[{"attempt_id": "not-a-uuid"}])
    assert latest_observation(conn, host_id)["observation_id"] == first.observation_id
    assert conn.execute(
        "SELECT session_id,retired_at FROM host_observation_sessions WHERE host_id=%s", (host_id,)
    ).fetchall() == [("old", None)]


def test_session_identity_is_scoped_to_the_host(host):
    conn, host_id = host
    other = "durable-session-" + uuid.uuid4().hex
    DatabaseOps.upsert_host(conn, {"host_id": other}, backend="postgres")
    observe(conn, host_id)
    observe(conn, host_id, "new")
    observe(conn, other)
    assert latest_observation(conn, other)["session_id"] == "old"
    assert latest_observation(conn, host_id)["session_id"] == "new"


def test_dictionary_rows_follow_the_same_retirement_fence(host):
    conn, host_id = host
    conn.row_factory = dict_row
    observe(conn, host_id)
    observe(conn, host_id, "new")
    prune_host_snapshots(conn, host_id)
    observe(conn, host_id, generation=2)
    assert latest_observation(conn, host_id) is None
    observe(conn, host_id, "new", 2)
    assert latest_observation(conn, host_id)["session_id"] == "new"


@pytest.mark.parametrize(
    "change",
    [
        "highest_generation=0",
        "retired_at=NULL",
        "first_seen_at=clock_timestamp()",
        "session_id='renamed'",
    ],
)
def test_database_refuses_regressing_known_session_fences(host, change):
    conn, host_id = host
    observe(conn, host_id)
    observe(conn, host_id, "new")
    with pytest.raises(CheckViolation), conn.transaction():
        conn.execute(
            "UPDATE host_observation_sessions SET "
            + change
            + " WHERE host_id=%s AND session_id='old'",
            (host_id,),
        )


def test_database_refuses_two_live_sessions_for_one_host(host):
    conn, host_id = host
    observe(conn, host_id)
    with pytest.raises(UniqueViolation), conn.transaction():
        conn.execute(
            "INSERT INTO host_observation_sessions(host_id,session_id) VALUES(%s,'another')",
            (host_id,),
        )


@pytest.mark.parametrize("service", ["api", "scheduler", "reconciler"])
def test_runtime_can_ingest_but_cannot_erase_session_fences(host, provisioned_roles, service):
    from psycopg import sql
    from control_plane.db_roles import SERVICE_ROLES

    conn, host_id = host
    conn.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(SERVICE_ROLES[service].name)))
    observe(conn, host_id)
    observe(conn, host_id, "new")
    assert latest_observation(conn, host_id)["session_id"] == "new"
    with pytest.raises(psycopg.errors.InsufficientPrivilege), conn.transaction():
        conn.execute("DELETE FROM host_observation_sessions WHERE host_id=%s", (host_id,))


def test_runtime_audit_detects_public_delete_grant(host, provisioned_roles):
    from control_plane.db_role_audit import role_violations

    conn, _ = host
    conn.execute("GRANT DELETE ON host_observation_sessions TO PUBLIC")
    assert any(
        "erase durable fences" in issue
        for issue in role_violations(conn, "api", check_session=False)
    )


def test_migration_backfills_batched_sessions_and_replays_without_losing_evidence(empty_db):
    _alembic(empty_db, "upgrade", "141")
    with _connect(empty_db) as conn:
        for i in range(205):
            host_id = f"migration-session-{i:04}"
            DatabaseOps.upsert_host(conn, {"host_id": host_id}, backend="postgres")
            conn.execute(
                "UPDATE hosts SET observation_session_id='current' WHERE host_id=%s", (host_id,)
            )
            for session, gen in [("old", 1), ("old", 2), ("current", 3)]:
                conn.execute(
                    "INSERT INTO host_observations(host_id,session_id,inventory_generation) "
                    "VALUES(%s,%s,%s)",
                    (host_id, session, gen),
                )
        DatabaseOps.upsert_host(conn, {"host_id": "history-already-pruned"}, backend="postgres")
        conn.execute(
            "UPDATE hosts SET observation_session_id='known' WHERE host_id='history-already-pruned'"
        )
        DatabaseOps.upsert_host(conn, {"host_id": "no-current-identity"}, backend="postgres")
        conn.execute(
            "INSERT INTO host_observations(host_id,session_id) VALUES('no-current-identity','historical')"
        )
    _alembic(empty_db, "upgrade", "142")
    with _connect(empty_db) as conn:
        assert conn.execute(
            "SELECT count(*),count(*) FILTER(WHERE retired_at IS NULL) "
            "FROM host_observation_sessions"
        ).fetchone() == (412, 206)
        assert conn.execute(
            "SELECT highest_generation,latest_observation_id FROM host_observation_sessions "
            "WHERE host_id='history-already-pruned'"
        ).fetchone() == (-1, None)
        assert conn.execute(
            "SELECT retired_at IS NOT NULL FROM host_observation_sessions "
            "WHERE host_id='no-current-identity'"
        ).fetchone() == (True,)
        first = conn.execute(
            "SELECT host_id,session_id,highest_generation,latest_observation_id,retired_at "
            "FROM host_observation_sessions ORDER BY host_id,session_id"
        ).fetchall()
        observe(conn, "migration-session-0000", "old", 1000)
        assert latest_observation(conn, "migration-session-0000")["observation_generation"] == 3
    _alembic(empty_db, "downgrade", "141")
    _alembic(empty_db, "upgrade", "142")
    with _connect(empty_db) as conn:
        assert (
            conn.execute(
                "SELECT host_id,session_id,highest_generation,latest_observation_id,retired_at "
                "FROM host_observation_sessions ORDER BY host_id,session_id"
            ).fetchall()
            == first
        )


def test_backfill_refuses_locked_hosts_then_resumes(empty_db):
    _alembic(empty_db, "upgrade", "141")
    with _connect(empty_db) as conn:
        DatabaseOps.upsert_host(conn, {"host_id": "locked-session"}, backend="postgres")
        conn.execute(
            "UPDATE hosts SET observation_session_id='current' WHERE host_id='locked-session'"
        )
    with _connect(empty_db) as blocker:
        blocker.execute("SELECT host_id FROM hosts WHERE host_id='locked-session' FOR UPDATE")
        result = _alembic(empty_db, "upgrade", "142", check=False)
        assert result.returncode != 0 and "locked hosts" in result.stderr
    _alembic(empty_db, "upgrade", "142")
    with _connect(empty_db) as conn:
        assert conn.execute("SELECT session_id FROM host_observation_sessions").fetchall() == [
            ("current",)
        ]


def test_old_writer_cannot_unretire_an_identity_by_downgrading_and_reupgrading(empty_db):
    _alembic(empty_db, "upgrade", "142")
    with _connect(empty_db) as conn:
        DatabaseOps.upsert_host(conn, {"host_id": "rollback-session"}, backend="postgres")
        observe(conn, "rollback-session")
        observe(conn, "rollback-session", "new")
    _alembic(empty_db, "downgrade", "141")
    with _connect(empty_db) as conn:
        conn.execute(
            "UPDATE hosts SET observation_session_id='old' WHERE host_id='rollback-session'"
        )
    result = _alembic(empty_db, "upgrade", "142", check=False)
    assert result.returncode != 0 and "disagrees with current host identity" in result.stderr
    with _connect(empty_db) as conn:
        assert conn.execute(
            "SELECT retired_at IS NOT NULL FROM host_observation_sessions "
            "WHERE host_id='rollback-session' AND session_id='old'"
        ).fetchone() == (True,)
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == ("141",)


def test_competing_new_sessions_leave_one_live_identity_and_old_reports_cannot_reclaim_it():
    host_id = "session-race-" + uuid.uuid4().hex
    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    with psycopg.connect(dsn) as conn:
        DatabaseOps.upsert_host(conn, {"host_id": host_id}, backend="postgres")
        observe(conn, host_id)
    barrier = threading.Barrier(2)

    def send(session):
        with psycopg.connect(dsn, options="-c statement_timeout=5000") as conn:
            barrier.wait(timeout=5)
            return observe(conn, host_id, session).observation_id

    try:
        with ThreadPoolExecutor(max_workers=2) as threads:
            futures = [threads.submit(send, name) for name in ["left", "right"]]
            assert all(future.result(timeout=10) for future in futures)
        with psycopg.connect(dsn) as conn:
            current = authority(conn, host_id)[0]
            assert current in {"left", "right"}
            assert conn.execute(
                "SELECT session_id FROM host_observation_sessions "
                "WHERE host_id=%s AND retired_at IS NULL",
                (host_id,),
            ).fetchall() == [(current,)]
            prune_host_snapshots(conn, host_id)
            for session in {"old", "left", "right"} - {current}:
                observe(conn, host_id, session, 1000000)
            assert authority(conn, host_id)[0] == current
            assert latest_observation(conn, host_id) is None
    finally:
        with psycopg.connect(dsn) as conn:
            conn.execute("DELETE FROM reconciliation_queue WHERE resource_id=%s", (host_id,))
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (host_id,))
