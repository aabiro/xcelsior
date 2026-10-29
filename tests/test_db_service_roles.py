"""Per-service PostgreSQL roles — real privilege enforcement tests.

Data-architecture companion §4.2 ("avoid a single unrestricted application
role") and §4.3 (per-service pools, application_name, session timeouts).

These are not structural tests. Every denial in
``control_plane.db_roles.DENIAL_CONTRACT`` is proven by *actually
executing* the forbidden statement as that role and requiring PostgreSQL
to reject it, and every positive right is proven by executing a permitted
statement as the same role. The DDL-authority claim (companion §4.4 rule 1)
is proven by attempting a real ``CREATE TABLE`` as the API runtime role.

The suite provisions the roles into the test database itself, so it also
serves as the provisioner's integration test.
"""

from __future__ import annotations

import os

import pytest

from control_plane import db_roles

try:
    import psycopg
    from psycopg.errors import InsufficientPrivilege

    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _dsn = os.environ.get("XCELSIOR_POSTGRES_DSN") or ""
        _can_create_role = bool(
            _c.execute(
                "SELECT rolsuper OR rolcreaterole FROM pg_roles WHERE rolname = current_user"
            ).fetchone()[0]
        )
except Exception as _exc:  # pragma: no cover - skip path
    pytestmark = pytest.mark.skip(f"no postgres available: {_exc}")
    _pool = None
    _can_create_role = False
else:
    if not _can_create_role:  # pragma: no cover - skip path
        pytestmark = pytest.mark.skip(
            "connecting role lacks CREATEROLE; run provision_db_roles.py as an admin"
        )


# ── Domain ownership drift guard ──────────────────────────────────────


def test_every_live_table_is_claimed_by_exactly_one_domain():
    """A new migration cannot create a relation no role can reach.

    Grants are generated from ``TABLE_DOMAINS``. A table absent from that
    map gets no grant at all, so a service cut over to its own role would
    fail at runtime with ``permission denied``. Catching it here makes the
    failure a red test instead of a production incident.
    """
    with _pool.connection() as conn:
        rows = conn.execute("""
            SELECT schemaname, tablename
              FROM pg_tables
             WHERE schemaname IN ('public', 'storage')
            """).fetchall()

    unclaimed: list[str] = []
    for schema, table in rows:
        qualified = f"{schema}.{table}"
        if db_roles.domain_of(qualified) is None:
            unclaimed.append(qualified)

    assert not unclaimed, (
        "tables not assigned to a logical domain in control_plane.db_roles "
        f"(companion §4.2): {sorted(unclaimed)}"
    )


def test_domains_are_disjoint():
    """One authority per fact — no table may belong to two domains."""
    seen: dict[str, str] = {}
    for domain, tables in db_roles.TABLE_DOMAINS.items():
        for table in tables:
            assert table not in seen, f"{table} claimed by both {seen[table]} and {domain}"
            seen[table] = domain


def test_exactly_one_role_holds_ddl_authority():
    """Companion §4.4 rule 1: Alembic (the migrator role) owns all DDL."""
    ddl_roles = [k for k, r in db_roles.SERVICE_ROLES.items() if r.ddl]
    assert ddl_roles == ["migrator"]
    assert db_roles.SERVICE_ROLES["migrator"].runtime is False
    assert "migrator" not in db_roles.runtime_role_keys()


# ── Provisioning + live privilege enforcement ─────────────────────────


@pytest.fixture(scope="module")
def provisioned_roles():
    """Provision the service roles and grant membership for verification.

    Membership is required to ``SET ROLE`` in the assertions; it is
    revoked on teardown so the test database is not left with an
    over-privileged application role.
    """
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "provision_db_roles",
        pathlib.Path(__file__).resolve().parent.parent / "scripts" / "provision_db_roles.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    with psycopg.connect(dsn, autocommit=False) as conn:
        plan = module.build_plan(conn, login_services=set(), passwords={})
        with conn.cursor() as cur:
            for sql, params in plan:
                cur.execute(sql, params or None)
        conn.commit()

    names = [r.name for r in db_roles.SERVICE_ROLES.values()]
    with psycopg.connect(dsn, autocommit=True) as conn:
        for name in names:
            conn.execute(f'GRANT "{name}" TO CURRENT_USER')
    try:
        yield names
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            for name in names:
                try:
                    conn.execute(f'REVOKE "{name}" FROM CURRENT_USER')
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass


def _as_role(conn, role_name: str):
    conn.execute(f'SET ROLE "{role_name}"')


def _updatable_column(conn, table: str) -> str:
    """A real, assignable column on ``table`` (skips generated/identity)."""
    schema, _, bare = table.partition(".") if "." in table else ("public", "", table)
    if not bare:
        schema, bare = "public", table
    row = conn.execute(
        """
        SELECT column_name
          FROM information_schema.columns
         WHERE table_schema = %s
           AND table_name = %s
           AND is_generated = 'NEVER'
           AND is_identity = 'NO'
         ORDER BY ordinal_position
         LIMIT 1
        """,
        (schema, bare),
    ).fetchone()
    assert row, f"no updatable column found on {table}"
    return str(row[0])


def _probe_sql(conn, table: str, verb: str) -> str:
    """A statement that exercises ``verb`` on ``table`` but touches no row."""
    if verb == "UPDATE":
        col = _updatable_column(conn, table)
        return f'UPDATE {table} SET "{col}" = "{col}" WHERE false'
    if verb == "INSERT":
        return f"INSERT INTO {table} SELECT * FROM {table} WHERE false"
    if verb == "SELECT":
        return f"SELECT 1 FROM {table} WHERE false"
    raise AssertionError(f"unsupported verb {verb}")


@pytest.mark.parametrize(
    "role_key,table,verb",
    db_roles.DENIAL_CONTRACT,
    ids=[f"{r}-cannot-{v.lower()}-{t}" for r, t, v in db_roles.DENIAL_CONTRACT],
)
def test_denial_contract_is_enforced_by_postgres(provisioned_roles, role_key, table, verb):
    """Execute the forbidden statement for real and require rejection.

    ``WHERE false`` means no row is ever touched — PostgreSQL checks
    table privileges before execution, so a permitted role reaches
    "0 rows" and a denied role raises ``InsufficientPrivilege``.

    The catalog assertion pins *which* privilege is missing, so a probe
    that happened to fail for an adjacent reason cannot pass as proof.
    """
    role = db_roles.SERVICE_ROLES[role_key]
    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]

    with _pool.connection() as conn:
        granted = conn.execute(
            "SELECT has_table_privilege(%s, %s, %s)", (role.name, table, verb)
        ).fetchone()[0]
    assert granted is False, f"{role.name} unexpectedly holds {verb} on {table}"

    with psycopg.connect(dsn, autocommit=False) as conn:
        sql = _probe_sql(conn, table, verb)
        _as_role(conn, role.name)
        with pytest.raises(InsufficientPrivilege) as excinfo:
            conn.execute(sql)
        conn.rollback()

    assert "permission denied" in str(excinfo.value).lower()


#: ``(role_key, table, verb)`` rights each service genuinely needs. Least
#: privilege must still be *sufficient* privilege — a role that cannot do
#: its job is an outage, not a security win.
POSITIVE_CONTRACT = (
    # The scheduler owns placement: claim, bind, and record the intent.
    ("scheduler", "jobs", "UPDATE"),
    ("scheduler", "job_attempts", "INSERT"),
    ("scheduler", "gpu_device_allocations", "INSERT"),
    ("scheduler", "placement_leases", "INSERT"),
    ("scheduler", "agent_commands", "INSERT"),
    ("scheduler", "outbox_events", "INSERT"),
    # ...and read the money/marketplace facts placement revalidates.
    ("scheduler", "wallets", "SELECT"),
    ("scheduler", "gpu_pricing", "SELECT"),
    # The reconciler repairs control state.
    ("reconciler", "gpu_device_allocations", "UPDATE"),
    ("reconciler", "reconciliation_findings", "INSERT"),
    ("reconciler", "placement_leases", "UPDATE"),
    # Billing owns the ledger and meters.
    ("billing", "wallets", "UPDATE"),
    ("billing", "usage_meters", "INSERT"),
    ("billing", "wallet_holds", "UPDATE"),
    ("billing", "jobs", "SELECT"),
    # Read-only can read everything.
    ("readonly", "jobs", "SELECT"),
    ("readonly", "wallets", "SELECT"),
    # The projector settles the outbox and reads the world.
    ("projector", "outbox_events", "UPDATE"),
    ("projector", "jobs", "SELECT"),
    # The API is the broad tenant-facing writer.
    ("api", "jobs", "INSERT"),
    ("api", "users", "UPDATE"),
    ("api", "wallet_holds", "INSERT"),
)


@pytest.mark.parametrize(
    "role_key,table,verb",
    POSITIVE_CONTRACT,
    ids=[f"{r}-can-{v.lower()}-{t}" for r, t, v in POSITIVE_CONTRACT],
)
def test_positive_rights_are_granted(provisioned_roles, role_key, table, verb):
    """Least privilege must still be sufficient privilege."""
    role = db_roles.SERVICE_ROLES[role_key]
    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    with psycopg.connect(dsn, autocommit=False) as conn:
        sql = _probe_sql(conn, table, verb)
        _as_role(conn, role.name)
        conn.execute(sql)  # must not raise
        conn.rollback()


def test_no_runtime_role_can_run_ddl(provisioned_roles):
    """Companion §4.4 rule 1, enforced by the database rather than by habit.

    ``_ensure_pg_tables`` is seed-only on Alembic-managed databases (A1.6
    tail). This proves that even a code regression that reintroduced
    runtime DDL would be stopped by PostgreSQL for every runtime role.
    """
    for key in db_roles.runtime_role_keys():
        role = db_roles.SERVICE_ROLES[key]
        with psycopg.connect(os.environ["XCELSIOR_POSTGRES_DSN"], autocommit=False) as conn:
            _as_role(conn, role.name)
            with pytest.raises(InsufficientPrivilege):
                conn.execute("CREATE TABLE ddl_probe_should_fail (id int)")
            conn.rollback()


def test_migrator_can_run_ddl(provisioned_roles):
    """The one role that may create objects, actually can."""
    migrator = db_roles.SERVICE_ROLES["migrator"]
    with psycopg.connect(os.environ["XCELSIOR_POSTGRES_DSN"], autocommit=False) as conn:
        _as_role(conn, migrator.name)
        conn.execute("CREATE TABLE ddl_probe_migrator (id int)")
        conn.rollback()  # never persist the probe table


def test_provisioning_is_idempotent(provisioned_roles):
    """Running the provisioner twice converges instead of erroring."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "provision_db_roles",
        pathlib.Path(__file__).resolve().parent.parent / "scripts" / "provision_db_roles.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    for _ in range(2):
        with psycopg.connect(dsn, autocommit=False) as conn:
            plan = module.build_plan(conn, login_services=set(), passwords={})
            with conn.cursor() as cur:
                for sql, params in plan:
                    cur.execute(sql, params or None)
            conn.commit()

    # Privileges still exactly as contracted after repeat runs.
    with _pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT has_table_privilege('xcelsior_scheduler', 'wallets', 'UPDATE')"
            ).fetchone()[0]
            is False
        )
        assert (
            conn.execute(
                "SELECT has_table_privilege('xcelsior_scheduler', 'jobs', 'UPDATE')"
            ).fetchone()[0]
            is True
        )


def test_provisioner_refuses_an_over_privileged_existing_role(provisioned_roles):
    """A role named least-privilege must not silently be a superuser."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "provision_db_roles",
        pathlib.Path(__file__).resolve().parent.parent / "scripts" / "provision_db_roles.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    # Only an attribute the *connecting* admin itself holds can be handed
    # out; the suite already requires CREATEROLE, so use that one.
    with psycopg.connect(dsn, autocommit=True) as conn:
        try:
            conn.execute('ALTER ROLE "xcelsior_readonly" CREATEROLE')
        except InsufficientPrivilege:  # pragma: no cover - superuser-only cluster
            pytest.skip("connecting role cannot grant CREATEROLE for the probe")
        try:
            with psycopg.connect(dsn, autocommit=False) as probe:
                with pytest.raises(module.OverPrivilegedRole) as excinfo:
                    module.build_plan(probe, login_services=set(), passwords={})
            assert "CREATEROLE" in str(excinfo.value)
        finally:
            conn.execute('ALTER ROLE "xcelsior_readonly" NOCREATEROLE')


# ── Connection identity (companion §4.3) ──────────────────────────────


def test_application_name_and_timeouts_land_on_the_real_session(monkeypatch):
    """The pool's connect options must reach the PostgreSQL session."""
    monkeypatch.setenv("XCELSIOR_SERVICE", "scheduler")
    monkeypatch.delenv("XCELSIOR_PG_APPLICATION_NAME", raising=False)
    kwargs = db_roles.connection_kwargs()
    assert kwargs["application_name"] == "xcelsior-scheduler"

    with psycopg.connect(os.environ["XCELSIOR_POSTGRES_DSN"], **kwargs) as conn:
        app = conn.execute("SHOW application_name").fetchone()[0]
        stmt = conn.execute("SHOW statement_timeout").fetchone()[0]
        lock = conn.execute("SHOW lock_timeout").fetchone()[0]
        idle = conn.execute("SHOW idle_in_transaction_session_timeout").fetchone()[0]

    assert app == "xcelsior-scheduler"
    # Scheduler reservation pool: "small, low timeout, no long queries".
    assert stmt == "15s"
    assert lock == "3s"
    assert idle == "1min"


def test_service_dsn_is_opt_in(monkeypatch):
    """Provisioning roles must not change any deployment's behaviour."""
    monkeypatch.setenv("XCELSIOR_SERVICE", "scheduler")
    monkeypatch.delenv("XCELSIOR_POSTGRES_DSN_SCHEDULER", raising=False)
    shared = os.environ["XCELSIOR_POSTGRES_DSN"]
    assert db_roles.resolve_service_dsn() == shared

    monkeypatch.setenv("XCELSIOR_POSTGRES_DSN_SCHEDULER", "postgresql://s@h/db")
    assert db_roles.resolve_service_dsn() == "postgresql://s@h/db"
    # Unrelated services are unaffected by another service's cutover.
    assert db_roles.resolve_service_dsn("billing") == shared


def test_resolve_postgres_dsn_honours_service_override(monkeypatch):
    """db.resolve_postgres_dsn is the single entrypoint services use."""
    import db as db_module

    monkeypatch.setenv("XCELSIOR_SERVICE", "reconciler")
    monkeypatch.setenv("XCELSIOR_POSTGRES_DSN_RECONCILER", "postgresql://r@h/db")
    assert db_module.resolve_postgres_dsn() == "postgresql://r@h/db"

    monkeypatch.delenv("XCELSIOR_POSTGRES_DSN_RECONCILER")
    assert db_module.resolve_postgres_dsn() == os.environ["XCELSIOR_POSTGRES_DSN"]


def test_migrator_session_has_no_statement_timeout():
    """A long expand migration must not be killed by a pool default."""
    opts = db_roles.session_options("migrator")
    assert "statement_timeout=0" in opts
    assert "lock_timeout=0" in opts
