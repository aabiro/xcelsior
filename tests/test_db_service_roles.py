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


def test_strict_dsn_resolution_never_falls_back(monkeypatch):
    import db

    monkeypatch.setenv("XCELSIOR_DB_ROLES_REQUIRED", "1")
    monkeypatch.setenv("XCELSIOR_SERVICE", "scheduler")
    monkeypatch.delenv("XCELSIOR_POSTGRES_DSN_SCHEDULER", raising=False)
    with pytest.raises(RuntimeError, match="shared database fallback refused"):
        db.resolve_postgres_dsn()
    monkeypatch.setenv("XCELSIOR_SERVICE", "misspelled")
    with pytest.raises(RuntimeError, match="XCELSIOR_SERVICE"):
        db.resolve_postgres_dsn()


@pytest.mark.parametrize("service", db_roles.runtime_role_keys())
def test_real_runtime_login_has_no_excess_authority(provisioned_roles, monkeypatch, service):
    from control_plane.db_role_audit import role_violations
    from psycopg.conninfo import make_conninfo

    role = db_roles.SERVICE_ROLES[service]
    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(psycopg.sql.SQL('ALTER ROLE {} LOGIN PASSWORD {}').format(psycopg.sql.Identifier(role.name), psycopg.sql.Literal('isolated-role-password')))
    try:
        # This is an actual separate login, not SET ROLE on a privileged pool.
        with psycopg.connect(make_conninfo(dsn, user=role.name, password='isolated-role-password')) as conn:
            assert role_violations(conn, service) == []
            conn.execute("SELECT version_num FROM alembic_version").fetchone()
            conn.commit()
            with pytest.raises(InsufficientPrivilege):
                conn.execute("CREATE TABLE public.forbidden_runtime_ddl (id int)")
            conn.rollback()
            with pytest.raises(InsufficientPrivilege):
                conn.execute("UPDATE alembic_version SET version_num=version_num")
            conn.rollback()
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(f'ALTER ROLE "{role.name}" NOLOGIN PASSWORD NULL')


def test_effective_public_grants_and_membership_are_detected(provisioned_roles):
    from control_plane.db_role_audit import role_violations

    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    with psycopg.connect(dsn) as conn:
        conn.execute("GRANT UPDATE ON wallets TO PUBLIC")
        assert any("wallets" in issue for issue in role_violations(conn, "readonly", check_session=False))
        conn.rollback()
        conn.execute("GRANT xcelsior_api TO xcelsior_billing")
        assert any("member" in issue for issue in role_violations(conn, "billing", check_session=False))
        conn.rollback()


def test_scoped_pools_use_distinct_logins_and_restore_after_failure(provisioned_roles, monkeypatch):
    import db
    from psycopg.conninfo import make_conninfo

    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    monkeypatch.setenv("XCELSIOR_DB_ROLES_REQUIRED", "1")
    monkeypatch.setenv("XCELSIOR_SERVICE", "api")
    for service in ("billing", "reconciler"):
        role = db_roles.SERVICE_ROLES[service].name
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(psycopg.sql.SQL('ALTER ROLE {} LOGIN PASSWORD {}').format(psycopg.sql.Identifier(role), psycopg.sql.Literal('isolated-role-password')))
        monkeypatch.setenv(db_roles.service_dsn_env_var(service), make_conninfo(dsn, user=role, password='isolated-role-password'))
    pools = []
    try:
        for service in ("billing", "reconciler", "billing"):
            with db_roles.service_scope(service):
                pool = db._get_pg_pool()
                pools.append(pool)
                with pool.connection() as conn:
                    assert conn.execute("SELECT current_user").fetchone()[0] == db_roles.SERVICE_ROLES[service].name
                    conn.row_factory = __import__("psycopg.rows", fromlist=["dict_row"]).dict_row
        assert pools[0] is pools[2] and pools[0] is not pools[1]
        with pytest.raises(ValueError, match="test failure"):
            with db_roles.service_scope("billing"):
                raise ValueError("test failure")
        assert db_roles.current_service() == "api"
        with pytest.raises(ValueError, match="invalid runtime"):
            with db_roles.service_scope("migrator"):
                pass
    finally:
        for pool in set(pools):
            pool.close()
        db._scoped_pg_pools.clear()
        with psycopg.connect(dsn, autocommit=True) as conn:
            for service in ("billing", "reconciler"):
                conn.execute(f'ALTER ROLE "{db_roles.SERVICE_ROLES[service].name}" NOLOGIN PASSWORD NULL')


def test_runtime_cannot_advance_another_domains_sequence(provisioned_roles):
    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    with psycopg.connect(dsn) as conn:
        sequences = conn.execute("SELECT pg_get_serial_sequence('public.agent_commands','id')").fetchone()
        assert sequences and sequences[0], "expected control-domain sequence"
        if sequences[0]:
            conn.execute("SET ROLE xcelsior_billing")
            with pytest.raises(InsufficientPrivilege):
                conn.execute("SELECT nextval(%s)", (sequences[0],))


def test_login_provisioning_accepts_quoted_password_without_sql_injection(provisioned_roles):
    from scripts.provision_db_roles import apply_plan, build_plan

    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    with psycopg.connect(dsn) as conn:
        plan = build_plan(conn, login_services={"readonly"}, passwords={"readonly": "probe'; ALTER ROLE xcelsior_readonly SUPERUSER; --"})
        assert all("probe'" not in statement for statement, _ in plan)
        apply_plan(conn, plan)
        assert conn.execute("SELECT rolcanlogin, rolsuper FROM pg_roles WHERE rolname='xcelsior_readonly'").fetchone() == (True, False)
        conn.rollback()


def test_future_objects_do_not_inherit_blanket_public_or_runtime_rights(provisioned_roles):
    from scripts.provision_db_roles import apply_plan, build_plan

    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    with psycopg.connect(dsn) as conn:
        conn.execute("ALTER DEFAULT PRIVILEGES GRANT ALL ON TABLES TO PUBLIC")
        conn.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO xcelsior_readonly")
        apply_plan(conn, build_plan(conn, login_services=set(), passwords={}))
        conn.execute("CREATE TABLE public.security_new_relation(id int)")
        assert not conn.execute("SELECT has_table_privilege('xcelsior_readonly','security_new_relation','SELECT,INSERT,UPDATE,DELETE')").fetchone()[0]
        conn.rollback()


def test_column_only_billing_escalation_is_detected(provisioned_roles):
    from control_plane.db_role_audit import role_violations

    with psycopg.connect(os.environ["XCELSIOR_POSTGRES_DSN"]) as conn:
        conn.execute("GRANT UPDATE(balance_micros) ON wallets TO xcelsior_readonly")
        assert any("wallets" in issue for issue in role_violations(conn, "readonly", check_session=False))
        conn.rollback()


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


def test_strict_scheduled_task_executes_with_its_own_login(provisioned_roles, monkeypatch):
    import db
    from control_plane import scheduled_tasks
    from psycopg.conninfo import make_conninfo

    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    observed = []
    monkeypatch.setenv("XCELSIOR_DB_ROLES_REQUIRED", "1")
    monkeypatch.setenv("XCELSIOR_SERVICE", "reconciler")
    monkeypatch.setattr(db, "_pg_pool", None)
    monkeypatch.setattr(db, "_pg_pool_identity", None)
    monkeypatch.setattr(db, "_scoped_pg_pools", {})
    monkeypatch.setattr(scheduled_tasks, "_registry", {})
    monkeypatch.setattr(scheduled_tasks, "_task_services", {})
    with psycopg.connect(dsn, autocommit=True) as admin:
        for service in ("billing", "reconciler"):
            role = db_roles.SERVICE_ROLES[service].name
            admin.execute(f'ALTER ROLE "{role}" LOGIN')
            monkeypatch.setenv(db_roles.service_dsn_env_var(service), make_conninfo(dsn, user=role))

    def task():
        with db.pg_connection() as conn:
            observed.append(conn.execute("SELECT current_user, session_user").fetchone())
            # Read authority alone is insufficient: the task must write using
            # its billing identity, then the executor must settle as reconciler.
            conn.execute("UPDATE wallets SET balance_micros = balance_micros WHERE false")

    try:
        scheduled_tasks.register_task("identity-proof", task, 60, service="billing")
        assert scheduled_tasks.claim_and_run_tasks("identity-test") == 1
        assert observed == [("xcelsior_billing", "xcelsior_billing")]
        with db.pg_connection() as conn:
            assert conn.execute("SELECT current_user").fetchone() == ("xcelsior_reconciler",)
            assert conn.execute(
                "SELECT last_status, claim_owner FROM scheduled_tasks WHERE task_name='identity-proof'"
            ).fetchone() == ("succeeded", None)
    finally:
        for pool in {db._pg_pool, *db._scoped_pg_pools.values()} - {None}:
            pool.close()
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute("DELETE FROM scheduled_tasks WHERE task_name='identity-proof'")
            for service in ("billing", "reconciler"):
                admin.execute(f'ALTER ROLE "{db_roles.SERVICE_ROLES[service].name}" NOLOGIN')


def test_external_partitions_run_as_migrator_and_retain_runtime_grants(provisioned_roles):
    from psycopg import sql
    from psycopg.conninfo import make_conninfo
    from scripts.maintain_db_partitions import maintain

    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    with psycopg.connect(dsn, autocommit=True) as admin:
        original_owner = admin.execute("SELECT current_user").fetchone()[0]
        schema_owners = admin.execute(
            "SELECT nspname, pg_get_userbyid(nspowner) FROM pg_namespace "
            "WHERE nspname IN ('public', 'storage')"
        ).fetchall()
        admin.execute("ALTER ROLE xcelsior_migrator LOGIN")
        # The disposable cluster's bootstrap role also owns system objects;
        # transfer only application schemas/tables, as a real cutover must.
        for schema, _ in schema_owners:
            admin.execute(sql.SQL("ALTER SCHEMA {} OWNER TO xcelsior_migrator").format(sql.Identifier(schema)))
        for schema, table in admin.execute(
            "SELECT schemaname, tablename FROM pg_tables WHERE schemaname IN ('public','storage')"
        ).fetchall():
            admin.execute(sql.SQL("ALTER TABLE {}.{} OWNER TO xcelsior_migrator").format(sql.Identifier(schema), sql.Identifier(table)))
        admin.execute("ALTER SEQUENCE placement_fencing_token_seq OWNER TO xcelsior_migrator")
    try:
        with psycopg.connect(make_conninfo(dsn, user="xcelsior_migrator")) as conn:
            maintain(conn)
            maintain(conn)
        with psycopg.connect(dsn) as conn:
            assert conn.execute(
                "SELECT last_status FROM scheduled_tasks WHERE task_name='telemetry_partition_maintenance'"
            ).fetchone() == ("succeeded",)
            assert conn.execute(
                "SELECT has_table_privilege('xcelsior_reconciler', 'telemetry_samples', 'INSERT'), "
                "has_table_privilege('xcelsior_readonly', 'telemetry_samples', 'INSERT')"
            ).fetchone() == (True, False)
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            for schema, table in admin.execute(
                "SELECT schemaname, tablename FROM pg_tables WHERE schemaname IN ('public','storage')"
            ).fetchall():
                admin.execute(sql.SQL("ALTER TABLE {}.{} OWNER TO {}").format(sql.Identifier(schema), sql.Identifier(table), sql.Identifier(original_owner)))
            for schema, owner in schema_owners:
                admin.execute(sql.SQL("ALTER SCHEMA {} OWNER TO {}").format(sql.Identifier(schema), sql.Identifier(owner)))
            admin.execute(sql.SQL("ALTER SEQUENCE placement_fencing_token_seq OWNER TO {}").format(sql.Identifier(original_owner)))
            admin.execute("ALTER ROLE xcelsior_migrator NOLOGIN")


def test_only_control_writers_can_advance_the_placement_fence(provisioned_roles):
    from control_plane.db_role_audit import role_violations

    with psycopg.connect(os.environ["XCELSIOR_POSTGRES_DSN"]) as conn:
        conn.execute("SET ROLE xcelsior_scheduler")
        assert conn.execute("SELECT nextval('placement_fencing_token_seq')").fetchone()[0] > 0
        conn.execute("RESET ROLE")
        conn.execute("GRANT USAGE ON SEQUENCE placement_fencing_token_seq TO xcelsior_billing")
        assert any("placement_fencing_token_seq" in issue for issue in role_violations(conn, "billing", check_session=False))
        conn.rollback()
        conn.execute("SET ROLE xcelsior_billing")
        with pytest.raises(InsufficientPrivilege):
            conn.execute("SELECT nextval('placement_fencing_token_seq')")


@pytest.mark.asyncio
async def test_ssh_gateway_uses_real_restricted_login(provisioned_roles, monkeypatch):
    from psycopg.conninfo import make_conninfo
    import ssh_gateway

    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    monkeypatch.setenv("XCELSIOR_DB_ROLES_REQUIRED", "1")
    monkeypatch.setenv("XCELSIOR_POSTGRES_DSN_SSH_GATEWAY", make_conninfo(dsn, user="xcelsior_ssh_gateway"))
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute("ALTER ROLE xcelsior_ssh_gateway LOGIN")
    try:
        async with await ssh_gateway._get_pg_conn() as conn:
            assert await (await conn.execute("SELECT current_user")).fetchone() == ("xcelsior_ssh_gateway",)
            await conn.execute("SELECT job_id FROM jobs LIMIT 1")
            with pytest.raises(InsufficientPrivilege):
                await conn.execute("SELECT * FROM api_keys LIMIT 1")
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute("ALTER ROLE xcelsior_ssh_gateway NOLOGIN")


def test_restricted_api_can_observe_another_roles_connection_identity(provisioned_roles):
    from control_plane import security_metrics

    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    with psycopg.connect(dsn, application_name="xcelsior-billing") as other:
        with psycopg.connect(dsn) as observer:
            observer.execute("SET ROLE xcelsior_api")
            snapshot = security_metrics.collect(observer)
            assert snapshot["database_identity_mismatches"] >= 1
            observer.rollback()
