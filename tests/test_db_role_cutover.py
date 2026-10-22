"""Ownership, migration and restore acceptance on disposable PostgreSQL only."""

import json
import os
from pathlib import Path
import subprocess
import sys

import psycopg
from psycopg.conninfo import make_conninfo
from psycopg.errors import InsufficientPrivilege
from psycopg import sql
import pytest
from sqlalchemy.engine import make_url

from migrations.connection import migration_dsn, require_migration_identity
from scripts.transfer_db_ownership import MIGRATOR, ownership_plan, transfer
from tests.test_db_service_roles import provisioned_roles

ROOT = Path(__file__).resolve().parents[1]


def test_migrations_fail_closed_without_the_dedicated_credential(monkeypatch):
    monkeypatch.setenv("XCELSIOR_DB_ROLES_REQUIRED", "1")
    monkeypatch.delenv("XCELSIOR_POSTGRES_DSN_MIGRATOR", raising=False)
    with pytest.raises(RuntimeError, match="shared migration fallback refused"):
        migration_dsn()
    monkeypatch.setenv("XCELSIOR_POSTGRES_DSN_MIGRATOR", "dedicated")
    assert migration_dsn("fallback") == "dedicated"
    for current, login in [("xcelsior_api", "xcelsior_api"), (MIGRATOR, "admin")]:
        with pytest.raises(RuntimeError, match="direct xcelsior_migrator login"):
            require_migration_identity(current, login)
    require_migration_identity(MIGRATOR, MIGRATOR)


def test_ownership_refuses_the_wrong_database_and_unknown_relations(provisioned_roles):
    with psycopg.connect(os.environ["XCELSIOR_POSTGRES_DSN"]) as conn:
        database, owner = conn.execute("SELECT current_database(),current_user").fetchone()
        with pytest.raises(RuntimeError, match="unexpected database"):
            ownership_plan(
                conn, expected_database="postgres", expected_port=conn.info.port, from_owner=owner
            )
        with pytest.raises(RuntimeError, match="unexpected PostgreSQL port"):
            ownership_plan(
                conn, expected_database=database, expected_port=conn.info.port + 1, from_owner=owner
            )
        conn.execute("CREATE TABLE public.unclassified_ownership_probe(id int)")
        with pytest.raises(RuntimeError, match="unclassified application relation"):
            ownership_plan(
                conn, expected_database=database, expected_port=conn.info.port, from_owner=owner
            )
        conn.rollback()


def test_transfer_supports_real_migrations_partitions_backups_and_owner_rollback(provisioned_roles):
    from scripts.maintain_db_partitions import maintain
    from control_plane.db_role_audit import role_violations

    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    plan = []
    with psycopg.connect(dsn, autocommit=True) as admin:
        database, old = admin.execute("SELECT current_database(),current_user").fetchone()
        system_owners = admin.execute(
            "SELECT oid,datdba FROM pg_database WHERE datname IN ('postgres','template0','template1') ORDER BY oid"
        ).fetchall()
        system_relation = admin.execute(
            "SELECT relowner FROM pg_class WHERE oid='pg_class'::regclass"
        ).fetchone()
        before = admin.execute("SELECT version_num FROM alembic_version").fetchall()
        admin.execute("ALTER ROLE xcelsior_migrator LOGIN")
        try:
            with admin.transaction():
                plan = ownership_plan(
                    admin, expected_database=database, expected_port=admin.info.port, from_owner=old
                )
                assert plan
                transfer(admin, plan, backup_owner=old)
            assert (
                ownership_plan(
                    admin, expected_database=database, expected_port=admin.info.port, from_owner=old
                )
                == []
            )
            for service in (
                "api",
                "scheduler",
                "reconciler",
                "billing",
                "projector",
                "ssh_gateway",
            ):
                assert role_violations(admin, service, check_session=False) == []
            assert (
                admin.execute(
                    "SELECT oid,datdba FROM pg_database WHERE datname IN ('postgres','template0','template1') ORDER BY oid"
                ).fetchall()
                == system_owners
            )
            assert (
                admin.execute(
                    "SELECT relowner FROM pg_class WHERE oid='pg_class'::regclass"
                ).fetchone()
                == system_relation
            )
            migrator_dsn = make_conninfo(dsn, user=MIGRATOR)
            with psycopg.connect(migrator_dsn) as migrator:
                migrator.execute("CREATE TABLE public.backup_grant_probe(id serial)")
                backup_oid = migrator.execute(
                    "SELECT oid FROM pg_roles WHERE rolname=%s", (old,)
                ).fetchone()[0]
                assert migrator.execute(
                    "SELECT privilege_type FROM pg_class c, LATERAL aclexplode(c.relacl) acl "
                    "WHERE c.oid='backup_grant_probe'::regclass AND acl.grantee=%s",
                    (backup_oid,),
                ).fetchall() == [("SELECT",)]
                assert migrator.execute(
                    "SELECT has_table_privilege('xcelsior_api','backup_grant_probe','SELECT')"
                ).fetchone() == (False,)
                migrator.execute("DROP TABLE public.backup_grant_probe")
                migrator.commit()
                maintain(migrator)
            env = os.environ.copy()
            # A valid migration DSN must win over every shared/runtime alias.
            env.update(
                XCELSIOR_DB_ROLES_REQUIRED="1",
                XCELSIOR_POSTGRES_DSN_MIGRATOR=make_url(dsn)
                .difference_update_query(["user", "password"])
                .set(username=MIGRATOR)
                .render_as_string(hide_password=False),
                XCELSIOR_POSTGRES_DSN="postgresql://invalid@127.0.0.1:1/wrong",
            )
            migrated = subprocess.run(
                [sys.executable, "-m", "alembic", "upgrade", "head"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=45,
            )
            assert migrated.returncode == 0, migrated.stdout + migrated.stderr
            assert admin.execute("SELECT version_num FROM alembic_version").fetchall() == before
            admin.execute("SET ROLE xcelsior_api")
            with pytest.raises(InsufficientPrivilege):
                admin.execute("ALTER TABLE users ADD COLUMN forbidden_cutover_ddl text")
            admin.execute("RESET ROLE")
        finally:
            admin.execute("RESET ROLE")
            # Transfer any partitions created by maintenance too, then restore
            # the original object/schema/database owners from the saved plan.
            for schema, table in admin.execute(
                "SELECT schemaname,tablename FROM pg_tables WHERE schemaname IN ('public','storage')"
            ).fetchall():
                admin.execute(
                    sql.SQL("ALTER TABLE {}.{} OWNER TO {}").format(
                        sql.Identifier(schema), sql.Identifier(table), sql.Identifier(old)
                    )
                )
            for step in reversed(plan):
                admin.execute(step["rollback"])
            admin.execute("ALTER ROLE xcelsior_migrator NOLOGIN")


def test_strict_restore_without_credential_stops_before_any_database_command(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("XCELSIOR_", "PG"))}
    env.update(XCELSIOR_DB_ROLES_REQUIRED="1", XCELSIOR_BACKUP_DIR=str(tmp_path))
    result = subprocess.run(
        ["bash", str(ROOT / "api/scripts/restore-db-drill.sh")],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "strict-role restore requires XCELSIOR_POSTGRES_DSN_MIGRATOR" in result.stderr
    assert "creating isolated database" not in result.stdout


def test_compose_cutover_removes_shared_owner_and_migration_credentials(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("XCELSIOR_", "COMPOSE_"))}
    role_dsns = {
        key: f"postgresql://xcelsior_{key}:fixture-{key}@127.0.0.1:5434/xcelsior"
        for key in ("api", "scheduler", "reconciler", "billing", "projector", "ssh_gateway")
    }
    env.update({f"XCELSIOR_POSTGRES_DSN_{key.upper()}": value for key, value in role_dsns.items()})
    env.update(
        XCELSIOR_POSTGRES_PASSWORD="fixture-owner-secret",
        XCELSIOR_POSTGRES_DSN_MIGRATOR="fixture-migration-secret",
        XCELSIOR_POSTGRES_ADMIN_DSN="fixture-admin-secret",
    )
    empty_env = tmp_path / "empty.env"
    empty_env.touch()
    rendered = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(empty_env),
            "--profile",
            "*",
            "-f",
            str(ROOT / "docker-compose.yml"),
            "-f",
            str(ROOT / "docker-compose.db-roles.yml"),
            "config",
            "--format",
            "json",
        ],
        env=env,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert rendered.returncode == 0, rendered.stderr
    services = json.loads(rendered.stdout)["services"]
    for name, role in {
        "api": "api",
        "api-blue": "api",
        "scheduler-worker": "scheduler",
        "bg-worker": "reconciler",
        "ssh-gateway": "ssh_gateway",
        "ssh-gateway-blue": "ssh_gateway",
    }.items():
        actual = services[name]["environment"]
        assert actual["XCELSIOR_DB_ROLES_REQUIRED"] == "1"
        assert actual["XCELSIOR_SERVICE"] == role
        for alias in ("XCELSIOR_POSTGRES_DSN", "XCELSIOR_PG_DSN", "DATABASE_URL"):
            assert actual[alias] == role_dsns[role]
        assert all(
            secret not in json.dumps(actual)
            for secret in (
                "fixture-owner-secret",
                "fixture-migration-secret",
                "fixture-admin-secret",
            )
        )
    env.pop("XCELSIOR_POSTGRES_DSN_BILLING")
    refused = subprocess.run(
        rendered.args, env=env, cwd=ROOT, text=True, capture_output=True, timeout=30
    )
    assert refused.returncode != 0
    assert "Billing role DSN required" in refused.stderr


def test_partition_oneshot_pins_live_image_and_keeps_credentials_out_of_argv(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    observed = tmp_path / "docker.json"
    fake = binaries / "docker"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "if sys.argv[1]=='ps':\n"
        " print('running-api-generation api-blue')\n"
        "elif sys.argv[1]=='inspect':\n"
        " print('sha256:'+'a'*64)\n"
        "else:\n"
        f" with open({str(observed)!r},'w') as out:\n"
        "  json.dump({'argv':sys.argv[1:],'credential':os.environ.get('XCELSIOR_POSTGRES_DSN_MIGRATOR')},out)\n"
    )
    fake.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("XCELSIOR_")}
    env.update(
        PATH=str(binaries) + ":" + env["PATH"],
        XCELSIOR_POSTGRES_DSN_MIGRATOR="private-fixture-credential",
        XCELSIOR_NODE_EXPORTER_TEXTFILE_DIR=str(tmp_path / "metrics"),
    )
    result = subprocess.run(
        ["bash", str(ROOT / "api/scripts/run-db-partition-maintenance.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    invocation = json.loads(observed.read_text())
    assert invocation["credential"] == "private-fixture-credential"
    assert "private-fixture-credential" not in json.dumps(invocation["argv"])
    assert invocation["argv"][-2:] == [
        "sha256:" + "a" * 64,
        "api/scripts/maintain_db_partitions.py",
    ]
    assert "--read-only" in invocation["argv"]
    metrics = tmp_path / "metrics"
    assert (
        "last_run_success 1" in (metrics / "xcelsior_partition_maintenance_status.prom").read_text()
    )
    success_before = (metrics / "xcelsior_partition_maintenance_success.prom").read_text()
    observed.unlink()
    env.pop("XCELSIOR_POSTGRES_DSN_MIGRATOR")
    refused = subprocess.run(result.args, env=env, capture_output=True, text=True, timeout=10)
    assert refused.returncode != 0
    assert not observed.exists()
    assert (
        "last_run_success 0" in (metrics / "xcelsior_partition_maintenance_status.prom").read_text()
    )
    assert (metrics / "xcelsior_partition_maintenance_success.prom").read_text() == success_before
