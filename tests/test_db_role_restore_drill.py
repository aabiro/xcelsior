"""Run the full native dump/restore/migrate drill under the migration login."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import psycopg
from sqlalchemy.engine import make_url

from tests._db_helpers import allow_test_login
from tests.test_db_service_roles import provisioned_roles

ROOT = Path(__file__).resolve().parents[1]


def test_real_strict_role_restore_is_disposable_and_never_migrates_the_source(
    provisioned_roles, tmp_path
):
    dsn = os.environ["XCELSIOR_POSTGRES_DSN"]
    source_url = make_url(dsn)
    migrate_url = source_url.difference_update_query(["user", "password"]).set(
        username="xcelsior_migrator"
    )
    with psycopg.connect(dsn) as source:
        pg_bin = Path("/usr/lib/postgresql") / str(source.info.server_version // 10000) / "bin"
    assert (pg_bin / "pg_dump").is_file(), "matching PostgreSQL client tools are required"
    backups = tmp_path / "backups"
    backups.mkdir()
    dump = backups / "xcelsior_fixture.dump"
    subprocess.run(
        [str(pg_bin / "pg_dump"), "--dbname", dsn, "--format=custom", "--file", str(dump)],
        check=True,
        timeout=30,
    )
    dump.with_suffix(".dump.sha256").write_text(
        hashlib.sha256(dump.read_bytes()).hexdigest() + "  " + dump.name + "\n"
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    identity = binaries / "id"
    identity.write_text('#!/bin/sh\nif [ "$1" = "-u" ]; then echo 1; else echo postgres; fi\n')
    identity.chmod(0o755)
    observed = tmp_path / "docker.json"
    docker = binaries / "docker"
    docker.write_text(
        f"#!{sys.executable}\nimport json,os,subprocess,sys\n"
        f"with open({str(observed)!r},'w') as out:\n"
        " json.dump(sys.argv[1:],out)\n"
        "assert '-e' in sys.argv and 'XCELSIOR_POSTGRES_DSN_MIGRATOR' in sys.argv\n"
        "raise SystemExit(subprocess.run([sys.executable,'-',sys.argv[-1]],env=os.environ.copy()).returncode)\n"
    )
    docker.chmod(0o755)
    env = os.environ.copy()
    env.update(
        PATH=str(binaries) + ":" + str(pg_bin) + ":" + env["PATH"],
        PGHOST=source_url.query.get("host", source_url.host or "localhost"),
        PGPORT=str(source_url.port or 5432),
        PGUSER=source_url.query.get("user", source_url.username or "security_test"),
        XCELSIOR_DB_ROLES_REQUIRED="1",
        XCELSIOR_POSTGRES_DSN_MIGRATOR=migrate_url.render_as_string(hide_password=False),
        XCELSIOR_PROJECT_DIR=str(ROOT),
        XCELSIOR_BACKUP_DIR=str(backups),
        XCELSIOR_NODE_EXPORTER_TEXTFILE_DIR=str(tmp_path / "metrics"),
    )
    with psycopg.connect(dsn, autocommit=True) as admin:
        before_revision = admin.execute("SELECT version_num FROM alembic_version").fetchall()
        before_databases = admin.execute(
            "SELECT datname FROM pg_database ORDER BY datname"
        ).fetchall()
        before_jobs = admin.execute("SELECT count(*) FROM jobs").fetchone()
        allow_test_login(admin, "xcelsior_migrator", dsn)
        try:
            result = subprocess.run(
                ["bash", str(ROOT / "api/scripts/restore-db-drill.sh")],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            evidence_files = list((backups / "restore-evidence").glob("*.json"))
            assert len(evidence_files) == 1
            evidence = json.loads(evidence_files[0].read_text())
            assert evidence["schema_revision"] == before_revision[0][0]
            assert all(evidence["invariants"].values())
            assert (
                admin.execute("SELECT version_num FROM alembic_version").fetchall()
                == before_revision
            )
            assert (
                admin.execute("SELECT datname FROM pg_database ORDER BY datname").fetchall()
                == before_databases
            )
            assert admin.execute("SELECT count(*) FROM jobs").fetchone() == before_jobs
            assert "XCELSIOR_POSTGRES_DSN_MIGRATOR" in json.loads(observed.read_text())
            assert env["XCELSIOR_POSTGRES_DSN_MIGRATOR"] not in observed.read_text()
        finally:
            admin.execute("ALTER ROLE xcelsior_migrator NOLOGIN PASSWORD NULL")
