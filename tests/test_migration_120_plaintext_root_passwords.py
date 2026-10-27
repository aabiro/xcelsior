"""Migration 120 removes stored root passwords from instances that are not running."""

import importlib.util
import uuid
from pathlib import Path
from types import SimpleNamespace

from db import DatabaseOps, get_engine

MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "api"
    / "migrations"
    / "versions"
    / "120_drop_plaintext_root_passwords.py"
)


def _migration():
    spec = importlib.util.spec_from_file_location("migration_120", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_running_instances_keep_their_password(monkeypatch):
    tag = uuid.uuid4().hex[:6]
    ssh = {
        "host": "100.64.0.9",
        "port": 36249,
        "root_password": "Zr8q2mWv9cXk4TnLp7Ad",
        "sshd": True,
    }
    rows = {
        f"m120st{tag}": ("stopped", True),
        f"m120fa{tag}": ("failed", True),
        f"m120te{tag}": ("terminated", True),
        f"m120ru{tag}": ("running", False),
    }
    with get_engine().transaction() as (conn, backend):
        for job_id, (status, _) in rows.items():
            DatabaseOps.upsert_job(
                conn,
                {
                    "job_id": job_id,
                    "name": "m120",
                    "status": status,
                    "priority": 0,
                    "ssh_status": dict(ssh),
                },
                backend=backend,
            )
        DatabaseOps.upsert_job(
            conn,
            {
                "job_id": f"m120en{tag}",
                "name": "m120",
                "status": "stopped",
                "priority": 0,
                "ssh_status": {"root_password_encrypted": "gAAAA-kept"},
            },
            backend=backend,
        )
    try:
        migration = _migration()
        with get_engine().transaction() as (conn, _backend):
            monkeypatch.setattr(migration, "op", SimpleNamespace(execute=conn.execute))
            migration.upgrade()
        with get_engine().connection() as (conn, backend):
            for job_id, (_status, scrubbed) in rows.items():
                stored = DatabaseOps.get_job(conn, job_id, backend=backend)["ssh_status"]
                assert ("root_password" in stored) is not scrubbed, job_id
                assert stored["port"] == 36249
            kept = DatabaseOps.get_job(conn, f"m120en{tag}", backend=backend)["ssh_status"]
            assert kept == {"root_password_encrypted": "gAAAA-kept"}
    finally:
        with get_engine().transaction() as (conn, _backend):
            conn.execute("DELETE FROM jobs WHERE job_id LIKE %s", (f"m120%{tag}",))
