"""The deploy host injects a private migration credential into one-shot jobs."""

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "migration_entrypoint", ROOT / "api/scripts/run_db_migrations.py"
)
entrypoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entrypoint)


@pytest.mark.parametrize("strict,credential", [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize("target", ["head", "126", "134", "135"])
def test_deploy_credential_injection_is_one_shot_and_fails_closed(
    tmp_path, monkeypatch, strict, credential, target
):
    for key in list(os.environ):
        if key.startswith("XCELSIOR_"):
            monkeypatch.delenv(key)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    observed = tmp_path / "run.json"
    configuration = json.dumps(
        {"services": {"api": {"environment": {"XCELSIOR_DB_ROLES_REQUIRED": str(int(strict))}}}}
    )
    fake = binaries / "docker"
    fake.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "if sys.argv[1:3]==['compose','config']:\n"
        f" print({configuration!r})\n"
        "else:\n"
        f" with open({str(observed)!r},'w') as out:\n"
        "  json.dump({'argv':sys.argv[1:],'credential':os.environ.get('XCELSIOR_POSTGRES_DSN_MIGRATOR')},out)\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(binaries) + ":" + os.environ["PATH"])
    private = tmp_path / "maintenance.env"
    marker = tmp_path / "executed"
    if credential:
        private.write_text(
            f"XCELSIOR_POSTGRES_DSN_MIGRATOR='private-fixture'\nUNRELATED=$(touch {marker})\n"
        )
        private.chmod(0o600)
    if strict and not credential:
        with pytest.raises(RuntimeError, match="private migration credential"):
            entrypoint.run(tmp_path, private, target=target)
        assert not observed.exists()
    else:
        assert entrypoint.run(tmp_path, private, target=target) == 0
        invocation = json.loads(observed.read_text())
        assert invocation["credential"] == ("private-fixture" if credential else None)
        assert "private-fixture" not in json.dumps(invocation["argv"])
        assert invocation["argv"][-6:] == ["api", "python", "-m", "alembic", "upgrade", target]
        if credential:
            assert "XCELSIOR_POSTGRES_DSN_MIGRATOR" in invocation["argv"]
            private.chmod(0o644)
            with pytest.raises(RuntimeError, match="must be private"):
                entrypoint.run(tmp_path, private)
    assert not marker.exists()


@pytest.mark.parametrize("target", ["+2", "134:135", "134;echo unsafe", "base", "1"])
def test_staged_target_refuses_relative_or_unreviewed_revision_shapes(tmp_path, target):
    with pytest.raises(RuntimeError, match="absolute three-digit"):
        entrypoint.run(tmp_path, tmp_path / "missing", target=target)


@pytest.mark.parametrize("migration_exit", [0, 7])
def test_catalog_import_shares_the_one_shot_credential_and_waits_for_migration(tmp_path, monkeypatch, migration_exit):
    from types import SimpleNamespace

    private = tmp_path / "maintenance.env"
    private.write_text("XCELSIOR_POSTGRES_DSN_MIGRATOR='private-catalog-fixture'\n")
    private.chmod(0o600)
    calls = []

    def run_command(command, **kwargs):
        if command[1:3] == ["compose", "config"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"services": {"api": {"environment": {"XCELSIOR_DB_ROLES_REQUIRED": "1"}}}}))
        calls.append((command, kwargs["env"]))
        return SimpleNamespace(returncode=migration_exit if "alembic" in command else 0)

    monkeypatch.setattr(entrypoint.subprocess, "run", run_command)
    assert entrypoint.run(tmp_path, private, import_compute_catalog=True) == migration_exit
    assert len(calls) == (1 if migration_exit else 2)
    for command, env in calls:
        assert "private-catalog-fixture" not in json.dumps(command)
        assert env["XCELSIOR_POSTGRES_DSN_MIGRATOR"] == "private-catalog-fixture"
    if not migration_exit:
        assert calls[1][0][-2:] == ["--import-dsn-env", "XCELSIOR_POSTGRES_DSN_MIGRATOR"]
        assert "api/scripts/sync_stripe_service_tiers.py" in calls[1][0]
    with pytest.raises(RuntimeError, match="requires migration head"):
        entrypoint.run(tmp_path, private, target="145", import_compute_catalog=True)
