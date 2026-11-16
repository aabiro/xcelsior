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
        f" with open({str(observed)!r},'a') as out:\n"
        "  out.write(json.dumps({'argv':sys.argv[1:],'credential':os.environ.get('XCELSIOR_POSTGRES_DSN_MIGRATOR')})+'\\n')\n"
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
        invocations = [json.loads(line) for line in observed.read_text().splitlines()]
        invocation = invocations[0]
        assert invocation["credential"] == ("private-fixture" if credential else None)
        assert "private-fixture" not in json.dumps([each["argv"] for each in invocations])
        assert invocation["argv"][-6:] == ["api", "python", "-m", "alembic", "upgrade", target]
        # The runtime grants follow the migration in the same one-shot shape,
        # and only where there is a migration credential to grant with.
        assert len(invocations) == (2 if credential else 1)
        if credential:
            assert "XCELSIOR_POSTGRES_DSN_MIGRATOR" in invocation["argv"]
            grants = invocations[1]
            assert grants["credential"] == "private-fixture"
            assert (
                grants["argv"][: invocation["argv"].index("api")]
                == invocation["argv"][: invocation["argv"].index("api")]
            )
            assert grants["argv"][-4:] == [
                "api",
                "python",
                "api/scripts/maintain_db_partitions.py",
                "--grants-only",
            ]
            private.chmod(0o644)
            with pytest.raises(RuntimeError, match="must be private"):
                entrypoint.run(tmp_path, private)
    assert not marker.exists()


@pytest.mark.parametrize("target", ["+2", "134:135", "134;echo unsafe", "base", "1"])
def test_staged_target_refuses_relative_or_unreviewed_revision_shapes(tmp_path, target):
    with pytest.raises(RuntimeError, match="absolute three-digit"):
        entrypoint.run(tmp_path, tmp_path / "missing", target=target)


@pytest.mark.parametrize("migration_exit", [0, 7])
def test_catalog_import_shares_the_one_shot_credential_and_waits_for_migration(
    tmp_path, monkeypatch, migration_exit
):
    from types import SimpleNamespace

    private = tmp_path / "maintenance.env"
    private.write_text("XCELSIOR_POSTGRES_DSN_MIGRATOR='private-catalog-fixture'\n")
    private.chmod(0o600)
    calls = []

    def run_command(command, **kwargs):
        if command[1:3] == ["compose", "config"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"services": {"api": {"environment": {"XCELSIOR_DB_ROLES_REQUIRED": "1"}}}}
                ),
            )
        calls.append((command, kwargs["env"]))
        return SimpleNamespace(returncode=migration_exit if "alembic" in command else 0)

    monkeypatch.setattr(entrypoint.subprocess, "run", run_command)
    assert entrypoint.run(tmp_path, private, import_compute_catalog=True) == migration_exit
    assert len(calls) == (1 if migration_exit else 3)
    for command, env in calls:
        assert "private-catalog-fixture" not in json.dumps(command)
        assert env["XCELSIOR_POSTGRES_DSN_MIGRATOR"] == "private-catalog-fixture"
    if not migration_exit:
        assert calls[1][0][-2:] == ["api/scripts/maintain_db_partitions.py", "--grants-only"]
        assert calls[2][0][-2:] == ["--import-dsn-env", "XCELSIOR_POSTGRES_DSN_MIGRATOR"]
        assert "api/scripts/sync_stripe_service_tiers.py" in calls[2][0]
    with pytest.raises(RuntimeError, match="requires migration 147 or later"):
        entrypoint.run(tmp_path, private, target="145", import_compute_catalog=True)


def test_a_staged_release_stops_before_the_contract_and_still_imports_the_catalog(
    tmp_path, monkeypatch
):
    # Release one of a contract (149 drops a column the old release writes):
    # migrate to 148, import the catalog, and leave 149 for the next deploy.
    from types import SimpleNamespace

    private = tmp_path / "maintenance.env"
    private.write_text("XCELSIOR_POSTGRES_DSN_MIGRATOR='private-catalog-fixture'\n")
    private.chmod(0o600)
    calls = []

    def run_command(command, **kwargs):
        if command[1:3] == ["compose", "config"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"services": {"api": {"environment": {"XCELSIOR_DB_ROLES_REQUIRED": "1"}}}}
                ),
            )
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(entrypoint.subprocess, "run", run_command)
    assert entrypoint.run(tmp_path, private, target="148", import_compute_catalog=True) == 0
    assert calls[0][-3:] == ["alembic", "upgrade", "148"]
    assert calls[1][-1] == "--grants-only"
    assert "api/scripts/sync_stripe_service_tiers.py" in calls[2]


@pytest.mark.parametrize("grants_exit", [0, 5])
def test_runtime_grants_follow_the_migration_and_a_failure_stops_the_deploy(
    tmp_path, monkeypatch, grants_exit
):
    """A table the release created must be usable before its API goes live.

    The grants run from this release's image, right after Alembic and before
    the catalog import; if they fail, nothing after them runs and the deploy
    aborts before activation (deploy.sh treats any non-zero as fatal).
    """
    from types import SimpleNamespace

    private = tmp_path / "maintenance.env"
    private.write_text("XCELSIOR_POSTGRES_DSN_MIGRATOR='private-grants-fixture'\n")
    private.chmod(0o600)
    calls = []

    def run_command(command, **kwargs):
        if command[1:3] == ["compose", "config"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"services": {"api": {"environment": {"XCELSIOR_DB_ROLES_REQUIRED": "1"}}}}
                ),
            )
        calls.append(command)
        return SimpleNamespace(returncode=grants_exit if "--grants-only" in command else 0)

    monkeypatch.setattr(entrypoint.subprocess, "run", run_command)
    assert entrypoint.run(tmp_path, private, import_compute_catalog=True) == grants_exit
    assert [call[-1] for call in calls[:2]] == ["head", "--grants-only"]
    assert len(calls) == (3 if grants_exit == 0 else 2)


def test_without_a_migration_credential_there_is_nothing_to_grant_with(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls = []

    def run_command(command, **kwargs):
        if command[1:3] == ["compose", "config"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"services": {"api": {"environment": {"XCELSIOR_DB_ROLES_REQUIRED": "0"}}}}
                ),
            )
        calls.append(command)
        return SimpleNamespace(returncode=0)

    for key in list(os.environ):
        if key.startswith("XCELSIOR_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(entrypoint.subprocess, "run", run_command)
    assert entrypoint.run(tmp_path, tmp_path / "missing.env") == 0
    assert len(calls) == 1 and calls[0][-2:] == ["upgrade", "head"]
