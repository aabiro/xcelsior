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
def test_deploy_credential_injection_is_one_shot_and_fails_closed(
    tmp_path, monkeypatch, strict, credential
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
            entrypoint.run(tmp_path, private)
        assert not observed.exists()
    else:
        assert entrypoint.run(tmp_path, private) == 0
        invocation = json.loads(observed.read_text())
        assert invocation["credential"] == ("private-fixture" if credential else None)
        assert "private-fixture" not in json.dumps(invocation["argv"])
        assert invocation["argv"][-6:] == ["api", "python", "-m", "alembic", "upgrade", "head"]
        if credential:
            assert "XCELSIOR_POSTGRES_DSN_MIGRATOR" in invocation["argv"]
            private.chmod(0o644)
            with pytest.raises(RuntimeError, match="must be private"):
                entrypoint.run(tmp_path, private)
    assert not marker.exists()
