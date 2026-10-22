"""The restore admin tools must use the configured application cluster."""

import hashlib
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("port", ["5434", "not-a-port"])
def test_restore_tools_choose_dotenv_port_without_executing_dotenv(tmp_path, port):
    project = tmp_path / "project"
    project.mkdir()
    marker = tmp_path / "executed"
    observed = tmp_path / "observed"
    (project / ".env").write_text(f"XCELSIOR_POSTGRES_PORT={port}\nUNRELATED=$(touch {marker})\n")
    backup = tmp_path / "backups"
    backup.mkdir()
    dump = backup / "xcelsior_test.dump"
    dump.write_bytes(b"fixture dump")
    (backup / "xcelsior_test.dump.sha256").write_text(
        hashlib.sha256(dump.read_bytes()).hexdigest() + "  xcelsior_test.dump\n"
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, body in {
        "id": 'if [ "$1" = "-u" ]; then echo 1; else echo postgres; fi',
        "createdb": f'printf "%s" "${{PGPORT-unset}}" > "{observed}"; exit 42',
        "dropdb": "exit 0",
    }.items():
        target = binaries / name
        target.write_text("#!/bin/sh\n" + body + "\n")
        target.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("XCELSIOR_", "PG"))}
    env.update(
        PATH=str(binaries) + ":" + env["PATH"],
        XCELSIOR_PROJECT_DIR=str(project),
        XCELSIOR_BACKUP_DIR=str(backup),
        XCELSIOR_NODE_EXPORTER_TEXTFILE_DIR=str(tmp_path / "metrics"),
    )
    result = subprocess.run(
        ["bash", str(ROOT / "api/scripts/restore-db-drill.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0  # fake createdb intentionally stops before database mutation
    assert not marker.exists()
    if port.isdigit():
        assert observed.read_text() == port
    else:
        assert not observed.exists()
        assert "invalid PostgreSQL port" in result.stderr
