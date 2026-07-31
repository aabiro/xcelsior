"""Executable safety gates for the production database backup workflow."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def test_backup_reads_dotenv_as_data_and_rotates_sidecars(tmp_path: Path) -> None:
    project = tmp_path / "project"
    scripts = project / "scripts"
    fake_bin = tmp_path / "bin"
    backup_dir = tmp_path / "backups"
    metrics_dir = tmp_path / "metrics"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    backup_dir.mkdir()

    script = scripts / "backup-db.sh"
    shutil.copy2(ROOT / "scripts" / "backup-db.sh", script)

    executed_marker = tmp_path / "dotenv-was-executed"
    (project / ".env").write_text(
        "\n".join(
            [
                "XCELSIOR_POSTGRES_DB=restore_fixture",
                "XCELSIOR_POSTGRES_USER=fixture_user",
                "XCELSIOR_POSTGRES_HOST=127.0.0.1",
                "XCELSIOR_POSTGRES_PORT=5432",
                "XCELSIOR_POSTGRES_PASSWORD=not-a-real-secret",
                "XCELSIOR_SCHEDULER_CANARY_GPU_MODELS=rtx 2060,nvidia geforce rtx 2060",
                f"UNRELATED_VALUE=$(touch {executed_marker})",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _write_executable(
        fake_bin / "pg_dump",
        """#!/usr/bin/env bash
set -euo pipefail
output=
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-f" ]]; then output="$2"; shift 2; else shift; fi
done
[[ -n "$output" ]]
printf 'valid fixture dump\\n' >"$output"
""",
    )
    _write_executable(
        fake_bin / "pg_restore",
        "#!/usr/bin/env bash\nset -euo pipefail\n[[ \"$1\" == \"--list\" ]]\n",
    )

    old_dump = backup_dir / "restore_fixture_old.dump"
    old_dump.write_text("old", encoding="utf-8")
    old_checksum = Path(f"{old_dump}.sha256")
    old_checksum.write_text("old", encoding="utf-8")
    old_time = time.time() - 20 * 86400
    os.utime(old_dump, (old_time, old_time))
    os.utime(old_checksum, (old_time, old_time))

    env = os.environ.copy()
    for key in (
        "XCELSIOR_POSTGRES_DB",
        "XCELSIOR_POSTGRES_USER",
        "XCELSIOR_POSTGRES_HOST",
        "XCELSIOR_POSTGRES_PORT",
        "XCELSIOR_POSTGRES_PASSWORD",
    ):
        env.pop(key, None)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "XCELSIOR_BACKUP_DIR": str(backup_dir),
            "XCELSIOR_BACKUP_RETAIN_DAYS": "14",
            "XCELSIOR_NODE_EXPORTER_TEXTFILE_DIR": str(metrics_dir),
        }
    )
    result = subprocess.run(
        [str(script)],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "restore_fixture_" in result.stdout
    assert not executed_marker.exists()
    assert not old_dump.exists()
    assert not old_checksum.exists()

    dumps = list(backup_dir.glob("restore_fixture_*.dump"))
    assert len(dumps) == 1
    assert Path(f"{dumps[0]}.sha256").is_file()
    assert oct(dumps[0].stat().st_mode & 0o777) == "0o600"
    success_metrics = (metrics_dir / "xcelsior_backup_success.prom").read_text(
        encoding="utf-8"
    )
    assert "xcelsior_backup_last_success_timestamp_seconds" in success_metrics
    assert "xcelsior_backup_last_size_bytes" in success_metrics
