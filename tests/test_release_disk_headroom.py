"""Release disk checks protect shared application writes without deleting data."""

import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "api/scripts/check_release_disk.py"
spec = importlib.util.spec_from_file_location("release_disk", SCRIPT)
assert spec and spec.loader
disk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(disk)


@pytest.mark.parametrize("available,passed", [(24, True), (23, True), (22, False), (0, False)])
def test_incoming_build_budget_must_leave_ten_percent_for_other_services(
    tmp_path, monkeypatch, available, passed
):
    monkeypatch.setattr(
        disk.os,
        "statvfs",
        lambda path: SimpleNamespace(f_blocks=150, f_bavail=available, f_frsize=disk.GIB),
    )
    result = disk.check_headroom([tmp_path / "release", tmp_path / "docker"], 8 * disk.GIB)
    assert result["passed"] is passed
    assert len(result["filesystems"]) == 1
    assert result["filesystems"][0]["reserved_bytes"] == 15 * disk.GIB
    assert not list(tmp_path.iterdir())


def test_small_filesystem_keeps_absolute_five_gib_reserve(tmp_path, monkeypatch):
    monkeypatch.setattr(
        disk.os,
        "statvfs",
        lambda path: SimpleNamespace(f_blocks=20, f_bavail=4, f_frsize=disk.GIB),
    )
    assert not disk.check_headroom([tmp_path], 0)["passed"]


@pytest.mark.parametrize("budget", [-1, True, "8GB"])
def test_invalid_budget_never_disables_the_guard(tmp_path, budget):
    with pytest.raises(ValueError, match="nonnegative integer"):
        disk.check_headroom([tmp_path], budget)


def test_probe_error_refuses_release(tmp_path, monkeypatch, capsys):
    def unavailable(path):
        raise OSError("capacity probe unavailable")

    monkeypatch.setattr(disk.os, "statvfs", unavailable)
    assert disk.main(["--path", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["passed"] is False


def test_actual_cli_refuses_an_unfunded_budget_without_creating_files(tmp_path):
    before = list(tmp_path.iterdir())
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--path", str(tmp_path), "--required-bytes", str(2**63)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["read_only"] is True
    assert list(tmp_path.iterdir()) == before


def test_deploy_guards_backup_sync_mirrors_and_build_before_release_writes():
    from tests.test_deploy_runtime_readiness import _function

    for name in ("backup_current", "sync_code", "deploy_docker"):
        assert _function(name).splitlines()[1].strip() == "require_release_disk_headroom"
    assert 'require_release_disk_headroom "$host"' in _function("sync_code_mirror_host")
    guard = _function("require_release_disk_headroom")
    assert '"$SCRIPT_DIR/check_release_disk.py"' in guard
    assert '|| error "Release blocked:' in guard


@pytest.mark.parametrize("operation", ["backup_current", "sync_code", "deploy_docker"])
def test_failed_remote_capacity_probe_stops_before_any_release_write(tmp_path, operation):
    from tests.test_deploy_runtime_readiness import _function

    marker = tmp_path / "release-write"
    script = (
        "set -euo pipefail\nREMOTE_HOST=isolated-host\n"
        f"SCRIPT_DIR={shlex.quote(str(SCRIPT.parent))}\n"
        "ssh_cmd_host() { cat >/dev/null; return 1; }\n"
        "error() { exit 21; }\n"
        f"log() {{ touch {shlex.quote(str(marker))}; }}\n"
        + _function("require_release_disk_headroom")
        + _function(operation)
        + operation
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 21
    assert not marker.exists()
