"""Failed migration/promotion must not mark unactivated inputs as deployed.

Run the deployment shell with fake transport and real persisted fingerprints.
No production host, Docker daemon or external network is used.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[1] / "api/scripts/deploy.sh"


def _function(name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", DEPLOY.read_text(), re.M | re.S)
    assert match, name
    return match.group(0)


def _run(
    directory: Path, failure: str, *, pull_only: bool = False
) -> subprocess.CompletedProcess[str]:
    script = r"""
set -euo pipefail
TARGET_ENV=prod
DEPLOY_BUILD_API=true
DEPLOY_BUILD_FRONTEND=true
DEPLOY_RUNTIME_CHANGED=true
DEPLOY_BUILD_CACHE=1
XCELSIOR_DEPLOY_SKIP_MCP=1
ready_calls=0
log() { :; }
warn() { :; }
success() { :; }
error() { printf '%s\n' "$*" >&2; exit 1; }
ensure_remote_deploy_state_dir() { :; }
require_release_disk_headroom() { :; }  # Capacity is covered by its own real CLI gate.
remote_docker_build_prefix() { printf 'true;'; }
validate_build_env() { :; }
verify_frontend_desktop_runtime_on_remote() { :; }
collect_frontend_build_args() { :; }
read_remote_api_colour() { cat "$fixture/colour"; }
store_remote_api_colour() { printf '%s' "$1" > "$fixture/colour"; }
read_remote_mcp_colour() { echo green; }
store_remote_mcp_colour() { :; }
deploy_ssh_gateway_blue_green() { :; }
remote_deploy_state_dir() { echo /tmp/deploy-state; }
store_remote_deploy_hashes() {
    cp "$fixture/desired" "$fixture/applied"
    printf 'persist\n' >> "$fixture/events"
}
wait_for_api_readiness() {
    ready_calls=$((ready_calls + 1))
    [[ "$failure" != standby || "$ready_calls" != 1 ]] || return 1
    [[ "$failure" != readiness || "$ready_calls" != 2 ]] || return 1
}
ssh_cmd_host() {
    shift
    printf '%s\n' "$*" >> "$fixture/events"
    case "$*" in
        *'release_images.py stage-compose'*)
            [[ "$failure" != stage ]] || return 1
            read -r release
            [[ "$release" == candidate ]] || return 1
            echo 'docker-compose.yml:/tmp/deploy-state/release.yml'
            ;;
        *'docker compose --profile blue pull'*) [[ "$failure" != pull ]] ;;
        *'release_images.py verify-pull'*) [[ "$failure" != revision ]] ;;
        *'run_db_migrations.py'*) [[ "$failure" != migration ]] ;;
        *'up -d --no-deps scheduler-worker'*) [[ "$failure" != scheduler ]] ;;
        *'up -d --no-deps frontend'*) [[ "$failure" != frontend ]] ;;
        *'MemAvailable'*) echo 1024 ;;
        *'docker ps'*) echo xcelsior-jaeger-1 ;;
        *'/readyz'*) echo '{"ok":true,"nfs_volumes":{"ok":true}}' ;;
        *'/paypal/enabled'*) echo '{"enabled":true}' ;;
        *) return 0 ;;
    esac
}
"""
    script += f"\nfixture={shlex.quote(str(directory))}\nfailure={shlex.quote(failure)}\n"
    script += f"PROJECT_DIR={shlex.quote(str(DEPLOY.parents[2]))}\nREMOTE_HOST=fixture\n"
    if pull_only:
        script += "DEPLOY_RELEASE_CONFIG=candidate\nXCELSIOR_DEPLOY_SKIP_MCP=0\n"
    script += _function("ssh_cmd") + _function("persist_deploy_inputs") + _function("deploy_docker")
    # Same order as the production callers. Failure exits before persistence.
    script += "\ndeploy_docker\npersist_deploy_inputs\n"
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=15)


@pytest.mark.parametrize("failure", ["migration", "standby", "scheduler", "frontend", "readiness"])
def test_failed_release_keeps_previous_inputs_until_successful_retry(tmp_path, failure):
    previous = "api=old-api\nfrontend=old-frontend\nruntime=old-runtime\nnginx=old-nginx\n"
    desired = previous.replace("old-", "new-")
    (tmp_path / "applied").write_text(previous)
    (tmp_path / "desired").write_text(desired)
    (tmp_path / "colour").write_text("green")

    failed = _run(tmp_path, failure)
    assert failed.returncode != 0, failed.stdout
    assert (tmp_path / "applied").read_text() == previous, (
        "retry would skip images/runtime whose activation did not finish"
    )
    assert "persist\n" not in (tmp_path / "events").read_text()
    if failure == "migration":
        assert "run_db_migrations.py" in (tmp_path / "events").read_text()
        assert (tmp_path / "colour").read_text() == "green"

    retried = _run(tmp_path, "none")
    assert retried.returncode == 0, retried.stderr
    assert (tmp_path / "applied").read_text() == desired
    events = (tmp_path / "events").read_text()
    assert events.count("persist\n") == 1
    assert events.endswith("persist\n")


@pytest.mark.parametrize("failure", ["stage", "pull", "revision", "none"])
def test_pull_only_release_gates_migrations_and_uses_pinned_compose(tmp_path, failure):
    (tmp_path / "desired").write_text("new fingerprints\n")
    (tmp_path / "colour").write_text("green")
    result = _run(tmp_path, failure, pull_only=True)
    events = (tmp_path / "events").read_text()
    assert "docker compose --profile blue build" not in events
    assert "up -d --build" not in events
    if failure != "none":
        assert result.returncode != 0
        assert "run_db_migrations.py" not in events
        assert "persist\n" not in events
        return
    assert result.returncode == 0, result.stderr
    assert (
        events.index("pull --policy always api frontend mcp")
        < events.index("release_images.py verify-pull")
        < events.index("run_db_migrations.py")
    )
    prefix = "export COMPOSE_FILE=docker-compose.yml:/tmp/deploy-state/release.yml;"
    assert all(
        command.startswith(prefix)
        for command in events.splitlines()
        if "run_db_migrations.py" in command or " up -d " in command
    )
    assert "up -d --no-deps mcp-blue" in events
    assert events.endswith("persist\n")


def test_code_backups_keep_environment_private_even_with_existing_directory(tmp_path):
    opt = tmp_path / "opt"
    source = opt / "xcelsior"
    source.mkdir(parents=True)
    (source / "api.py").write_text("# fake deployment\n")
    (source / ".env").write_text("SYNTHETIC_SECRET=test-only\n")
    backups = opt / "backups"
    backups.mkdir(mode=0o755)
    script = (
        "set -euo pipefail\n"
        'log() { :; }; success() { :; }; sudo() { "$@"; }\n'
        "ssh_cmd() { source /dev/stdin; }\n"
        "require_release_disk_headroom() { :; }\n"
        f"DEPLOY_DIR={shlex.quote(str(source))}\n"
        f"BACKUP_DIR={shlex.quote(str(backups))}\n"
        + _function("backup_current").replace("-C /opt ", f"-C {shlex.quote(str(opt))} ")
        + "\nbackup_current\n"
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    archives = list(backups.glob("xcelsior_*.tar.gz"))
    assert len(archives) == 1
    assert backups.stat().st_mode & 0o777 == 0o700
    assert archives[0].stat().st_mode & 0o777 == 0o600
