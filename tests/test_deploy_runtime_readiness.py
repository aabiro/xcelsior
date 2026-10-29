"""Exercise deployment gates against a real HTTP dependency failure.

A green /healthz used to promote an API whose SSH bind was an empty directory,
breaking both persistent storage and every web terminal.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _function(name: str) -> str:
    script = (ROOT / "scripts/deploy.sh").read_text()
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", script, re.M | re.S)
    assert match
    return match.group(0)


@pytest.mark.parametrize("statuses,expected", [([503, 503], 1), ([503, 200], 0), ([200], 0)])
def test_promotion_waits_for_dependencies(statuses: list[int], expected: int) -> None:
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            status = statuses[min(len(requests) - 1, len(statuses) - 1)]
            self.send_response(status if self.path == "/readyz" else 200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, format: str, *args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'ssh_cmd() { bash -c "$1"; }\nsleep() { :; }\n'
                    + _function("wait_for_api_readiness")
                    + f"wait_for_api_readiness {server.server_port} {len(statuses)}",
                ],
                capture_output=True,
                timeout=15,
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert result.returncode == expected
    assert requests == ["/readyz"] * len(statuses)


def test_readiness_gates_the_swap_and_final_success() -> None:
    script = _function("deploy_docker")
    standby = script.index('if ! wait_for_api_readiness "$standby_port"; then')
    swap = script.index('log "Swapping nginx upstream')
    assert standby < swap
    assert 'error "Blue-green swap aborted' in script[standby:swap]
    assert 'if ! wait_for_api_readiness "$final_port" 15; then' in script
    assert 'error "Live API failed /readyz' in script


@pytest.mark.parametrize("deploy_home", ["/root", "/home/linuxuser"])
def test_explicit_runtime_key_does_not_follow_deploying_user(deploy_home: str) -> None:
    env = {
        **os.environ,
        "DOCKER_CONFIG": os.environ.get("DOCKER_CONFIG", str(Path.home() / ".docker")),
        "HOME": deploy_home,
        "XCELSIOR_SSH_KEY_HOST_PATH": "/runtime-credentials/platform-key",
        "XCELSIOR_SSH_KNOWN_HOSTS_HOST_PATH": "/runtime-credentials/known_hosts",
    }
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(ROOT / "docker-compose.yml"),
            "--env-file",
            "/dev/null",
            "--profile",
            "blue",
            "config",
            "--format",
            "json",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, "Compose configuration failed"
    services = json.loads(result.stdout)["services"]
    for name in ("api", "api-blue", "bg-worker", "scheduler-worker"):
        mounts = {volume["target"]: volume for volume in services[name]["volumes"]}
        key = mounts["/run/secrets/ssh_key"]
        assert key["source"] == "/runtime-credentials/platform-key", name
        assert key["read_only"] is True
        # Compose omits false values from normalized JSON; short bind syntax
        # instead resolves to create_host_path=true.
        assert key.get("bind", {}).get("create_host_path", False) is False
    hosts = mounts["/home/xcelsior/.ssh/known_hosts"]
    assert hosts["source"] == "/runtime-credentials/known_hosts"
    assert hosts.get("bind", {}).get("create_host_path", False) is False
