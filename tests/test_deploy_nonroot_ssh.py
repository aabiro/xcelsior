"""Deployment keeps the SSH login non-root and elevates the complete operation."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[1] / "api/scripts/deploy.sh"


def _function(name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", DEPLOY.read_text(), re.M | re.S)
    assert match
    return match.group(0)


@pytest.mark.parametrize("user", ["linuxuser", "root"])
@pytest.mark.parametrize("stdin_script", [False, True])
def test_remote_shell_preserves_quoting_and_stdin(tmp_path, user, stdin_script):
    trace = tmp_path / "sudo-used"
    literal = 'quotes " $HOME $(echo unexpected) `echo unexpected`\nsecond line'
    payload = "printf '%s' " + shlex.quote(literal)
    script = "\n".join(
        [
            "set -euo pipefail",
            f"REMOTE_USER={user}",
            "SSH_BASE_OPTS=()",
            _function("_remote_target"),
            _function("ssh_cmd_host"),
            # Execute the command exactly as an SSH shell would. No network or
            # actual privilege elevation is needed for this shell contract.
            'ssh() { test "$1" = "$REMOTE_USER@test-host"; shift; if [[ $# -eq 0 ]]; then bash -s; else bash -c "$*"; fi; }',
            'sudo() { test "$1" = -n; shift; printf used > "$SUDO_TRACE"; "$@"; }',
            "export -f sudo",
            'ssh_cmd_host test-host' + ('' if stdin_script else ' "$REMOTE_PAYLOAD"'),
        ]
    )
    result = subprocess.run(
        ["bash", "-c", script],
        input=payload if stdin_script else "",
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "SUDO_TRACE": str(trace), "REMOTE_PAYLOAD": payload},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == literal
    assert trace.exists() == (user != "root")


@pytest.mark.parametrize("user", ["linuxuser", "root"])
def test_rsync_elevates_the_receiver_without_changing_the_login(user):
    script = "\n".join(
        [
            "set -euo pipefail",
            f"REMOTE_USER={user}",
            "RSYNC_BASE_OPTS=()",
            "RSYNC_COMPRESS_OPTS=()",
            _function("_remote_target"),
            _function("rsync_to_host"),
            "_rsync_shell_for_host() { printf ssh; }",
            "rsync() { printf '%s\\n' \"$@\"; }",
            "rsync_to_host test-host 'source dir/' /opt/xcelsior_new/ --delete",
        ]
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    receiver = "rsync" if user == "root" else "sudo -n rsync"
    assert f"--rsync-path={receiver}" in args
    assert args[-2:] == ["source dir/", f"{user}@test-host:/opt/xcelsior_new/"]
