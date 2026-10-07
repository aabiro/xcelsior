"""A deploy rebuilds and reinstalls only what changed since the last one.

`detect_deploy_inputs` compares local hashes of the API, frontend, nginx and
runtime inputs with the ones the previous deploy recorded on the host. The
remote side was read by a function started with `&`, which filled its array
in a background subshell that then exited. Every previous hash read as empty,
so every deploy logged `api_build=true frontend_build=true nginx=true
runtime=true` (all of them did on 2026-10-06): it rebuilt both images and
reinstalled nginx whatever changed, and "nothing to deploy" never fired.

This runs the real function with SSH and hashing stubbed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "scripts" / "deploy.sh"


def _function(name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", DEPLOY.read_text(), re.M | re.S)
    assert match, f"{name} not found in deploy.sh"
    return match.group(0)


LOCAL = {"api": "A1", "frontend": "F1", "nginx": "N1", "runtime": "R1"}


def _decide(remote: dict[str, str]) -> dict[str, str]:
    remote_lines = "".join(f"{k}|{v}\\n" for k, v in remote.items())
    script = "\n".join(
        [
            "set -u",
            f"PROJECT_DIR={ROOT}",
            f"ENV_FILE={ROOT}/.env",
            "log() { :; }",
            "_deploy_mark() { :; }",
            f"ssh_cmd() {{ printf '{remote_lines}'; }}",
            # The api, nginx and runtime subsets are told apart by their first argument.
            'hash_repo_subset() { case "$1" in .dockerignore) echo A1 ;; nginx) echo N1 ;; *) echo R1 ;; esac; }',
            "frontend_build_hash() { echo F1; }",
            *(
                _function(n)
                for n in (
                    "remote_deploy_meta_dir",
                    "fetch_remote_deploy_hashes",
                    "parse_remote_deploy_hashes",
                    "load_remote_deploy_hash",
                    "detect_deploy_inputs",
                )
            ),
            "detect_deploy_inputs",
            'echo "api=$DEPLOY_BUILD_API frontend=$DEPLOY_BUILD_FRONTEND nginx=$DEPLOY_INSTALL_NGINX runtime=$DEPLOY_RUNTIME_CHANGED"',
        ]
    )
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return dict(pair.split("=") for pair in r.stdout.split())


def test_nothing_changed_means_nothing_is_rebuilt():
    assert _decide(LOCAL) == {"api": "false", "frontend": "false", "nginx": "false", "runtime": "false"}


@pytest.mark.parametrize("changed", sorted(LOCAL))
def test_only_the_changed_input_is_redone(changed):
    decided = _decide({**LOCAL, changed: "OLD"})
    assert decided == {k: ("true" if k == changed else "false") for k in LOCAL}


def test_a_first_deploy_with_no_record_does_everything():
    assert set(_decide({}).values()) == {"true"}
