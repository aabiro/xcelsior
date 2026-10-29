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
import shlex
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
            "runtime_build_hash() { echo R1; }",
            *(
                _function(n)
                for n in (
                    "api_build_hash",
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
    assert _decide(LOCAL) == {
        "api": "false",
        "frontend": "false",
        "nginx": "false",
        "runtime": "false",
    }


@pytest.mark.parametrize("changed", sorted(LOCAL))
def test_only_the_changed_input_is_redone(changed):
    decided = _decide({**LOCAL, changed: "OLD"})
    assert decided == {k: ("true" if k == changed else "false") for k in LOCAL}


def test_a_first_deploy_with_no_record_does_everything():
    assert set(_decide({}).values()) == {"true"}


def _hash(script: str) -> str:
    result = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize(
    "relative", ["control_plane/scheduler/config.py", "uv.lock", "spot/pricing.py"]
)
def test_runtime_module_and_lock_changes_require_a_new_image(tmp_path, relative):
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("before")
    script = (
        f"PROJECT_DIR={shlex.quote(str(tmp_path))}\n"
        + _function("hash_repo_subset")
        + _function("api_build_hash")
        + "api_build_hash"
    )
    before = _hash(script)
    target.write_text("after")
    assert _hash(script) != before


def test_generated_caches_do_not_trigger_a_rebuild(tmp_path):
    source = tmp_path / "control_plane"
    source.mkdir()
    (source / "runtime.py").write_text("source")
    script = (
        f"PROJECT_DIR={shlex.quote(str(tmp_path))}\n"
        + _function("hash_repo_subset")
        + _function("api_build_hash")
        + "api_build_hash"
    )
    before = _hash(script)
    (source / "__pycache__").mkdir()
    (source / "__pycache__/runtime.pyc").write_bytes(b"generated")
    assert _hash(script) == before


def test_preserved_production_env_controls_runtime_hash(tmp_path):
    local, remote = tmp_path / "local.env", tmp_path / "remote.env"
    local.write_text("XCELSIOR_SCHEDULER_MODE=shadow\n")
    remote.write_text(local.read_text())
    script = (
        f"ENV_FILE={shlex.quote(str(local))}\nSKIP_PROD_ENV_PUSH=1\n"
        "hash_repo_subset() { echo compose-hash; }\n"
        f"ssh_cmd() {{ sha256sum {shlex.quote(str(remote))} | cut -d' ' -f1; }}\n"
        + _function("runtime_build_hash")
        + "runtime_build_hash"
    )
    before = _hash(script)
    local.write_text("XCELSIOR_SCHEDULER_MODE=paused\n")
    assert _hash(script) == before, "undeployed developer settings must not drive a rollout"
    remote.write_text("XCELSIOR_SCHEDULER_MODE=canary\n")
    assert _hash(script) != before, "a production-only mode change must restart the services"
