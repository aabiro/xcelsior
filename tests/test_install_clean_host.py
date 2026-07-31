"""Clean-host install integration test (X2) — verifies B1 venv + requirements."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = ROOT / "scripts" / "install.sh"
REQUIREMENTS = ROOT / "scripts" / "worker-requirements.txt"


@pytest.mark.skipif(
    not Path("/usr/bin/python3").exists(),
    reason="python3 required",
)
def test_worker_requirements_install_in_fresh_venv():
    """Simulates clean-host pip install of pinned worker deps."""
    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "venv"
        subprocess.run(["python3", "-m", "venv", str(venv)], check=True)
        pip = venv / "bin" / "pip"
        subprocess.run([str(pip), "install", "-r", str(REQUIREMENTS)], check=True, capture_output=True)
        py = venv / "bin" / "python3"
        subprocess.run(
            [str(py), "-c", "import requests, dotenv, prometheus_client, cryptography, psutil"],
            check=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )


def test_install_sh_has_platform_guard_and_npx_fallback():
    text = INSTALL_SH.read_text()
    wizard_version = json.loads((ROOT / "wizard" / "package.json").read_text())["version"]
    assert "check_platform" in text
    assert f'WIZARD_PACKAGE="@xcelsior-gpu/wizard@{wizard_version}"' in text
    assert 'npx --yes "$WIZARD_PACKAGE"' in text
    assert "@latest" not in text
    assert "falling back to direct agent install" in text
    assert "worker-requirements.txt" in text
    assert "verify_agent_signature" in text


def _run_installer_function(
    function: str,
    fake_commands: dict[str, str],
    *,
    include_system_commands: bool = True,
) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmp:
        bin_dir = Path(tmp)
        for name, body in fake_commands.items():
            command = bin_dir / name
            command.write_text(f"#!/usr/bin/env bash\n{body}\n")
            command.chmod(0o755)
        command_path = f"{bin_dir}:/usr/bin:/bin" if include_system_commands else str(bin_dir)
        return subprocess.run(
            ["/bin/bash", "-c", f"source {INSTALL_SH}; {function}"],
            text=True,
            capture_output=True,
            env={**os.environ, "PATH": command_path},
        )


def test_install_preflight_fails_when_nvidia_smi_is_missing():
    result = _run_installer_function(
        "check_nvidia",
        {},
        include_system_commands=False,
    )
    assert result.returncode != 0
    assert "nvidia-smi is required" in result.stdout


def test_install_preflight_fails_when_nvidia_smi_cannot_detect_gpu():
    result = _run_installer_function("check_nvidia", {"nvidia-smi": "exit 1"})
    assert result.returncode != 0
    assert "could not detect a usable NVIDIA GPU" in result.stdout


def test_install_preflight_fails_without_nvidia_docker_runtime():
    docker = """
if [ "$1" = "info" ] && [ "${2:-}" = "--format" ]; then
    printf '%s\\n' '{"runc":{}}'
    exit 0
fi
if [ "$1" = "info" ]; then
    exit 0
fi
exit 1
"""
    result = _run_installer_function("check_docker", {"docker": docker})
    assert result.returncode != 0
    assert "NVIDIA Container Toolkit/runtime is required" in result.stdout


def test_install_preflight_accepts_detected_gpu_and_nvidia_runtime():
    docker = """
if [ "$1" = "info" ] && [ "${2:-}" = "--format" ]; then
    printf '%s\\n' '{"nvidia":{},"runc":{}}'
    exit 0
fi
if [ "$1" = "info" ]; then
    exit 0
fi
exit 1
"""
    result = _run_installer_function(
        "check_nvidia; check_docker",
        {
            "nvidia-smi": "printf '%s\\n' 'NVIDIA RTX 4090'",
            "docker": docker,
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NVIDIA Container Toolkit/runtime detected" in result.stdout


def test_install_status_check_reports_pending_without_claiming_admission():
    with tempfile.TemporaryDirectory() as tmp:
        env_file = Path(tmp) / "worker.env"
        env_file.write_text(
            "XCELSIOR_API=https://example.invalid\n"
            "XCELSIOR_HOST_ID=pending-host\n"
        )
        result = _run_installer_function(
            f"ENV_FILE={env_file}; verify_host_online",
            {
                "curl": (
                    "printf '%s\\n' "
                    "'{\"host\":{\"status\":\"pending\",\"admitted\":false}}'"
                ),
            },
        )

    assert result.returncode != 0
    assert "pending authoritative verification" in result.stdout
    assert "not listed or eligible for work" in result.stdout
    assert "authoritatively admitted and active" not in result.stdout


def test_install_status_check_requires_admitted_flag_for_active_success():
    with tempfile.TemporaryDirectory() as tmp:
        env_file = Path(tmp) / "worker.env"
        env_file.write_text(
            "XCELSIOR_API=https://example.invalid\n"
            "XCELSIOR_HOST_ID=verified-host\n"
        )
        result = _run_installer_function(
            f"ENV_FILE={env_file}; verify_host_online",
            {
                "curl": (
                    "printf '%s\\n' "
                    "'{\"host\":{\"status\":\"active\",\"admitted\":true}}'"
                ),
            },
        )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "authoritatively admitted and active" in result.stdout
