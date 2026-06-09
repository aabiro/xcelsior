"""Clean-host install integration test (X2) — verifies B1 venv + requirements."""

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
    assert "check_platform" in text
    assert "npx @xcelsior-gpu/wizard@latest" in text
    assert "falling back to direct agent install" in text
    assert "worker-requirements.txt" in text
    assert "verify_agent_signature" in text