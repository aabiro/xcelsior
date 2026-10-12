"""Clean-host install integration test (X2) — verifies B1 venv + requirements."""

import json
import base64
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
        subprocess.run(
            [str(pip), "install", "-r", str(REQUIREMENTS)], check=True, capture_output=True
        )
        py = venv / "bin" / "python3"
        subprocess.run(
            [
                str(py),
                "-c",
                "import requests, dotenv, prometheus_client, cryptography, psutil, pynvml",
            ],
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


@pytest.mark.parametrize("tamper", [False, True])
def test_installer_verifies_ed25519_bytes(tmp_path, tamper):
    """Exercise the actual OpenSSL verifier, including rejection of tampering."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    public = tmp_path / "public.pem"
    public.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    data = b"signed worker release\n"
    payload = tmp_path / "worker.py"
    payload.write_bytes(data + (b"modified\n" if tamper else b""))
    signature = base64.b64encode(key.sign(data)).decode()
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; AGENT_SIGNING_PUB_EMBEDDED="$(cat "$2")"; '
            'verify_agent_signature "$3" "$4"',
            "verify",
            str(INSTALL_SH),
            str(public),
            str(payload),
            signature,
        ],
        text=True,
        capture_output=True,
    )
    assert (result.returncode == 0) is not tamper, result.stdout + result.stderr


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
            "XCELSIOR_API=https://example.invalid\n" "XCELSIOR_HOST_ID=pending-host\n"
        )
        result = _run_installer_function(
            f"ENV_FILE={env_file}; verify_host_online",
            {
                "curl": ("printf '%s\\n' " '\'{"host":{"status":"pending","admitted":false}}\''),
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
            "XCELSIOR_API=https://example.invalid\n" "XCELSIOR_HOST_ID=verified-host\n"
        )
        result = _run_installer_function(
            f"ENV_FILE={env_file}; verify_host_online",
            {
                "curl": ("printf '%s\\n' " '\'{"host":{"status":"active","admitted":true}}\''),
            },
        )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "authoritatively admitted and active" in result.stdout


def test_installer_runs_main_when_piped():
    # `curl -fsSL .../install.sh | bash -s -- --help`: BASH_SOURCE[0] is empty there.
    result = subprocess.run(
        ["/bin/bash", "-s", "--", "--help"],
        input=INSTALL_SH.read_text(),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout


def test_installer_does_not_run_main_when_sourced():
    # The wizard sources install.sh for its functions.
    result = subprocess.run(
        ["/bin/bash", "-c", f"source {INSTALL_SH}; echo sourced", "xcelsior-worker-install"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "sourced"


def test_systemd_unit_uses_state_directory(tmp_path):
    unit = tmp_path / "xcelsior-worker.service"
    result = _run_installer_function(
        "install_systemd",
        {
            # `sudo install -m 0644 <tmp> /etc/systemd/system/xcelsior-worker.service`;
            # install_systemd also installs other files (egress units) first.
            "sudo": (
                'if [ "$1" = install ]; then case "${@: -1}" in '
                f'*/xcelsior-worker.service) cp "${{@: -2:1}}" {unit};; esac; fi'
            ),
            "systemctl": "exit 0",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    text = unit.read_text()
    assert "StateDirectory=xcelsior\n" in text
    # A missing ReadWritePaths entry fails the first start with 226/NAMESPACE.
    rw = next(line for line in text.splitlines() if line.startswith("ReadWritePaths="))
    assert "/var/lib/xcelsior" not in rw
    # Issue #65: a dedicated user, no access to anyone's home, root steps via xcelsior-privd.
    for line in (
        "User=xcelsior",
        "Group=xcelsior",
        "NoNewPrivileges=true",
        "ProtectHome=true",
        "RuntimeDirectory=xcelsior",
        "ExecStart=/opt/xcelsior/venv/bin/python3 /opt/xcelsior/worker_agent.py",
        "Wants=network-online.target xcelsior-privd.socket",
    ):
        assert line + "\n" in text, line


def test_privileged_helper_units_and_root_config(tmp_path):
    home = tmp_path / "home"
    (home / ".xcelsior").mkdir(parents=True)
    (home / ".xcelsior" / "worker.env").write_text(
        "XCELSIOR_NFS_SERVER=100.64.0.1\nXCELSIOR_NFS_EXPORT_BASE=/srv/nfs/volumes\n"
    )
    out = tmp_path / "out"
    out.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # sudo: copy installed files into out/, capture tee'd config, ignore the rest.
    (bin_dir / "sudo").write_text(f"""#!/usr/bin/env bash
case "$1" in
  install) [ "$2" = -d ] && exit 0; src="${{@: -2:1}}"; dst="${{@: -1}}"; cp "$src" "{out}/$(basename "$dst")" ;;
  test) exit 1 ;;
  tee) cat > "{out}/$(basename "$2")" ;;
  sed) shift; sed "${{@:1:$#-1}}" "{out}/$(basename "${{@: -1}}")" ;;
esac
exit 0
""")
    (bin_dir / "sudo").chmod(0o755)
    helper = ROOT / "worker_privileged.py"
    result = subprocess.run(
        ["/bin/bash", "-c", f"source {INSTALL_SH}; install_privd {helper}"],
        text=True,
        capture_output=True,
        env={**os.environ, "HOME": str(home), "PATH": f"{bin_dir}:/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    socket_unit = (out / "xcelsior-privd.socket").read_text()
    for line in (
        "ListenStream=/run/xcelsior-privd.sock",
        "SocketGroup=xcelsior",
        "SocketMode=0660",
        "Accept=yes",
    ):
        assert line in socket_unit
    service = (out / "xcelsior-privd@.service").read_text()
    assert "ExecStart=/usr/bin/python3 /usr/local/libexec/xcelsior/worker_privileged.py" in service
    assert "StandardInput=socket" in service and "ProtectSystem" not in service
    assert (out / "worker_privileged.py").read_bytes() == helper.read_bytes()
    conf = (out / "privileged.conf").read_text()
    assert "NFS_SERVERS=100.64.0.1\n" in conf
    assert "NFS_EXPORT_ROOTS=/exports,/srv/nfs/volumes\n" in conf
    assert "VOLUME_DIR=\n" not in conf  # empty values are dropped so the helper's defaults apply


def test_rented_container_protection_is_installed_with_the_worker(tmp_path):
    """install_systemd (wizard, manual and --update paths) sets up egress isolation."""
    calls = tmp_path / "calls.log"
    units = tmp_path / "units"
    units.mkdir()
    result = _run_installer_function(
        "install_systemd",
        {
            "sudo": (
                f'echo "$*" >> {calls}; '
                'if [ "$1" = install ]; then case "${@: -1}" in '
                f'/etc/systemd/system/*) cp "${{@: -2:1}}" {units}/"$(basename "${{@: -1}}")";; esac; fi'
            ),
            "systemctl": "exit 0",
            "squid": "exit 0",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    log = calls.read_text()
    # The packaged proxy listens everywhere; ours binds only the tenant gateway.
    assert "systemctl mask squid.service" in log
    # Root runs the provisioner from a root-owned copy, never the worker's lib dir.
    assert "install -m 0644 -o root -g root" in log
    assert "/usr/local/lib/xcelsior/egress_isolation.py" in log
    provision = (units / "xcelsior-egress.service").read_text()
    assert (
        "ExecStart=/usr/bin/python3 /usr/local/lib/xcelsior/egress_isolation.py provision"
        in provision
    )
    assert "OnUnitInactiveSec=2min" in (units / "xcelsior-egress.timer").read_text()
    proxy = (units / "xcelsior-egress-proxy.service").read_text()
    assert "User=proxy" in proxy and "/etc/xcelsior/egress/squid.conf" in proxy
    assert "Network protection for rented containers is active" in result.stdout
    # The worker unit is still installed after it.
    assert (units / "xcelsior-worker.service").exists()
