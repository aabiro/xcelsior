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
            command.write_text(f"#!/bin/bash\n{body}\n")
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
    assert "nvidia-smi is not installed" in result.stdout
    assert "Install a supported NVIDIA driver" in result.stdout


def test_install_preflight_fails_when_nvidia_smi_cannot_detect_gpu():
    result = _run_installer_function("check_nvidia", {"nvidia-smi": "exit 1"})
    assert result.returncode != 0
    assert "did not list a usable NVIDIA GPU" in result.stdout
    assert "Seat a supported NVIDIA GPU" in result.stdout


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
    assert "NVIDIA container runtime is not installed" in result.stdout
    assert "Install the NVIDIA Container Toolkit" in result.stdout


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


def _verify_online(hosts_json: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmp:
        env_file = Path(tmp) / "worker.env"
        env_file.write_text("XCELSIOR_HOST_ID=h1\nXCELSIOR_API_TOKEN=xcel_ai_key\n")
        return _run_installer_function(
            f"ENV_FILE={env_file}; sleep() {{ :; }}; verify_host_online",
            {"curl": f"printf '%s\\n' '{hosts_json}'"},
        )


def test_install_status_check_reports_pending_without_claiming_admission():
    # Connected (fresh heartbeat) but not admitted: the provider is done; admission is ours.
    result = _verify_online(
        '{"hosts":[{"host_id":"h1","last_seen":9999999999,"status":"pending","admitted":false}]}'
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "connected (heartbeat received)" in result.stdout
    assert "pending admission review" in result.stdout
    assert "admitted and" not in result.stdout


def test_install_status_check_requires_admitted_flag_for_active_success():
    result = _verify_online(
        '{"hosts":[{"host_id":"h1","last_seen":9999999999,"status":"active","admitted":true}]}'
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "connected, admitted and active" in result.stdout


def test_install_status_check_fails_without_a_fresh_heartbeat():
    result = _verify_online(
        '{"hosts":[{"host_id":"h1","last_seen":1,"status":"active","admitted":true}]}'
    )
    assert result.returncode != 0
    assert "No heartbeat from h1" in result.stdout


UNAME_LINUX = """
case "$1" in
  -s) printf '%s\\n' Linux ;;
  -m) printf '%s\\n' x86_64 ;;
esac
"""

UNAME_MAC = """
case "$1" in
  -s) printf '%s\\n' Darwin ;;
  -m) printf '%s\\n' arm64 ;;
esac
"""

UNAME_PPC = """
case "$1" in
  -s) printf '%s\\n' Linux ;;
  -m) printf '%s\\n' ppc64le ;;
esac
"""


def test_install_messages_match_the_shared_requirement_copy():
    shell = INSTALL_SH.read_text()
    wizard = (ROOT / "wizard" / "src" / "host-requirements.ts").read_text()
    frontend = (ROOT / "frontend" / "src" / "lib" / "host-setup-requirements.ts").read_text()
    assert wizard == frontend
    for sentence in (
        "This machine is macOS (Darwin), so it cannot host an Xcelsior worker yet.",
        "not Linux, so it cannot host an Xcelsior worker yet.",
        "which this worker does not support.",
        "systemd is not available on this machine",
        "python3 is not installed, so the worker cannot run.",
        "curl is not installed, so the installer cannot download the worker.",
        "openssl is not installed, so the installer cannot verify the worker signature.",
        "docker is not installed, so rented GPUs cannot start containers.",
        "nvidia-smi is not installed, so no NVIDIA driver was found.",
        "did not list a usable NVIDIA GPU",
        "Docker is installed but it is not running",
        "NVIDIA container runtime is not installed",
        "No host IP could be detected",
        "Tailscale is optional",
    ):
        assert sentence in shell
        assert sentence in wizard


def test_install_preflight_rejects_macos():
    result = _run_installer_function("check_platform", {"uname": UNAME_MAC})
    assert result.returncode != 0
    assert "macOS (Darwin)" in result.stdout
    assert "Use a Linux host" in result.stdout
    assert "WSL2" in result.stdout


def test_install_preflight_rejects_other_non_linux():
    uname = """
case "$1" in
  -s) printf '%s\\n' FreeBSD ;;
  -m) printf '%s\\n' amd64 ;;
esac
"""
    result = _run_installer_function("check_platform", {"uname": uname})
    assert result.returncode != 0
    assert "FreeBSD" in result.stdout
    assert "not Linux" in result.stdout
    assert "Use a Linux host" in result.stdout


def test_install_preflight_rejects_unsupported_cpu():
    result = _run_installer_function(
        "check_platform",
        {"uname": UNAME_PPC, "systemctl": "exit 0"},
        include_system_commands=False,
    )
    assert result.returncode != 0
    assert "ppc64le" in result.stdout
    assert "amd64 (x86_64) or arm64 (aarch64)" in result.stdout


def test_install_preflight_rejects_missing_systemd():
    result = _run_installer_function(
        "check_platform",
        {"uname": UNAME_LINUX},
        include_system_commands=False,
    )
    assert result.returncode != 0
    assert "systemctl was not found" in result.stdout
    assert "WSL2" in result.stdout


@pytest.mark.parametrize(
    ("missing", "problem", "step"),
    [
        ("python3", "python3 is not installed", "Install Python 3"),
        ("docker", "docker is not installed", "Install Docker Engine"),
        ("curl", "curl is not installed", "Install curl"),
        ("openssl", "openssl is not installed", "Install openssl"),
    ],
)
def test_install_preflight_names_each_missing_command(missing, problem, step):
    commands = {
        "python3": "exit 0",
        "docker": "exit 0",
        "curl": "exit 0",
        "openssl": "exit 0",
    }
    del commands[missing]
    result = _run_installer_function("check_deps", commands, include_system_commands=False)
    assert result.returncode != 0
    assert problem in result.stdout
    assert step in result.stdout


def test_install_preflight_rejects_docker_that_is_not_running():
    result = _run_installer_function(
        "check_docker",
        {"docker": "exit 1"},
        include_system_commands=False,
    )
    assert result.returncode != 0
    assert "not running" in result.stdout
    assert 'usermod -aG docker "$USER"' in result.stdout


def test_install_preflight_rejects_missing_host_ip():
    result = _run_installer_function(
        "detect_network",
        {"curl": "exit 1"},
        include_system_commands=False,
    )
    assert result.returncode != 0
    assert "No host IP could be detected" in result.stdout
    assert "Tailscale is optional" in result.stdout


def test_install_preflight_accepts_linux_systemd_and_supported_cpu():
    result = _run_installer_function(
        "check_platform",
        {"uname": UNAME_LINUX, "systemctl": "exit 0"},
        include_system_commands=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
        "Wants=network-online.target xcelsior-privd.socket xcelsior-svid.timer",
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
    # Squid's messages reach the journal through -d 1 (cache_log is /dev/null).
    assert "squid -N -d 1 -f" in proxy
    # Boot order: the proxy first, then the check, and no cycle between them.
    assert "After=docker.service xcelsior-egress-proxy.service" in provision
    assert "Wants=xcelsior-egress-proxy.service" in provision
    assert "xcelsior-egress.service" not in proxy
    assert "Network protection for rented containers is active" in result.stdout
    # The worker unit is still installed after it.
    assert (units / "xcelsior-worker.service").exists()
