#!/usr/bin/env python3
"""Sign worker distribution files, or verify using only the public key."""

from __future__ import annotations

from pathlib import Path
import argparse

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent if _HERE.parent.name == "api" else _HERE.parent
KEY = _HERE / "agent-signing.key"
SIGNABLE = (
    "worker_agent.py",
    "security.py",
    "nvml_telemetry.py",
    "worker_image_cache.py",
    "worker_nfs.py",
    "worker_nvme_cache.py",
    "worker_luks_volumes.py",
    "env_config.py",
    "criu_hosts.py",
    "egress_isolation.py",
    "worker_privileged.py",
    "hardware_isolation.py",
    "worker-requirements.txt",
)

_PKG_FOR_FILE = {
    "worker_agent.py": "workers",
    "security.py": "auth",
    "nvml_telemetry.py": "workers",
    "worker_image_cache.py": "workers",
    "worker_nfs.py": "workers",
    "worker_nvme_cache.py": "workers",
    "worker_luks_volumes.py": "workers",
    "env_config.py": "core",
    "criu_hosts.py": "workers",
    "egress_isolation.py": "hosts",
    "worker_privileged.py": "workers",
    "hardware_isolation.py": "workers",
}


def _scripts_dir() -> Path:
    return (ROOT / "api" / "scripts") if (ROOT / "api" / "scripts").is_dir() else (ROOT / "scripts")


def source(name: str) -> Path:
    if name == "worker-requirements.txt":
        return _scripts_dir() / name
    pkg = _PKG_FOR_FILE.get(name, "")
    if pkg and (ROOT / pkg / name).exists():
        return ROOT / pkg / name
    return ROOT / name


def verify() -> None:
    scripts = _scripts_dir()
    public = serialization.load_pem_public_key((scripts / "agent-signing.pub").read_bytes())
    if not isinstance(public, Ed25519PublicKey):
        raise SystemExit("agent-signing.pub is not Ed25519")
    for name in SIGNABLE:
        public.verify((scripts / f"{name}.sig").read_bytes(), source(name).read_bytes())
    print(f"Verified {len(SIGNABLE)} worker distribution signatures")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    if parser.parse_args().verify:
        verify()
        return
    if not KEY.exists():
        raise SystemExit(f"Missing signing key: {KEY}")
    key = serialization.load_pem_private_key(KEY.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise SystemExit("agent-signing.key is not Ed25519")
    scripts = _scripts_dir()
    expected = serialization.load_pem_public_key((scripts / "agent-signing.pub").read_bytes())
    if key.public_key().public_bytes_raw() != expected.public_bytes_raw():
        raise SystemExit("Signing key does not match the published worker trust root")
    for name in SIGNABLE:
        src = source(name)
        sig = scripts / f"{name}.sig"
        data = src.read_bytes()
        sig.write_bytes(key.sign(data))
        print(f"Signed {src} -> {sig}")


if __name__ == "__main__":
    main()
