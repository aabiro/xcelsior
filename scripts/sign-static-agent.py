#!/usr/bin/env python3
"""Sign worker distribution files, or verify using only the public key."""

from __future__ import annotations

from pathlib import Path
import argparse

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

ROOT = Path(__file__).resolve().parent.parent
KEY = Path(__file__).resolve().parent / "agent-signing.key"
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
    "worker-requirements.txt",
)


def source(name: str) -> Path:
    return ROOT / ("scripts" if name == "worker-requirements.txt" else "") / name


def verify() -> None:
    public = serialization.load_pem_public_key((ROOT / "scripts/agent-signing.pub").read_bytes())
    if not isinstance(public, Ed25519PublicKey):
        raise SystemExit("agent-signing.pub is not Ed25519")
    for name in SIGNABLE:
        public.verify((ROOT / "scripts" / f"{name}.sig").read_bytes(), source(name).read_bytes())
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
    scripts = ROOT / "scripts"
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
