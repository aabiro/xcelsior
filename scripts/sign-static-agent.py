#!/usr/bin/env python3
"""Sign worker_agent.py for install.sh fail-closed verification (B6)."""

from __future__ import annotations

from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parent.parent
KEY = Path(__file__).resolve().parent / "agent-signing.key"
SIGNABLE = ("worker_agent.py", "security.py", "nvml_telemetry.py")


def main() -> None:
    if not KEY.exists():
        raise SystemExit(f"Missing signing key: {KEY}")
    key = serialization.load_pem_private_key(KEY.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise SystemExit("agent-signing.key is not Ed25519")
    scripts = Path(__file__).resolve().parent
    for name in SIGNABLE:
        src = ROOT / name
        sig = scripts / f"{name}.sig"
        data = src.read_bytes()
        sig.write_bytes(key.sign(data))
        print(f"Signed {src} -> {sig}")


if __name__ == "__main__":
    main()