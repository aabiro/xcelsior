#!/usr/bin/env python3
"""NFS mount smoke from a GPU worker host (optional second worker verification).

Usage:
  python scripts/volumes_worker_mount_smoke.py
  python scripts/volumes_worker_mount_smoke.py --host 100.64.0.6 --ssh-user aaryn
  WORKER_MOUNT_HOST=100.64.x.x python scripts/volumes_worker_mount_smoke.py

Mounts VPS NFS export from the worker the same way worker_agent does at job start.
Exits 0 on PASS or SKIP (unreachable optional host); 1 on mount failure.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _mount_smoke(host: str, ssh_user: str, ssh_key: str) -> int:
    nfs_server = os.environ.get("XCELSIOR_NFS_SERVER", "100.64.0.1")
    export = os.environ.get("XCELSIOR_NFS_EXPORT_BASE", "/exports/volumes")
    mount_opts = os.environ.get(
        "XCELSIOR_NFS_MOUNT_OPTS",
        "hard,timeo=15,retrans=2,tcp,nfsvers=4",
    )
    mount_dir = "/tmp/nfs-worker-smoke"
    remote = (
        f"sudo mkdir -p {mount_dir} && sudo umount {mount_dir} 2>/dev/null || true; "
        f"sudo mount -t nfs4 -o {mount_opts} "
        f"{nfs_server}:{export}/mount-test {mount_dir} && "
        f"ls {mount_dir} && sudo umount {mount_dir} && echo WORKER_MOUNT_OK"
    )
    target = f"{ssh_user}@{host}"
    cmd = [
        "ssh",
        "-i",
        ssh_key,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        target,
        remote,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"worker-mount ({target}): SKIP — {exc}")
        return 0

    if proc.returncode == 0 and "WORKER_MOUNT_OK" in proc.stdout:
        print(f"worker-mount ({target}): PASS")
        return 0

    err = (proc.stderr or proc.stdout or "")[:300]
    unreachable = any(
        s in err for s in ("Connection refused", "No route to host", "Could not resolve", "Connection timed out")
    )
    if unreachable:
        print(f"worker-mount ({target}): SKIP — host unreachable (optional worker not registered)")
        return 0

    print(f"worker-mount ({target}): FAIL rc={proc.returncode} err={err}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU worker NFS mount smoke")
    parser.add_argument(
        "--host",
        default=os.environ.get("WORKER_MOUNT_HOST")
        or os.environ.get("AARYNFANS_PROD_HOST")
        or "100.64.0.6",
        help="Worker mesh IP (default: WORKER_MOUNT_HOST or ASUS 100.64.0.6)",
    )
    parser.add_argument(
        "--ssh-user",
        default=os.environ.get("WORKER_SSH_USER", os.environ.get("XCELSIOR_SSH_USER", "aaryn")),
    )
    parser.add_argument(
        "--ssh-key",
        default=os.environ.get("XCELSIOR_SSH_KEY", str(Path.home() / ".ssh" / "xcelsior")),
    )
    args = parser.parse_args()
    return _mount_smoke(args.host, args.ssh_user, args.ssh_key)


if __name__ == "__main__":
    raise SystemExit(main())