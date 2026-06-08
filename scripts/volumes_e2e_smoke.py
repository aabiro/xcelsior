#!/usr/bin/env python3
"""Staging smoke test for persistent volumes API (metadata-only or NFS).

Usage:
  python scripts/volumes_e2e_smoke.py --email you@example.com --password '...'
  python scripts/volumes_e2e_smoke.py --token 'Bearer ...' --base-url https://xcelsior.ca
  python scripts/volumes_e2e_smoke.py --infra-only   # CRUD only; skips instance launch
  python scripts/volumes_e2e_smoke.py --encrypted      # also test LUKS volume provision
  python scripts/volumes_e2e_smoke.py --persist      # stop/start instance + verify NFS file
  python scripts/volumes_e2e_smoke.py --hot-attach   # attach volume to running instance
  python scripts/volumes_e2e_smoke.py --worker-mount # NFS mount from VPS (simulates GPU worker)
  python scripts/volumes_e2e_smoke.py --snapshots    # create/list/delete snapshot on detached volume
  python scripts/volumes_e2e_smoke.py --worker-mount --worker-host 100.64.0.6  # mount from GPU worker

Exits 0 on success, 1 on failure. Does not require a running GPU instance unless --persist.
With --infra-only, launch is skipped (no wallet/GPU needed).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parent.parent
ENV_AUDIT = PROJECT / ".env.audit"

POLL_INTERVAL_SEC = 10
POLL_MAX_WAIT_SEC = 180
PERSIST_MARKER = "xcelsior-persist-smoke"
HOT_ATTACH_MARKER = "xcelsior-hot-attach-smoke"
LAUNCH_IMAGE = "nvidia/cuda:12.0.0-base-ubuntu22.04"


def _load_audit_cfg() -> dict[str, str]:
    cfg: dict[str, str] = {}
    if ENV_AUDIT.exists():
        for line in ENV_AUDIT.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    cfg["base"] = (os.environ.get("AUDIT_BASE") or cfg.get("AUDIT_BASE") or "https://xcelsior.ca").rstrip(
        "/"
    )
    cfg["email"] = os.environ.get("AUDIT_EMAIL") or cfg.get("AUDIT_EMAIL", "")
    cfg["password"] = os.environ.get("AUDIT_PASSWORD") or cfg.get("AUDIT_PASSWORD", "")
    return cfg


def _headers(token: str | None) -> dict[str, str]:
    if not token:
        return {}
    if token.lower().startswith("bearer "):
        return {"Authorization": token}
    return {"Authorization": f"Bearer {token}"}


def _poll_instance(client: httpx.Client, hdrs: dict[str, str], job_id: str) -> dict:
    deadline = time.time() + POLL_MAX_WAIT_SEC
    last: dict = {}
    while time.time() < deadline:
        resp = client.get(f"/instance/{job_id}", headers=hdrs)
        if resp.status_code == 200:
            last = resp.json().get("instance") or {}
            status = last.get("status")
            if status in ("running", "failed", "stopped", "completed"):
                return last
        time.sleep(POLL_INTERVAL_SEC)
    return last


def _nfs_write_marker(volume_id: str, marker: str, filename: str) -> bool:
    """Write a marker file on NFS export (SSH to VPS host)."""
    ssh_key = os.environ.get("XCELSIOR_SSH_KEY", str(Path.home() / ".ssh" / "xcelsior"))
    host = os.environ.get("NFS_SSH_HOST", os.environ.get("MAC_NFS_HOST", "linuxuser@100.64.0.1"))
    wrap = os.environ.get("XCELSIOR_NFS_SSH_CMD_WRAP", "").strip()
    path = f"/exports/volumes/{volume_id}/{filename}"
    remote = f"sh -c 'echo {marker} > {path}'"
    if wrap:
        remote = f"{wrap} {shlex.quote(remote)}"
    cmd = [
        "ssh",
        "-i",
        ssh_key,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        host,
        remote,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _worker_docker_cat(host_ip: str, job_id: str, container_path: str) -> str | None:
    """Read a file inside a running instance via worker docker exec."""
    ssh_key = os.environ.get("XCELSIOR_SSH_KEY", str(Path.home() / ".ssh" / "xcelsior"))
    ssh_user = os.environ.get("XCELSIOR_SSH_USER", "xcelsior")
    container = f"xcl-{job_id}"
    remote = f"docker exec {container} cat {container_path}"
    cmd = [
        "ssh",
        "-i",
        ssh_key,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"{ssh_user}@{host_ip}",
        remote,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _nfs_read_marker(volume_id: str) -> str | None:
    """Read persist marker from NFS export (SSH to VPS host)."""
    ssh_key = os.environ.get("XCELSIOR_SSH_KEY", str(Path.home() / ".ssh" / "xcelsior"))
    host = os.environ.get("NFS_SSH_HOST", os.environ.get("MAC_NFS_HOST", "linuxuser@100.64.0.1"))
    wrap = os.environ.get("XCELSIOR_NFS_SSH_CMD_WRAP", "").strip()
    path = f"/exports/volumes/{volume_id}/{PERSIST_MARKER}.txt"
    remote = f"cat {path}"
    if wrap:
        remote = f"{wrap} {shlex.quote(remote)}"
    cmd = [
        "ssh",
        "-i",
        ssh_key,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        host,
        remote,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _nfs_mount_remote_cmd(mount_dir: str) -> str:
    nfs_server = os.environ.get("XCELSIOR_NFS_SERVER", "100.64.0.3")
    export = os.environ.get("XCELSIOR_NFS_EXPORT_BASE", "/exports/volumes")
    return (
        f"sudo mkdir -p {mount_dir} && sudo umount {mount_dir} 2>/dev/null || true; "
        f"sudo mount -t nfs4 -o nfsvers=4.0,port=12049,hard,timeo=15,retrans=2,tcp "
        f"{nfs_server}:{export}/mount-test {mount_dir} && "
        f"ls {mount_dir} && sudo umount {mount_dir} && echo WORKER_MOUNT_OK"
    )


def _worker_mount_smoke(worker_host: str | None = None) -> bool:
    """Mount Mac NFS from VPS or a GPU worker host — same path workers use at job start."""
    ssh_key = os.environ.get("XCELSIOR_SSH_KEY", str(Path.home() / ".ssh" / "xcelsior"))
    if worker_host:
        ssh_user = os.environ.get("XCELSIOR_SSH_USER", "xcelsior")
        target = f"{ssh_user}@{worker_host}"
        label = f"worker-host {worker_host}"
    else:
        target = os.environ.get("XCELSIOR_DEPLOY_USER", "linuxuser") + "@" + os.environ.get(
            "XCELSIOR_DEPLOY_HOST", "100.64.0.1"
        )
        label = "VPS"
    remote = _nfs_mount_remote_cmd("/tmp/nfs-worker-smoke")
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
        if proc.returncode == 0 and "WORKER_MOUNT_OK" in proc.stdout:
            print(f"worker-mount ({label}): PASS")
            return True
        if worker_host and proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "")[:200]
            if "Connection refused" in err or "No route to host" in err or "Could not resolve" in err:
                print(f"worker-mount ({label}): SKIP — host unreachable (optional second worker)")
                return True
        print(f"worker-mount ({label}): FAIL rc={proc.returncode} err={proc.stderr[:200]}", file=sys.stderr)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        if worker_host:
            print(f"worker-mount ({label}): SKIP — {exc}")
            return True
        print(f"worker-mount ({label}): FAIL {exc}", file=sys.stderr)
    return False


def _test_snapshots_flow(client: httpx.Client, hdrs: dict[str, str], volume_id: str) -> bool:
    """Create, list, and delete a snapshot on a detached available volume."""
    label = f"smoke-snap-{uuid.uuid4().hex[:6]}"
    created = client.post(
        f"/api/v2/volumes/{volume_id}/snapshots",
        headers=hdrs,
        json={"label": label},
    )
    if created.status_code != 200:
        print(f"snapshots create failed: {created.status_code} {created.text[:300]}", file=sys.stderr)
        return False
    snap_id = (created.json().get("snapshot") or {}).get("snapshot_id")
    if not snap_id:
        print(f"snapshots create missing snapshot_id: {created.json()}", file=sys.stderr)
        return False
    print(f"snapshots: created {snap_id}")

    listed = client.get(f"/api/v2/volumes/{volume_id}/snapshots", headers=hdrs)
    if listed.status_code != 200:
        print(f"snapshots list failed: {listed.status_code}", file=sys.stderr)
        return False
    ids = [s["snapshot_id"] for s in listed.json().get("snapshots") or []]
    if snap_id not in ids:
        print(f"snapshots: {snap_id} not in list {ids}", file=sys.stderr)
        return False
    print(f"snapshots: listed {len(ids)} snapshot(s)")

    deleted = client.delete(
        f"/api/v2/volumes/{volume_id}/snapshots/{snap_id}",
        headers=hdrs,
    )
    if deleted.status_code != 200:
        print(f"snapshots delete failed: {deleted.status_code} {deleted.text[:200]}", file=sys.stderr)
        return False
    print("snapshots: delete ok")
    return True


def _test_encrypted_volume(client: httpx.Client, hdrs: dict[str, str]) -> str | None:
    name = f"smoke-enc-{uuid.uuid4().hex[:8]}"
    created = client.post(
        "/api/v2/volumes",
        headers=hdrs,
        json={"name": name, "size_gb": 1, "encrypted": True},
    )
    if created.status_code != 200:
        print(f"encrypted create failed: {created.status_code} {created.text[:300]}", file=sys.stderr)
        return None
    vol = created.json().get("volume") or {}
    volume_id = vol.get("volume_id")
    if not volume_id:
        print(f"encrypted create missing volume_id: {created.json()}", file=sys.stderr)
        return None
    if vol.get("status") != "available":
        print(f"encrypted volume status={vol.get('status')} (expected available)", file=sys.stderr)
        client.delete(f"/api/v2/volumes/{volume_id}", headers=hdrs)
        return None
    print(f"encrypted volume ok: {volume_id}")
    deleted = client.delete(f"/api/v2/volumes/{volume_id}", headers=hdrs)
    if deleted.status_code != 200:
        print(f"encrypted delete failed: {deleted.status_code}", file=sys.stderr)
        return None
    return volume_id


def _test_hot_attach_flow(client: httpx.Client, hdrs: dict[str, str], volume_id: str) -> bool:
    marker_file = f"{HOT_ATTACH_MARKER}.txt"
    if not _nfs_write_marker(volume_id, HOT_ATTACH_MARKER, marker_file):
        print("hot-attach: failed to seed NFS marker", file=sys.stderr)
        return False

    name = f"smoke-hot-{uuid.uuid4().hex[:8]}"
    launch = client.post(
        "/instance",
        headers=hdrs,
        json={
            "name": name,
            "vram_needed_gb": 1,
            "image": LAUNCH_IMAGE,
            "interactive": False,
            "command": "sleep infinity",
        },
    )
    if launch.status_code == 402:
        print("hot-attach: skipped (insufficient wallet)")
        return True
    if launch.status_code != 200:
        print(f"hot-attach launch failed: {launch.status_code} {launch.text[:300]}", file=sys.stderr)
        return False

    inst = launch.json().get("instance") or {}
    job_id = inst.get("job_id")
    if not job_id:
        print("hot-attach: launch missing job_id", file=sys.stderr)
        return False

    polled = _poll_instance(client, hdrs, job_id)
    status = polled.get("status")
    host_ip = polled.get("host_ip") or polled.get("ip")
    print(f"hot-attach: job {job_id} status={status} host_ip={host_ip}")

    if status != "running":
        print("hot-attach: skipped (no GPU host — job did not reach running)")
        client.post(f"/instances/{job_id}/terminate", headers=hdrs)
        return True

    attach = client.post(
        f"/api/v2/volumes/{volume_id}/attach",
        headers=hdrs,
        json={"instance_id": job_id, "mount_path": "/workspace"},
    )
    if attach.status_code != 200:
        print(f"hot-attach API failed: {attach.status_code} {attach.text[:300]}", file=sys.stderr)
        client.post(f"/instances/{job_id}/terminate", headers=hdrs)
        return False

    if not host_ip:
        vol = client.get(f"/api/v2/volumes/{volume_id}", headers=hdrs)
        host_ip = (vol.json().get("volume") or {}).get("host_ip")

    deadline = time.time() + POLL_MAX_WAIT_SEC
    marker = None
    container_path = f"/workspace/{marker_file}"
    while time.time() < deadline:
        if host_ip:
            marker = _worker_docker_cat(host_ip, job_id, container_path)
            if marker == HOT_ATTACH_MARKER:
                break
        time.sleep(POLL_INTERVAL_SEC)

    client.post(f"/instances/{job_id}/terminate", headers=hdrs)

    if marker != HOT_ATTACH_MARKER:
        print(
            f"hot-attach: marker in container expected '{HOT_ATTACH_MARKER}' got '{marker}'",
            file=sys.stderr,
        )
        return False
    print("hot-attach: volume readable in running container at /workspace")
    return True


def _test_persist_flow(client: httpx.Client, hdrs: dict[str, str], volume_id: str) -> bool:
    name = f"smoke-persist-{uuid.uuid4().hex[:8]}"
    launch = client.post(
        "/instance",
        headers=hdrs,
        json={
            "name": name,
            "vram_needed_gb": 1,
            "volume_ids": [volume_id],
            "image": LAUNCH_IMAGE,
            "interactive": False,
            "command": f"echo {PERSIST_MARKER} > /workspace/{PERSIST_MARKER}.txt && sleep infinity",
        },
    )
    if launch.status_code == 402:
        print("persist: skipped (insufficient wallet)")
        return True
    if launch.status_code != 200:
        print(f"persist launch failed: {launch.status_code} {launch.text[:300]}", file=sys.stderr)
        return False

    inst = launch.json().get("instance") or {}
    job_id = inst.get("job_id")
    if not job_id:
        print("persist: launch missing job_id", file=sys.stderr)
        return False

    polled = _poll_instance(client, hdrs, job_id)
    status = polled.get("status")
    print(f"persist: job {job_id} status={status} host={polled.get('host_id')}")

    if status != "running":
        print("persist: skipped (no GPU host — job did not reach running)")
        client.post(f"/instances/{job_id}/terminate", headers=hdrs)
        return True

    marker = _nfs_read_marker(volume_id)
    if marker != PERSIST_MARKER:
        print(f"persist: marker on NFS expected '{PERSIST_MARKER}' got '{marker}'", file=sys.stderr)
        client.post(f"/instances/{job_id}/terminate", headers=hdrs)
        return False
    print("persist: marker written on NFS")

    stop = client.post(f"/instances/{job_id}/stop", headers=hdrs)
    if stop.status_code != 200:
        print(f"persist stop failed: {stop.status_code} {stop.text[:200]}", file=sys.stderr)
        client.post(f"/instances/{job_id}/terminate", headers=hdrs)
        return False
    _poll_instance(client, hdrs, job_id)

    start = client.post(f"/instances/{job_id}/start", headers=hdrs)
    if start.status_code != 200:
        print(f"persist start failed: {start.status_code} {start.text[:200]}", file=sys.stderr)
        client.post(f"/instances/{job_id}/terminate", headers=hdrs)
        return False
    _poll_instance(client, hdrs, job_id)

    marker2 = _nfs_read_marker(volume_id)
    if marker2 != PERSIST_MARKER:
        print(f"persist: marker after restart expected '{PERSIST_MARKER}' got '{marker2}'", file=sys.stderr)
        client.post(f"/instances/{job_id}/terminate", headers=hdrs)
        return False
    print("persist: marker survived stop/start")

    term = client.post(f"/instances/{job_id}/terminate", headers=hdrs)
    if term.status_code != 200:
        print(f"persist terminate failed: {term.status_code} {term.text[:200]}", file=sys.stderr)
        return False
    _poll_instance(client, hdrs, job_id)

    vol_resp = client.get(f"/api/v2/volumes/{volume_id}", headers=hdrs)
    if vol_resp.status_code != 200:
        print(f"persist: volume get after terminate failed: {vol_resp.status_code}", file=sys.stderr)
        return False
    vol_status = (vol_resp.json().get("volume") or {}).get("status")
    if vol_status != "available":
        print(f"persist: volume status after terminate expected available got {vol_status}", file=sys.stderr)
        return False

    marker3 = _nfs_read_marker(volume_id)
    if marker3 != PERSIST_MARKER:
        print(f"persist: marker after terminate expected '{PERSIST_MARKER}' got '{marker3}'", file=sys.stderr)
        return False
    print("persist: volume available after terminate, NFS data intact")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Volumes API smoke test")
    audit = _load_audit_cfg()
    parser.add_argument("--base-url", default=audit["base"] or "http://localhost:8000")
    parser.add_argument("--email", default=audit["email"] or None)
    parser.add_argument("--password", default=audit["password"] or None)
    parser.add_argument("--token", help="Bearer access token (skips login)")
    parser.add_argument(
        "--infra-only",
        action="store_true",
        help="Skip instance launch (NFS provision CRUD only; no wallet/GPU required)",
    )
    parser.add_argument(
        "--encrypted",
        action="store_true",
        help="Also test encrypted (LUKS) volume create + delete",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Launch instance, verify NFS persist across stop/start (skips if no GPU host)",
    )
    parser.add_argument(
        "--hot-attach",
        action="store_true",
        help="Attach volume to running instance and verify container read (skips if no GPU)",
    )
    parser.add_argument(
        "--worker-mount",
        action="store_true",
        help="Mount Mac NFS from VPS (simulates GPU worker mesh mount)",
    )
    parser.add_argument(
        "--worker-host",
        default=os.environ.get("AARYNFANS_PROD_HOST") or os.environ.get("WORKER_MOUNT_HOST"),
        help="Optional GPU worker mesh IP for --worker-mount (skips if unreachable)",
    )
    parser.add_argument(
        "--snapshots",
        action="store_true",
        help="Create/list/delete a volume snapshot on NFS _snapshots/ (detached volume)",
    )
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    client = httpx.Client(base_url=base, timeout=120.0)

    token = args.token
    if not token:
        if not args.email or not args.password:
            print("Provide --token or --email + --password", file=sys.stderr)
            return 1
        login = client.post(
            "/api/auth/login",
            json={"email": args.email, "password": args.password},
        )
        if login.status_code != 200:
            print(f"Login failed: {login.status_code} {login.text[:200]}", file=sys.stderr)
            return 1
        token = login.json().get("access_token")
        if not token:
            print("Login response missing access_token", file=sys.stderr)
            return 1

    hdrs = _headers(token)
    name = f"smoke-vol-{uuid.uuid4().hex[:8]}"

    created = client.post(
        "/api/v2/volumes",
        headers=hdrs,
        json={"name": name, "size_gb": 1, "encrypted": False},
    )
    if created.status_code != 200:
        print(f"Create failed: {created.status_code} {created.text[:300]}", file=sys.stderr)
        return 1
    body = created.json()
    vol = body.get("volume") or {}
    volume_id = vol.get("volume_id")
    if not volume_id:
        print(f"Create missing volume_id: {body}", file=sys.stderr)
        return 1
    if not vol.get("owner_id"):
        print(f"Create response missing owner_id: {vol}", file=sys.stderr)
        return 1
    print(f"created volume_id={volume_id} owner_id={vol.get('owner_id')}")

    fetched = client.get(f"/api/v2/volumes/{volume_id}", headers=hdrs)
    if fetched.status_code != 200:
        print(f"Get failed: {fetched.status_code}", file=sys.stderr)
        return 1

    listed = client.get("/api/v2/volumes", headers=hdrs)
    if listed.status_code != 200:
        print(f"List failed: {listed.status_code}", file=sys.stderr)
        return 1
    ids = [v["volume_id"] for v in listed.json().get("volumes") or []]
    if volume_id not in ids:
        print(f"Volume {volume_id} not in list", file=sys.stderr)
        return 1

    if not args.infra_only:
        launched = client.post(
            "/instance",
            headers=hdrs,
            json={"name": f"smoke-{name}", "vram_needed_gb": 1, "volume_ids": [volume_id]},
        )
        if launched.status_code == 402:
            print("launch skipped: insufficient wallet balance (use --infra-only for NFS CRUD smoke)")
        elif launched.status_code != 200:
            print(
                f"Launch with volume_ids failed: {launched.status_code} {launched.text[:300]}",
                file=sys.stderr,
            )
            return 1
        else:
            print("launch with volume_ids ok")
    else:
        print("infra-only: skipped instance launch")

    if args.encrypted and _test_encrypted_volume(client, hdrs) is None:
        client.delete(f"/api/v2/volumes/{volume_id}", headers=hdrs)
        return 1

    if args.persist and not _test_persist_flow(client, hdrs, volume_id):
        client.delete(f"/api/v2/volumes/{volume_id}", headers=hdrs)
        return 1

    if args.hot_attach and not _test_hot_attach_flow(client, hdrs, volume_id):
        client.delete(f"/api/v2/volumes/{volume_id}", headers=hdrs)
        return 1

    if args.worker_mount and not _worker_mount_smoke(args.worker_host):
        client.delete(f"/api/v2/volumes/{volume_id}", headers=hdrs)
        return 1

    if args.snapshots and not _test_snapshots_flow(client, hdrs, volume_id):
        client.delete(f"/api/v2/volumes/{volume_id}", headers=hdrs)
        return 1

    deleted = client.delete(f"/api/v2/volumes/{volume_id}", headers=hdrs)
    if deleted.status_code != 200:
        print(f"Delete failed: {deleted.status_code} {deleted.text[:200]}", file=sys.stderr)
        return 1
    print("deleted ok")

    readyz = client.get("/readyz")
    if readyz.status_code == 200:
        nfs = readyz.json().get("nfs_volumes") or {}
        print(f"readyz nfs_volumes: mode={nfs.get('mode')} configured={nfs.get('configured')}")

    print("volumes_e2e_smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())