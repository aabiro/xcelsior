# Xcelsior Worker Agent — NFS mount support + hot-mounting volumes into
# running containers.
#
# Extracted from worker_agent.py (NFS support section). Fully self-contained:
# only touches subprocess/os/shlex and the shared "xcelsior-worker" logger, no
# coupling to worker_agent's mutable agent-wide state.

import logging
import os
import shlex
import subprocess

log = logging.getLogger("xcelsior-worker")

MANAGED_VOLUME_HOST_DIR = "/mnt/xcelsior-volumes"


def _nfs_mount_fstype() -> str:
    """Return ``mount -t`` type for configured NFS mount options."""
    opts = os.environ.get(
        "XCELSIOR_NFS_MOUNT_OPTS",
        "hard,timeo=600,retrans=3,rsize=1048576,wsize=1048576,noatime,nosuid,nodev,_netdev,tcp",
    )
    return "nfs4" if "nfsvers=4" in opts else "nfs"


def _mount_nfs(server, path, mount_point):
    """Mount an NFS share for shared model/data storage.

    Args:
        server: NFS server hostname or IP
        path: NFS export path (e.g., /exports/models)
        mount_point: Local mount point (e.g., /mnt/xcelsior-nfs)

    Returns:
        True if mounted successfully, False otherwise.
    """
    try:
        os.makedirs(mount_point, exist_ok=True)

        # Check if already mounted
        r = subprocess.run(
            ["mountpoint", "-q", mount_point],
            capture_output=True,
            timeout=5,
        )
        if r.returncode == 0:
            return True  # Already mounted

        # Mount NFS
        # Use `hard` mount + long timeo/retrans so I/O blocks-and-retries on
        # NFS server reboot instead of silently returning errors (which would
        # cause data corruption on write-back caches). Do not change to `soft`
        # without an explicit durability review.
        mount_cmd = [
            "mount",
            "-t",
            _nfs_mount_fstype(),
            "-o",
            os.environ.get(
                "XCELSIOR_NFS_MOUNT_OPTS",
                "hard,timeo=600,retrans=3,rsize=1048576,wsize=1048576,noatime,nosuid,nodev,_netdev,tcp",
            ),
            f"{server}:{path}",
            mount_point,
        ]
        r = subprocess.run(mount_cmd, capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            log.warning("NFS mount failed: %s", r.stderr.strip())
            return False

        return True
    except Exception as e:
        log.warning("NFS mount error: %s", e)
        return False


def _unmount_nfs(mount_point):
    """Unmount an NFS share (best-effort, lazy unmount)."""
    try:
        subprocess.run(
            ["umount", "-l", mount_point],
            capture_output=True,
            timeout=10,
        )
        log.info("NFS unmounted: %s", mount_point)
    except Exception as e:
        log.debug("NFS unmount failed (non-fatal): %s", e)


def _container_pid(container_name: str) -> int | None:
    """Return the host PID of a running container, or None."""
    try:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Pid}}", container_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return None
        pid = int((r.stdout or "0").strip() or 0)
        return pid if pid > 0 else None
    except (subprocess.TimeoutExpired, ValueError):
        return None


def _nsenter_bind_mount(pid: int, host_src: str, container_dst: str, mode: str = "rw") -> bool:
    """Bind-mount host_src into a running container's mount namespace."""
    safe_src = shlex.quote(host_src)
    safe_dst = shlex.quote(container_dst)
    if mode == "ro":
        mount_script = (
            f"mkdir -p {safe_dst} && "
            f"mount --bind {safe_src} {safe_dst} && "
            f"mount -o remount,bind,ro {safe_dst}"
        )
    else:
        mount_script = f"mkdir -p {safe_dst} && mount --bind {safe_src} {safe_dst}"
    try:
        r = subprocess.run(
            ["nsenter", "-t", str(pid), "-m", "--", "sh", "-c", mount_script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode != 0:
            log.warning(
                "nsenter bind mount failed pid=%s %s→%s: %s",
                pid,
                host_src,
                container_dst,
                (r.stderr or "")[:200],
            )
            return False
        return True
    except subprocess.TimeoutExpired:
        log.warning("nsenter bind mount timed out pid=%s dst=%s", pid, container_dst)
        return False


def _nsenter_lazy_umount(pid: int, container_dst: str) -> None:
    """Lazy-unmount a path inside a container's mount namespace (best-effort)."""
    safe_dst = shlex.quote(container_dst)
    try:
        subprocess.run(
            ["nsenter", "-t", str(pid), "-m", "--", "umount", "-l", safe_dst],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        pass


def _hot_mount_volume(args: dict) -> bool:
    """NFS-mount a managed volume on the host and bind it into a running container."""
    job_id = str(args.get("job_id") or "")
    volume_id = str(args.get("volume_id") or "")
    container_path = str(args.get("container_path") or "/workspace")
    mode = str(args.get("mode") or "rw")
    if not job_id or not volume_id:
        log.warning("mount_volume missing job_id/volume_id: %r", args)
        return False

    container_name = str(args.get("container_name") or f"xcl-{job_id}")
    inspect = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if inspect.returncode != 0 or inspect.stdout.strip() != "true":
        log.warning("mount_volume container %s not running", container_name)
        return False

    host_mount = f"{MANAGED_VOLUME_HOST_DIR}/{volume_id}"
    nfs_server = str(args.get("nfs_server") or os.environ.get("XCELSIOR_NFS_SERVER", ""))
    nfs_export_base = str(
        args.get("nfs_export_base") or os.environ.get("XCELSIOR_NFS_EXPORT_BASE", "/exports/volumes")
    )
    if not nfs_server:
        log.warning("mount_volume: XCELSIOR_NFS_SERVER not configured")
        return False

    nfs_path = f"{nfs_export_base}/{volume_id}"
    if not _mount_nfs(nfs_server, nfs_path, host_mount):
        log.warning("mount_volume: NFS mount failed for %s", volume_id)
        return False

    pid = _container_pid(container_name)
    if not pid:
        log.warning("mount_volume: no PID for %s", container_name)
        _unmount_nfs(host_mount)
        return False

    if not _nsenter_bind_mount(pid, host_mount, container_path, mode=mode):
        _unmount_nfs(host_mount)
        return False

    log.info(
        "Hot-mounted volume %s at %s in %s (host %s)",
        volume_id,
        container_path,
        container_name,
        host_mount,
    )
    return True


def _hot_unmount_volume(args: dict) -> bool:
    """Unbind a managed volume from a container and lazy-unmount the host NFS mount."""
    job_id = str(args.get("job_id") or "")
    volume_id = str(args.get("volume_id") or "")
    container_path = str(args.get("container_path") or "/workspace")
    if not job_id or not volume_id:
        log.warning("unmount_volume missing job_id/volume_id: %r", args)
        return False

    container_name = str(args.get("container_name") or f"xcl-{job_id}")
    host_mount = f"{MANAGED_VOLUME_HOST_DIR}/{volume_id}"

    pid = _container_pid(container_name)
    if pid:
        _nsenter_lazy_umount(pid, container_path)

    _unmount_nfs(host_mount)
    log.info("Hot-unmounted volume %s from %s at %s", volume_id, container_name, container_path)
    return True
