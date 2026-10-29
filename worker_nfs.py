# Xcelsior Worker Agent — NFS mount support + hot-mounting volumes into
# running containers.
#
# Extracted from worker_agent.py (NFS support section). Fully self-contained:
# only touches subprocess/os, the root helper client (worker_privileged) and the
# shared "xcelsior-worker" logger, no coupling to worker_agent's mutable state.

import logging
import os
import subprocess

import worker_privileged

log = logging.getLogger("xcelsior-worker")

# Mounting, unmounting and entering a job's mount namespace need root; the worker
# asks the root helper (worker_privileged, xcelsior-privd.socket) to do them.
MANAGED_VOLUME_HOST_DIR = worker_privileged.NFS_VOLUME_ROOT


def _mount_nfs(server, path, mount_point):
    """Mount an NFS share for shared model/data storage.

    Args:
        server: NFS server hostname or IP (must be on the mesh or in NFS_SERVERS)
        path: NFS export path (e.g., /exports/models)
        mount_point: /mnt/xcelsior-nfs, or /mnt/xcelsior-volumes/<volume id>

    Returns:
        True if mounted successfully, False otherwise.
    """
    # Mount options (`hard`, long timeo/retrans: I/O blocks and retries through an
    # NFS server reboot instead of returning errors) live in privileged.conf.
    if mount_point.startswith(MANAGED_VOLUME_HOST_DIR + "/"):
        volume_id = mount_point[len(MANAGED_VOLUME_HOST_DIR) + 1 :]
    elif os.path.normpath(mount_point) == worker_privileged.NFS_SHARED_MOUNT:
        volume_id = None
    else:
        log.warning(
            "NFS mount refused: %s is neither %s nor under %s",
            mount_point,
            worker_privileged.NFS_SHARED_MOUNT,
            MANAGED_VOLUME_HOST_DIR,
        )
        return False
    try:
        reply = worker_privileged.call("nfs_mount", server=server, export=path, volume_id=volume_id)
    except Exception as e:
        log.warning("NFS mount error: %s", e)
        return False
    if not reply.get("ok"):
        log.warning("NFS mount failed: %s", reply.get("error"))
        return False
    return True


def _unmount_nfs(mount_point):
    """Unmount an NFS share (best-effort, lazy unmount)."""
    try:
        reply = worker_privileged.call("umount", path=mount_point, lazy=True)
        if reply.get("ok"):
            log.info("NFS unmounted: %s", mount_point)
        else:
            log.debug("NFS unmount failed (non-fatal): %s", reply.get("error"))
    except Exception as e:
        log.debug("NFS unmount failed (non-fatal): %s", e)


def _container_bind(
    container_name: str, host_src: str, container_dst: str, mode: str = "rw"
) -> bool:
    """Mount host_src at container_dst inside a running job container."""
    try:
        reply = worker_privileged.call(
            "container_attach",
            container=container_name,
            source=host_src,
            dst=container_dst,
            mode=mode,
        )
    except Exception as e:
        reply = {"ok": False, "error": str(e)}
    if not reply.get("ok"):
        log.warning(
            "bind mount into %s failed %s→%s: %s",
            container_name,
            host_src,
            container_dst,
            str(reply.get("error"))[:200],
        )
        return False
    return True


def _container_unbind(container_name: str, container_dst: str) -> None:
    """Lazy-unmount a path inside a running job container (best-effort)."""
    try:
        worker_privileged.call("container_detach", container=container_name, dst=container_dst)
    except Exception as e:
        log.debug("unbind %s:%s failed (non-fatal): %s", container_name, container_dst, e)


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
        args.get("nfs_export_base")
        or os.environ.get("XCELSIOR_NFS_EXPORT_BASE", "/exports/volumes")
    )
    if not nfs_server:
        log.warning("mount_volume: XCELSIOR_NFS_SERVER not configured")
        return False

    nfs_path = f"{nfs_export_base}/{volume_id}"
    if not _mount_nfs(nfs_server, nfs_path, host_mount):
        log.warning("mount_volume: NFS mount failed for %s", volume_id)
        return False

    if not _container_bind(container_name, host_mount, container_path, mode=mode):
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

    _container_unbind(container_name, container_path)
    _unmount_nfs(host_mount)
    log.info("Hot-unmounted volume %s from %s at %s", volume_id, container_name, container_path)
    return True
