# Xcelsior Worker Agent — LUKS encrypted volume provisioning.
#
# Extracted from worker_agent.py. Per Phase 10.2: encrypted at rest via LUKS
# with a per-volume key, giving cryptographic erasure on volume delete
# (destroy the LUKS key). The root helper (worker_privileged,
# xcelsior-privd.socket) does the work: cryptsetup, mkfs and mount need root,
# and the keys under VOLUME_KEY_DIR never become readable by the worker.

import logging
import os

import worker_privileged

log = logging.getLogger("xcelsior-worker")

VOLUME_BASE_DIR = os.environ.get("XCELSIOR_VOLUME_DIR", "/mnt/xcelsior/volumes")
VOLUME_KEY_DIR = os.environ.get("XCELSIOR_VOLUME_KEY_DIR", "/etc/xcelsior/volume-keys")


def _helper(op: str, volume_id: str, **args) -> dict:
    try:
        reply = worker_privileged.call(op, volume_id=volume_id, **args)
    except Exception as e:
        reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if not reply.get("ok"):
        log.error("Volume %s: %s failed: %s", volume_id, op, reply.get("error"))
    return reply


def provision_encrypted_volume(volume_id: str, size_gb: int) -> bool:
    """Create a LUKS-encrypted volume with a per-volume key.

    The helper creates a sparse backing file, a random 256-bit key, formats it
    LUKS2 (aes-xts-plain64, 512-bit key), opens it, creates ext4 and mounts it at
    /mnt/xcelsior/volumes/{volume_id}. It cleans up after itself on failure.

    Returns True on success, False on failure.
    """
    reply = _helper("luks_provision", volume_id, size_gb=int(size_gb))
    if reply.get("ok"):
        log.info(
            "Volume %s provisioned: %dGB LUKS2+ext4 at %s", volume_id, size_gb, reply["mount_point"]
        )
    return bool(reply.get("ok"))


def attach_encrypted_volume(volume_id: str) -> str | None:
    """Open and mount an existing LUKS volume. Returns mount path or None."""
    reply = _helper("luks_attach", volume_id)
    if not reply.get("ok"):
        return None
    log.info("Volume %s attached at %s", volume_id, reply["mount_point"])
    return reply["mount_point"]


def detach_encrypted_volume(volume_id: str) -> bool:
    """Unmount and close a LUKS volume."""
    ok = bool(_helper("luks_detach", volume_id).get("ok"))
    if ok:
        log.info("Volume %s detached", volume_id)
    return ok


def destroy_encrypted_volume(volume_id: str) -> bool:
    """Cryptographic erasure: detach, shred the LUKS key, then remove the backing file.

    Once the key is shredded, the data is irrecoverable regardless
    of whether the backing file still exists.
    """
    reply = _helper("luks_destroy", volume_id)
    if not reply.get("ok"):
        log.critical(
            "SECURITY: destroy failed for volume %s — key may remain on disk: %s",
            volume_id,
            reply.get("error"),
        )
        return False
    log.info("Volume %s destroyed (LUKS key shredded)", volume_id)
    return True


def _cleanup_partial_volume(volume_id: str):
    """Best-effort cleanup after a failed provisioning attempt."""
    _helper("luks_cleanup", volume_id)
