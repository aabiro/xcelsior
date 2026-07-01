# Xcelsior Worker Agent — LUKS encrypted volume provisioning.
#
# Extracted from worker_agent.py. Per Phase 10.2: encrypted at rest via LUKS
# with a per-volume key, giving cryptographic erasure on volume delete
# (destroy the LUKS key). Fully self-contained: only touches os/subprocess
# and the shared "xcelsior-worker" logger.

import logging
import os
import subprocess
from contextlib import suppress

log = logging.getLogger("xcelsior-worker")

VOLUME_BASE_DIR = os.environ.get("XCELSIOR_VOLUME_DIR", "/mnt/xcelsior/volumes")
VOLUME_KEY_DIR = os.environ.get("XCELSIOR_VOLUME_KEY_DIR", "/etc/xcelsior/volume-keys")


def _ensure_volume_dirs():
    """Create base directories for volumes and keys (if running as root)."""
    os.makedirs(VOLUME_BASE_DIR, exist_ok=True)
    os.makedirs(VOLUME_KEY_DIR, mode=0o700, exist_ok=True)


def provision_encrypted_volume(volume_id: str, size_gb: int) -> bool:
    """Create a LUKS-encrypted volume with a per-volume key.

    Steps:
      1. Create a sparse file as the backing store
      2. Generate a random 256-bit LUKS key
      3. Format with LUKS2 using the key
      4. Open the LUKS device
      5. Create ext4 filesystem
      6. Mount to /mnt/xcelsior/volumes/{volume_id}

    Returns True on success, False on failure.
    """
    _ensure_volume_dirs()

    backing_file = os.path.join(VOLUME_BASE_DIR, f"{volume_id}.img")
    key_file = os.path.join(VOLUME_KEY_DIR, f"{volume_id}.key")
    mapper_name = f"xcelsior-vol-{volume_id[:12]}"
    mount_point = os.path.join(VOLUME_BASE_DIR, volume_id)

    try:
        # 1. Create sparse backing file
        subprocess.run(
            ["truncate", "-s", f"{size_gb}G", backing_file],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # 2. Generate per-volume random key (256-bit)
        key_bytes = os.urandom(32)
        old_umask = os.umask(0o077)
        try:
            with open(key_file, "wb") as f:
                f.write(key_bytes)
        finally:
            os.umask(old_umask)

        # 3. LUKS format the backing file
        subprocess.run(
            [
                "cryptsetup",
                "luksFormat",
                "--batch-mode",
                "--type",
                "luks2",
                "--key-file",
                key_file,
                "--cipher",
                "aes-xts-plain64",
                "--key-size",
                "512",
                "--hash",
                "sha256",
                backing_file,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # 4. Open LUKS device
        subprocess.run(
            [
                "cryptsetup",
                "luksOpen",
                "--key-file",
                key_file,
                backing_file,
                mapper_name,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # 5. Create ext4 filesystem
        dm_path = f"/dev/mapper/{mapper_name}"
        subprocess.run(
            ["mkfs.ext4", "-q", "-L", f"vol-{volume_id[:8]}", dm_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # 6. Mount
        os.makedirs(mount_point, exist_ok=True)
        subprocess.run(
            ["mount", dm_path, mount_point],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        log.info("Volume %s provisioned: %dGB LUKS2+ext4 at %s", volume_id, size_gb, mount_point)
        return True

    except subprocess.CalledProcessError as e:
        log.error("Volume provisioning failed for %s: %s (stderr: %s)", volume_id, e, e.stderr)
        # Cleanup partial state. Call via worker_agent so callers/tests that
        # patch worker_agent._cleanup_partial_volume still take effect.
        import worker_agent

        worker_agent._cleanup_partial_volume(volume_id)
        return False
    except Exception as e:
        log.error("Volume provisioning error for %s: %s", volume_id, e)
        import worker_agent

        worker_agent._cleanup_partial_volume(volume_id)
        return False


def attach_encrypted_volume(volume_id: str) -> str | None:
    """Open and mount an existing LUKS volume. Returns mount path or None."""
    backing_file = os.path.join(VOLUME_BASE_DIR, f"{volume_id}.img")
    key_file = os.path.join(VOLUME_KEY_DIR, f"{volume_id}.key")
    mapper_name = f"xcelsior-vol-{volume_id[:12]}"
    mount_point = os.path.join(VOLUME_BASE_DIR, volume_id)
    dm_path = f"/dev/mapper/{mapper_name}"

    if not os.path.exists(backing_file) or not os.path.exists(key_file):
        log.error("Volume %s: backing file or key not found", volume_id)
        return None

    try:
        # Check if already open
        if not os.path.exists(dm_path):
            subprocess.run(
                ["cryptsetup", "luksOpen", "--key-file", key_file, backing_file, mapper_name],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

        # Mount if not already mounted
        os.makedirs(mount_point, exist_ok=True)
        r = subprocess.run(["mountpoint", "-q", mount_point], capture_output=True, timeout=5)
        if r.returncode != 0:
            subprocess.run(
                ["mount", dm_path, mount_point],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

        log.info("Volume %s attached at %s", volume_id, mount_point)
        return mount_point

    except subprocess.CalledProcessError as e:
        log.error("Volume attach failed for %s: %s", volume_id, e.stderr)
        return None


def detach_encrypted_volume(volume_id: str) -> bool:
    """Unmount and close a LUKS volume."""
    mapper_name = f"xcelsior-vol-{volume_id[:12]}"
    mount_point = os.path.join(VOLUME_BASE_DIR, volume_id)
    dm_path = f"/dev/mapper/{mapper_name}"

    try:
        # Unmount
        r = subprocess.run(["mountpoint", "-q", mount_point], capture_output=True, timeout=5)
        if r.returncode == 0:
            subprocess.run(
                ["umount", mount_point],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

        # Close LUKS device
        if os.path.exists(dm_path):
            subprocess.run(
                ["cryptsetup", "luksClose", mapper_name],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

        log.info("Volume %s detached", volume_id)
        return True

    except subprocess.CalledProcessError as e:
        log.error("Volume detach failed for %s: %s", volume_id, e.stderr)
        return False


def destroy_encrypted_volume(volume_id: str) -> bool:
    """Cryptographic erasure: destroy LUKS key then remove backing file.

    Once the key is shredded, the data is irrecoverable regardless
    of whether the backing file still exists.
    """
    key_file = os.path.join(VOLUME_KEY_DIR, f"{volume_id}.key")
    backing_file = os.path.join(VOLUME_BASE_DIR, f"{volume_id}.img")
    mount_point = os.path.join(VOLUME_BASE_DIR, volume_id)

    # Detach first (best-effort). Call via worker_agent so callers/tests that
    # patch worker_agent.detach_encrypted_volume still take effect.
    import worker_agent

    worker_agent.detach_encrypted_volume(volume_id)

    try:
        # Destroy key — cryptographic erasure (overwrite with random then delete)
        if os.path.exists(key_file):
            r = subprocess.run(
                ["shred", "-u", "-z", "-n", "3", key_file],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if r.returncode != 0:
                log.critical(
                    "SECURITY: shred failed for volume %s key — key may remain on disk: %s",
                    volume_id,
                    r.stderr,
                )
                return False
            log.info("Volume %s: LUKS key destroyed (cryptographic erasure)", volume_id)

        # Remove backing file
        if os.path.exists(backing_file):
            os.remove(backing_file)

        # Remove mount point directory
        if os.path.isdir(mount_point):
            os.rmdir(mount_point)

        log.info("Volume %s destroyed", volume_id)
        return True

    except Exception as e:
        log.error("Volume destroy failed for %s: %s", volume_id, e)
        return False


def _cleanup_partial_volume(volume_id: str):
    """Best-effort cleanup after a failed provisioning attempt."""
    mapper_name = f"xcelsior-vol-{volume_id[:12]}"
    dm_path = f"/dev/mapper/{mapper_name}"
    key_file = os.path.join(VOLUME_KEY_DIR, f"{volume_id}.key")
    backing_file = os.path.join(VOLUME_BASE_DIR, f"{volume_id}.img")
    mount_point = os.path.join(VOLUME_BASE_DIR, volume_id)

    with suppress(Exception):
        subprocess.run(["umount", mount_point], capture_output=True, timeout=10)
    with suppress(Exception):
        if os.path.exists(dm_path):
            subprocess.run(
                ["cryptsetup", "luksClose", mapper_name], capture_output=True, timeout=10
            )
    with suppress(Exception):
        if os.path.exists(key_file):
            subprocess.run(
                ["shred", "-u", "-z", "-n", "3", key_file],
                capture_output=True,
                timeout=30,
            )
    with suppress(Exception):
        if os.path.exists(backing_file):
            os.remove(backing_file)
    with suppress(Exception):
        if os.path.isdir(mount_point):
            os.rmdir(mount_point)
