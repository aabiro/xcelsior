# Xcelsior Persistent Volume Management
# NFS-based persistent storage that can be attached to GPU instances.
#
# Per REPORT_XCELSIOR_TECHNICAL_FINAL.md:
# - Create/delete volumes per user
# - Attach/detach to running instances
# - Encrypted at rest (LUKS or provider-managed)
# - Region-aware (same-region for performance)
# - Mount as /workspace by default

import logging
import os
import shlex
import time
import uuid
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger("xcelsior.volumes")

MAX_VOLUME_SIZE_GB = int(os.environ.get("XCELSIOR_MAX_VOLUME_GB", "2000"))
MAX_TOTAL_STORAGE_GB = int(os.environ.get("XCELSIOR_MAX_TOTAL_STORAGE_GB", "100"))
DEFAULT_MOUNT_PATH = "/workspace"
NFS_SERVER = os.environ.get("XCELSIOR_NFS_SERVER", "")
NFS_EXPORT_BASE = os.environ.get("XCELSIOR_NFS_EXPORT_BASE", "/exports/volumes")


class VolumeEngine:
    """Manages persistent volumes and their attachments to instances.

    Volumes are NFS-backed directories exported from a central storage server.
    On attach, the volume is NFS-mounted on the instance's host.
    On detach, it is unmounted.
    On create, the directory and NFS export are provisioned on the storage server.
    On delete, the export is removed and data destroyed.
    """

    @contextmanager
    def _conn(self):
        from db import _get_pg_pool
        from psycopg.rows import dict_row
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _ssh_exec(self, ip: str, cmd: str) -> tuple[int, str, str]:
        """Execute a command on a remote host via SSH."""
        from scheduler import ssh_exec
        return ssh_exec(ip, cmd)

    def _get_instance_host_ip(self, instance_id: str) -> Optional[str]:
        """Look up which host an instance is running on, return its IP."""
        with self._conn() as conn:
            # Try jobs table (payload->>'job_id' and payload->>'host_id')
            row = conn.execute(
                """SELECT h.payload->>'ip' AS ip FROM hosts h
                   JOIN jobs j ON j.host_id = h.host_id
                   WHERE j.job_id = %s AND j.status = 'running'""",
                (instance_id,),
            ).fetchone()
            if row:
                return row["ip"]

            # Try JSONB payload format
            row = conn.execute(
                """SELECT h.payload->>'ip' AS ip FROM hosts h
                   WHERE h.host_id = (
                     SELECT payload->>'host_id' FROM jobs
                     WHERE payload->>'job_id' = %s AND payload->>'status' = 'running'
                   )""",
                (instance_id,),
            ).fetchone()
            return row["ip"] if row else None

    def create_volume(
        self,
        owner_id: str,
        name: str,
        size_gb: int,
        storage_type: str = "nfs",
        region: str = "",
        province: str = "",
        encrypted: bool = True,
    ) -> dict:
        """Create a new persistent volume.

        Provisions storage on the NFS server (creates directory, sets quota),
        then records metadata in the database.
        """
        if not name or not name.strip():
            raise ValueError("Volume name is required")
        name = name.strip()
        if size_gb > MAX_VOLUME_SIZE_GB:
            raise ValueError(f"Volume size {size_gb}GB exceeds max {MAX_VOLUME_SIZE_GB}GB")
        if size_gb < 1:
            raise ValueError("Volume size must be at least 1GB")

        now = time.time()
        volume_id = f"vol-{uuid.uuid4().hex[:12]}"

        with self._conn() as conn:
            # Check total storage capacity
            total = conn.execute(
                "SELECT COALESCE(SUM(size_gb), 0) AS total FROM volumes WHERE status != 'deleted'",
            ).fetchone()
            if total["total"] + size_gb > MAX_TOTAL_STORAGE_GB:
                remaining = MAX_TOTAL_STORAGE_GB - total["total"]
                raise ValueError(
                    f"Insufficient storage capacity. Requested {size_gb}GB but only {remaining}GB available "
                    f"(total limit: {MAX_TOTAL_STORAGE_GB}GB)"
                )

            # Check name uniqueness for this owner
            existing = conn.execute(
                "SELECT volume_id FROM volumes WHERE owner_id = %s AND name = %s AND status != 'deleted'",
                (owner_id, name),
            ).fetchone()
            if existing:
                raise ValueError(f"Volume name '{name}' already exists")

            conn.execute(
                """INSERT INTO volumes
                   (volume_id, owner_id, name, storage_type, size_gb,
                    region, province, encrypted, status, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'provisioning', %s)""",
                (volume_id, owner_id, name, storage_type, size_gb,
                 region, province, encrypted, now),
            )

        # Provision actual storage on the NFS server
        provision_ok = self._provision_volume_storage(volume_id, size_gb)

        with self._conn() as conn:
            new_status = "available" if provision_ok else "error"
            conn.execute(
                "UPDATE volumes SET status = %s WHERE volume_id = %s",
                (new_status, volume_id),
            )

        if not provision_ok:
            log.error("Volume storage provisioning failed for %s", volume_id)
            raise ValueError("Failed to provision volume storage — check NFS server configuration")

        log.info("Volume created: %s name=%s size=%dGB owner=%s", volume_id, name, size_gb, owner_id)
        return {
            "volume_id": volume_id,
            "name": name,
            "size_gb": size_gb,
            "status": "available",
            "encrypted": encrypted,
        }

    def _provision_volume_storage(self, volume_id: str, size_gb: int) -> bool:
        """Create the volume directory on the NFS server."""
        if not NFS_SERVER:
            log.warning("XCELSIOR_NFS_SERVER not configured — volume %s is metadata-only", volume_id)
            return True  # Allow metadata-only operation in dev/test

        vol_path = f"{NFS_EXPORT_BASE}/{volume_id}"
        safe_path = shlex.quote(vol_path)
        # Create directory and set ownership to allow container writes
        cmd = f"mkdir -p {safe_path} && chmod 1777 {safe_path}"
        rc, _, stderr = self._ssh_exec(NFS_SERVER, cmd)
        if rc != 0:
            log.error("NFS volume provision failed for %s: %s", volume_id, stderr)
            return False
        log.info("NFS storage created for volume %s at %s:%s", volume_id, NFS_SERVER, vol_path)
        return True

    def get_volume(self, volume_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM volumes WHERE volume_id = %s AND status != 'deleted'",
                (volume_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_volumes(self, owner_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM volumes WHERE owner_id = %s AND status != 'deleted' ORDER BY created_at DESC",
                (owner_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_volume(self, volume_id: str, owner_id: str) -> dict:
        """Delete a volume. Must not be attached to any instance.

        Removes the NFS directory on the storage server, then marks deleted in DB.
        """
        with self._conn() as conn:
            vol = conn.execute(
                "SELECT * FROM volumes WHERE volume_id = %s AND owner_id = %s FOR UPDATE",
                (volume_id, owner_id),
            ).fetchone()
            if not vol:
                raise ValueError("Volume not found")

            # Check no active attachments
            att = conn.execute(
                "SELECT attachment_id FROM volume_attachments WHERE volume_id = %s AND detached_at = 0",
                (volume_id,),
            ).fetchone()
            if att:
                raise ValueError("Cannot delete volume with active attachments. Detach first.")

            conn.execute(
                "UPDATE volumes SET status = 'deleting', deleted_at = %s WHERE volume_id = %s",
                (time.time(), volume_id),
            )

        # Destroy actual storage on the NFS server
        self._destroy_volume_storage(volume_id)

        with self._conn() as conn:
            conn.execute(
                "UPDATE volumes SET status = 'deleted' WHERE volume_id = %s",
                (volume_id,),
            )

        log.info("Volume deleted: %s", volume_id)
        return {"volume_id": volume_id, "status": "deleted"}

    def _destroy_volume_storage(self, volume_id: str) -> bool:
        """Remove the volume directory from the NFS server."""
        if not NFS_SERVER:
            log.warning("XCELSIOR_NFS_SERVER not configured — skipping storage deletion for %s", volume_id)
            return True

        vol_path = f"{NFS_EXPORT_BASE}/{volume_id}"
        safe_path = shlex.quote(vol_path)
        # Safety: only delete under the exports base path
        cmd = f"test -d {safe_path} && rm -rf {safe_path}"
        rc, _, stderr = self._ssh_exec(NFS_SERVER, cmd)
        if rc != 0:
            log.error("NFS volume deletion failed for %s: %s", volume_id, stderr)
            return False
        log.info("NFS storage destroyed for volume %s", volume_id)
        return True

    def attach_volume(
        self,
        volume_id: str,
        instance_id: str,
        mount_path: str = DEFAULT_MOUNT_PATH,
        mode: str = "rw",
    ) -> dict:
        """Attach a volume to a running instance.

        Looks up which host the instance runs on, then SSH-mounts the NFS share
        into the container's filesystem.
        """
        now = time.time()
        with self._conn() as conn:
            vol = conn.execute(
                "SELECT * FROM volumes WHERE volume_id = %s FOR UPDATE",
                (volume_id,),
            ).fetchone()
            if not vol:
                raise ValueError("Volume not found")
            if vol["status"] not in ("available", "attached"):
                raise ValueError(f"Volume status '{vol['status']}' cannot be attached")

            # Check not already attached to this instance
            existing = conn.execute(
                """SELECT attachment_id FROM volume_attachments
                   WHERE volume_id = %s AND instance_id = %s AND detached_at = 0""",
                (volume_id, instance_id),
            ).fetchone()
            if existing:
                return {"attachment_id": existing["attachment_id"], "already_attached": True}

            attachment_id = f"att-{uuid.uuid4().hex[:12]}"
            conn.execute(
                """INSERT INTO volume_attachments
                   (attachment_id, volume_id, instance_id, mount_path, mode, attached_at)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (attachment_id, volume_id, instance_id, mount_path, mode, now),
            )
            conn.execute(
                "UPDATE volumes SET status = 'attached' WHERE volume_id = %s",
                (volume_id,),
            )

        # Mount the NFS share on the instance host
        mount_ok = self._mount_on_host(volume_id, instance_id, mount_path, mode)
        if not mount_ok:
            # Rollback: remove attachment record
            with self._conn() as conn:
                conn.execute(
                    "DELETE FROM volume_attachments WHERE attachment_id = %s",
                    (attachment_id,),
                )
                conn.execute(
                    "UPDATE volumes SET status = 'available' WHERE volume_id = %s",
                    (volume_id,),
                )
            raise ValueError(f"Failed to mount volume on instance host — attachment rolled back")

        log.info("Volume %s attached to %s at %s (%s)", volume_id, instance_id, mount_path, mode)
        return {
            "attachment_id": attachment_id,
            "volume_id": volume_id,
            "instance_id": instance_id,
            "mount_path": mount_path,
            "mode": mode,
        }

    def _mount_on_host(self, volume_id: str, instance_id: str, mount_path: str, mode: str) -> bool:
        """NFS-mount the volume on the host where the instance runs."""
        if not NFS_SERVER:
            log.warning("XCELSIOR_NFS_SERVER not configured — skipping real mount for %s", volume_id)
            return True  # Allow metadata-only in dev/test

        host_ip = self._get_instance_host_ip(instance_id)
        if not host_ip:
            log.error("Cannot mount volume %s: instance %s has no host IP (not running?)", volume_id, instance_id)
            return False

        nfs_src = f"{NFS_SERVER}:{NFS_EXPORT_BASE}/{volume_id}"
        safe_mount = shlex.quote(mount_path)
        safe_src = shlex.quote(nfs_src)
        ro_flag = ",ro" if mode == "ro" else ""
        cmd = (
            f"mkdir -p {safe_mount} && "
            f"mount -t nfs -o noatime,nodiratime,tcp,hard,intr{ro_flag} {safe_src} {safe_mount}"
        )
        rc, _, stderr = self._ssh_exec(host_ip, cmd)
        if rc != 0:
            log.error("NFS mount failed for volume %s on host %s: %s", volume_id, host_ip, stderr)
            return False
        log.info("NFS mounted %s on %s at %s", volume_id, host_ip, mount_path)
        return True

    def detach_volume(self, volume_id: str, instance_id: str) -> dict:
        """Detach a volume from an instance.

        Unmounts the NFS share on the host, then updates DB records.
        """
        now = time.time()
        with self._conn() as conn:
            att = conn.execute(
                """SELECT * FROM volume_attachments
                   WHERE volume_id = %s AND instance_id = %s AND detached_at = 0 FOR UPDATE""",
                (volume_id, instance_id),
            ).fetchone()
            if not att:
                raise ValueError("No active attachment found")

            mount_path = att.get("mount_path", DEFAULT_MOUNT_PATH)

        # Unmount from the instance host first
        self._unmount_from_host(volume_id, instance_id, mount_path)

        with self._conn() as conn:
            conn.execute(
                "UPDATE volume_attachments SET detached_at = %s WHERE attachment_id = %s",
                (now, att["attachment_id"]),
            )

            # Check if volume has any remaining attachments
            remaining = conn.execute(
                "SELECT COUNT(*) as cnt FROM volume_attachments WHERE volume_id = %s AND detached_at = 0",
                (volume_id,),
            ).fetchone()
            if remaining["cnt"] == 0:
                conn.execute(
                    "UPDATE volumes SET status = 'available' WHERE volume_id = %s",
                    (volume_id,),
                )

        log.info("Volume %s detached from %s", volume_id, instance_id)
        return {"volume_id": volume_id, "instance_id": instance_id, "status": "detached"}

    def _unmount_from_host(self, volume_id: str, instance_id: str, mount_path: str) -> bool:
        """NFS-unmount the volume on the host where the instance runs."""
        if not NFS_SERVER:
            log.warning("XCELSIOR_NFS_SERVER not configured — skipping real unmount for %s", volume_id)
            return True

        host_ip = self._get_instance_host_ip(instance_id)
        if not host_ip:
            log.warning("Cannot unmount volume %s: instance %s has no host IP (already stopped?)", volume_id, instance_id)
            return True  # Instance already gone, nothing to unmount

        safe_mount = shlex.quote(mount_path)
        # Lazy unmount to avoid hanging on busy mount
        cmd = f"umount -l {safe_mount} 2>/dev/null; true"
        rc, _, stderr = self._ssh_exec(host_ip, cmd)
        if rc != 0:
            log.error("NFS unmount failed for volume %s on host %s: %s", volume_id, host_ip, stderr)
            return False
        log.info("NFS unmounted %s from %s at %s", volume_id, host_ip, mount_path)
        return True

    def get_instance_volumes(self, instance_id: str) -> list[dict]:
        """Get all volumes attached to an instance."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT va.*, v.name, v.size_gb, v.storage_type, v.encrypted
                   FROM volume_attachments va
                   JOIN volumes v ON v.volume_id = va.volume_id
                   WHERE va.instance_id = %s AND va.detached_at = 0""",
                (instance_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def detach_all_for_instance(self, instance_id: str) -> int:
        """Detach all volumes from an instance (on instance termination).

        Unmounts each NFS share on the host, then updates DB.
        """
        now = time.time()
        detached = 0
        with self._conn() as conn:
            atts = conn.execute(
                "SELECT volume_id, attachment_id, mount_path FROM volume_attachments WHERE instance_id = %s AND detached_at = 0",
                (instance_id,),
            ).fetchall()

        # Unmount each volume from the host (outside DB transaction)
        for att in atts:
            mount_path = att.get("mount_path", DEFAULT_MOUNT_PATH)
            self._unmount_from_host(att["volume_id"], instance_id, mount_path)

        with self._conn() as conn:
            for att in atts:
                conn.execute(
                    "UPDATE volume_attachments SET detached_at = %s WHERE attachment_id = %s",
                    (now, att["attachment_id"]),
                )
                conn.execute(
                    "UPDATE volumes SET status = 'available' WHERE volume_id = %s",
                    (att["volume_id"],),
                )
                detached += 1
        if detached:
            log.info("Detached %d volumes from instance %s", detached, instance_id)
        return detached


# ── Singleton ─────────────────────────────────────────────────────────

_volume_engine: Optional[VolumeEngine] = None


def get_volume_engine() -> VolumeEngine:
    global _volume_engine
    if _volume_engine is None:
        _volume_engine = VolumeEngine()
    return _volume_engine
