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
import time
import uuid
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger("xcelsior.volumes")

MAX_VOLUME_SIZE_GB = int(os.environ.get("XCELSIOR_MAX_VOLUME_GB", "2000"))
DEFAULT_MOUNT_PATH = "/workspace"


class VolumeEngine:
    """Manages persistent volumes and their attachments to instances."""

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
        """Create a new persistent volume."""
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
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'available', %s)""",
                (volume_id, owner_id, name, storage_type, size_gb,
                 region, province, encrypted, now),
            )

        log.info("Volume created: %s name=%s size=%dGB owner=%s", volume_id, name, size_gb, owner_id)
        return {
            "volume_id": volume_id,
            "name": name,
            "size_gb": size_gb,
            "status": "available",
            "encrypted": encrypted,
        }

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
        """Delete a volume. Must not be attached to any instance."""
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
                "UPDATE volumes SET status = 'deleted', deleted_at = %s WHERE volume_id = %s",
                (time.time(), volume_id),
            )

        log.info("Volume deleted: %s", volume_id)
        return {"volume_id": volume_id, "status": "deleted"}

    def attach_volume(
        self,
        volume_id: str,
        instance_id: str,
        mount_path: str = DEFAULT_MOUNT_PATH,
        mode: str = "rw",
    ) -> dict:
        """Attach a volume to a running instance."""
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

        log.info("Volume %s attached to %s at %s (%s)", volume_id, instance_id, mount_path, mode)
        return {
            "attachment_id": attachment_id,
            "volume_id": volume_id,
            "instance_id": instance_id,
            "mount_path": mount_path,
            "mode": mode,
        }

    def detach_volume(self, volume_id: str, instance_id: str) -> dict:
        """Detach a volume from an instance."""
        now = time.time()
        with self._conn() as conn:
            att = conn.execute(
                """SELECT * FROM volume_attachments
                   WHERE volume_id = %s AND instance_id = %s AND detached_at = 0 FOR UPDATE""",
                (volume_id, instance_id),
            ).fetchone()
            if not att:
                raise ValueError("No active attachment found")

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
        """Detach all volumes from an instance (on instance termination)."""
        now = time.time()
        detached = 0
        with self._conn() as conn:
            atts = conn.execute(
                "SELECT volume_id, attachment_id FROM volume_attachments WHERE instance_id = %s AND detached_at = 0",
                (instance_id,),
            ).fetchall()
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
