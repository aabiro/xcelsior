# Xcelsior Persistent Volume Management
# NFS-based persistent storage that can be attached to GPU instances.
#
# Per REPORT_XCELSIOR_TECHNICAL_FINAL.md:
# - Create/delete volumes per user
# - Attach/detach to running instances
# - Encrypted at rest (LUKS or provider-managed)
# - Region-aware (same-region for performance)
# - Mount as /workspace by default

import base64
import logging
import os
import re
import shlex
import subprocess
import time
import uuid
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger("xcelsior.volumes")

VOLUME_PRICE_PER_GB_MONTH_CAD = 0.03
MAX_VOLUME_SIZE_GB = int(os.environ.get("XCELSIOR_MAX_VOLUME_GB", "2000"))
MAX_TOTAL_STORAGE_GB = int(os.environ.get("XCELSIOR_MAX_TOTAL_STORAGE_GB", "2000"))
DEFAULT_MOUNT_PATH = "/workspace"
NFS_SERVER = os.environ.get("XCELSIOR_NFS_SERVER", "")
NFS_EXPORT_BASE = os.environ.get("XCELSIOR_NFS_EXPORT_BASE", "/exports/volumes")

# Canonical NFS mount options — used by volumes.py, scheduler.py, worker_agent.py.
# hard: I/O blocks-and-retries on NFS server reboot (never silently fails).
#   Required for data durability on GPU checkpoints / training state.
# timeo=600: 60-second initial RPC timeout (in deciseconds); retrans=3: 3 retries.
# rsize/wsize=1M: large buffers for GPU checkpoint throughput.
# _netdev: wait for network before mounting at boot.
NFS_MOUNT_OPTS = (
    "hard,timeo=600,retrans=3,rsize=1048576,wsize=1048576,noatime,nosuid,nodev,_netdev,tcp"
)


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

    def _ssh_exec(self, ip: str, cmd: str, timeout: int = 30) -> tuple[int, str, str]:
        """Execute a command on a remote host via SSH."""
        from scheduler import ssh_exec

        return ssh_exec(ip, cmd, timeout=timeout)

    def _ssh_exec_with_retry(
        self, ip: str, cmd: str, *, max_retries: int = 3, timeout: int = 30
    ) -> tuple[int, str, str]:
        """SSH exec with exponential backoff on timeout / connection refused.

        Retries only on transient failures (timeout, connection refused rc=255).
        Non-zero exit from the remote command itself is NOT retried.
        """
        delays = [2, 4, 8]  # seconds between retries
        for attempt in range(max_retries):
            rc, out, err = self._ssh_exec(ip, cmd, timeout=timeout)
            # rc == 255 with specific stderr patterns = SSH-level failure (not remote cmd)
            ssh_transient = rc == 255 and (
                "timeout" in err.lower()
                or "connection refused" in err.lower()
                or "no route to host" in err.lower()
            )
            if rc != 255 or not ssh_transient:
                return rc, out, err
            if attempt < max_retries - 1:
                delay = delays[min(attempt, len(delays) - 1)]
                log.warning(
                    "SSH transient failure to %s (attempt %d/%d), retrying in %ds: %s",
                    ip,
                    attempt + 1,
                    max_retries,
                    delay,
                    err[:120],
                )
                time.sleep(delay)
        return rc, out, err  # last attempt's result

    # ── SSH with stdin (for piping LUKS keys) ──────────────────────────

    def _ssh_exec_with_stdin(
        self, ip: str, cmd: str, stdin_data: bytes, *, timeout: int = 60
    ) -> tuple[int, str, str]:
        """SSH exec that pipes stdin_data to the remote command.

        Used for LUKS operations where the key must be piped via stdin
        (never written to disk on the remote host).
        """
        from scheduler import SSH_KEY_PATH, SSH_USER

        full_cmd = [
            "ssh",
            "-i",
            SSH_KEY_PATH,
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            f"{SSH_USER}@{ip}",
            cmd,
        ]
        try:
            result = subprocess.run(
                full_cmd,
                input=stdin_data,
                capture_output=True,
                timeout=timeout,
            )
            return (
                result.returncode,
                result.stdout.decode(errors="replace").strip(),
                result.stderr.decode(errors="replace").strip(),
            )
        except subprocess.TimeoutExpired:
            log.warning("SSH+stdin command timed out after %ds to %s", timeout, ip)
            return 255, "", f"ssh timeout after {timeout}s"

    # ── Volume Encryption Key Management ─────────────────────────────
    # Keys are 256-bit random, Fernet-encrypted, stored in DB.
    # LUKS key material never touches disk on the NFS server — piped via stdin.

    @staticmethod
    def _generate_volume_key() -> bytes:
        """Generate a 256-bit random key for LUKS encryption."""
        return os.urandom(32)

    @staticmethod
    def _encrypt_key(raw_key: bytes) -> str:
        """Fernet-encrypt a raw LUKS key for DB storage."""
        from security import encrypt_secret

        return encrypt_secret(base64.b64encode(raw_key).decode())

    @staticmethod
    def _decrypt_key(ciphertext: str) -> bytes:
        """Recover raw LUKS key from Fernet ciphertext."""
        from security import decrypt_secret

        return base64.b64decode(decrypt_secret(ciphertext))

    def _store_key(self, conn, volume_id: str, raw_key: bytes) -> None:
        """Encrypt and store a LUKS key in the database."""
        ct = self._encrypt_key(raw_key)
        conn.execute(
            "UPDATE volumes SET key_ciphertext = %s WHERE volume_id = %s",
            (ct, volume_id),
        )

    def _retrieve_key(self, conn, volume_id: str) -> bytes | None:
        """Retrieve and decrypt the LUKS key for a volume. Returns None if no key."""
        row = conn.execute(
            "SELECT key_ciphertext FROM volumes WHERE volume_id = %s",
            (volume_id,),
        ).fetchone()
        if not row or not row["key_ciphertext"]:
            return None
        return self._decrypt_key(row["key_ciphertext"])

    def _destroy_key(self, conn, volume_id: str) -> None:
        """Destroy the stored encryption key (cryptographic erasure)."""
        conn.execute(
            "UPDATE volumes SET key_ciphertext = '' WHERE volume_id = %s",
            (volume_id,),
        )
        log.info("Volume %s: encryption key destroyed (cryptographic erasure)", volume_id)

    def _emit_event(
        self, event_type: str, volume_id: str, actor: str = "", data: dict | None = None
    ):
        """Best-effort emit a volume lifecycle event into the tamper-evident event store."""
        try:
            from events import Event, EventStore

            store = EventStore()
            store.append(
                Event(
                    event_type=event_type,
                    entity_type="volume",
                    entity_id=volume_id,
                    actor=f"user:{actor}" if actor else "system",
                    data=data or {},
                )
            )
        except Exception as e:
            log.warning("Failed to emit volume event %s for %s: %s", event_type, volume_id, e)

    # ── Volume State Machine ─────────────────────────────────────────
    #   provisioning ──→ available ──→ attached ──→ available
    #        │               │                          │
    #        ↓               ↓                          ↓
    #      error          deleting ──→ deleted        (detach)
    #        │               │
    #        ↓               ↓
    #   provisioning    available (rollback on NFS fail)
    #
    _VALID_TRANSITIONS: dict[str, set[str]] = {
        "provisioning": {"available", "error"},
        "available": {"attached", "deleting"},
        "attached": {"available"},
        "deleting": {"deleted", "available"},  # available = rollback if NFS delete fails
        "error": {"provisioning", "deleting"},  # user can retry or delete errored volumes
        # "deleted" is terminal — no outgoing transitions
    }

    def _transition_status(
        self, conn, volume_id: str, new_status: str, *, current: str | None = None
    ) -> str:
        """Atomically transition a volume's status with guard validation.

        If `current` is provided, it's used as the expected current status
        (avoids a SELECT). Otherwise, the current status is read from DB.

        Returns the new status. Raises ValueError on invalid transition.
        """
        if current is None:
            row = conn.execute(
                "SELECT status FROM volumes WHERE volume_id = %s FOR UPDATE",
                (volume_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"Volume {volume_id} not found")
            current = row["status"]

        allowed = self._VALID_TRANSITIONS.get(current, set())
        if new_status not in allowed:
            raise ValueError(
                f"Invalid volume transition: {current} → {new_status} "
                f"(allowed: {', '.join(sorted(allowed)) or 'none — terminal state'})"
            )

        conn.execute(
            "UPDATE volumes SET status = %s WHERE volume_id = %s",
            (new_status, volume_id),
        )
        log.debug("Volume %s: %s → %s", volume_id, current, new_status)
        return new_status

    def _get_instance_host_ip(self, instance_id: str) -> Optional[str]:
        """Look up which host an instance was/is running on, return its IP.

        Does NOT filter by job status — volumes must be unmountable even after
        the job reaches a terminal state (completed, failed, terminated).
        """
        with self._conn() as conn:
            row = conn.execute(
                """SELECT h.payload->>'ip' AS ip FROM hosts h
                   JOIN jobs j ON j.host_id = h.host_id
                   WHERE j.job_id = %s AND j.host_id IS NOT NULL""",
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
        if len(name) > 128:
            raise ValueError("Volume name must be 128 characters or fewer")
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$", name):
            raise ValueError(
                "Volume name must start with alphanumeric and contain only letters, digits, hyphens, underscores, and dots"
            )
        if size_gb > MAX_VOLUME_SIZE_GB:
            raise ValueError(f"Volume size {size_gb}GB exceeds max {MAX_VOLUME_SIZE_GB}GB")
        if size_gb < 1:
            raise ValueError("Volume size must be at least 1GB")

        now = time.time()
        volume_id = f"vol-{uuid.uuid4().hex[:12]}"

        with self._conn() as conn:
            # Lock user's volume rows to prevent TOCTOU race, then sum outside the lock
            conn.execute(
                "SELECT volume_id FROM volumes WHERE owner_id = %s AND status != 'deleted' FOR UPDATE",
                (owner_id,),
            )
            total = conn.execute(
                "SELECT COALESCE(SUM(size_gb), 0) AS total FROM volumes WHERE owner_id = %s AND status != 'deleted'",
                (owner_id,),
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
                (
                    volume_id,
                    owner_id,
                    name,
                    storage_type,
                    size_gb,
                    region,
                    province,
                    encrypted,
                    now,
                ),
            )

        # Provision actual storage on the NFS server
        raw_key = self._generate_volume_key() if encrypted else None
        provision_ok = self._provision_volume_storage(
            volume_id,
            size_gb,
            encrypted=encrypted,
            raw_key=raw_key,
        )

        with self._conn() as conn:
            new_status = "available" if provision_ok else "error"
            self._transition_status(conn, volume_id, new_status, current="provisioning")
            if provision_ok and encrypted and raw_key:
                self._store_key(conn, volume_id, raw_key)

        if not provision_ok:
            log.warning(
                "Volume storage provisioning deferred for %s — NFS unreachable, will retry on attach",
                volume_id,
            )
            # Status already set to 'error' above — user can retry via UI

        vol_status = "available" if provision_ok else "error"
        log.info(
            "Volume created: %s name=%s size=%dGB owner=%s status=%s",
            volume_id,
            name,
            size_gb,
            owner_id,
            vol_status,
        )
        self._emit_event(
            "volume.created",
            volume_id,
            actor=owner_id,
            data={
                "name": name,
                "size_gb": size_gb,
                "region": region,
                "encrypted": encrypted,
            },
        )
        return {
            "volume_id": volume_id,
            "name": name,
            "size_gb": size_gb,
            "status": vol_status,
            "encrypted": encrypted,
        }

    def _provision_volume_storage(
        self,
        volume_id: str,
        size_gb: int,
        *,
        encrypted: bool = False,
        raw_key: bytes | None = None,
    ) -> bool:
        """Create the volume directory/image on the NFS server.

        For encrypted volumes: creates a sparse LUKS2 loopback image,
        formats with ext4, and mounts at the export path. Key is piped via
        stdin and never written to disk on the NFS server.
        For unencrypted: simple mkdir -p.
        """
        if not NFS_SERVER:
            log.warning(
                "XCELSIOR_NFS_SERVER not configured — volume %s is metadata-only", volume_id
            )
            return True  # Allow metadata-only operation in dev/test

        vol_path = f"{NFS_EXPORT_BASE}/{volume_id}"
        safe_path = shlex.quote(vol_path)

        if not encrypted:
            # Plain unencrypted NFS directory
            cmd = f"mkdir -p {safe_path} && chmod 1777 {safe_path}"
            rc, _, stderr = self._ssh_exec_with_retry(NFS_SERVER, cmd)
            if rc != 0:
                log.error("NFS volume provision failed for %s: %s", volume_id, stderr)
                return False
            log.info("NFS storage created for volume %s at %s:%s", volume_id, NFS_SERVER, vol_path)
            return True

        # ── Encrypted volume: LUKS2 loopback ──────────────────────────
        if not raw_key:
            log.error("Cannot provision encrypted volume %s without a key", volume_id)
            return False

        img_path = f"{vol_path}.img"
        safe_img = shlex.quote(img_path)
        mapper = f"xcelsior-vol-{volume_id[:12]}"
        safe_mapper = shlex.quote(mapper)

        # Step 0: Clean up stale LUKS state from any previous failed attempt
        self._ssh_exec_with_retry(
            NFS_SERVER,
            f"sudo umount -l {safe_path} 2>/dev/null; "
            f"sudo cryptsetup luksClose {safe_mapper} 2>/dev/null; true",
        )

        # Step 1: Create sparse image file
        rc, _, err = self._ssh_exec_with_retry(
            NFS_SERVER,
            f"truncate -s {size_gb}G {safe_img}",
        )
        if rc != 0:
            log.error("LUKS truncate failed for %s: %s", volume_id, err)
            return False

        # Step 2: luksFormat — key piped via stdin
        # --key-size 512 = AES-256-XTS (256-bit AES + 256-bit XTS tweak)
        rc, _, err = self._ssh_exec_with_stdin(
            NFS_SERVER,
            f"sudo cryptsetup luksFormat --batch-mode --type luks2 "
            f"--key-file /dev/stdin --key-size 512 "
            f"--cipher aes-xts-plain64 --hash sha256 {safe_img}",
            raw_key,
            timeout=120,
        )
        if rc != 0:
            log.error("LUKS luksFormat failed for %s: %s", volume_id, err)
            self._ssh_exec_with_retry(NFS_SERVER, f"rm -f {safe_img}")
            return False

        # Step 3: luksOpen — key piped via stdin
        rc, _, err = self._ssh_exec_with_stdin(
            NFS_SERVER,
            f"sudo cryptsetup luksOpen --key-file /dev/stdin {safe_img} {safe_mapper}",
            raw_key,
            timeout=60,
        )
        if rc != 0:
            log.error("LUKS luksOpen failed for %s: %s", volume_id, err)
            self._ssh_exec_with_retry(NFS_SERVER, f"rm -f {safe_img}")
            return False

        # Step 4: mkfs.ext4
        rc, _, err = self._ssh_exec_with_retry(
            NFS_SERVER,
            f"sudo mkfs.ext4 -q -L vol-{volume_id[:8]} /dev/mapper/{safe_mapper}",
        )
        if rc != 0:
            log.error("LUKS mkfs failed for %s: %s", volume_id, err)
            self._ssh_exec_with_retry(
                NFS_SERVER, f"sudo cryptsetup luksClose {safe_mapper} 2>/dev/null; rm -f {safe_img}"
            )
            return False

        # Step 5: Mount
        rc, _, err = self._ssh_exec_with_retry(
            NFS_SERVER,
            f"mkdir -p {safe_path} && sudo mount /dev/mapper/{safe_mapper} {safe_path} && chmod 1777 {safe_path}",
        )
        if rc != 0:
            log.error("LUKS mount failed for %s: %s", volume_id, err)
            self._ssh_exec_with_retry(
                NFS_SERVER,
                f"sudo cryptsetup luksClose {safe_mapper} 2>/dev/null; rm -f {safe_img}; rmdir {safe_path} 2>/dev/null",
            )
            return False

        log.info(
            "Encrypted NFS storage created for volume %s: LUKS2 %dGB at %s:%s",
            volume_id,
            size_gb,
            NFS_SERVER,
            vol_path,
        )
        return True

    def retry_provision(self, volume_id: str, owner_id: str) -> dict:
        """Retry provisioning for a volume stuck in 'error' status.

        Only the owner can retry. Transitions error → provisioning → available/error.
        For encrypted volumes, re-uses the existing key from the DB (or generates new).
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT volume_id, owner_id, status, size_gb, encrypted, key_ciphertext FROM volumes WHERE volume_id = %s",
                (volume_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"Volume {volume_id} not found")
            if row["owner_id"] != owner_id:
                raise PermissionError("Not authorised to retry this volume")
            if row["status"] != "error":
                raise ValueError(f"Volume is '{row['status']}', not 'error' — cannot retry")

            self._transition_status(conn, volume_id, "provisioning", current="error")

        # Prepare encryption key if needed
        raw_key = None
        is_encrypted = bool(row["encrypted"])
        if is_encrypted:
            if row["key_ciphertext"]:
                raw_key = self._decrypt_key(row["key_ciphertext"])
            else:
                raw_key = self._generate_volume_key()

        provision_ok = self._provision_volume_storage(
            volume_id,
            row["size_gb"],
            encrypted=is_encrypted,
            raw_key=raw_key,
        )

        with self._conn() as conn:
            new_status = "available" if provision_ok else "error"
            self._transition_status(conn, volume_id, new_status, current="provisioning")
            # Store key if we generated a new one and provisioning succeeded
            if provision_ok and raw_key and not row["key_ciphertext"]:
                self._store_key(conn, volume_id, raw_key)

        if not provision_ok:
            raise RuntimeError("Re-provisioning failed — NFS server may be unavailable")

        self._emit_event("volume.retried", volume_id, actor=owner_id)
        return {"volume_id": volume_id, "status": "available"}

    # Public-facing columns (excludes encryption_key_id, mount_path_host, deleted_at)
    _PUBLIC_COLS = (
        "volume_id, owner_id, name, storage_type, size_gb, "
        "region, province, encrypted, status, created_at"
    )

    def get_volume(self, volume_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT {self._PUBLIC_COLS} FROM volumes WHERE volume_id = %s AND status != 'deleted'",
                (volume_id,),
            ).fetchone()
            if not row:
                return None
            vol = dict(row)
            # Enrich with current attachment info
            att = conn.execute(
                "SELECT instance_id FROM volume_attachments WHERE volume_id = %s AND detached_at = 0 LIMIT 1",
                (volume_id,),
            ).fetchone()
            vol["attached_to"] = att["instance_id"] if att else None
            return vol

    def rename_volume(self, volume_id: str, owner_id: str, new_name: str) -> dict:
        """Rename a volume. Only the owner can rename."""
        new_name = (new_name or "").strip()
        if not new_name:
            raise ValueError("Volume name is required")
        if len(new_name) > 128:
            raise ValueError("Volume name must be 128 characters or fewer")
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$", new_name):
            raise ValueError(
                "Volume name must start with alphanumeric and contain only letters, digits, hyphens, underscores, and dots"
            )

        with self._conn() as conn:
            row = conn.execute(
                "SELECT volume_id, owner_id, name, status FROM volumes WHERE volume_id = %s AND status != 'deleted'",
                (volume_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"Volume {volume_id} not found")
            if row["owner_id"] != owner_id:
                raise PermissionError("Not authorised to rename this volume")
            # Check name uniqueness for this owner
            existing = conn.execute(
                "SELECT volume_id FROM volumes WHERE owner_id = %s AND name = %s AND status != 'deleted' AND volume_id != %s",
                (owner_id, new_name, volume_id),
            ).fetchone()
            if existing:
                raise ValueError(f"Volume name '{new_name}' already exists")
            conn.execute(
                "UPDATE volumes SET name = %s WHERE volume_id = %s",
                (new_name, volume_id),
            )

        log.info("Volume renamed: %s → %s (owner=%s)", volume_id, new_name, owner_id)
        self._emit_event("volume.renamed", volume_id, actor=owner_id, data={"name": new_name})
        return {"volume_id": volume_id, "name": new_name}

    def list_volumes(self, owner_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {self._PUBLIC_COLS} FROM volumes WHERE owner_id = %s AND status != 'deleted' ORDER BY created_at DESC",
                (owner_id,),
            ).fetchall()
            vols = [dict(r) for r in rows]
            if not vols:
                return vols
            # Batch-fetch current attachments
            vol_ids = [v["volume_id"] for v in vols]
            atts = conn.execute(
                "SELECT volume_id, instance_id FROM volume_attachments WHERE volume_id = ANY(%s) AND detached_at = 0",
                (vol_ids,),
            ).fetchall()
            att_map = {a["volume_id"]: a["instance_id"] for a in atts}
            for v in vols:
                v["attached_to"] = att_map.get(v["volume_id"])
            return vols

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

            # Validate transition via state machine
            current = vol["status"]
            allowed = self._VALID_TRANSITIONS.get(current, set())
            if "deleting" not in allowed:
                raise ValueError(
                    f"Cannot delete volume in '{current}' state "
                    f"(allowed transitions: {', '.join(sorted(allowed)) or 'none'})"
                )
            conn.execute(
                "UPDATE volumes SET status = 'deleting', deleted_at = %s WHERE volume_id = %s",
                (time.time(), volume_id),
            )

        # Destroy actual storage on the NFS server
        destroyed = self._destroy_volume_storage(volume_id, encrypted=bool(vol.get("encrypted")))
        if not destroyed:
            # Revert from 'deleting' back to 'available' — storage still exists
            with self._conn() as conn:
                self._transition_status(conn, volume_id, "available", current="deleting")
                conn.execute(
                    "UPDATE volumes SET deleted_at = 0 WHERE volume_id = %s",
                    (volume_id,),
                )
            raise RuntimeError(f"Failed to destroy storage for volume {volume_id}")

        with self._conn() as conn:
            self._transition_status(conn, volume_id, "deleted", current="deleting")
            if vol.get("encrypted"):
                self._destroy_key(conn, volume_id)
            # Cascade: soft-delete snapshot rows so they don't appear in
            # listings or get restored to a non-existent volume.
            conn.execute(
                "UPDATE volume_snapshots SET deleted_at = %s "
                "WHERE volume_id = %s AND deleted_at = 0",
                (time.time(), volume_id),
            )

        # Best-effort: remove the snapshot dir on NFS too. Ignored on failure
        # because the rows are already soft-deleted and the next storage
        # janitor pass can sweep orphans.
        if NFS_SERVER:
            snap_dir = f"{NFS_EXPORT_BASE}/_snapshots/{volume_id}"
            try:
                self._ssh_exec_with_retry(
                    NFS_SERVER,
                    f"case {shlex.quote(snap_dir)} in "
                    f"{shlex.quote(NFS_EXPORT_BASE)}/_snapshots/*) "
                    f"rm -rf --one-file-system {shlex.quote(snap_dir)};; "
                    f"esac",
                )
            except Exception as e:
                log.warning("Snapshot dir cleanup failed for %s: %s", volume_id, e)

        log.info("Volume deleted: %s", volume_id)
        self._emit_event("volume.deleted", volume_id, actor=owner_id)
        return {"volume_id": volume_id, "status": "deleted"}

    def _destroy_volume_storage(self, volume_id: str, *, encrypted: bool = False) -> bool:
        """Remove the volume storage from the NFS server.

        For encrypted volumes: unmount, close LUKS, delete image file.
        For unencrypted: remove directory.
        """
        if not NFS_SERVER:
            log.warning(
                "XCELSIOR_NFS_SERVER not configured — skipping storage deletion for %s", volume_id
            )
            return True

        vol_path = f"{NFS_EXPORT_BASE}/{volume_id}"
        safe_path = shlex.quote(vol_path)

        if encrypted:
            img_path = f"{vol_path}.img"
            safe_img = shlex.quote(img_path)
            mapper = f"xcelsior-vol-{volume_id[:12]}"
            safe_mapper = shlex.quote(mapper)

            # Unmount (lazy to handle busy)
            self._ssh_exec_with_retry(NFS_SERVER, f"sudo umount -l {safe_path} 2>/dev/null; true")
            # Close LUKS device
            self._ssh_exec_with_retry(
                NFS_SERVER, f"sudo cryptsetup luksClose {safe_mapper} 2>/dev/null; true"
            )
            # Remove backing image and mount point
            rc, _, err = self._ssh_exec_with_retry(
                NFS_SERVER,
                f"rm -f {safe_img} && rmdir {safe_path} 2>/dev/null; true",
            )
            if rc != 0:
                log.error("NFS encrypted volume cleanup failed for %s: %s", volume_id, err)
                return False
            log.info(
                "Encrypted NFS storage destroyed for volume %s (LUKS + image removed)", volume_id
            )
            return True

        # Unencrypted: safe rm -rf with symlink traversal guard
        cmd = (
            f"real=$(readlink -f {safe_path}) && "
            f'[[ "$real" == {shlex.quote(NFS_EXPORT_BASE)}/* ]] && '
            f'rm -rf --one-file-system "$real"'
        )
        rc, _, stderr = self._ssh_exec_with_retry(NFS_SERVER, cmd)
        if rc != 0:
            log.error("NFS volume deletion failed for %s: %s", volume_id, stderr)
            return False
        log.info("NFS storage destroyed for volume %s", volume_id)
        return True

    def reopen_luks_volume(self, volume_id: str) -> bool:
        """Re-open and mount a LUKS volume after NFS server reboot.

        Retrieves the encryption key from the database, opens the LUKS
        device, and re-mounts it at the export path.
        """
        if not NFS_SERVER:
            return False
        with self._conn() as conn:
            raw_key = self._retrieve_key(conn, volume_id)
        if not raw_key:
            log.error("Cannot reopen volume %s: no encryption key in database", volume_id)
            return False

        vol_path = f"{NFS_EXPORT_BASE}/{volume_id}"
        img_path = f"{vol_path}.img"
        mapper = f"xcelsior-vol-{volume_id[:12]}"
        safe_img = shlex.quote(img_path)
        safe_path = shlex.quote(vol_path)
        safe_mapper = shlex.quote(mapper)

        # luksOpen — key piped via stdin
        rc, _, err = self._ssh_exec_with_stdin(
            NFS_SERVER,
            f"sudo cryptsetup luksOpen --key-file /dev/stdin {safe_img} {safe_mapper}",
            raw_key,
        )
        if rc != 0:
            log.error("luksOpen failed for %s: %s", volume_id, err)
            return False

        # Mount
        rc, _, err = self._ssh_exec_with_retry(
            NFS_SERVER,
            f"mkdir -p {safe_path} && sudo mount /dev/mapper/{safe_mapper} {safe_path}",
        )
        if rc != 0:
            log.error("Mount failed for %s: %s", volume_id, err)
            return False

        log.info("Volume %s LUKS device reopened and mounted successfully", volume_id)
        return True

    # ── P2.5 Snapshots ────────────────────────────────────────────────
    # Instant copy-on-write snapshots via `cp --reflink=auto`. On XFS or
    # btrfs this is O(1); on ext4 it falls back to a sparse copy (still no
    # rsync traffic across the wire, since everything happens server-side
    # on the NFS host). Snapshots require the volume to be detached so the
    # on-disk state is consistent — matches the RunPod/Vast model.

    def create_snapshot(
        self, volume_id: str, owner_id: str, label: str = ""
    ) -> dict:
        """Create an instant CoW snapshot. Volume must be detached."""
        # Soft cap: prevents runaway snapshot growth (each one consumes a
        # row + a server-side reflink that may diverge over time). Matches
        # RunPod's per-volume snapshot ceiling.
        MAX_SNAPSHOTS_PER_VOLUME = 50
        label = (label or "").strip()[:128]
        with self._conn() as conn:
            vol = conn.execute(
                "SELECT volume_id, owner_id, status, encrypted "
                "FROM volumes WHERE volume_id=%s AND deleted_at=0",
                (volume_id,),
            ).fetchone()
            if not vol or vol["owner_id"] != owner_id:
                raise ValueError("Volume not found")
            if vol["status"] != "available":
                raise ValueError(
                    f"Volume must be detached to snapshot (current status: {vol['status']})"
                )
            count_row = conn.execute(
                "SELECT COUNT(*) AS n FROM volume_snapshots "
                "WHERE volume_id=%s AND deleted_at=0",
                (volume_id,),
            ).fetchone()
            if count_row and count_row["n"] >= MAX_SNAPSHOTS_PER_VOLUME:
                raise ValueError(
                    f"Snapshot limit reached ({MAX_SNAPSHOTS_PER_VOLUME}); delete old snapshots first"
                )

        snap_id = f"snap-{uuid.uuid4().hex[:16]}"
        created_at = time.time()
        size_bytes = 0

        if NFS_SERVER:
            snap_dir = f"{NFS_EXPORT_BASE}/_snapshots/{volume_id}"
            self._ssh_exec_with_retry(
                NFS_SERVER, f"mkdir -p {shlex.quote(snap_dir)}"
            )
            if vol["encrypted"]:
                src = f"{NFS_EXPORT_BASE}/{volume_id}.img"
                dst = f"{snap_dir}/{snap_id}.img"
                cmd = (
                    f"cp --reflink=auto --sparse=always "
                    f"{shlex.quote(src)} {shlex.quote(dst)} && "
                    f"stat -c%s {shlex.quote(dst)}"
                )
            else:
                src = f"{NFS_EXPORT_BASE}/{volume_id}"
                dst = f"{snap_dir}/{snap_id}"
                cmd = (
                    f"cp -a --reflink=auto {shlex.quote(src)} {shlex.quote(dst)} && "
                    f"du -sb {shlex.quote(dst)} | cut -f1"
                )
            rc, out, err = self._ssh_exec_with_retry(NFS_SERVER, cmd, timeout=300)
            if rc != 0:
                raise RuntimeError(f"Snapshot failed: {err.strip()[:200]}")
            try:
                size_bytes = int(out.strip().split()[0])
            except (ValueError, IndexError):
                size_bytes = 0

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO volume_snapshots "
                "(snapshot_id, volume_id, owner_id, label, size_bytes, status, created_at) "
                "VALUES (%s, %s, %s, %s, %s, 'ready', %s)",
                (snap_id, volume_id, owner_id, label, size_bytes, created_at),
            )
        self._emit_event(
            "volume.snapshot.created",
            volume_id,
            actor=owner_id,
            data={"snapshot_id": snap_id, "label": label, "size_bytes": size_bytes},
        )
        log.info("Snapshot %s created for volume %s (%d bytes)", snap_id, volume_id, size_bytes)
        return {
            "snapshot_id": snap_id,
            "volume_id": volume_id,
            "label": label,
            "size_bytes": size_bytes,
            "status": "ready",
            "created_at": created_at,
        }

    def list_snapshots(self, volume_id: str, owner_id: str) -> list[dict]:
        with self._conn() as conn:
            vol = conn.execute(
                "SELECT owner_id FROM volumes WHERE volume_id=%s AND deleted_at=0",
                (volume_id,),
            ).fetchone()
            if not vol or vol["owner_id"] != owner_id:
                raise PermissionError("Volume not found")
            rows = conn.execute(
                "SELECT snapshot_id, volume_id, label, size_bytes, status, created_at "
                "FROM volume_snapshots WHERE volume_id=%s AND deleted_at=0 "
                "ORDER BY created_at DESC",
                (volume_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def restore_snapshot(
        self, volume_id: str, owner_id: str, snapshot_id: str
    ) -> dict:
        """Restore volume data from a snapshot. Volume must be detached.

        Current data is preserved as a `.pre-restore-{ts}` sibling so the
        operator can recover if the wrong snapshot was chosen.
        """
        with self._conn() as conn:
            vol = conn.execute(
                "SELECT volume_id, owner_id, status, encrypted "
                "FROM volumes WHERE volume_id=%s AND deleted_at=0",
                (volume_id,),
            ).fetchone()
            if not vol or vol["owner_id"] != owner_id:
                raise ValueError("Volume not found")
            if vol["status"] != "available":
                raise ValueError(
                    f"Volume must be detached to restore (current status: {vol['status']})"
                )
            snap = conn.execute(
                "SELECT snapshot_id FROM volume_snapshots "
                "WHERE snapshot_id=%s AND volume_id=%s AND deleted_at=0",
                (snapshot_id, volume_id),
            ).fetchone()
            if not snap:
                raise ValueError("Snapshot not found")

        if NFS_SERVER:
            snap_dir = f"{NFS_EXPORT_BASE}/_snapshots/{volume_id}"
            ts = int(time.time())
            if vol["encrypted"]:
                src = f"{snap_dir}/{snapshot_id}.img"
                dst = f"{NFS_EXPORT_BASE}/{volume_id}.img"
                backup = f"{dst}.pre-restore-{ts}"
                cmd = (
                    f"cp --reflink=auto {shlex.quote(dst)} {shlex.quote(backup)} && "
                    f"cp --reflink=auto --sparse=always "
                    f"{shlex.quote(src)} {shlex.quote(dst)}"
                )
            else:
                src = f"{snap_dir}/{snapshot_id}"
                dst = f"{NFS_EXPORT_BASE}/{volume_id}"
                backup = f"{dst}.pre-restore-{ts}"
                # `mv -T` (no-target-dir) avoids the trap where if `backup`
                # exists as a dir, mv would nest inside it. If `dst` is
                # missing for any reason, fall through to the cp-only path.
                cmd = (
                    f"if [ -e {shlex.quote(dst)} ]; then "
                    f"mv -T {shlex.quote(dst)} {shlex.quote(backup)}; fi && "
                    f"cp -a --reflink=auto {shlex.quote(src)} {shlex.quote(dst)}"
                )
            rc, _, err = self._ssh_exec_with_retry(NFS_SERVER, cmd, timeout=600)
            if rc != 0:
                raise RuntimeError(f"Restore failed: {err.strip()[:200]}")
        self._emit_event(
            "volume.snapshot.restored",
            volume_id,
            actor=owner_id,
            data={"snapshot_id": snapshot_id},
        )
        log.info("Volume %s restored from snapshot %s", volume_id, snapshot_id)
        return {"volume_id": volume_id, "snapshot_id": snapshot_id, "status": "restored"}

    def delete_snapshot(
        self, volume_id: str, owner_id: str, snapshot_id: str
    ) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT vs.snapshot_id, v.owner_id AS vol_owner, v.encrypted "
                "FROM volume_snapshots vs JOIN volumes v ON v.volume_id = vs.volume_id "
                "WHERE vs.snapshot_id=%s AND vs.volume_id=%s AND vs.deleted_at=0",
                (snapshot_id, volume_id),
            ).fetchone()
            if not row or row["vol_owner"] != owner_id:
                raise ValueError("Snapshot not found")

        if NFS_SERVER:
            snap_dir = f"{NFS_EXPORT_BASE}/_snapshots/{volume_id}"
            if row["encrypted"]:
                path = f"{snap_dir}/{snapshot_id}.img"
                cmd = f"rm -f {shlex.quote(path)}"
            else:
                # Guard: only delete paths under the snapshot dir.
                path = f"{snap_dir}/{snapshot_id}"
                cmd = (
                    f"case {shlex.quote(path)} in "
                    f"{shlex.quote(NFS_EXPORT_BASE)}/_snapshots/*) "
                    f"rm -rf --one-file-system {shlex.quote(path)};; "
                    f"*) echo refuse; exit 1;; esac"
                )
            self._ssh_exec_with_retry(NFS_SERVER, cmd)

        with self._conn() as conn:
            conn.execute(
                "UPDATE volume_snapshots SET deleted_at=%s WHERE snapshot_id=%s",
                (time.time(), snapshot_id),
            )
        self._emit_event(
            "volume.snapshot.deleted",
            volume_id,
            actor=owner_id,
            data={"snapshot_id": snapshot_id},
        )
        return {"snapshot_id": snapshot_id, "status": "deleted"}

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
            self._transition_status(conn, volume_id, "attached", current="available")

        # Mount the NFS share on the instance host
        mount_ok = self._mount_on_host(volume_id, instance_id, mount_path, mode)
        if not mount_ok:
            # Rollback: remove attachment record
            with self._conn() as conn:
                conn.execute(
                    "DELETE FROM volume_attachments WHERE attachment_id = %s",
                    (attachment_id,),
                )
                self._transition_status(conn, volume_id, "available", current="attached")
            raise ValueError(f"Failed to mount volume on instance host — attachment rolled back")

        log.info("Volume %s attached to %s at %s (%s)", volume_id, instance_id, mount_path, mode)
        self._emit_event(
            "volume.attached",
            volume_id,
            data={
                "instance_id": instance_id,
                "mount_path": mount_path,
                "mode": mode,
            },
        )
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
            log.warning(
                "XCELSIOR_NFS_SERVER not configured — skipping real mount for %s", volume_id
            )
            return True  # Allow metadata-only in dev/test

        host_ip = self._get_instance_host_ip(instance_id)
        if not host_ip:
            log.error(
                "Cannot mount volume %s: instance %s has no host IP (not running?)",
                volume_id,
                instance_id,
            )
            return False

        nfs_src = f"{NFS_SERVER}:{NFS_EXPORT_BASE}/{volume_id}"
        safe_mount = shlex.quote(mount_path)
        safe_src = shlex.quote(nfs_src)
        ro_flag = ",ro" if mode == "ro" else ""
        cmd = (
            f"mkdir -p {safe_mount} && "
            f"mount -t nfs -o {NFS_MOUNT_OPTS}{ro_flag} {safe_src} {safe_mount}"
        )
        rc, _, stderr = self._ssh_exec_with_retry(host_ip, cmd)
        if rc != 0:
            log.error("NFS mount failed for volume %s on host %s: %s", volume_id, host_ip, stderr)
            return False
        log.info("NFS mounted %s on %s at %s", volume_id, host_ip, mount_path)
        return True

    def detach_volume(self, volume_id: str, instance_id: str = None) -> dict:
        """Detach a volume from an instance.

        If instance_id is None, finds and detaches the active attachment.
        Unmounts the NFS share on the host, then updates DB records atomically.
        """
        now = time.time()
        with self._conn() as conn:
            if instance_id:
                att = conn.execute(
                    """SELECT * FROM volume_attachments
                       WHERE volume_id = %s AND instance_id = %s AND detached_at = 0 FOR UPDATE""",
                    (volume_id, instance_id),
                ).fetchone()
            else:
                att = conn.execute(
                    """SELECT * FROM volume_attachments
                       WHERE volume_id = %s AND detached_at = 0 FOR UPDATE""",
                    (volume_id,),
                ).fetchone()
            if not att:
                raise ValueError("No active attachment found")

            instance_id = att["instance_id"]
            mount_path = att.get("mount_path", DEFAULT_MOUNT_PATH)

            # Best-effort unmount (failures must not block DB cleanup)
            try:
                self._unmount_from_host(volume_id, instance_id, mount_path)
            except Exception as e:
                log.warning(
                    "Unmount failed for volume %s on instance %s: %s", volume_id, instance_id, e
                )

            # Atomic DB update within same transaction
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
                self._transition_status(conn, volume_id, "available", current="attached")

        log.info("Volume %s detached from %s", volume_id, instance_id)
        self._emit_event("volume.detached", volume_id, data={"instance_id": instance_id})
        return {"volume_id": volume_id, "instance_id": instance_id, "status": "detached"}

    def _unmount_from_host(self, volume_id: str, instance_id: str, mount_path: str) -> bool:
        """NFS-unmount the volume on the host where the instance runs."""
        if not NFS_SERVER:
            log.warning(
                "XCELSIOR_NFS_SERVER not configured — skipping real unmount for %s", volume_id
            )
            return True

        host_ip = self._get_instance_host_ip(instance_id)
        if not host_ip:
            log.warning(
                "Cannot unmount volume %s: instance %s has no host IP (already stopped?)",
                volume_id,
                instance_id,
            )
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

    def get_volume_host_ids(self, volume_ids: list[str]) -> set[str]:
        """Return set of host_ids where any of the given volumes are currently attached.

        Queries volume_attachments → jobs → hosts to find which hosts
        have these volumes mounted. Used for data-gravity scheduling.
        """
        if not volume_ids:
            return set()
        host_ids: set[str] = set()
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT DISTINCT j.host_id
                   FROM volume_attachments va
                   JOIN jobs j ON j.job_id = va.instance_id
                   WHERE va.volume_id = ANY(%s)
                     AND va.detached_at = 0
                     AND j.host_id IS NOT NULL
                     AND j.status = 'running'""",
                (list(volume_ids),),
            ).fetchall()
            for r in rows:
                if r["host_id"]:
                    host_ids.add(r["host_id"])
        return host_ids

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

        Unmounts each NFS share on the host (best-effort), then marks
        all attachments as detached and volumes as available atomically.
        """
        now = time.time()
        detached = 0
        with self._conn() as conn:
            atts = conn.execute(
                "SELECT volume_id, attachment_id, mount_path FROM volume_attachments WHERE instance_id = %s AND detached_at = 0",
                (instance_id,),
            ).fetchall()

            # Best-effort unmount (failures must not block DB cleanup)
            for att in atts:
                mount_path = att.get("mount_path", DEFAULT_MOUNT_PATH)
                try:
                    self._unmount_from_host(att["volume_id"], instance_id, mount_path)
                except Exception as e:
                    log.warning(
                        "Unmount failed for volume %s on instance %s: %s",
                        att["volume_id"],
                        instance_id,
                        e,
                    )

            # Atomic DB update — all volumes detached in one transaction
            for att in atts:
                conn.execute(
                    "UPDATE volume_attachments SET detached_at = %s WHERE attachment_id = %s",
                    (now, att["attachment_id"]),
                )
                # Only set volume back to available if no other active attachments remain
                remaining = conn.execute(
                    "SELECT COUNT(*) as cnt FROM volume_attachments WHERE volume_id = %s AND detached_at = 0",
                    (att["volume_id"],),
                ).fetchone()
                if remaining["cnt"] == 0:
                    self._transition_status(conn, att["volume_id"], "available", current="attached")
                detached += 1
            conn.commit()
        if detached:
            log.info("Detached %d volumes from instance %s", detached, instance_id)
        return detached

    def cleanup_stale_volumes(self, max_age_seconds: int = 600) -> int:
        """Clean up volumes stuck in transient states.

        - 'provisioning' for too long → 'error' (user can retry)
        - 'deleting' for too long → 'error' (user can retry delete)

        Volumes crash between INSERT and NFS export, or between NFS delete
        and DB update, stay in transient states forever. This sweeps them.
        """
        cutoff = time.time() - max_age_seconds
        total = 0
        with self._conn() as conn:
            r1 = conn.execute(
                "UPDATE volumes SET status = 'error' WHERE status = 'provisioning' AND created_at < %s",
                (cutoff,),
            )
            total += r1.rowcount
            r2 = conn.execute(
                "UPDATE volumes SET status = 'error' WHERE status = 'deleting' AND deleted_at < %s AND deleted_at > 0",
                (cutoff,),
            )
            total += r2.rowcount
            conn.commit()
        if total:
            log.warning(
                "Cleaned up %d stale volumes (provisioning=%d, deleting=%d, age > %ds)",
                total,
                r1.rowcount,
                r2.rowcount,
                max_age_seconds,
            )
        return total

    # Keep old name as alias for backward compatibility
    cleanup_stale_provisioning = cleanup_stale_volumes

    def reconcile_orphaned_attachments(self) -> int:
        """Find and fix orphaned volume states.

        Two cases:
        1. Volume status = 'attached' but no active attachment rows exist
           → set back to 'available'
        2. Active attachment rows referencing jobs in terminal states
           (completed, failed, terminated, cancelled) → close the attachment

        Called from the billing cycle to self-heal before charging.
        """
        fixed = 0
        now = time.time()
        with self._conn() as conn:
            # Case 1: volumes marked 'attached' with zero active attachments
            orphaned_vols = conn.execute(
                """SELECT v.volume_id FROM volumes v
                   WHERE v.status = 'attached'
                   AND NOT EXISTS (
                       SELECT 1 FROM volume_attachments a
                       WHERE a.volume_id = v.volume_id AND a.detached_at = 0
                   )""",
            ).fetchall()
            for ov in orphaned_vols:
                conn.execute(
                    "UPDATE volumes SET status = 'available' WHERE volume_id = %s",
                    (ov["volume_id"],),
                )
                log.warning(
                    "Reconciled orphaned volume %s: attached→available (no active attachments)",
                    ov["volume_id"],
                )
                fixed += 1

            # Case 2: active attachments to dead instances
            stale_atts = conn.execute(
                """SELECT a.attachment_id, a.volume_id, a.instance_id
                   FROM volume_attachments a
                   JOIN jobs j ON j.job_id = a.instance_id
                   WHERE a.detached_at = 0
                   AND j.status IN ('completed', 'failed', 'terminated', 'cancelled')""",
            ).fetchall()
            for sa in stale_atts:
                conn.execute(
                    "UPDATE volume_attachments SET detached_at = %s WHERE attachment_id = %s",
                    (now, sa["attachment_id"]),
                )
                # Check if volume has any remaining active attachments
                remaining = conn.execute(
                    "SELECT COUNT(*) as cnt FROM volume_attachments WHERE volume_id = %s AND detached_at = 0",
                    (sa["volume_id"],),
                ).fetchone()
                if remaining["cnt"] == 0:
                    conn.execute(
                        "UPDATE volumes SET status = 'available' WHERE volume_id = %s AND status = 'attached'",
                        (sa["volume_id"],),
                    )
                log.warning(
                    "Reconciled stale attachment %s: volume %s was attached to dead instance %s",
                    sa["attachment_id"],
                    sa["volume_id"],
                    sa["instance_id"],
                )
                fixed += 1
            conn.commit()
        if fixed:
            log.info("Reconciled %d orphaned volume attachments", fixed)
        return fixed


# ── Singleton ─────────────────────────────────────────────────────────

_volume_engine: Optional[VolumeEngine] = None


def get_volume_engine() -> VolumeEngine:
    global _volume_engine
    if _volume_engine is None:
        _volume_engine = VolumeEngine()
    return _volume_engine
