# Xcelsior S3-Compatible Artifact Storage
# Implements REPORT_FEATURE_FINAL.md:
#   - Presigned URLs as the battle-tested artifact distribution primitive
#   - Two-tier storage: B2 (Canada East/Toronto) for residency, R2 for cache
#   - Uniform S3 interface for future self-hosted migration
#   - Workers never get long-lived storage credentials
#
# Storage backends:
#   - Backblaze B2 (CA East / Toronto): Canada-resident, compliance-ready
#   - Cloudflare R2: Zero egress, ideal for public/cached artifacts
#   - Local/MinIO: Self-hosted fallback for development
#
# Artifact types:
#   - Model weights (input)
#   - Job outputs (results, checkpoints)
#   - Container images (OCI layers)
#   - Logs and telemetry exports

import hashlib
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

log = logging.getLogger("xcelsior")


# ── Storage Backends ─────────────────────────────────────────────────


class StorageBackend(str, Enum):
    B2 = "b2"  # Backblaze B2 — Canada East (Toronto)
    R2 = "r2"  # Cloudflare R2 — zero egress
    LOCAL = "local"  # Local filesystem / MinIO
    S3 = "s3"  # Generic S3-compatible


class StorageUnavailable(RuntimeError):
    """A configured remote artifact backend cannot be reached or
    initialized (data-architecture companion §6.6). Raised instead of
    silently degrading to local ``file://`` storage — a misconfigured
    production backend must fail loudly, never masquerade as healthy."""


class ArtifactType(str, Enum):
    MODEL_WEIGHTS = "model_weights"
    JOB_OUTPUT = "job_output"
    CHECKPOINT = "checkpoint"
    CONTAINER_IMAGE = "container_image"
    LOG = "log"
    TELEMETRY = "telemetry"
    DATASET = "dataset"


class ResidencyPolicy(str, Enum):
    """Where artifacts should be stored based on compliance requirements."""

    CANADA_ONLY = "canada_only"  # B2 CA East only
    CANADA_PREFERRED = "canada_preferred"  # B2 primary, R2 cache
    ANY = "any"  # R2 or B2, lowest cost


# ── Configuration ────────────────────────────────────────────────────


@dataclass
class StorageConfig:
    """Configuration for a storage backend."""

    backend: str = StorageBackend.LOCAL
    local_dir: str = ""
    endpoint_url: str = ""
    bucket: str = "xcelsior-artifacts"
    region: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    presign_expiry_sec: int = 3600  # 1 hour default
    max_upload_size_mb: int = 10240  # 10 GB default
    residency: str = ResidencyPolicy.ANY

    @classmethod
    def from_env(cls, prefix: str = "XCELSIOR_STORAGE") -> "StorageConfig":
        """Load configuration from environment variables."""
        return cls(
            backend=os.environ.get(f"{prefix}_BACKEND", "local"),
            local_dir=os.environ.get(f"{prefix}_LOCAL_DIR", ""),
            endpoint_url=os.environ.get(f"{prefix}_ENDPOINT_URL", ""),
            bucket=os.environ.get(f"{prefix}_BUCKET", "xcelsior-artifacts"),
            region=os.environ.get(f"{prefix}_REGION", ""),
            access_key_id=os.environ.get(f"{prefix}_ACCESS_KEY_ID", ""),
            secret_access_key=os.environ.get(f"{prefix}_SECRET_ACCESS_KEY", ""),
            presign_expiry_sec=int(os.environ.get(f"{prefix}_PRESIGN_EXPIRY", "3600")),
            max_upload_size_mb=int(os.environ.get(f"{prefix}_MAX_UPLOAD_MB", "10240")),
            residency=os.environ.get(f"{prefix}_RESIDENCY", "any"),
        )


# ── Artifact Metadata ────────────────────────────────────────────────


@dataclass
class ArtifactMeta:
    """Metadata for a stored artifact."""

    artifact_id: str = ""
    artifact_type: str = ArtifactType.JOB_OUTPUT
    key: str = ""  # S3 object key
    bucket: str = ""
    backend: str = StorageBackend.LOCAL
    size_bytes: int = 0
    sha256: str = ""
    content_type: str = "application/octet-stream"
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    job_id: str = ""
    host_id: str = ""
    owner: str = ""
    residency_region: str = ""  # Where the data physically resides
    residency_country: str = ""
    tags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Storage Client ────────────────────────────────────────────────────
# Abstracts S3-compatible operations. Uses boto3 if available,
# falls back to urllib for presigned URL generation.


class StorageClient:
    """S3-compatible storage client with presigned URL support.

    This is the uniform interface that works with B2, R2, MinIO,
    and any S3-compatible backend.
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig.from_env()
        self._client = None

    def _get_client(self):
        """Lazy-initialize the boto3 S3 client.

        Returns ``None`` ONLY for the explicit local backend (the
        development/filesystem profile). For any remote backend a missing
        SDK or a client-init failure is fatal — it raises
        :class:`StorageUnavailable` rather than returning ``None``, which
        every caller would otherwise treat as "use local file:// storage"
        (companion §6.6: no silent remote→local fallback).
        """
        if self._client is not None:
            return self._client

        # Local backend doesn't use remote S3 — filesystem path (dev profile).
        if self.config.backend == StorageBackend.LOCAL:
            return None

        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError as exc:
            raise StorageUnavailable(
                f"artifact backend is '{self.config.backend}' but boto3 is not "
                "installed; refusing to fall back to local file:// storage"
            ) from exc

        try:
            kwargs = {
                "service_name": "s3",
                "aws_access_key_id": self.config.access_key_id,
                "aws_secret_access_key": self.config.secret_access_key,
                "config": BotoConfig(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
            }
            if self.config.endpoint_url:
                kwargs["endpoint_url"] = self.config.endpoint_url
            if self.config.region:
                kwargs["region_name"] = self.config.region

            self._client = boto3.client(**kwargs)
            return self._client
        except Exception as exc:
            raise StorageUnavailable(
                f"failed to initialize '{self.config.backend}' storage client: "
                f"{exc}"
            ) from exc

    def generate_upload_url(
        self,
        key: str,
        content_type: str = "application/octet-stream",
        expiry_sec: Optional[int] = None,
        max_size_mb: Optional[int] = None,
    ) -> dict:
        """Generate a presigned URL for uploading an artifact.

        Workers use this to upload directly to storage without
        needing long-lived credentials. This is the recommended
        pattern from both AWS and Cloudflare R2 docs.

        Returns:
            {"url": str, "method": "PUT", "headers": dict,
             "expires_at": float, "key": str}
        """
        client = self._get_client()
        if not client:
            return self._local_upload_url(key)

        expiry = expiry_sec or self.config.presign_expiry_sec

        try:
            url = client.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": self.config.bucket,
                    "Key": key,
                    "ContentType": content_type,
                },
                ExpiresIn=expiry,
            )

            return {
                "url": url,
                "method": "PUT",
                "headers": {"Content-Type": content_type},
                "expires_at": time.time() + expiry,
                "key": key,
                "bucket": self.config.bucket,
                "backend": self.config.backend,
            }
        except Exception as e:
            log.error("Failed to generate upload URL: %s", e)
            raise

    def generate_download_url(
        self,
        key: str,
        expiry_sec: Optional[int] = None,
    ) -> dict:
        """Generate a presigned URL for downloading an artifact.

        Workers and clients use this to pull artifacts without
        needing storage credentials.

        Returns:
            {"url": str, "method": "GET", "expires_at": float, "key": str}
        """
        client = self._get_client()
        if not client:
            return self._local_download_url(key)

        expiry = expiry_sec or self.config.presign_expiry_sec

        try:
            url = client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.config.bucket,
                    "Key": key,
                },
                ExpiresIn=expiry,
            )

            return {
                "url": url,
                "method": "GET",
                "expires_at": time.time() + expiry,
                "key": key,
                "bucket": self.config.bucket,
                "backend": self.config.backend,
            }
        except Exception as e:
            log.error("Failed to generate download URL: %s", e)
            raise

    def head_object(self, key: str) -> Optional[dict]:
        """Check if an object exists and get its metadata."""
        client = self._get_client()
        if not client:
            return self._local_head(key)

        try:
            response = client.head_object(
                Bucket=self.config.bucket,
                Key=key,
            )
            return {
                "key": key,
                "size_bytes": response.get("ContentLength", 0),
                "content_type": response.get("ContentType", ""),
                "last_modified": response.get("LastModified"),
                "etag": response.get("ETag", "").strip('"'),
            }
        except Exception:
            return None

    def put_object(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> bool:
        """Upload object bytes server-side (API-mediated uploads)."""
        client = self._get_client()
        if not client:
            return self._local_put(key, data)

        try:
            client.put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
            return True
        except Exception as e:
            log.error("Failed to put %s: %s", key, e)
            raise

    def delete_object(self, key: str) -> bool:
        """Delete an artifact."""
        client = self._get_client()
        if not client:
            return self._local_delete(key)

        try:
            client.delete_object(
                Bucket=self.config.bucket,
                Key=key,
            )
            return True
        except Exception as e:
            log.error("Failed to delete %s: %s", key, e)
            return False

    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> list[dict]:
        """List objects with a prefix."""
        client = self._get_client()
        if not client:
            return self._local_list(prefix)

        try:
            response = client.list_objects_v2(
                Bucket=self.config.bucket,
                Prefix=prefix,
                MaxKeys=max_keys,
            )
            return [
                {
                    "key": obj["Key"],
                    "size_bytes": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                }
                for obj in response.get("Contents", [])
            ]
        except Exception as e:
            # A listing error is an error — never return an empty list that
            # a caller would report as "no artifacts" (companion §6.6).
            log.error("Failed to list objects: %s", e)
            raise StorageUnavailable(f"list_objects failed: {e}") from e

    # ── Local filesystem fallback ────────────────────────────────────

    def _local_dir(self):
        d = self.config.local_dir or os.path.join(os.path.dirname(__file__), "artifacts")
        os.makedirs(d, exist_ok=True)
        return d

    def _local_upload_url(self, key):
        path = os.path.join(self._local_dir(), key.replace("/", os.sep))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return {
            "url": f"file://{path}",
            "method": "PUT",
            "headers": {},
            "expires_at": time.time() + 3600,
            "key": key,
            "bucket": "local",
            "backend": "local",
        }

    def _local_download_url(self, key):
        path = os.path.join(self._local_dir(), key.replace("/", os.sep))
        return {
            "url": f"file://{path}",
            "method": "GET",
            "expires_at": time.time() + 3600,
            "key": key,
            "bucket": "local",
            "backend": "local",
        }

    def _local_head(self, key):
        path = os.path.join(self._local_dir(), key.replace("/", os.sep))
        if not os.path.exists(path):
            return None
        stat = os.stat(path)
        return {
            "key": key,
            "size_bytes": stat.st_size,
            "content_type": "application/octet-stream",
            "last_modified": stat.st_mtime,
        }

    def _local_put(self, key, data):
        path = os.path.join(self._local_dir(), key.replace("/", os.sep))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return True

    def _local_delete(self, key):
        path = os.path.join(self._local_dir(), key.replace("/", os.sep))
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return False

    def _local_list(self, prefix):
        base = self._local_dir()
        results = []
        for root, _, files in os.walk(base):
            for f in files:
                full = os.path.join(root, f)
                key = os.path.relpath(full, base).replace(os.sep, "/")
                if key.startswith(prefix):
                    stat = os.stat(full)
                    results.append(
                        {
                            "key": key,
                            "size_bytes": stat.st_size,
                            "last_modified": time.ctime(stat.st_mtime),
                        }
                    )
        return results


# ── Artifact Manager ─────────────────────────────────────────────────
# Higher-level interface that handles routing between storage backends
# based on residency policy, fully integrated with PostgreSQL storage catalog schema.


_CACHE_AUTO: Any = object()  # sentinel: auto-detect cache from env


class ArtifactManager:
    """Manages artifact storage with residency-aware backend routing and PostgreSQL catalog state tracking."""

    def __init__(
        self,
        primary: Optional[StorageClient] = None,
        cache: Optional[StorageClient] = _CACHE_AUTO,
    ):
        self.primary = primary or StorageClient(StorageConfig.from_env("XCELSIOR_STORAGE"))
        # Cache backend (R2) — optional
        if cache is _CACHE_AUTO:
            cache_endpoint = os.environ.get("XCELSIOR_CACHE_ENDPOINT_URL", "")
            self.cache = (
                StorageClient(StorageConfig.from_env("XCELSIOR_CACHE")) if cache_endpoint else None
            )
        else:
            self.cache = cache

    def _is_db_active(self) -> bool:
        """Check if PostgreSQL storage catalog backend is active."""
        from db import DB_BACKEND
        if DB_BACKEND != "postgres":
            return False
        try:
            from control_plane.db import control_plane_transaction
            return True
        except ImportError:
            return False

    def _make_key(
        self,
        artifact_type: str,
        job_id: str = "",
        filename: str = "",
    ) -> str:
        """Generate a structured S3 key (used primarily for legacy fallback)."""
        parts = [artifact_type]
        if job_id:
            parts.append(job_id)
        if filename:
            parts.append(filename)
        else:
            parts.append(f"{int(time.time())}_{os.urandom(4).hex()}")
        return "/".join(parts)

    def request_upload(
        self,
        artifact_type: str,
        job_id: str = "",
        filename: str = "",
        content_type: str = "application/octet-stream",
        residency: str = ResidencyPolicy.ANY,
        owner_user_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> dict:
        """Request a presigned upload URL for an artifact.

        The caller (agent/worker) uses this URL to upload directly
        to storage without ever receiving storage credentials.
        """
        import uuid

        # Resolve tenant_id and job_id for database
        tenant_id = job_id or "system-tenant"
        db_job_id = None
        if job_id and not job_id.startswith("user-"):
            db_job_id = job_id
            # Try to resolve tenant_id from jobs table
            if self._is_db_active():
                from control_plane.db import control_plane_transaction
                try:
                    with control_plane_transaction() as conn:
                        row = conn.execute(
                            "SELECT tenant_id FROM jobs WHERE job_id = %s", (job_id,)
                        ).fetchone()
                        if row:
                            tenant_id = row[0]
                except Exception:
                    pass

        # Generate artifact ID and key
        artifact_id = uuid.uuid4()
        upload_session_id = uuid.uuid4()

        # Check if the job actually exists in PostgreSQL before writing catalog tables
        db_job_exists = False
        if db_job_id and self._is_db_active():
            from control_plane.db import control_plane_transaction
            try:
                with control_plane_transaction() as conn:
                    row = conn.execute(
                        "SELECT 1 FROM jobs WHERE job_id = %s", (db_job_id,)
                    ).fetchone()
                    if row:
                        db_job_exists = True
            except Exception:
                pass

        write_to_db = self._is_db_active() and (db_job_id is None or db_job_exists)

        # Route to appropriate backend based on residency
        if residency == ResidencyPolicy.CANADA_ONLY:
            client = self.primary  # B2 CA East
        elif residency == ResidencyPolicy.CANADA_PREFERRED:
            client = self.primary  # B2 preferred, R2 fallback
        else:
            client = self.cache if self.cache else self.primary

        bucket = client.config.bucket
        provider = client.config.backend
        region = client.config.region or "ca-central-1"

        if write_to_db:
            # Use immutable key containing the random artifact_id for complete catalog isolation
            key = f"{artifact_type}/{job_id}/{artifact_id}_{filename}" if job_id else f"{artifact_type}/{artifact_id}_{filename}"
            from control_plane.db import control_plane_transaction
            idem_key = idempotency_key or f"idem-{job_id}-{filename}-{artifact_id.hex[:8]}"
            with control_plane_transaction() as conn:
                # 1. Insert into storage.artifacts (state requested)
                conn.execute(
                    """INSERT INTO storage.artifacts (
                           artifact_id, tenant_id, owner_user_id, job_id, attempt_id,
                           artifact_type, logical_name, state, primary_provider,
                           primary_bucket, object_key, content_type, residency_region,
                           retention_class, retain_until
                       ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'requested', %s, %s, %s, %s, %s, 'standard',
                                 clock_timestamp() + interval '90 days')""",
                    (
                        artifact_id,
                        tenant_id,
                        owner_user_id,
                        db_job_id,
                        uuid.UUID(attempt_id) if attempt_id else None,
                        artifact_type,
                        filename,
                        provider,
                        bucket,
                        key,
                        content_type,
                        region,
                    ),
                )
                # 2. Insert into storage.artifact_upload_sessions
                conn.execute(
                    """INSERT INTO storage.artifact_upload_sessions (
                           upload_session_id, artifact_id, tenant_id, principal_id,
                           expires_at, idempotency_key
                       ) VALUES (%s, %s, %s, %s, clock_timestamp() + interval '1 hour', %s)""",
                    (
                        upload_session_id,
                        artifact_id,
                        tenant_id,
                        owner_user_id or "system",
                        idem_key,
                    ),
                )
        else:
            # Fall back to legacy key structure if DB is not active (for simple unit tests)
            key = self._make_key(artifact_type, job_id, filename)

        result = client.generate_upload_url(key, content_type)
        result["artifact_type"] = artifact_type
        result["residency"] = residency
        result["artifact_id"] = str(artifact_id)
        result["upload_session_id"] = str(upload_session_id)
        return result

    def finalize_upload(self, session_id: str, checksum: Optional[str] = None) -> dict:
        """Finalize an upload session, perform HEAD check, and atomically update state to available."""
        from control_plane.db import control_plane_transaction
        import uuid

        try:
            sess_uuid = uuid.UUID(str(session_id))
        except ValueError:
            raise ValueError(f"Invalid upload_session_id: {session_id}")

        with control_plane_transaction() as conn:
            # 1. Look up the upload session
            sess = conn.execute(
                """SELECT artifact_id, tenant_id, principal_id, expires_at, completed_at
                   FROM storage.artifact_upload_sessions
                   WHERE upload_session_id = %s FOR UPDATE""",
                (sess_uuid,),
            ).fetchone()
            if not sess:
                raise ValueError(f"Upload session {session_id} not found")

            artifact_id, tenant_id, principal_id, expires_at, completed_at = sess
            if completed_at:
                # Already finalized, return current state
                art = conn.execute(
                    """SELECT artifact_id, logical_name, object_key, size_bytes, state
                       FROM storage.artifacts WHERE artifact_id = %s""",
                    (artifact_id,),
                ).fetchone()
                if art is None:
                    # The session's FK guarantees this row; its absence
                    # means the catalog was mutated out from under us.
                    raise RuntimeError(
                        f"Upload session {session_id} references missing "
                        f"artifact {artifact_id}"
                    )
                return {
                    "artifact_id": str(art[0]),
                    "logical_name": art[1],
                    "object_key": art[2],
                    "size_bytes": art[3],
                    "state": art[4],
                }

            # 2. Look up the artifact metadata
            art = conn.execute(
                """SELECT primary_provider, primary_bucket, object_key, logical_name, state
                   FROM storage.artifacts WHERE artifact_id = %s FOR UPDATE""",
                (artifact_id,),
            ).fetchone()
            if not art:
                raise ValueError(f"Artifact {artifact_id} not found for session {session_id}")

            provider, bucket, key, logical_name, state = art

            # Route to the appropriate client
            client = self.cache if provider == "r2" and self.cache else self.primary

            # 3. Perform provider HEAD check
            meta = client.head_object(key)
            if not meta:
                # Update artifact state to abandoned
                conn.execute(
                    "UPDATE storage.artifacts SET state = 'abandoned' WHERE artifact_id = %s",
                    (artifact_id,),
                )
                raise RuntimeError(
                    f"Object {key} not found on provider {provider} bucket {bucket}"
                )

            size_bytes = meta.get("size_bytes", 0)

            # 4. Atomically transition state to 'available'
            now_row = conn.execute("SELECT clock_timestamp()").fetchone()
            assert now_row is not None  # SELECT of a constant always returns a row
            now = now_row[0]
            conn.execute(
                """UPDATE storage.artifacts
                   SET state = 'available',
                       size_bytes = %s,
                       sha256 = COALESCE(sha256, %s),
                       available_at = %s,
                       version = version + 1
                   WHERE artifact_id = %s""",
                (size_bytes, checksum or meta.get("etag"), now, artifact_id),
            )

            # Mark session complete
            conn.execute(
                """UPDATE storage.artifact_upload_sessions
                   SET completed_at = %s
                   WHERE upload_session_id = %s""",
                (now, sess_uuid),
            )

            # Insert replica tracking
            conn.execute(
                """INSERT INTO storage.artifact_replicas (
                       artifact_id, provider, bucket, object_key, state, verified_at
                   ) VALUES (%s, %s, %s, %s, 'active', %s)
                   ON CONFLICT (artifact_id, provider, bucket) DO UPDATE
                   SET state = 'active', verified_at = %s""",
                (artifact_id, provider, bucket, key, now, now),
            )

            return {
                "artifact_id": str(artifact_id),
                "logical_name": logical_name,
                "object_key": key,
                "size_bytes": size_bytes,
                "state": "available",
            }

    def request_download(
        self,
        key: str,
        prefer_cache: bool = True,
    ) -> dict:
        """Request a presigned download URL for a specific storage key."""
        if prefer_cache and self.cache:
            meta = self.cache.head_object(key)
            if meta:
                return self.cache.generate_download_url(key)

        return self.primary.generate_download_url(key)

    def request_download_by_id(self, artifact_id: str) -> dict:
        """Request a presigned download URL for an artifact by its database ID."""
        import uuid
        try:
            art_uuid = uuid.UUID(str(artifact_id))
        except ValueError:
            raise ValueError(f"Invalid artifact_id: {artifact_id}")

        from control_plane.db import control_plane_transaction
        with control_plane_transaction() as conn:
            art = conn.execute(
                """SELECT object_key, primary_provider, primary_bucket, state, logical_name
                   FROM storage.artifacts WHERE artifact_id = %s""",
                (art_uuid,),
            ).fetchone()

        if not art:
            raise KeyError(f"Artifact {artifact_id} not found")

        key, provider, bucket, state, logical_name = art
        if state != "available":
            raise ValueError(f"Artifact {artifact_id} is not available (current state: {state})")

        # Route to appropriate client
        client = self.cache if provider == "r2" and self.cache else self.primary
        result = client.generate_download_url(key)
        result["logical_name"] = logical_name
        result["artifact_id"] = str(artifact_id)
        return result

    def download_url_for(
        self,
        artifact_type: str,
        job_id: str,
        filename: str,
        prefer_cache: bool = True,
    ) -> dict:
        """Presigned download URL for an artifact, addressed by logical name and job ID."""
        if self._is_db_active():
            from control_plane.db import control_plane_transaction
            with control_plane_transaction() as conn:
                row = conn.execute(
                    """SELECT object_key FROM storage.artifacts
                       WHERE job_id = %s AND artifact_type = %s AND logical_name = %s AND state = 'available'
                       ORDER BY created_at DESC LIMIT 1""",
                    (job_id, artifact_type, filename),
                ).fetchone()
                if row:
                    return self.request_download(row[0], prefer_cache=prefer_cache)

        # Fallback to key guess for legacy files or non-DB environments
        key = self._make_key(artifact_type, job_id, filename)
        return self.request_download(key, prefer_cache=prefer_cache)

    def get_job_artifacts(self, job_id: str) -> list[dict]:
        """List all available artifacts for a job."""
        if not self._is_db_active():
            # Legacy fallback directory walk / listing
            results = []
            for atype in ArtifactType:
                prefix = f"{atype.value}/{job_id}/"
                results.extend(self.primary.list_objects(prefix))
            return results

        # Database-backed listing
        from control_plane.db import control_plane_transaction
        with control_plane_transaction() as conn:
            if job_id.startswith("user-"):
                rows = conn.execute(
                    """SELECT artifact_id, object_key, logical_name, artifact_type, size_bytes, created_at
                       FROM storage.artifacts
                       WHERE tenant_id = %s AND job_id IS NULL AND state = 'available'
                       ORDER BY created_at DESC""",
                    (job_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT artifact_id, object_key, logical_name, artifact_type, size_bytes, created_at
                       FROM storage.artifacts
                       WHERE job_id = %s AND state = 'available'
                       ORDER BY created_at DESC""",
                    (job_id,),
                ).fetchall()

        db_results = [
            {
                "artifact_id": str(r[0]),
                "key": r[1],
                "filename": r[2],
                "artifact_type": r[3],
                "size_bytes": r[4],
                "last_modified": r[5].timestamp() if r[5] else time.time(),
            }
            for r in rows
        ]

        if not db_results:
            # Fallback to filesystem listing if database returned nothing
            # (essential for raw filesystem-based unit tests and legacy files)
            results = []
            for atype in ArtifactType:
                prefix = f"{atype.value}/{job_id}/"
                try:
                    results.extend(self.primary.list_objects(prefix))
                except Exception:
                    pass
            return results

        return db_results

    def cleanup_expired(self, older_than_sec: int = 604800) -> int:
        """Delete artifacts older than threshold. Returns count deleted."""
        # 1. Database-backed cleanup if active
        if self._is_db_active():
            from control_plane.db import control_plane_transaction
            import uuid

            # Transition expired requested/uploading upload sessions to abandoned
            with control_plane_transaction() as conn:
                expired_sessions = conn.execute(
                    """SELECT upload_session_id, artifact_id 
                       FROM storage.artifact_upload_sessions 
                       WHERE completed_at IS NULL AND expires_at < clock_timestamp()"""
                ).fetchall()

                abandoned_count = 0
                for r in expired_sessions:
                    sess_id, art_id = r[0], r[1]
                    conn.execute(
                        "UPDATE storage.artifact_upload_sessions SET completed_at = clock_timestamp() WHERE upload_session_id = %s",
                        (sess_id,)
                    )
                    conn.execute(
                        "UPDATE storage.artifacts SET state = 'abandoned' WHERE artifact_id = %s AND state IN ('requested', 'uploading')",
                        (art_id,)
                    )
                    abandoned_count += 1

                if abandoned_count > 0:
                    log.info("CLEANUP DB: marked %d expired upload sessions as abandoned", abandoned_count)

            # Process artifact_deletion_jobs
            with control_plane_transaction() as conn:
                jobs = conn.execute(
                    """SELECT deletion_id, artifact_id, reason, requested_by
                       FROM storage.artifact_deletion_jobs
                       WHERE state IN ('requested', 'delete_failed') AND next_attempt_at <= clock_timestamp()
                       LIMIT 50"""
                ).fetchall()

                for r in jobs:
                    del_id, art_id = r[0], r[1]
                    conn.execute(
                        "UPDATE storage.artifact_deletion_jobs SET state = 'claimed', attempt_count = attempt_count + 1 WHERE deletion_id = %s",
                        (del_id,)
                    )

                    try:
                        art = conn.execute(
                            "SELECT object_key, primary_provider, primary_bucket FROM storage.artifacts WHERE artifact_id = %s",
                            (art_id,)
                        ).fetchone()

                        if art:
                            key = art[0]
                            self.primary.delete_object(key)
                            conn.execute(
                                "UPDATE storage.artifacts SET state = 'deleted', deleted_at = clock_timestamp() WHERE artifact_id = %s",
                                (art_id,)
                            )
                            conn.execute(
                                "UPDATE storage.artifact_replicas SET state = 'deleted' WHERE artifact_id = %s",
                                (art_id,)
                            )

                        conn.execute(
                            "UPDATE storage.artifact_deletion_jobs SET state = 'completed', completed_at = clock_timestamp() WHERE deletion_id = %s",
                            (del_id,)
                        )
                        log.info("CLEANUP DB: processed deletion job %s for artifact %s", del_id, art_id)
                    except Exception as e:
                        next_attempt = time.time() + 300
                        conn.execute(
                            """UPDATE storage.artifact_deletion_jobs 
                               SET state = 'delete_failed', next_attempt_at = to_timestamp(%s), last_error = %s 
                               WHERE deletion_id = %s""",
                            (next_attempt, str(e), del_id)
                        )
                        log.error("CLEANUP DB: deletion job %s failed: %s", del_id, e)

        # 2. Legacy filesystem/S3-level cleanup fallback
        cutoff = time.time() - older_than_sec
        deleted = 0
        for atype in ArtifactType:
            objects = self.primary.list_objects(f"{atype.value}/")
            for obj in objects:
                if isinstance(obj.get("last_modified"), (int, float)):
                    if obj["last_modified"] < cutoff:
                        if self.primary.delete_object(obj["key"]):
                            deleted += 1
        log.info("CLEANUP: deleted %d expired artifacts", deleted)
        return deleted


# ── Singleton ─────────────────────────────────────────────────────────

_artifact_manager: Optional[ArtifactManager] = None


def get_artifact_manager() -> ArtifactManager:
    global _artifact_manager
    if _artifact_manager is None:
        _artifact_manager = ArtifactManager()
    return _artifact_manager
