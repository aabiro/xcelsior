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
from typing import Optional

log = logging.getLogger("xcelsior")


# ── Storage Backends ─────────────────────────────────────────────────

class StorageBackend(str, Enum):
    B2 = "b2"           # Backblaze B2 — Canada East (Toronto)
    R2 = "r2"           # Cloudflare R2 — zero egress
    LOCAL = "local"     # Local filesystem / MinIO
    S3 = "s3"           # Generic S3-compatible


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
    CANADA_ONLY = "canada_only"          # B2 CA East only
    CANADA_PREFERRED = "canada_preferred" # B2 primary, R2 cache
    ANY = "any"                          # R2 or B2, lowest cost


# ── Configuration ────────────────────────────────────────────────────

@dataclass
class StorageConfig:
    """Configuration for a storage backend."""
    backend: str = StorageBackend.LOCAL
    endpoint_url: str = ""
    bucket: str = "xcelsior-artifacts"
    region: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    presign_expiry_sec: int = 3600    # 1 hour default
    max_upload_size_mb: int = 10240   # 10 GB default
    residency: str = ResidencyPolicy.ANY

    @classmethod
    def from_env(cls, prefix: str = "XCELSIOR_STORAGE") -> "StorageConfig":
        """Load configuration from environment variables."""
        return cls(
            backend=os.environ.get(f"{prefix}_BACKEND", "local"),
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
    key: str = ""                     # S3 object key
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
    residency_region: str = ""        # Where the data physically resides
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
        """Lazy-initialize boto3 S3 client."""
        if self._client is not None:
            return self._client

        try:
            import boto3
            from botocore.config import Config as BotoConfig

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
        except ImportError:
            log.warning("boto3 not installed — storage operations will fail. "
                        "Install with: pip install boto3")
            return None

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
            log.error("Failed to list objects: %s", e)
            return []

    # ── Local filesystem fallback ────────────────────────────────────

    def _local_dir(self):
        d = os.path.join(os.path.dirname(__file__), "artifacts")
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
                    results.append({
                        "key": key,
                        "size_bytes": stat.st_size,
                        "last_modified": time.ctime(stat.st_mtime),
                    })
        return results


# ── Artifact Manager ─────────────────────────────────────────────────
# Higher-level interface that handles routing between storage backends
# based on residency policy.

class ArtifactManager:
    """Manages artifact storage with residency-aware backend routing.

    Two-tier strategy from the report:
    - B2 CA East for Canada-only compliance artifacts
    - R2 for egress-heavy, non-sensitive cached artifacts
    - Uniform presigned URL interface for both
    """

    def __init__(
        self,
        primary: Optional[StorageClient] = None,
        cache: Optional[StorageClient] = None,
    ):
        self.primary = primary or StorageClient(StorageConfig.from_env("XCELSIOR_STORAGE"))
        # Cache backend (R2) — optional
        cache_endpoint = os.environ.get("XCELSIOR_CACHE_ENDPOINT_URL", "")
        if cache_endpoint:
            self.cache = cache or StorageClient(StorageConfig.from_env("XCELSIOR_CACHE"))
        else:
            self.cache = None

    def _make_key(
        self,
        artifact_type: str,
        job_id: str = "",
        filename: str = "",
    ) -> str:
        """Generate a structured S3 key."""
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
    ) -> dict:
        """Request a presigned upload URL for an artifact.

        The caller (agent/worker) uses this URL to upload directly
        to storage without ever receiving storage credentials.
        """
        key = self._make_key(artifact_type, job_id, filename)

        # Route to appropriate backend based on residency
        if residency == ResidencyPolicy.CANADA_ONLY:
            client = self.primary  # B2 CA East
        elif residency == ResidencyPolicy.CANADA_PREFERRED:
            client = self.primary  # B2 preferred, R2 fallback
        else:
            client = self.cache if self.cache else self.primary

        result = client.generate_upload_url(key, content_type)
        result["artifact_type"] = artifact_type
        result["residency"] = residency
        return result

    def request_download(
        self,
        key: str,
        prefer_cache: bool = True,
    ) -> dict:
        """Request a presigned download URL.

        If the artifact exists in cache (R2), serve from there
        for zero-egress downloads. Otherwise, serve from primary (B2).
        """
        if prefer_cache and self.cache:
            meta = self.cache.head_object(key)
            if meta:
                return self.cache.generate_download_url(key)

        return self.primary.generate_download_url(key)

    def get_job_artifacts(self, job_id: str) -> list[dict]:
        """List all artifacts for a job."""
        results = []
        for atype in ArtifactType:
            prefix = f"{atype.value}/{job_id}/"
            results.extend(self.primary.list_objects(prefix))
        return results

    def cleanup_expired(self, older_than_sec: int = 604800) -> int:
        """Delete artifacts older than threshold. Returns count deleted."""
        cutoff = time.time() - older_than_sec
        deleted = 0
        for atype in ArtifactType:
            objects = self.primary.list_objects(f"{atype.value}/")
            for obj in objects:
                # Check timestamp from key or metadata
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
