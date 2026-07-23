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


def _provider_error_code(exc: Exception) -> str:
    """The provider's error code for a boto/S3 exception, or ''.

    Companion §6.6: "Every provider method distinguishes not-found,
    precondition-failed, permission-denied, throttled, and unavailable."
    One helper so every method classifies identically instead of each
    reimplementing the `response['Error']['Code']` lookup (and drifting).
    """
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if code:
            return str(code)
        http = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if http:
            return str(http)
    return ""


_NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


class _FinalizeRejected(Exception):
    """Internal: a finalize verification failed.

    Carries the state the artifact should end in so the wrapper can write
    it *outside* the transaction that is about to roll back.
    """

    def __init__(
        self,
        artifact_id,
        state: str | None,
        message: str,
        *,
        error_type: type[Exception] = RuntimeError,
    ):
        super().__init__(message)
        self.artifact_id = artifact_id
        self.state = state
        self.message = message
        self.error_type = error_type

    def as_error(self) -> Exception:
        return self.error_type(self.message)


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
        except Exception as exc:
            # Distinguish "definitely absent" from "could not tell".
            #
            # Returning None for both let a transient 429/503 be read as
            # "the object is not there", and `finalize_upload` marks an
            # artifact `abandoned` on a None — so a throttle during
            # finalize discarded a perfectly good upload. DA§6.6: "Every
            # provider method distinguishes not-found, precondition-failed,
            # permission-denied, throttled, and unavailable."
            if _provider_error_code(exc) in _NOT_FOUND_CODES:
                return None
            raise StorageUnavailable(
                f"HEAD {key} failed on {self.config.backend}: {exc}"
            ) from exc

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
            # A genuine delete failure must RAISE, not return False.
            # The deletion worker does not check this return value — it
            # relies on an exception to mark the job `delete_failed` for
            # retry. Returning False let a failed delete be recorded as a
            # completed one, so the catalog said `deleted` while the bytes
            # remained: an orphan the inventory scan would later have to
            # find (companion §6.6, §12.4).
            #
            # An already-absent object is idempotent success: DELETE's
            # desired end state (the object is gone) already holds.
            if _provider_error_code(e) in _NOT_FOUND_CODES:
                return True
            log.error("Failed to delete %s: %s", key, e)
            raise StorageUnavailable(f"delete_object failed for {key}: {e}") from e

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

    def reconcile_inventory(self, scan_limit: int = 500) -> dict:
        """Compare the catalog against provider reality, report-only.

        Two drift classes matter (companion §6.6, §12.4):

        - **missing bytes** — a row is `available` but the object is gone.
          A user downloading it gets a 404 while the catalog insists the
          artifact exists. This is the correctness case.
        - **orphan bytes** — an object exists with no catalog row (or one
          that is not live). Cost, not correctness; a cleanup candidate
          only after a safety window, never deleted from a request path.

        Findings are recorded, never auto-remediated (B0.3 rule 17): a
        reconciler deleting bytes or flipping states off an untrusted
        scan is how a transient provider blip becomes data loss. This runs
        as a durable scheduled task, off the request path (§6.6: "Orphan
        scanning ... is not in the request path").

        Only the `missing bytes` scan runs against a remote provider that
        can HEAD. Orphan detection needs a provider inventory export and is
        left to the operator tooling until the adapter split (B9.2a) lands
        a paginated lister.
        """
        if not self._is_db_active():
            return {"scanned": 0, "missing_bytes": 0, "skipped": "catalog inactive"}

        from control_plane.db import control_plane_transaction
        from psycopg.types.json import Jsonb

        missing = 0
        scanned = 0
        with control_plane_transaction() as conn:
            rows = conn.execute(
                """SELECT artifact_id, object_key, primary_provider, tenant_id
                     FROM storage.artifacts
                    WHERE state IN ('available', 'expiring')
                      AND legal_hold = false
                    ORDER BY available_at NULLS FIRST
                    LIMIT %s""",
                (max(1, scan_limit),),
            ).fetchall()

            for artifact_id, object_key, provider, tenant_id in rows:
                scanned += 1
                client = self.cache if provider == "r2" and self.cache else self.primary
                try:
                    meta = client.head_object(object_key)
                except StorageUnavailable:
                    # Ambiguous — do not accuse the catalog of drift on a
                    # provider we could not reach.
                    continue
                if meta is not None:
                    continue

                # Object is definitively absent while the row says available.
                existing = conn.execute(
                    """SELECT 1 FROM reconciliation_findings
                        WHERE resource_type = 'artifact'
                          AND resource_id = %s
                          AND finding_type = 'missing_bytes'
                          AND resolved_at IS NULL LIMIT 1""",
                    (str(artifact_id),),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        """INSERT INTO reconciliation_findings
                               (resource_type, resource_id, tenant_id,
                                finding_type, severity, summary, desired,
                                observed, action_taken)
                           VALUES ('artifact', %s, %s, 'missing_bytes',
                                   'error', %s, %s, %s, 'report_only')""",
                        (
                            str(artifact_id), tenant_id,
                            f"artifact {artifact_id} is 'available' but its "
                            f"object {object_key} is missing from the provider",
                            Jsonb({"state": "available"}),
                            Jsonb({"object_present": False}),
                        ),
                    )
                    missing += 1

        if missing:
            log.warning("artifact inventory: %d available rows with missing bytes", missing)
        return {"scanned": scanned, "missing_bytes": missing}

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
        """Finalize an upload session: HEAD, verify, then mark available.

        Rejections are applied in their own transaction. The verification
        work runs inside ``control_plane_transaction``, and raising from
        there rolls the whole block back — so a state write made just
        before the raise was silently discarded. That is why the
        pre-existing "mark abandoned when the object is missing" never
        persisted: the UPDATE and the RuntimeError were in the same
        transaction, and the error message said one thing while the
        database kept saying `requested`.
        """
        try:
            return self._finalize_upload_txn(session_id, checksum)
        except _FinalizeRejected as rejection:
            if rejection.state:
                self._mark_artifact_state(rejection.artifact_id, rejection.state)
            raise rejection.as_error()

    def _mark_artifact_state(self, artifact_id, state: str) -> None:
        """Record a terminal verification outcome in its own transaction."""
        from control_plane.db import control_plane_transaction

        try:
            with control_plane_transaction() as conn:
                conn.execute(
                    "UPDATE storage.artifacts SET state = %s, version = version + 1 "
                    "WHERE artifact_id = %s",
                    (state, artifact_id),
                )
        except Exception as exc:
            log.error(
                "failed to record artifact %s state=%s: %s", artifact_id, state, exc
            )

    def _finalize_upload_txn(self, session_id: str, checksum: Optional[str] = None) -> dict:
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
                """SELECT artifact_id, tenant_id, principal_id, expires_at,
                          completed_at, expected_size_bytes, expected_sha256
                   FROM storage.artifact_upload_sessions
                   WHERE upload_session_id = %s FOR UPDATE""",
                (sess_uuid,),
            ).fetchone()
            if not sess:
                raise ValueError(f"Upload session {session_id} not found")

            (
                artifact_id, tenant_id, principal_id, expires_at,
                completed_at, expected_size, expected_sha256,
            ) = sess
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
                raise _FinalizeRejected(
                    artifact_id,
                    "abandoned",
                    f"Object {key} not found on provider {provider} bucket {bucket}",
                )

            size_bytes = meta.get("size_bytes", 0)

            # 3b. Verify the object against what the session declared.
            #
            # `expected_size_bytes` and `expected_sha256` were recorded when
            # the upload was authorized and, before this, were never read
            # again — so a truncated, substituted, or corrupt object became
            # `available` unchallenged. DA§6.2: "The finalize operation
            # performs a provider HEAD, checks generation/version, expected
            # size, content type, and checksum, then transitions the
            # PostgreSQL row atomically."
            #
            # A mismatch is `corrupt`, not `abandoned`: the bytes exist and
            # are wrong, which is a different operational question from an
            # upload that never arrived.

            if expected_size is not None and int(size_bytes) != int(expected_size):
                raise _FinalizeRejected(
                    artifact_id,
                    "corrupt",
                    f"Artifact {artifact_id} size mismatch: session declared "
                    f"{expected_size} bytes, provider reports {size_bytes}",
                )

            supplied = (checksum or "").strip().lower() or None
            expected = (expected_sha256 or "").strip().lower() or None
            if expected and supplied and supplied != expected:
                raise _FinalizeRejected(
                    artifact_id,
                    "corrupt",
                    f"Artifact {artifact_id} checksum mismatch: session "
                    f"declared {expected}, upload reported {supplied}",
                )
            if expected and not supplied:
                raise _FinalizeRejected(
                    artifact_id,
                    None,
                    f"Artifact {artifact_id} requires a checksum at finalize: "
                    f"the upload session declared an expected sha256",
                    error_type=ValueError,
                )

            # Only a real content hash goes in `sha256`. A provider ETag is
            # not one — for a multipart upload it is not a hash of the
            # content at all — so storing it here would make the column
            # unverifiable.
            content_sha256 = supplied or expected

            # 4. Atomically transition state to 'available'
            now_row = conn.execute("SELECT clock_timestamp()").fetchone()
            assert now_row is not None  # SELECT of a constant always returns a row
            now = now_row[0]
            conn.execute(
                """UPDATE storage.artifacts
                   SET state = 'available',
                       size_bytes = %s,
                       sha256 = COALESCE(sha256, %s),
                       object_generation = COALESCE(object_generation, %s),
                       available_at = %s,
                       version = version + 1
                   WHERE artifact_id = %s""",
                (size_bytes, content_sha256, meta.get("etag") or None,
                 now, artifact_id),
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
                            "SELECT object_key, primary_provider, primary_bucket, "
                            "legal_hold, retain_until FROM storage.artifacts "
                            "WHERE artifact_id = %s",
                            (art_id,)
                        ).fetchone()

                        # Legal hold overrides lifecycle deletion (DA§6.5,
                        # §8.8). This is a terminal refusal, not a retry:
                        # backing off would re-attempt the delete every
                        # cycle for the life of the hold and bury the
                        # signal that someone asked for it.
                        if art and art[3]:
                            conn.execute(
                                """UPDATE storage.artifact_deletion_jobs
                                      SET state = 'delete_failed',
                                          last_error = %s,
                                          next_attempt_at =
                                              clock_timestamp() + interval '1 day'
                                    WHERE deletion_id = %s""",
                                (
                                    "refused: artifact is under legal hold",
                                    del_id,
                                ),
                            )
                            log.warning(
                                "CLEANUP DB: refusing deletion of artifact %s "
                                "— legal hold is in force",
                                art_id,
                            )
                            continue

                        # Retention floor: the catalog, not the object's
                        # age, decides when bytes may go.
                        if art and art[4] is not None:
                            still_retained = conn.execute(
                                "SELECT %s > clock_timestamp()", (art[4],)
                            ).fetchone()
                            if still_retained and still_retained[0]:
                                conn.execute(
                                    """UPDATE storage.artifact_deletion_jobs
                                          SET state = 'delete_failed',
                                              last_error = %s,
                                              next_attempt_at = %s
                                        WHERE deletion_id = %s""",
                                    (
                                        "deferred: retain_until has not passed",
                                        art[4],
                                        del_id,
                                    ),
                                )
                                log.info(
                                    "CLEANUP DB: deferring deletion of %s until %s",
                                    art_id, art[4],
                                )
                                continue

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

            # The catalog is the deletion authority once it is active. The
            # object-listing sweep below must not also run: it deletes by
            # object age with no catalog lookup, so it would remove bytes
            # for an artifact that is `available`, under legal hold, or
            # inside its retention window — and leave the catalog row
            # claiming the object still exists.
            #
            # DA§6.5: "Provider lifecycle rules are a safety net and cost
            # mechanism, not the only business workflow. PostgreSQL
            # determines eligibility."
            return abandoned_count

        # 2. Legacy object-age sweep — development/pre-catalog only.
        #
        # Reached only when the storage catalog is inactive, i.e. there is
        # no PostgreSQL row to consult. Deleting by prefix listing cannot
        # honour legal hold, retention, or artifact state, which is why it
        # is not allowed to run alongside the catalog.
        cutoff = time.time() - older_than_sec
        deleted = 0
        for atype in ArtifactType:
            objects = self.primary.list_objects(f"{atype.value}/")
            for obj in objects:
                if isinstance(obj.get("last_modified"), (int, float)):
                    if obj["last_modified"] < cutoff:
                        if self.primary.delete_object(obj["key"]):
                            deleted += 1
        log.info("CLEANUP: deleted %d expired artifacts (no catalog)", deleted)
        return deleted


# ── Singleton ─────────────────────────────────────────────────────────

_artifact_manager: Optional[ArtifactManager] = None


def get_artifact_manager() -> ArtifactManager:
    global _artifact_manager
    if _artifact_manager is None:
        _artifact_manager = ArtifactManager()
    return _artifact_manager
