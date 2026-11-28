"""Private S3-compatible storage used by control-plane maintenance jobs."""

from dataclasses import dataclass
from contextlib import closing
from hashlib import file_digest
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import urlsplit

import boto3
from botocore.config import Config


class ObjectStorageError(RuntimeError):
    """An operation did not produce an independently verifiable object."""


@dataclass(frozen=True)
class StoredObject:
    key: str
    size_bytes: int
    sha256: str


def _fingerprint(path: Path) -> tuple[int, str]:
    with path.open("rb") as source:
        return os.fstat(source.fileno()).st_size, file_digest(source, "sha256").hexdigest()


class ObjectStoreBackend:
    def __init__(self, client: Any, bucket: str):
        self.client = client
        self.bucket = bucket

    @classmethod
    def from_env(cls) -> "ObjectStoreBackend":
        bucket = os.environ.get("XCELSIOR_OBJECT_STORE_BUCKET", "").strip()
        if not bucket:
            raise ObjectStorageError("XCELSIOR_OBJECT_STORE_BUCKET is required")
        endpoint = os.environ.get("XCELSIOR_OBJECT_STORE_ENDPOINT", "").strip()
        if endpoint:
            url = urlsplit(endpoint)
            if url.scheme != "https" or not url.hostname or url.username or url.password:
                raise ObjectStorageError("Object storage requires a private bucket over HTTPS")
            if url.path not in ("", "/") or url.query or url.fragment:
                raise ObjectStorageError("Object storage endpoint must be an HTTPS origin")
        access_key = os.environ.get("XCELSIOR_OBJECT_STORE_ACCESS_KEY", "")
        secret_key = os.environ.get("XCELSIOR_OBJECT_STORE_SECRET_KEY", "")
        token = os.environ.get("XCELSIOR_OBJECT_STORE_SESSION_TOKEN", "")
        if bool(access_key) != bool(secret_key) or (token and not access_key):
            raise ObjectStorageError("Incomplete object storage credentials")
        if endpoint and not access_key:
            # The default AWS role must not be used to sign calls to a custom endpoint.
            raise ObjectStorageError("A custom endpoint requires its own scoped credentials")
        client = boto3.client(
            "s3",
            endpoint_url=endpoint or None,
            region_name=os.environ.get("XCELSIOR_OBJECT_STORE_REGION") or None,
            aws_access_key_id=access_key or None,
            aws_secret_access_key=secret_key or None,
            aws_session_token=token or None,
            config=Config(
                signature_version="s3v4",
                connect_timeout=5,
                read_timeout=30,
                retries={"mode": "standard", "total_max_attempts": 4},
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
                s3={"addressing_style": "path"} if endpoint else None,
            ),
        )
        return cls(client, bucket)

    def upload(self, key: str, source: Path) -> StoredObject:
        size, sha256 = _fingerprint(source)
        if size == 0:
            raise ObjectStorageError("Refusing to publish an empty object")
        self.client.upload_file(
            str(source), self.bucket, key, ExtraArgs={"Metadata": {"sha256": sha256}}
        )
        stored = self.stat(key)
        if (stored.size_bytes, stored.sha256) != (size, sha256):
            raise ObjectStorageError("Uploaded object metadata did not match the source")
        return stored

    def stat(self, key: str) -> StoredObject:
        result = self.client.head_object(Bucket=self.bucket, Key=key)
        sha256 = result.get("Metadata", {}).get("sha256", "")
        size = result["ContentLength"]
        if size <= 0 or not re.fullmatch("[a-f0-9]{64}", sha256):
            raise ObjectStorageError("Object has no valid SHA-256 or length")
        return StoredObject(key, size, sha256)

    def download(self, key: str, destination: Path) -> StoredObject:
        stored = self.stat(key)
        with NamedTemporaryFile(dir=destination.parent) as temporary:
            self.client.download_fileobj(self.bucket, key, temporary)
            temporary.flush()
            if _fingerprint(Path(temporary.name)) != (stored.size_bytes, stored.sha256):
                raise ObjectStorageError("Downloaded object checksum or length did not match")
            os.link(temporary.name, destination)
        return stored

    def verify(self, stored: StoredObject) -> None:
        from hashlib import sha256

        digest = sha256()
        size = 0
        response = self.client.get_object(Bucket=self.bucket, Key=stored.key)
        with closing(response["Body"]) as stream:
            for chunk in stream.iter_chunks(chunk_size=8 * 1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
        if (size, digest.hexdigest()) != (stored.size_bytes, stored.sha256):
            raise ObjectStorageError("Object read-back checksum or length did not match")

    def keys(self, prefix: str) -> list[str]:
        pages = self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=prefix
        )
        return [item["Key"] for page in pages for item in page.get("Contents", [])]
