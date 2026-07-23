"""Artifact finalize verification (Track B B9.2b, companion §6.2/§6.4/§6.6).

`DA§6.2`: "An object is usable only in `available`. The finalize operation
performs a provider HEAD, checks generation/version, expected size,
content type, and checksum, then transitions the PostgreSQL row
atomically."

Before this, finalize did the HEAD but nothing else. The session SELECT
fetched `artifact_id, tenant_id, principal_id, expires_at, completed_at`
— so `expected_size_bytes` and `expected_sha256` were written when the
upload was authorized and **never read again**. A truncated, substituted,
or corrupt object became `available` unchallenged, and the caller's
claimed checksum was stored verbatim without being compared to anything.

Two related defects fixed with it:

- `sha256` could be populated from a provider ETag, which for a multipart
  upload is not a hash of the content, making the column unverifiable;
- `head_object` returned None for *every* exception, and finalize reads
  None as "object absent" and marks the artifact `abandoned` — so a
  transient 429/503 during finalize discarded a good upload.
"""

from __future__ import annotations

import uuid

import pytest

from artifacts import StorageUnavailable, get_artifact_manager
from db import pg_transaction


@pytest.fixture
def upload_session():
    """An artifact plus an authorized upload session, with declared values."""
    created: list[str] = []

    def _make(*, expected_size: int | None = None,
              expected_sha256: str | None = None) -> tuple[str, str]:
        artifact_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        with pg_transaction() as conn:
            conn.execute(
                """INSERT INTO storage.artifacts
                   (artifact_id, tenant_id, artifact_type, logical_name, state,
                    primary_provider, primary_bucket, object_key, content_type,
                    residency_region, retention_class)
                   VALUES (%s, %s, 'output', %s, 'requested', 'local', 'test',
                           %s, 'application/octet-stream', 'ca-central',
                           'standard')""",
                (artifact_id, f"tenant-{uuid.uuid4().hex[:8]}",
                 f"obj-{artifact_id}", f"output/{artifact_id}.bin"),
            )
            conn.execute(
                """INSERT INTO storage.artifact_upload_sessions
                   (upload_session_id, artifact_id, tenant_id, principal_id,
                    expected_size_bytes, expected_sha256, expires_at,
                    idempotency_key)
                   VALUES (%s, %s, %s, 'tester', %s, %s,
                           clock_timestamp() + interval '1 hour', %s)""",
                (session_id, artifact_id, f"tenant-{uuid.uuid4().hex[:8]}",
                 expected_size, expected_sha256, f"idem-{session_id}"),
            )
        created.append(artifact_id)
        return artifact_id, session_id

    yield _make

    if created:
        with pg_transaction() as conn:
            conn.execute(
                "DELETE FROM storage.artifact_upload_sessions "
                "WHERE artifact_id = ANY(%s::uuid[])", (created,)
            )
            conn.execute(
                "DELETE FROM storage.artifact_replicas "
                "WHERE artifact_id = ANY(%s::uuid[])", (created,)
            )
            conn.execute(
                "DELETE FROM storage.artifacts WHERE artifact_id = ANY(%s::uuid[])",
                (created,),
            )


def _state(artifact_id: str) -> str | None:
    with pg_transaction() as conn:
        row = conn.execute(
            "SELECT state FROM storage.artifacts WHERE artifact_id = %s",
            (artifact_id,),
        ).fetchone()
    return None if row is None else row[0]


def _sha256(artifact_id: str) -> str | None:
    with pg_transaction() as conn:
        row = conn.execute(
            "SELECT sha256 FROM storage.artifacts WHERE artifact_id = %s",
            (artifact_id,),
        ).fetchone()
    return None if row is None else row[0]


def _head(monkeypatch, manager, *, size: int, etag: str = "deadbeef"):
    monkeypatch.setattr(
        manager.primary, "head_object",
        lambda key: {"key": key, "size_bytes": size,
                     "content_type": "application/octet-stream",
                     "last_modified": None, "etag": etag},
    )


SHA_A = "a" * 64
SHA_B = "b" * 64


def test_size_mismatch_marks_corrupt(upload_session, monkeypatch):
    """A truncated upload must not become `available`."""
    artifact_id, session_id = upload_session(expected_size=1024)
    manager = get_artifact_manager()
    _head(monkeypatch, manager, size=512)

    with pytest.raises(RuntimeError, match="size mismatch"):
        manager.finalize_upload(session_id)

    assert _state(artifact_id) == "corrupt", (
        "a size mismatch means the bytes exist and are wrong — that is "
        "`corrupt`, distinct from an upload that never arrived"
    )


def test_checksum_mismatch_marks_corrupt(upload_session, monkeypatch):
    """The declared checksum was previously never compared to anything."""
    artifact_id, session_id = upload_session(expected_sha256=SHA_A)
    manager = get_artifact_manager()
    _head(monkeypatch, manager, size=100)

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        manager.finalize_upload(session_id, checksum=SHA_B)

    assert _state(artifact_id) == "corrupt"


def test_missing_checksum_is_refused_when_one_was_declared(
    upload_session, monkeypatch
):
    """Declaring an expected hash and then not supplying one proves nothing."""
    artifact_id, session_id = upload_session(expected_sha256=SHA_A)
    manager = get_artifact_manager()
    _head(monkeypatch, manager, size=100)

    with pytest.raises(ValueError, match="requires a checksum"):
        manager.finalize_upload(session_id)

    assert _state(artifact_id) == "requested"


def test_matching_size_and_checksum_becomes_available(upload_session, monkeypatch):
    """The control: verification passing must still finalize.

    Without this, every test above would pass on a finalize that simply
    never succeeds.
    """
    artifact_id, session_id = upload_session(expected_size=100, expected_sha256=SHA_A)
    manager = get_artifact_manager()
    _head(monkeypatch, manager, size=100)

    manager.finalize_upload(session_id, checksum=SHA_A)

    assert _state(artifact_id) == "available"
    assert _sha256(artifact_id) == SHA_A


def test_provider_etag_is_not_stored_as_a_content_hash(
    upload_session, monkeypatch
):
    """An S3 ETag is not a SHA-256 — for multipart it hashes nothing useful.

    Storing it in the `sha256` column made that column unverifiable: a
    later integrity check comparing content against it would always fail.
    """
    artifact_id, session_id = upload_session()
    manager = get_artifact_manager()
    _head(monkeypatch, manager, size=100, etag="not-a-sha256")

    manager.finalize_upload(session_id)

    assert _state(artifact_id) == "available"
    assert _sha256(artifact_id) != "not-a-sha256", (
        "a provider ETag was stored in the sha256 column"
    )


def test_ambiguous_provider_error_does_not_abandon_the_upload(
    upload_session, monkeypatch
):
    """A throttle during finalize must not discard a good upload.

    `head_object` returned None for every exception, and finalize reads
    None as "object absent" and marks the artifact `abandoned`. A
    transient 429/503 therefore destroyed the catalog record of a
    successfully uploaded object (DA§6.6).
    """
    artifact_id, session_id = upload_session()
    manager = get_artifact_manager()

    def _throttled(key):
        raise StorageUnavailable("SlowDown: reduce request rate")

    monkeypatch.setattr(manager.primary, "head_object", _throttled)

    with pytest.raises(StorageUnavailable):
        manager.finalize_upload(session_id)

    assert _state(artifact_id) == "requested", (
        "an ambiguous provider failure abandoned the artifact; only a "
        "definite not-found may do that"
    )


def test_genuinely_missing_object_still_abandons(upload_session, monkeypatch):
    """The counterpart: a real absence is still handled."""
    artifact_id, session_id = upload_session()
    manager = get_artifact_manager()
    monkeypatch.setattr(manager.primary, "head_object", lambda key: None)

    with pytest.raises(RuntimeError, match="not found"):
        manager.finalize_upload(session_id)

    assert _state(artifact_id) == "abandoned"


# ── Provider error taxonomy (Track B B9.2a, companion §6.6/§12.4) ──


class _BotoError(Exception):
    """Mimics botocore.exceptions.ClientError's .response shape."""

    def __init__(self, code: str, http: int = 500):
        super().__init__(code)
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": http},
        }


def test_delete_of_missing_object_is_idempotent_success(monkeypatch):
    """DELETE's end state (object gone) already holds — not a failure."""
    from artifacts import StorageClient, StorageConfig, StorageBackend

    client = StorageClient(StorageConfig(backend=StorageBackend.S3, bucket="b"))

    class _Boto:
        def delete_object(self, **kw):
            raise _BotoError("NoSuchKey", http=404)

    monkeypatch.setattr(client, "_get_client", lambda: _Boto())
    assert client.delete_object("gone") is True


def test_delete_failure_raises_rather_than_returning_false(monkeypatch):
    """The bug: a False return was recorded as a successful delete.

    The deletion worker never checks the return value; it relies on an
    exception to mark the job `delete_failed` for retry. A silent False
    let the catalog say `deleted` while the bytes remained.
    """
    from artifacts import (
        StorageClient, StorageConfig, StorageBackend, StorageUnavailable,
    )

    client = StorageClient(StorageConfig(backend=StorageBackend.S3, bucket="b"))

    class _Boto:
        def delete_object(self, **kw):
            raise _BotoError("InternalError", http=500)

    monkeypatch.setattr(client, "_get_client", lambda: _Boto())
    with pytest.raises(StorageUnavailable):
        client.delete_object("key")


def test_head_distinguishes_absent_from_ambiguous(monkeypatch):
    from artifacts import (
        StorageClient, StorageConfig, StorageBackend, StorageUnavailable,
    )

    client = StorageClient(StorageConfig(backend=StorageBackend.S3, bucket="b"))

    class _Missing:
        def head_object(self, **kw):
            raise _BotoError("404", http=404)

    class _Throttled:
        def head_object(self, **kw):
            raise _BotoError("SlowDown", http=503)

    monkeypatch.setattr(client, "_get_client", lambda: _Missing())
    assert client.head_object("k") is None

    monkeypatch.setattr(client, "_get_client", lambda: _Throttled())
    with pytest.raises(StorageUnavailable):
        client.head_object("k")


def test_error_code_helper_reads_code_then_http_status():
    from artifacts import _provider_error_code

    assert _provider_error_code(_BotoError("NoSuchKey", http=404)) == "NoSuchKey"

    class _HttpOnly(Exception):
        response = {"ResponseMetadata": {"HTTPStatusCode": 404}}

    assert _provider_error_code(_HttpOnly()) == "404"
    assert _provider_error_code(ValueError("no response attr")) == ""
