"""Artifact deletion authority (Track B B9.2c, companion §6.5/§8.8).

Two compliance defects, both found by reading the deletion path against
`DA§6.5` ("Provider lifecycle rules are a safety net and cost mechanism,
not the only business workflow. PostgreSQL determines eligibility"):

1. The catalog deletion worker never consulted `legal_hold`. An artifact
   under hold was deleted like any other, which is exactly what a hold
   exists to prevent (`DA§8.8`: "Legal holds suspend deletion for the
   governed records and are auditable").
2. The legacy object-age sweep ran **unconditionally**, outside the
   catalog branch. It deleted bytes by prefix listing with no catalog
   lookup — so an `available` artifact, under hold, inside its retention
   window, lost its bytes if the object was older than the cutoff, and the
   catalog row was never updated to say so.

The second is the more dangerous: it needed no deletion request at all.
"""

from __future__ import annotations

import uuid

import pytest

from artifacts import get_artifact_manager
from db import pg_transaction


@pytest.fixture
def artifact():
    created: list[str] = []

    def _make(
        *,
        legal_hold: bool = False,
        retain_days: int | None = None,
        state: str = "available",
    ) -> str:
        artifact_id = str(uuid.uuid4())
        with pg_transaction() as conn:
            conn.execute(
                """INSERT INTO storage.artifacts
                   (artifact_id, tenant_id, artifact_type, logical_name, state,
                    primary_provider, primary_bucket, object_key, content_type,
                    residency_region, retention_class, legal_hold, retain_until)
                   VALUES (%s, %s, 'output', %s, %s, 'local', 'test',
                           %s, 'application/octet-stream', 'ca-central',
                           'standard', %s,
                           CASE WHEN %s::int IS NULL THEN NULL
                                ELSE clock_timestamp()
                                     + make_interval(days => %s::int)
                           END)""",
                (
                    artifact_id, f"tenant-{uuid.uuid4().hex[:8]}",
                    f"obj-{artifact_id}", state,
                    f"output/{artifact_id}.bin", legal_hold,
                    retain_days, retain_days,
                ),
            )
        created.append(artifact_id)
        return artifact_id

    yield _make

    if created:
        with pg_transaction() as conn:
            conn.execute(
                "DELETE FROM storage.artifact_deletion_jobs "
                "WHERE artifact_id = ANY(%s::uuid[])", (created,)
            )
            conn.execute(
                "DELETE FROM storage.artifacts WHERE artifact_id = ANY(%s::uuid[])",
                (created,),
            )


def _request_deletion(artifact_id: str) -> str:
    deletion_id = str(uuid.uuid4())
    with pg_transaction() as conn:
        conn.execute(
            """INSERT INTO storage.artifact_deletion_jobs
               (deletion_id, artifact_id, reason, requested_by, state)
               VALUES (%s, %s, 'test', 'tester', 'requested')""",
            (deletion_id, artifact_id),
        )
    return deletion_id


def _artifact_state(artifact_id: str) -> str | None:
    with pg_transaction() as conn:
        row = conn.execute(
            "SELECT state FROM storage.artifacts WHERE artifact_id = %s",
            (artifact_id,),
        ).fetchone()
    return None if row is None else row[0]


def _deletion_job(deletion_id: str) -> tuple[str, str | None]:
    with pg_transaction() as conn:
        row = conn.execute(
            "SELECT state, last_error FROM storage.artifact_deletion_jobs "
            "WHERE deletion_id = %s",
            (deletion_id,),
        ).fetchone()
    assert row is not None
    return row[0], row[1]


def test_legal_hold_blocks_deletion(artifact):
    """The defect: a held artifact was deleted like any other."""
    artifact_id = artifact(legal_hold=True)
    deletion_id = _request_deletion(artifact_id)

    get_artifact_manager().cleanup_expired()

    assert _artifact_state(artifact_id) == "available", (
        "an artifact under legal hold was deleted; a hold must override "
        "lifecycle deletion (DA§6.5, §8.8)"
    )
    state, error = _deletion_job(deletion_id)
    assert state == "delete_failed"
    assert error and "legal hold" in error.lower(), (
        f"the refusal must say why, got {error!r}"
    )


def test_retention_window_defers_deletion(artifact):
    """`retain_until` in the future is not yet eligible."""
    artifact_id = artifact(retain_days=30)
    deletion_id = _request_deletion(artifact_id)

    get_artifact_manager().cleanup_expired()

    assert _artifact_state(artifact_id) == "available"
    state, error = _deletion_job(deletion_id)
    assert state == "delete_failed"
    assert error and "retain_until" in error


def test_ordinary_artifact_is_deleted(artifact):
    """The control: without a hold or retention, deletion proceeds.

    Without this, the two tests above would pass on a worker that simply
    never deletes anything.
    """
    artifact_id = artifact()
    deletion_id = _request_deletion(artifact_id)

    get_artifact_manager().cleanup_expired()

    assert _artifact_state(artifact_id) == "deleted"
    state, _ = _deletion_job(deletion_id)
    assert state == "completed"


def test_catalog_active_suppresses_the_object_age_sweep(artifact, monkeypatch):
    """The worse defect: bytes deleted with no catalog lookup at all.

    The legacy sweep ran unconditionally and deleted by object age. An
    `available` artifact under legal hold lost its bytes without any
    deletion request, and the catalog row was never updated — leaving a
    row that claims an object which no longer exists.
    """
    artifact(legal_hold=True)
    manager = get_artifact_manager()
    if not manager._is_db_active():
        pytest.skip("storage catalog inactive in this environment")

    listed: list[str] = []
    deleted: list[str] = []
    monkeypatch.setattr(
        manager.primary, "list_objects",
        lambda prefix="", max_keys=1000: listed.append(prefix) or [],
    )
    monkeypatch.setattr(
        manager.primary, "delete_object",
        lambda key: deleted.append(key) or True,
    )

    manager.cleanup_expired()

    assert listed == [], (
        "the object-age sweep ran while the catalog was active; it cannot "
        "honour legal hold, retention, or artifact state because it never "
        "reads the catalog (DA§6.5)"
    )
    assert deleted == []
