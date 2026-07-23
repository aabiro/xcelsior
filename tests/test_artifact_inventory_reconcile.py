"""Artifact inventory reconciliation (Track B B9.2d, companion §6.6/§12.4).

An `available` artifact whose bytes have vanished is a 404 waiting to
happen: the catalog insists the object exists, a download tries to fetch
it, and the user gets an error the catalog never anticipated. Nothing on
the request path notices — only an off-path inventory scan does.

`DA§6.6`: "Orphan scanning compares provider inventory with catalog
exports asynchronously; it is not in the request path." Report-only:
recording drift, never deleting bytes or flipping states off an untrusted
scan (B0.3 rule 17) — a transient provider blip must not become data loss.
"""

from __future__ import annotations

import uuid

import pytest

from artifacts import StorageUnavailable, get_artifact_manager
from db import pg_transaction


@pytest.fixture
def artifact():
    created: list[str] = []

    def _make(*, state: str = "available", legal_hold: bool = False) -> tuple[str, str]:
        artifact_id = str(uuid.uuid4())
        object_key = f"output/{artifact_id}.bin"
        with pg_transaction() as conn:
            conn.execute(
                """INSERT INTO storage.artifacts
                   (artifact_id, tenant_id, artifact_type, logical_name, state,
                    primary_provider, primary_bucket, object_key, content_type,
                    residency_region, retention_class, legal_hold, available_at)
                   VALUES (%s, %s, 'output', %s, %s, 'local', 'test', %s,
                           'application/octet-stream', 'ca-central', 'standard',
                           %s, clock_timestamp())""",
                (artifact_id, f"tenant-{uuid.uuid4().hex[:8]}",
                 f"obj-{artifact_id}", state, object_key, legal_hold),
            )
        created.append(artifact_id)
        return artifact_id, object_key

    yield _make

    if created:
        with pg_transaction() as conn:
            conn.execute(
                "DELETE FROM reconciliation_findings WHERE resource_type = 'artifact' "
                "AND resource_id = ANY(%s)", (created,)
            )
            conn.execute(
                "DELETE FROM storage.artifacts WHERE artifact_id = ANY(%s::uuid[])",
                (created,),
            )


def _findings(artifact_id: str) -> list[tuple[str, str, str | None]]:
    with pg_transaction() as conn:
        rows = conn.execute(
            "SELECT finding_type, severity, resolved_at::text "
            "FROM reconciliation_findings "
            "WHERE resource_type = 'artifact' AND resource_id = %s",
            (artifact_id,),
        ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def test_missing_bytes_are_flagged(artifact, monkeypatch):
    """The correctness case: `available` row, object gone."""
    artifact_id, _ = artifact(state="available")
    manager = get_artifact_manager()
    if not manager._is_db_active():
        pytest.skip("catalog inactive")
    monkeypatch.setattr(manager.primary, "head_object", lambda key: None)

    result = manager.reconcile_inventory()

    assert result["missing_bytes"] >= 1
    assert ("missing_bytes", "error", None) in _findings(artifact_id)


def test_present_bytes_produce_no_finding(artifact, monkeypatch):
    """The control: an object that is really there is not drift."""
    artifact_id, key = artifact(state="available")
    manager = get_artifact_manager()
    if not manager._is_db_active():
        pytest.skip("catalog inactive")
    monkeypatch.setattr(
        manager.primary, "head_object",
        lambda k: {"key": k, "size_bytes": 100, "etag": "x"},
    )

    manager.reconcile_inventory()

    assert _findings(artifact_id) == []


def test_ambiguous_provider_error_is_not_treated_as_missing(artifact, monkeypatch):
    """A transient provider failure must not accuse the catalog of drift.

    Treating "could not tell" as "object gone" would let a provider outage
    manufacture a fleet of false missing-bytes findings — and, if this were
    ever wired to remediate, delete real data.
    """
    artifact_id, _ = artifact(state="available")
    manager = get_artifact_manager()
    if not manager._is_db_active():
        pytest.skip("catalog inactive")

    def _throttled(key):
        raise StorageUnavailable("SlowDown")

    monkeypatch.setattr(manager.primary, "head_object", _throttled)

    manager.reconcile_inventory()

    assert _findings(artifact_id) == [], (
        "an unreachable provider produced a drift finding; only a definite "
        "absence may"
    )


def test_findings_are_deduplicated(artifact, monkeypatch):
    artifact_id, _ = artifact(state="available")
    manager = get_artifact_manager()
    if not manager._is_db_active():
        pytest.skip("catalog inactive")
    monkeypatch.setattr(manager.primary, "head_object", lambda key: None)

    for _ in range(3):
        manager.reconcile_inventory()

    open_findings = [f for f in _findings(artifact_id) if f[2] is None]
    assert len(open_findings) == 1


def test_legal_hold_rows_are_not_scanned(artifact, monkeypatch):
    """A held artifact is out of the deletion/lifecycle path entirely."""
    artifact_id, _ = artifact(state="available", legal_hold=True)
    manager = get_artifact_manager()
    if not manager._is_db_active():
        pytest.skip("catalog inactive")
    monkeypatch.setattr(manager.primary, "head_object", lambda key: None)

    manager.reconcile_inventory()

    assert _findings(artifact_id) == []


def test_reconciler_does_not_mutate_the_artifact(artifact, monkeypatch):
    """Report-only: state is untouched (B0.3 rule 17)."""
    artifact_id, _ = artifact(state="available")
    manager = get_artifact_manager()
    if not manager._is_db_active():
        pytest.skip("catalog inactive")
    monkeypatch.setattr(manager.primary, "head_object", lambda key: None)

    manager.reconcile_inventory()

    with pg_transaction() as conn:
        row = conn.execute(
            "SELECT state FROM storage.artifacts WHERE artifact_id = %s",
            (artifact_id,),
        ).fetchone()
    assert row is not None and row[0] == "available", (
        "the inventory scan changed artifact state; it is report-only"
    )
