"""Track B B4.5 — signed Merkle checkpoints over the audit stream (§13.6/§12.2).

Proves the checkpoint integrity guarantees:
  * create → verify round-trips;
  * any change to a sealed interval's events is detected (root mismatch);
  * a missing manifest fails verification;
  * key rotation preserves verifiability of older manifests, but removing the
    signing key version a manifest used makes it unverifiable (as it should).
"""

from __future__ import annotations

import datetime as _dt
import json
import uuid

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _has = _c.execute("SELECT to_regclass('audit_checkpoints')").fetchone()[0] is not None
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("audit_checkpoints missing — upgrade >= 075")

from control_plane.audit_checkpoints import create_checkpoint, verify_checkpoint


@pytest.fixture(autouse=True)
def _clean_and_key(monkeypatch):
    # Deterministic keyring for the test.
    monkeypatch.setenv("XCELSIOR_AUDIT_SIGNING_KEYS", json.dumps({"v1": "secret-one"}))
    monkeypatch.setenv("XCELSIOR_AUDIT_SIGNING_ACTIVE", "v1")
    yield
    if _pool is None:
        return
    with _pool.connection() as conn:
        conn.execute("TRUNCATE audit_checkpoints")  # WORM row trigger doesn't fire on TRUNCATE
        conn.execute("TRUNCATE audit_events_v2")
        conn.commit()


def _insert_event(conn, *, seq: int, created_at: str) -> None:
    conn.execute(
        """INSERT INTO audit_events_v2
               (stream_type, stream_id, stream_sequence, event_type, event_hash, created_at)
           VALUES ('job', 's-cp', %s, 'job.v1.created', %s, %s)""",
        (seq, f"hash-{seq}", created_at),
    )


def _seed(n: int = 3) -> tuple[str, str]:
    start = "2026-07-10 00:00:00+00"
    end = "2026-07-11 00:00:00+00"
    with _pool.connection() as conn:
        for i in range(n):
            _insert_event(conn, seq=i, created_at=f"2026-07-10 0{i}:00:00+00")
        conn.commit()
    return start, end


def test_create_and_verify_roundtrip():
    start, end = _seed(3)
    with _pool.connection() as conn:
        cid = create_checkpoint(conn, interval_start=start, interval_end=end)
        conn.commit()
    with _pool.connection() as conn:
        ok, reason = verify_checkpoint(conn, cid)
    assert ok is True and reason == "ok"


def test_change_to_sealed_interval_is_detected():
    start, end = _seed(3)
    with _pool.connection() as conn:
        cid = create_checkpoint(conn, interval_start=start, interval_end=end)
        conn.commit()
    # Append a (back-dated) event INTO the sealed interval — audit rows are
    # append-only, so this is the only "tamper" possible, and it must be caught.
    with _pool.connection() as conn:
        _insert_event(conn, seq=99, created_at="2026-07-10 05:00:00+00")
        conn.commit()
    with _pool.connection() as conn:
        ok, reason = verify_checkpoint(conn, cid)
    assert ok is False and reason == "merkle_root_mismatch"


def test_missing_manifest_fails():
    with _pool.connection() as conn:
        ok, reason = verify_checkpoint(conn, str(uuid.uuid4()))
    assert ok is False and reason == "manifest_missing"


def test_key_rotation_preserves_old_manifest(monkeypatch):
    start, end = _seed(2)
    with _pool.connection() as conn:
        cid = create_checkpoint(conn, interval_start=start, interval_end=end)  # signed v1
        conn.commit()
    # Rotate: v2 active, v1 RETAINED → the old manifest still verifies.
    monkeypatch.setenv("XCELSIOR_AUDIT_SIGNING_KEYS", json.dumps({"v1": "secret-one", "v2": "secret-two"}))
    monkeypatch.setenv("XCELSIOR_AUDIT_SIGNING_ACTIVE", "v2")
    with _pool.connection() as conn:
        assert verify_checkpoint(conn, cid) == (True, "ok")
    # Remove v1 entirely → its manifest can no longer be verified (honest).
    monkeypatch.setenv("XCELSIOR_AUDIT_SIGNING_KEYS", json.dumps({"v2": "secret-two"}))
    with _pool.connection() as conn:
        ok, reason = verify_checkpoint(conn, cid)
    assert ok is False and reason == "unknown_key_version"


def test_checkpoints_chain_prev_manifest_hash():
    start, end = _seed(2)
    with _pool.connection() as conn:
        c1 = create_checkpoint(conn, interval_start=start, interval_end=end)
        conn.commit()
    with _pool.connection() as conn:
        c2 = create_checkpoint(conn, interval_start="2026-07-11 00:00:00+00", interval_end="2026-07-12 00:00:00+00")
        conn.commit()
    with _pool.connection() as conn:
        h1 = conn.execute("SELECT manifest_sha256 FROM audit_checkpoints WHERE checkpoint_id=%s", (c1,)).fetchone()[0]
        prev2 = conn.execute("SELECT prev_manifest_hash FROM audit_checkpoints WHERE checkpoint_id=%s", (c2,)).fetchone()[0]
    assert prev2 == h1  # the second manifest chains the first
