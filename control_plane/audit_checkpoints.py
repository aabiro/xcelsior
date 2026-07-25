"""Signed Merkle checkpoints over the audit stream (Track B B4.5, §13.6/§12.2).

A periodic checkpoint seals an interval of `audit_events_v2`: it computes a
Merkle root over the interval's `(event_id, event_hash)` leaves, chains the
previous manifest's hash, and signs the manifest with a versioned key. Verifying
recomputes the root from the (WORM) events and checks the signature with the
manifest's recorded key version — so:

  * a change to any sealed-interval event → recomputed root ≠ stored root;
  * a tampered manifest → recomputed manifest hash ≠ stored, or bad signature;
  * a missing manifest → verification fails;
  * key rotation → older manifests still verify, because each records the key
    version it was signed with.

Signing keys are a versioned keyring (like the OAuth JWT keyring) and are meant
to be administratively separate from the object-storage bucket the manifest is
also uploaded to in production.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import json
import os
from typing import Any

_EMPTY_LEAF = hashlib.sha256(b"xcelsior-audit-empty").hexdigest()


def _ts(value: Any) -> str:
    """Canonical UTC-ISO form for an interval bound, whether given as a string
    (create) or a DB timestamptz (verify), so the signed manifest is identical
    both ways."""
    v = value
    if isinstance(v, str):
        v = _dt.datetime.fromisoformat(v.strip().replace(" ", "T"))
    if getattr(v, "tzinfo", None) is None:
        v = v.replace(tzinfo=_dt.timezone.utc)
    return v.astimezone(_dt.timezone.utc).isoformat()


def _signing_keys() -> tuple[str, dict[str, str]]:
    """(active_key_version, {version: secret}). Env-driven, dev/test default.

    `XCELSIOR_AUDIT_SIGNING_KEYS` is JSON `{"v1": "secret", ...}`;
    `XCELSIOR_AUDIT_SIGNING_ACTIVE` names the active version. Older versions stay
    in the map so their manifests keep verifying after rotation.
    """
    raw = os.environ.get("XCELSIOR_AUDIT_SIGNING_KEYS", "").strip()
    keys: dict[str, str] = {}
    if raw:
        try:
            keys = {str(k): str(v) for k, v in json.loads(raw).items()}
        except Exception:
            keys = {}
    if not keys:
        keys = {"v1": os.environ.get("XCELSIOR_AUDIT_SIGNING_KEY", "dev-audit-key-not-for-prod")}
    active = os.environ.get("XCELSIOR_AUDIT_SIGNING_ACTIVE", "").strip() or sorted(keys)[-1]
    if active not in keys:
        active = sorted(keys)[-1]
    return active, keys


def _merkle_root(leaves: list[str]) -> str:
    """SHA-256 Merkle root over ordered leaves (odd node duplicates the last)."""
    if not leaves:
        return _EMPTY_LEAF
    level = [hashlib.sha256(leaf.encode()).hexdigest() for leaf in leaves]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i + 1] if i + 1 < len(level) else level[i]
            nxt.append(hashlib.sha256((a + b).encode()).hexdigest())
        level = nxt
    return level[0]


def _interval_rows(conn: Any, start: Any, end: Any) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT event_id, event_hash
          FROM audit_events_v2
         WHERE created_at >= %s AND created_at < %s
         ORDER BY created_at, event_id
        """,
        (start, end),
    ).fetchall()
    return [(str(r[0]), str(r[1] or "")) for r in rows]


def _manifest(
    *, start: Any, end: Any, merkle_root: str, row_count: int,
    first_event_id: str | None, last_event_id: str | None,
    prev_manifest_hash: str | None, schema_versions: dict[str, Any], key_version: str,
) -> dict[str, Any]:
    return {
        "interval_start": _ts(start),
        "interval_end": _ts(end),
        "merkle_root": merkle_root,
        "row_count": row_count,
        "first_event_id": first_event_id,
        "last_event_id": last_event_id,
        "prev_manifest_hash": prev_manifest_hash,
        "schema_versions": schema_versions,
        "signing_key_version": key_version,
    }


def _canonical(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"))


def _sign(manifest_sha256: str, secret: str) -> str:
    return hmac.new(secret.encode(), manifest_sha256.encode(), hashlib.sha256).hexdigest()


def create_checkpoint(conn: Any, *, interval_start: Any, interval_end: Any, schema_versions: dict[str, Any] | None = None) -> str:
    """Seal [interval_start, interval_end) into a signed manifest. Returns id."""
    leaves = _interval_rows(conn, interval_start, interval_end)
    merkle_root = _merkle_root([f"{eid}:{eh}" for eid, eh in leaves])
    prev = conn.execute(
        "SELECT manifest_sha256 FROM audit_checkpoints ORDER BY interval_end DESC, created_at DESC LIMIT 1"
    ).fetchone()
    prev_hash = prev[0] if prev else None
    active, keys = _signing_keys()
    manifest = _manifest(
        start=interval_start, end=interval_end, merkle_root=merkle_root,
        row_count=len(leaves),
        first_event_id=leaves[0][0] if leaves else None,
        last_event_id=leaves[-1][0] if leaves else None,
        prev_manifest_hash=prev_hash, schema_versions=schema_versions or {"audit_events_v2": 1},
        key_version=active,
    )
    manifest_sha256 = hashlib.sha256(_canonical(manifest).encode()).hexdigest()
    signature = _sign(manifest_sha256, keys[active])
    row = conn.execute(
        """
        INSERT INTO audit_checkpoints
            (interval_start, interval_end, merkle_root, row_count, first_event_id,
             last_event_id, prev_manifest_hash, manifest_sha256, schema_versions,
             signing_key_version, signature)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING checkpoint_id
        """,
        (
            interval_start, interval_end, merkle_root, len(leaves),
            manifest["first_event_id"], manifest["last_event_id"], prev_hash,
            manifest_sha256, json.dumps(manifest["schema_versions"]), active, signature,
        ),
    ).fetchone()
    return str(row[0])


def verify_checkpoint(conn: Any, checkpoint_id: str) -> tuple[bool, str]:
    """Recompute the root + manifest hash from the WORM events and verify the
    signature with the recorded key version. Returns (ok, reason)."""
    cp = conn.execute(
        """
        SELECT interval_start, interval_end, merkle_root, row_count, first_event_id,
               last_event_id, prev_manifest_hash, manifest_sha256, schema_versions,
               signing_key_version, signature
          FROM audit_checkpoints WHERE checkpoint_id = %s
        """,
        (checkpoint_id,),
    ).fetchone()
    if cp is None:
        return False, "manifest_missing"
    (start, end, merkle_root, row_count, first_id, last_id, prev_hash,
     manifest_sha256, schema_versions, key_version, signature) = cp

    leaves = _interval_rows(conn, start, end)
    recomputed_root = _merkle_root([f"{eid}:{eh}" for eid, eh in leaves])
    if recomputed_root != merkle_root or len(leaves) != row_count:
        return False, "merkle_root_mismatch"

    schema_versions = schema_versions if isinstance(schema_versions, dict) else json.loads(schema_versions or "{}")
    manifest = _manifest(
        start=start, end=end, merkle_root=merkle_root, row_count=row_count,
        first_event_id=str(first_id) if first_id else None,
        last_event_id=str(last_id) if last_id else None,
        prev_manifest_hash=prev_hash, schema_versions=schema_versions, key_version=key_version,
    )
    recomputed_sha = hashlib.sha256(_canonical(manifest).encode()).hexdigest()
    if recomputed_sha != manifest_sha256:
        return False, "manifest_tampered"

    _, keys = _signing_keys()
    secret = keys.get(str(key_version))
    if secret is None:
        return False, "unknown_key_version"
    if not hmac.compare_digest(_sign(manifest_sha256, secret), str(signature)):
        return False, "bad_signature"
    return True, "ok"


def audit_checkpoint_task(*, period_sec: int = 86_400) -> None:
    """Durable `scheduled_tasks` entry point — seal the previous period and
    verify the manifest just written (§12.2 verification on a schedule)."""
    import logging

    log = logging.getLogger("xcelsior")
    now = _dt.datetime.now(_dt.timezone.utc)
    end = now.replace(minute=0, second=0, microsecond=0)
    start = end - _dt.timedelta(seconds=period_sec)
    from control_plane.db import control_plane_transaction

    with control_plane_transaction() as conn:
        cid = create_checkpoint(conn, interval_start=start, interval_end=end)
    with control_plane_transaction() as conn:
        ok, reason = verify_checkpoint(conn, cid)
    if not ok:
        log.error("audit checkpoint %s failed self-verification: %s", cid, reason)
    else:
        log.debug("audit checkpoint %s sealed [%s, %s) and verified", cid, start, end)
