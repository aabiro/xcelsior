"""Routes: ssh."""

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request

from routes._deps import (
    _require_auth,
)
from db import UserStore
import hashlib as _hashlib

log = logging.getLogger(__name__)
router = APIRouter()

VALID_SSH_KEY_TYPES = {
    "ssh-rsa",
    "ssh-ed25519",
    "ecdsa-sha2-nistp256",
    "ecdsa-sha2-nistp384",
    "ecdsa-sha2-nistp521",
    "sk-ssh-ed25519@openssh.com",
    "sk-ecdsa-sha2-nistp256@openssh.com",
}


# ── Helper: _validate_ssh_public_key ──


def _validate_ssh_public_key(key_str: str) -> str:
    """Validate and normalize an SSH public key string. Returns the key type or raises."""
    import base64 as _b64, re as _re

    key_str = key_str.strip()
    # Remove any comment-only lines
    lines = [l.strip() for l in key_str.splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines:
        raise ValueError("Empty SSH key")
    key_str = lines[0]  # Use only the first key line
    parts = key_str.split(None, 2)
    if len(parts) < 2:
        raise ValueError("Invalid SSH public key format")
    key_type = parts[0]
    if key_type not in VALID_SSH_KEY_TYPES:
        raise ValueError(f"Unsupported key type: {key_type}")
    # Validate base64 data
    try:
        _b64.b64decode(parts[1], validate=True)
    except Exception as e:
        raise ValueError("Invalid base64 key data")
    return key_type


# ── Helper: _ssh_key_fingerprint ──


def _ssh_key_fingerprint(key_str: str) -> str:
    """Compute SHA-256 fingerprint of an SSH public key (like ssh-keygen -l)."""
    import base64 as _b64

    parts = key_str.strip().split(None, 2)
    raw = _b64.b64decode(parts[1])
    digest = _hashlib.sha256(raw).digest()
    fp = _b64.b64encode(digest).rstrip(b"=").decode()
    return f"SHA256:{fp}"


def _trigger_reinject_for_user(user: dict) -> int:
    """Find every running interactive instance owned by ``user`` and ask its
    host's worker agent to re-run ``_inject_ssh_keys``.

    "Just works" UX: when a user adds or removes an SSH key in Settings,
    every live interactive instance picks up the change within one
    worker drain cycle (≤30s) — no instance restart required.

    Best-effort: any failure is logged and swallowed so the public
    SSH-key API never fails because of a downstream queue/db hiccup.
    Returns the number of commands enqueued (for logging).
    """
    try:
        from db import _get_pg_pool
        from routes.agent import enqueue_agent_command
    except Exception as e:
        log.debug("reinject trigger: import failed: %s", e)
        return 0

    from routes._deps import _customer_ids_accessible_by_user

    email = user.get("email") or ""
    candidates = sorted(_customer_ids_accessible_by_user(user))
    if not candidates:
        return 0

    try:
        from control_plane.job_targets import classify_job_target

        pool = _get_pg_pool()
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT j.job_id, j.host_id, j.status, j.active_attempt_id,
                       j.payload->>'container_name' AS container_name,
                       EXISTS (
                           SELECT 1 FROM job_attempts a WHERE a.job_id = j.job_id
                       ) AS has_fenced_history
                  FROM jobs j
                 WHERE j.status = 'running'
                   AND j.host_id IS NOT NULL
                   AND j.payload->>'owner' = ANY(%s)
                   AND (j.payload->>'interactive')::boolean = true
                """,
                (candidates,),
            )
            rows = cur.fetchall() or []
    except Exception as e:
        log.warning("reinject trigger: job lookup failed for %s: %s", email, e)
        return 0

    enqueued = 0
    for row in rows:
        job_id = row[0]
        host_id = row[1]
        status = row[2] or "running"
        active_attempt_id = row[3]
        payload_cname = row[4]
        has_fenced = bool(row[5])
        target = classify_job_target(
            job_id=job_id,
            host_id=host_id,
            status=status,
            active_attempt_id=active_attempt_id,
            has_fenced_history=has_fenced,
            payload_container_name=payload_cname,
        )
        # Fenced-history without live attempt: do not guess legacy xcl-{job}.
        if not target.allows_unfenced_container_command:
            log.info(
                "reinject trigger: skip job=%s class=%s (no live residual target)",
                job_id,
                target.class_,
            )
            continue
        try:
            enqueue_agent_command(
                host_id,
                "reinject_shell",
                {"job_id": job_id, "container_name": target.container_name},
                created_by=f"ssh-key-change:{email}",
                ttl_sec=600,
            )
            enqueued += 1
        except Exception as e:
            # Worst case: queue full / db down — user can still relaunch
            # the instance. Don't fail the API call over this.
            log.warning(
                "reinject trigger: enqueue failed job=%s host=%s: %s",
                job_id,
                host_id,
                e,
            )
    if enqueued:
        log.info(
            "ssh key change for %s — enqueued reinject for %d running instance(s)",
            email,
            enqueued,
        )
    return enqueued


@router.post("/api/ssh/keys", tags=["SSH Keys"])
async def api_add_ssh_key(request: Request):
    """Upload a user SSH public key. Like GitHub/AWS key management.

    **Machine credentials may register a key, and that is a deliberate
    narrowing of the guard that used to stand here.**

    This route called `_require_user_grant`, which refuses
    `auth_type == "client_credentials"` outright. A Quick Connect token is
    exactly that (`oauth_service.py`), so the `register_ssh_key` tool — the
    step Gate P2's journey depends on — answered 403 for the credential this
    product tells people to paste. "Launch → connected, without a browser" was
    unreachable for one of the two credential types, and the tool shipped
    before anyone checked which helper the route used.

    `_require_user_grant` is the right guard for MFA, password and account
    deletion: state whose whole point is that a human is present. Registering a
    public key is not that. It is an ordinary scoped mutation of the caller's
    own resources, and the authorisation question is the one `_require_scope`
    asks — does *this credential* hold `ssh:write`.

    What the narrowing costs is attribution: a key added by an agent used to be
    indistinguishable from one a human pasted into the dashboard. Migration
    `099` adds the two columns that fix that, and they are written here. A key
    registered by a machine credential is attributable to the OAuth client that
    added it, so revoking that client identifies exactly which keys to remove.
    """
    from routes._deps import _require_scope

    user = _require_auth(request)
    _require_scope(user, "ssh:write")
    body = await request.json()
    name = body.get("name", "").strip() or "default"
    public_key = body.get("public_key", "").strip()
    if not public_key:
        raise HTTPException(400, "public_key is required")
    if len(name) > 100:
        raise HTTPException(400, "Key name too long (max 100 characters)")
    if len(public_key) > 16384:
        raise HTTPException(400, "Key too large")
    try:
        _validate_ssh_public_key(public_key)
    except ValueError as e:
        raise HTTPException(400, str(e))
    fingerprint = _ssh_key_fingerprint(public_key)
    # Check for duplicate fingerprint
    existing = UserStore.list_ssh_keys(user["email"])
    for k in existing:
        if k["fingerprint"] == fingerprint:
            raise HTTPException(409, "This key is already added")
    key_id = f"sshk-{uuid.uuid4().hex[:12]}"
    # Normalize: keep only the key line (type + data + optional comment)
    parts = public_key.strip().splitlines()[0].strip().split(None, 2)
    normalized = " ".join(parts[:2])
    if len(parts) > 2:
        normalized += " " + parts[2]
    # Who added this key. `None` for an interactive registration — which is the
    # truth for every row that predates migration 099, and the reason both
    # columns are nullable with no default rather than backfilled to something
    # that reads as a claim.
    from routes._deps import _MACHINE_AUTH_TYPES

    auth_type = str(user.get("auth_type") or "") or None
    machine = auth_type in _MACHINE_AUTH_TYPES
    key_data = {
        "id": key_id,
        "email": user["email"],
        "user_id": user["user_id"],
        "name": name,
        "public_key": normalized,
        "fingerprint": fingerprint,
        "created_at": time.time(),
        "registered_by_client_id": user.get("client_id") if machine else None,
        "registered_by_auth_type": auth_type,
    }
    UserStore.add_ssh_key(key_data)
    # "Just works" UX: push the new key into every running interactive
    # instance owned by this user so they don't need to relaunch.
    _trigger_reinject_for_user(user)
    return {
        "ok": True,
        "id": key_id,
        "name": name,
        "fingerprint": fingerprint,
    }


@router.get("/api/ssh/keys", tags=["SSH Keys"])
def api_list_ssh_keys(request: Request):
    """List the authenticated user's SSH public keys."""
    from routes._deps import _require_scope

    # Narrowed with the POST above, for one reason: a credential that can add a
    # key and not see what it added cannot verify its own work or clean up. The
    # scope split still does the restricting — Quick Connect holds `ssh:write`
    # and not `ssh:read`, so it registers and does not enumerate.
    user = _require_auth(request)
    _require_scope(user, "ssh:read")
    keys = UserStore.list_ssh_keys(user["email"])
    return {
        "ok": True,
        "keys": [
            {
                "id": k["id"],
                "name": k["name"],
                "fingerprint": k["fingerprint"],
                "public_key": k["public_key"],
                "created_at": k["created_at"],
            }
            for k in keys
        ],
    }


@router.delete("/api/ssh/keys/{key_id}", tags=["SSH Keys"])
def api_delete_ssh_key(key_id: str, request: Request):
    """Delete a user SSH public key by ID."""
    from routes._deps import _require_scope

    user = _require_auth(request)
    # Registration requires `ssh:write` and listing requires `ssh:read`, and this
    # required nothing — so deletion, the only destructive one of the three, was
    # the only one a narrowed machine credential could reach. It also calls
    # `_trigger_reinject_for_user` below, which revokes the key from running
    # instances, so an under-scoped caller could lock a user out of their own
    # machines. Same verb asymmetry `tests/test_conditional_scope_guard.py`
    # exists to catch, arriving through a missing line rather than a wrong one.
    _require_scope(user, "ssh:write")
    deleted = UserStore.delete_ssh_key(user["email"], key_id)
    if not deleted:
        raise HTTPException(404, "SSH key not found")
    # Revoke from running instances so the deleted key can no longer connect.
    _trigger_reinject_for_user(user)
    return {"ok": True}
