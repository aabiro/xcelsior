"""Per-host agent bearer tokens and field-wide rotation (blueprint §19.2).

    "bearer tokens remain only during migration, scoped to one host and
    rotated. Do not use one platform API token for the fleet long term."

The shared platform token (``api-token``/``api-admin``) is a single
credential that authenticates the entire fleet: stealing it from one GPU
host yields authority over every host. This module replaces it with a
per-host credential that is:

* **scoped** — a token authenticates exactly one ``host_id`` and cannot
  be replayed against another host, even by an otherwise valid worker;
* **hashed at rest** — only SHA-256 digests are stored, so a database
  read does not yield usable credentials;
* **rotatable without downtime** — rotation supersedes the old token with
  a bounded grace window rather than revoking it instantly, so a worker
  that has fetched a new token but not yet adopted it is never locked
  out; the grace window is what makes *field-wide* rotation safe;
* **revocable immediately** — revocation is terminal and takes effect on
  the next request, with no grace.

Rollout is a three-state flag, ``XCELSIOR_AGENT_HOST_TOKENS``:

``off``
    Host tokens are not accepted at all. Nothing changes.
``allow`` (default)
    Host tokens are accepted alongside the existing shared-bearer
    migration path. Accepting a strictly stronger credential cannot break
    a deployed fleet, so this is safe as the default.
``require``
    The shared fleet bearer is refused for host-scoped agent calls; only
    a per-host token or a verified gateway identity is accepted. **This
    is the completion of field-wide bearer rotation** and is the operator
    flip once every host holds a token.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

log = logging.getLogger("xcelsior.control_plane.agent_tokens")

#: Opaque token prefix. Makes credentials greppable in logs/secret
#: scanners and lets verification reject obviously-not-a-host-token
#: bearers without touching the database.
TOKEN_PREFIX = "xat_"

#: Bytes of entropy in the secret portion (256 bits).
_SECRET_BYTES = 32

#: Characters of the token stored in cleartext for display/lookup hints.
_PREFIX_LEN = 12

DEFAULT_TTL_DAYS = 90
DEFAULT_ROTATION_GRACE_MINUTES = 60

#: ``last_used_at`` is refreshed at most this often, so a busy worker's
#: poll loop does not turn every authenticated request into a write.
_LAST_USED_THROTTLE_SECONDS = 60


class AgentTokenError(Exception):
    """Base class for token administration failures."""


class UnknownHost(AgentTokenError):
    """Token operations require a registered host."""


class TokenRejected(AgentTokenError):
    """A presented token is not usable (unknown, expired, or revoked)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class IssuedToken:
    """A freshly minted credential. ``secret`` is returned exactly once."""

    token_id: str
    host_id: str
    secret: str
    prefix: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class VerifiedHostToken:
    """A presented token that authenticated successfully."""

    token_id: str
    host_id: str
    status: str
    expires_at: datetime

    @property
    def is_superseded(self) -> bool:
        """True while the holder is inside a rotation grace window."""
        return self.status == "superseded"


# ── Mode flag ─────────────────────────────────────────────────────────

_VALID_MODES = ("off", "allow", "require")


def host_token_mode() -> str:
    """Resolved ``XCELSIOR_AGENT_HOST_TOKENS`` mode.

    An unrecognised value fails **closed** to ``off`` rather than
    silently enabling an auth path an operator did not ask for.
    """
    raw = (os.environ.get("XCELSIOR_AGENT_HOST_TOKENS") or "allow").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return "allow"
    if raw in ("0", "false", "no"):
        return "off"
    if raw in _VALID_MODES:
        return raw
    log.warning("unrecognised XCELSIOR_AGENT_HOST_TOKENS=%r — failing closed to 'off'", raw)
    return "off"


def host_tokens_enabled() -> bool:
    return host_token_mode() in ("allow", "require")


def host_tokens_required() -> bool:
    """True once the shared fleet bearer must no longer be accepted."""
    return host_token_mode() == "require"


def token_ttl_days() -> int:
    try:
        value = int(os.environ.get("XCELSIOR_AGENT_TOKEN_TTL_DAYS", DEFAULT_TTL_DAYS))
    except ValueError:
        return DEFAULT_TTL_DAYS
    return value if value > 0 else DEFAULT_TTL_DAYS


def rotation_grace_minutes() -> int:
    try:
        value = int(
            os.environ.get(
                "XCELSIOR_AGENT_TOKEN_ROTATION_GRACE_MINUTES",
                DEFAULT_ROTATION_GRACE_MINUTES,
            )
        )
    except ValueError:
        return DEFAULT_ROTATION_GRACE_MINUTES
    return value if value >= 0 else DEFAULT_ROTATION_GRACE_MINUTES


# ── Secret handling ───────────────────────────────────────────────────


def looks_like_host_token(candidate: str | None) -> bool:
    """Cheap shape check — avoids a DB round trip for platform bearers."""
    return bool(candidate) and str(candidate).startswith(TOKEN_PREFIX)


def hash_token(secret: str) -> str:
    """SHA-256 hex digest of the presented secret.

    The stored value is a digest of a 256-bit random secret, so a slow
    KDF buys nothing here: there is no low-entropy pre-image to grind.
    Lookup by digest also keeps verification a single indexed read
    rather than a scan-and-compare over every row.
    """
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _mint_secret() -> tuple[str, str, str]:
    """(secret, prefix, hash) for a new credential."""
    secret = TOKEN_PREFIX + secrets.token_urlsafe(_SECRET_BYTES)
    return secret, secret[:_PREFIX_LEN], hash_token(secret)


# ── Persistence ───────────────────────────────────────────────────────


def _host_exists(conn, host_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM hosts WHERE host_id = %s", (host_id,)).fetchone()
    return row is not None


def issue_token(
    conn,
    host_id: str,
    *,
    issued_by: str | None = None,
    reason: str | None = None,
    ttl_days: int | None = None,
    grace_minutes: int | None = None,
) -> IssuedToken:
    """Mint a credential for ``host_id``, superseding any active one.

    Runs inside the caller's transaction. The previous active token is
    moved to ``superseded`` with a bounded grace window instead of being
    revoked, so issuing does not strand a running worker mid-poll.
    """
    if not _host_exists(conn, host_id):
        raise UnknownHost(f"host {host_id!r} is not registered")

    ttl = token_ttl_days() if ttl_days is None else ttl_days
    grace = rotation_grace_minutes() if grace_minutes is None else grace_minutes

    # Lock the host's token history so two concurrent issues cannot both
    # believe they superseded the same active row (the partial unique
    # index is the backstop, this makes the loser wait rather than error).
    conn.execute(
        "SELECT token_id FROM host_agent_tokens WHERE host_id = %s FOR UPDATE",
        (host_id,),
    ).fetchall()

    conn.execute(
        """
        UPDATE host_agent_tokens
           SET status = 'superseded',
               superseded_at = clock_timestamp(),
               expires_at = LEAST(
                   expires_at,
                   clock_timestamp() + make_interval(mins => %s)
               )
         WHERE host_id = %s
           AND status = 'active'
        """,
        (grace, host_id),
    )

    secret, prefix, digest = _mint_secret()
    row = conn.execute(
        """
        INSERT INTO host_agent_tokens
            (host_id, token_prefix, token_hash, status, expires_at,
             issued_by, issue_reason)
        VALUES (%s, %s, %s, 'active',
                clock_timestamp() + make_interval(days => %s), %s, %s)
        RETURNING token_id, expires_at
        """,
        (host_id, prefix, digest, ttl, issued_by, reason),
    ).fetchone()
    return IssuedToken(
        token_id=str(row[0]),
        host_id=host_id,
        secret=secret,
        prefix=prefix,
        expires_at=row[1],
    )


def verify_token(
    conn,
    secret: str,
    *,
    required_host_id: str | None = None,
    client_ip: str | None = None,
) -> VerifiedHostToken:
    """Authenticate ``secret``; raise :class:`TokenRejected` otherwise.

    A ``superseded`` token still authenticates until its grace deadline —
    that overlap is what lets an operator rotate the whole field without
    a synchronised restart. ``revoked`` never authenticates.
    """
    if not looks_like_host_token(secret):
        raise TokenRejected("not_a_host_token")

    row = conn.execute(
        """
        SELECT token_id, host_id, status, expires_at,
               expires_at <= clock_timestamp() AS past_deadline
          FROM host_agent_tokens
         WHERE token_hash = %s
        """,
        (hash_token(secret),),
    ).fetchone()
    if row is None:
        raise TokenRejected("unknown_token")

    token_id, host_id, status, expires_at, past_deadline = row
    if status == "revoked":
        raise TokenRejected("revoked")
    if status == "expired" or past_deadline:
        raise TokenRejected("expired")
    if status not in ("active", "superseded"):  # pragma: no cover - CHECK guards
        raise TokenRejected("not_usable")
    if required_host_id and str(host_id) != required_host_id:
        # A stolen worker credential must not be replayable against a
        # different host — this is the whole point of scoping.
        raise TokenRejected("host_mismatch")

    conn.execute(
        """
        UPDATE host_agent_tokens
           SET last_used_at = clock_timestamp(),
               last_used_ip = COALESCE(%s, last_used_ip)
         WHERE token_id = %s
           AND (last_used_at IS NULL
                OR last_used_at < clock_timestamp() - make_interval(secs => %s))
        """,
        (client_ip, token_id, _LAST_USED_THROTTLE_SECONDS),
    )
    return VerifiedHostToken(
        token_id=str(token_id),
        host_id=str(host_id),
        status=str(status),
        expires_at=expires_at,
    )


def rotate_token(
    conn,
    current_secret: str,
    *,
    client_ip: str | None = None,
    grace_minutes: int | None = None,
) -> IssuedToken:
    """Exchange a valid credential for a fresh one (worker-driven rotation).

    Rotating from an already-superseded token is allowed and idempotent
    in effect: the caller proves possession of *a* valid credential for
    the host, and receives the host's new active one.
    """
    verified = verify_token(conn, current_secret, client_ip=client_ip)
    return issue_token(
        conn,
        verified.host_id,
        issued_by=f"rotation:{verified.token_id}",
        reason="worker_rotation",
        grace_minutes=grace_minutes,
    )


def revoke_token(
    conn,
    *,
    token_id: str | None = None,
    host_id: str | None = None,
    reason: str = "operator_revocation",
) -> int:
    """Immediately revoke one token, or every live token for a host.

    Revocation has no grace window — it is the compromise response.
    Returns the number of rows revoked.
    """
    if not token_id and not host_id:
        raise AgentTokenError("revoke_token requires token_id or host_id")
    if token_id:
        cur = conn.execute(
            """
            UPDATE host_agent_tokens
               SET status = 'revoked',
                   revoked_at = clock_timestamp(),
                   revoked_reason = %s
             WHERE token_id = %s
               AND status IN ('active', 'superseded')
            """,
            (reason, token_id),
        )
    else:
        cur = conn.execute(
            """
            UPDATE host_agent_tokens
               SET status = 'revoked',
                   revoked_at = clock_timestamp(),
                   revoked_reason = %s
             WHERE host_id = %s
               AND status IN ('active', 'superseded')
            """,
            (reason, host_id),
        )
    return int(cur.rowcount or 0)


def expire_stale_tokens(conn, *, limit: int = 500) -> int:
    """Mark past-deadline tokens ``expired`` (durable scheduled task).

    Verification already refuses a past-deadline token, so this is
    hygiene rather than enforcement: it keeps the partial unique index
    and operator views honest about which credentials are live.
    """
    cur = conn.execute(
        """
        UPDATE host_agent_tokens
           SET status = 'expired'
         WHERE token_id IN (
               SELECT token_id
                 FROM host_agent_tokens
                WHERE status IN ('active', 'superseded')
                  AND expires_at <= clock_timestamp()
                ORDER BY expires_at
                FOR UPDATE SKIP LOCKED
                LIMIT %s
         )
        """,
        (limit,),
    )
    return int(cur.rowcount or 0)


def list_tokens(conn, host_id: str) -> list[dict[str, Any]]:
    """Operator view of a host's credential history (never the secret)."""
    rows = conn.execute(
        """
        SELECT token_id, host_id, token_prefix, status, issued_at, expires_at,
               last_used_at, last_used_ip, superseded_at, revoked_at,
               revoked_reason, issued_by, issue_reason
          FROM host_agent_tokens
         WHERE host_id = %s
         ORDER BY issued_at DESC
        """,
        (host_id,),
    ).fetchall()
    fields = (
        "token_id",
        "host_id",
        "token_prefix",
        "status",
        "issued_at",
        "expires_at",
        "last_used_at",
        "last_used_ip",
        "superseded_at",
        "revoked_at",
        "revoked_reason",
        "issued_by",
        "issue_reason",
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(zip(fields, row))
        for key in (
            "issued_at",
            "expires_at",
            "last_used_at",
            "superseded_at",
            "revoked_at",
        ):
            if isinstance(item.get(key), datetime):
                item[key] = item[key].isoformat()
        item["token_id"] = str(item["token_id"])
        out.append(item)
    return out


def rotation_coverage(conn, host_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """How far field-wide rotation has progressed.

    Operators need this before flipping ``XCELSIOR_AGENT_HOST_TOKENS`` to
    ``require``: flipping while a host has no live token locks that host
    out of the control plane.
    """
    params: tuple[Any, ...] = ()
    host_filter = ""
    if host_ids is not None:
        ids = tuple(host_ids)
        if not ids:
            return {"hosts": 0, "with_active_token": 0, "missing": [], "ready": True}
        host_filter = "WHERE h.host_id = ANY(%s)"
        params = (list(ids),)

    rows = conn.execute(
        f"""
        SELECT h.host_id,
               EXISTS (
                   SELECT 1 FROM host_agent_tokens t
                    WHERE t.host_id = h.host_id
                      AND t.status IN ('active', 'superseded')
                      AND t.expires_at > clock_timestamp()
               ) AS has_token
          FROM hosts h
          {host_filter}
        """,
        params or None,
    ).fetchall()
    missing = [str(r[0]) for r in rows if not r[1]]
    return {
        "hosts": len(rows),
        "with_active_token": len(rows) - len(missing),
        "missing": sorted(missing),
        "ready": not missing,
    }
