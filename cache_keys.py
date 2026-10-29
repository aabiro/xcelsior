"""Redis/Valkey key contract (Track B B9.1a, companion §5.4).

One helper builds every ephemeral cache key so the rules are enforced in
one place rather than re-decided at each call site:

    xc:{env}:v1:{family}[:{public}...][:{sha256(secret)}]

Every key carries the environment and a schema version, so two
deployments sharing an instance cannot collide and a format change is a
version bump rather than a silent reinterpretation of existing keys.

**Secret-bearing identifiers are hashed.** Companion §5.4: "Never put raw
bearer tokens, API keys, secrets, email addresses, prompts, or artifact
URLs in key names." A Redis key name is not private — it is visible to
`SCAN`, `MONITOR`, the slowlog, RDB/AOF files on disk, managed-service
support tooling, and any keyspace analyzer or metrics exporter. A key name
containing a live access token hands that token to every one of them.

`hash_token()` already existed in `oauth_service` and was already used for
refresh tokens; this module generalizes the same treatment to every
identifier that reaches a key name.
"""

from __future__ import annotations

import hashlib
import os
import re

NAMESPACE = "xc"
KEY_VERSION = "v1"

# A public segment must be a low-cardinality, non-sensitive token: a kind
# name, a fixed sub-family, a time bucket. Anything else belongs in the
# hashed `secret` position.
_PUBLIC_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def environment() -> str:
    """Deployment environment segment, e.g. `prod` / `test` / `dev`."""
    return (os.environ.get("XCELSIOR_ENV") or "dev").strip().lower() or "dev"


def opaque(value: str) -> str:
    """SHA-256 of a secret- or PII-bearing identifier.

    Deterministic, so a lookup finds the key a write created, and
    irreversible, so the key name discloses nothing.
    """
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def cache_key(family: str, *public: str, secret: str | None = None) -> str:
    """Build a namespaced key.

    Args:
        family: top-level domain — `oauth`, `ratelimit`, `presence`, `cache`.
        public: low-cardinality, non-sensitive segments. Rejected if they
            look like free-form data, which is the shape a leaked secret
            or an email address arrives in.
        secret: an identifier derived from a credential, an email, a
            prompt, or a URL. Hashed before it reaches the key name.

    Raises:
        ValueError: if a public segment is empty, over-long, or contains
            characters that suggest it is not actually a fixed token —
            `@` (email), `:` (would forge a segment boundary), or
            whitespace.
    """
    if not _PUBLIC_SEGMENT_RE.match(family):
        raise ValueError(f"invalid key family: {family!r}")

    parts = [NAMESPACE, environment(), KEY_VERSION, family]
    for segment in public:
        text = str(segment)
        if not _PUBLIC_SEGMENT_RE.match(text):
            raise ValueError(
                f"invalid public key segment {text!r} in family {family!r}. "
                "Public segments are fixed, low-cardinality tokens; anything "
                "derived from a credential, email, prompt, or URL must be "
                "passed as `secret=` so it is hashed (companion §5.4)."
            )
        parts.append(text)

    if secret is not None:
        parts.append(opaque(secret))
    return ":".join(parts)
