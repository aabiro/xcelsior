"""Validating a reported SSH host-key fingerprint.

The value a worker reports here is **served back to users** as the thing they
compare against before trusting an SSH connection. So it is attacker-controlled
text until proven otherwise, and the proof has to happen at the API boundary
rather than at the point of display — a renderer that has to remember to escape
is a renderer that will forget.

## Why the shape is exact rather than lenient

`ssh-keygen -lf` prints `SHA256:` followed by **43 characters** of unpadded
base64 (32 bytes → ceil(32/3)*4 = 44 with one `=` of padding, which OpenSSH
strips). That is not a range: any other length is not a SHA-256 fingerprint, and
accepting one would publish a value that can never match what `ssh-keyscan`
produces.

The refusals the plan names, and what each would cost if accepted:

| input | if accepted |
|---|---|
| `"yes"` | a user compares it, it never matches, and they learn to ignore the warning |
| `"SHA256:"` | the prefix alone looks authoritative and verifies nothing |
| 4 KB of junk | attacker-controlled text on a page, and a row that is mostly payload |

## Why empty is not a failure

An agent that reports nothing is normal — a non-interactive launch, an older
worker, a proxy-terminated host where the container legitimately holds no keys.
`None` means "unknown", and every consumer must keep saying so. A **wrong** value
is worse than none: `null` makes a model say "this cannot be verified", while a
plausible-but-wrong value makes it say "verified".
"""

from __future__ import annotations

import re

#: `SHA256:` + exactly 43 unpadded-base64 characters, anchored at both ends.
#: Anchored on purpose: an unanchored search would happily pull a valid-looking
#: fingerprint out of the middle of 4 KB of junk and store the junk's neighbour.
_FINGERPRINT = re.compile(r"^SHA256:[A-Za-z0-9+/]{43}$")

#: Nothing legitimate approaches this. A bound before the regex keeps a
#: pathological input from being handed to the matcher at all.
MAX_LENGTH = 128


def parse_host_key_fingerprint(raw: object) -> str | None:
    """The reported fingerprint if it is one, else `None`.

    Never raises and never returns a partial value: the caller stores exactly
    what this returns, so "unknown" has to be representable as itself rather
    than as an empty string that renders as a blank verification line.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if not isinstance(raw, str):
        return None
    candidate = raw.strip()
    if not candidate or len(candidate) > MAX_LENGTH:
        return None
    return candidate if _FINGERPRINT.match(candidate) else None


def fingerprint_is_valid(raw: object) -> bool:
    """Predicate form, for callers that only need the yes/no."""
    return parse_host_key_fingerprint(raw) is not None
