"""Environment resolution that fails closed.

Several security decisions were independently keyed on
``os.environ.get("XCELSIOR_ENV", "dev")`` — authentication, the JWT signing
secret, the signing algorithm, the Fernet key for stored secrets, SSH host-key
pinning on the terminal. Each treated an *absent* variable as development, so a
single missing env var disabled authentication, selected a signing secret that
lives in this source tree, made stored secrets recoverable from a constant, and
turned off MITM protection.

They also disagreed about what counts as production. ``AUTH_REQUIRED`` asked
``env not in {"dev", "development", "test"}`` while ``security.py`` asked
``env in ("production", "prod")``. A typo like ``prodution`` satisfied the first
(so authentication stayed on) and failed the second (so the insecure dev key was
used). The same value was production for one control and not the other.

This module is the single answer to "are we allowed to relax anything", and it
answers no unless explicitly told otherwise:

* unset, empty, or whitespace  → production
* an unrecognised value        → production
* ``dev`` / ``development`` / ``test`` / ``local`` → that value

Case and surrounding whitespace are normalised, so a ``.env`` line with a
trailing space does not silently flip a developer's machine into production
mode — the failure this guards against is the reverse, and being strict about
whitespace would only push people toward looser checks.

Deliberately importable from anywhere: standard library only, no project
imports, so it cannot participate in an import cycle.
"""

from __future__ import annotations

import os

#: The only values that may fall back to an insecure default — a committed
#: signing secret, a deterministic encryption key, authentication off. Staging
#: is deliberately absent: it is a real deployment holding real data.
RELAXED_ENVS: frozenset[str] = frozenset({"dev", "development", "test", "local"})

#: Deployments that are not development but are also not the production VPS.
#: They must never use an insecure fallback.
PREPROD_ENVS: frozenset[str] = frozenset({"staging", "preprod"})

#: Spellings that name the production deployment itself.
PRODUCTION_ENVS: frozenset[str] = frozenset({"production", "prod"})

#: Every value we recognise. Anything outside this set is a typo or an
#: environment nobody taught this module about, and is treated as production.
KNOWN_ENVS: frozenset[str] = RELAXED_ENVS | PREPROD_ENVS | PRODUCTION_ENVS

PRODUCTION = "production"


def resolve_env(raw: str | None = None) -> str:
    """The effective environment name. Unknown values resolve to production.

    Never raises: every caller is on a security path where refusing to answer
    would be less safe than answering strictly.

    `raw` exists so this can be exercised without mutating `os.environ`. Tests
    that swapped the variable to drive each branch leaked into unrelated
    modules — anything that reads the environment at call time and caches, such
    as the cache-key namespace, saw a different value mid-suite. A pure decision
    should be testable purely.
    """
    value = os.environ.get("XCELSIOR_ENV") if raw is None else raw
    normalized = (value or "").strip().lower()
    if normalized in KNOWN_ENVS:
        return normalized
    return PRODUCTION


def is_relaxed_env(raw: str | None = None) -> bool:
    """May this process fall back to an insecure default?

    True only for an explicitly named development context. This is the question
    behind the signing secret, the Fernet key, and `AUTH_REQUIRED` — and the
    answer for staging is **no**, because staging holds real data.

    Do not re-derive this with a membership test on the raw variable; that is
    the pattern that produced the defects this module exists to prevent.
    """
    return resolve_env(raw) in RELAXED_ENVS


def is_production(raw: str | None = None) -> bool:
    """Is this the production deployment itself?

    Narrower than ``not is_relaxed_env()``: staging is neither relaxed nor
    production. Use `is_relaxed_env()` for "may we weaken this"; use this only
    for "is this the production deployment", where the distinction is a
    deliberate capability rather than a security relaxation.

    **Known trap, deliberately left rather than papered over.** Two booleans
    encode three states, so ``if is_production(): strict() else: relaxed()``
    sends staging down the relaxed branch — fail-open for the environment
    holding real data, which is the defect class this module exists to close,
    arriving through the new API instead of `os.environ.get`. Every call site
    here is written as "relaxed, or not" precisely to avoid that shape.

    The distinction was justified by `routes/agent.py` honouring
    ``XCELSIOR_ALLOW_UNAUTH_AGENT`` on staging while production refuses it
    outright. That capability does not currently function: the flag is set in no
    ``.env``, compose file, systemd unit or shell history, it is hardcoded to
    "0" in CI, and the startup validator raises on it wherever enforcement is
    on — which includes staging. So the justification is documentation, not
    behaviour. Collapsing the two states, or making this an exhaustively-handled
    enum, is tracked separately; doing it here would bundle a design change into
    a fail-closed fix.
    """
    return resolve_env(raw) in PRODUCTION_ENVS
