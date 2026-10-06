"""Environment resolution that fails closed.

Several security decisions were independently keyed on
``os.environ.get("XCELSIOR_ENV", "dev")``. Unset, empty, or unrecognised
values resolve to production. Staging is neither relaxed nor production.

Use ``env_kind()`` and handle all three states. ``if is_production(): strict()
else: relaxed()`` is fail-open for staging and is the defect in issue #17.
"""

from __future__ import annotations

import os
from enum import Enum

RELAXED_ENVS: frozenset[str] = frozenset({"dev", "development", "test", "local"})
PREPROD_ENVS: frozenset[str] = frozenset({"staging", "preprod"})
PRODUCTION_ENVS: frozenset[str] = frozenset({"production", "prod"})
KNOWN_ENVS: frozenset[str] = RELAXED_ENVS | PREPROD_ENVS | PRODUCTION_ENVS
PRODUCTION = "production"


class EnvKind(str, Enum):
    """The three states two booleans were encoding.

    RELAXED may use insecure fallbacks. STAGING and PRODUCTION may not.
    Callers that only care about production capabilities must match PRODUCTION
    explicitly. A missing arm is a bug, not a relaxed default.
    """

    RELAXED = "relaxed"
    STAGING = "staging"
    PRODUCTION = "production"


def resolve_env(raw: str | None = None) -> str:
    value = os.environ.get("XCELSIOR_ENV") if raw is None else raw
    normalized = (value or "").strip().lower()
    if normalized in KNOWN_ENVS:
        return normalized
    return PRODUCTION


def env_kind(raw: str | None = None) -> EnvKind:
    resolved = resolve_env(raw)
    if resolved in RELAXED_ENVS:
        return EnvKind.RELAXED
    if resolved in PREPROD_ENVS:
        return EnvKind.STAGING
    return EnvKind.PRODUCTION


def is_relaxed_env(raw: str | None = None) -> bool:
    return env_kind(raw) is EnvKind.RELAXED


def is_production(raw: str | None = None) -> bool:
    return env_kind(raw) is EnvKind.PRODUCTION


def is_strict_env(raw: str | None = None) -> bool:
    """Staging and production. Use this anywhere the old else-branch was relaxed."""
    return env_kind(raw) is not EnvKind.RELAXED
