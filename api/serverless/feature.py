# Xcelsior — Serverless feature flag (global env + per-owner allowlist)

from __future__ import annotations

import os


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


def serverless_global_enabled() -> bool:
    """Global kill switch. Defaults off in prod, on in test."""
    if os.environ.get("XCELSIOR_ENV", "").lower() == "test":
        return _env_bool("XCELSIOR_SERVERLESS_ENABLED", "true")
    return _env_bool("XCELSIOR_SERVERLESS_ENABLED", "false")


def serverless_allowlist() -> set[str] | None:
    """None = no allowlist (all owners when globally enabled)."""
    raw = os.environ.get("XCELSIOR_SERVERLESS_ALLOWLIST", "").strip()
    if not raw:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def serverless_enabled_for_owner(owner_id: str) -> bool:
    if not serverless_global_enabled():
        return False
    allowlist = serverless_allowlist()
    if allowlist is None:
        return True
    return str(owner_id).strip() in allowlist


def serverless_feature_status(*, owner_id: str | None = None) -> dict:
    """Public status payload for /api/v2/serverless/enabled."""
    global_on = serverless_global_enabled()
    allowlist = serverless_allowlist()
    allowed = False
    if global_on and owner_id:
        allowed = serverless_enabled_for_owner(owner_id)
    elif global_on and allowlist is None:
        allowed = True
    return {
        "enabled": allowed,
        "global_enabled": global_on,
        "allowlist_active": allowlist is not None,
    }