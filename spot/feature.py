"""Spot instance feature flag — global kill switch for launch."""

from __future__ import annotations

import os


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


def spot_global_enabled() -> bool:
    """Global spot launch switch. Defaults on (test + prod)."""
    return _env_bool("XCELSIOR_SPOT_ENABLED", "true")


def spot_feature_status() -> dict:
    """Public status for UI and API callers."""
    enabled = spot_global_enabled()
    return {
        "enabled": enabled,
        "global_enabled": enabled,
        "message": (
            None
            if enabled
            else "Spot instances are temporarily unavailable. On-demand launches are unaffected."
        ),
    }