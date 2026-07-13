# Xcelsior — Path-based vanity slugs for serverless endpoints (Phase 15)

from __future__ import annotations

import re

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def endpoint_vanity_slug(name: str, endpoint_id: str) -> str:
    """Derive a stable URL-safe slug from the endpoint display name."""
    base = (name or "").strip().lower()
    slug = _SLUG_RE.sub("-", base).strip("-")[:48]
    if slug:
        return slug
    eid = (endpoint_id or "").strip().lower()
    return eid[:12] if eid else "endpoint"


def vanity_invoke_path(endpoint_id: str, slug: str) -> str:
    """Primary invoke path for serverless endpoints."""
    clean_slug = endpoint_vanity_slug(slug, endpoint_id)
    return f"/v1/serverless/{endpoint_id}/{clean_slug}"


def clean_endpoint_display_name(value: str) -> str:
    """Remove provider prefixes from stored user-facing endpoint names."""
    name = (value or "").strip()
    if "/" not in name:
        return name
    return [part for part in name.split("/") if part][-1] or name
