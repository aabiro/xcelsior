# Xcelsior — NFS model weight cache for serverless presets

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless.repo import ServerlessRepo

log = logging.getLogger("xcelsior.serverless.cache")

_CACHE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def cache_volume_name(model_ref: str, revision: str, region: str) -> str:
    safe_model = _CACHE_NAME_RE.sub("--", model_ref.replace("/", "--"))[:80]
    safe_rev = _CACHE_NAME_RE.sub("-", revision)[:32]
    return f"slvr-cache-{safe_model}-{safe_rev}-{region}"[:120]


def resolve_cache_volume(
    repo: ServerlessRepo,
    owner_id: str,
    model_ref: str,
    revision: str,
    region: str,
    *,
    size_gb: int = 50,
) -> str | None:
    """
    Find or provision an NFS weight-cache volume for (model_ref, revision, region).
    Returns volume_id or None if volumes subsystem unavailable.
    """
    if not model_ref:
        return None
    name = cache_volume_name(model_ref, revision, region)
    try:
        from volumes import get_volume_engine

        ve = get_volume_engine()
        existing = ve.list_volumes(owner_id)
        for vol in existing:
            if vol.get("name") == name and vol.get("status") not in ("deleted", "deleting"):
                return str(vol["volume_id"])
        created = ve.create_volume(
            owner_id=owner_id,
            name=name,
            size_gb=size_gb,
            region=region,
            storage_type="nfs",
        )
        return str(created.get("volume_id") or "")
    except Exception as e:
        log.warning("Cache volume resolve failed for %s: %s", name, e)
        return None


def attach_cache_to_endpoint_spec(spec, volume_id: str | None) -> None:
    if volume_id:
        spec.cache_volume_id = volume_id


def cache_replicate_regions() -> list[str]:
    raw = os.environ.get("XCELSIOR_SERVERLESS_CACHE_REPLICATE_REGIONS", "")
    return [r.strip() for r in raw.split(",") if r.strip()]


def replicate_cache_volumes(
    repo: ServerlessRepo,
    owner_id: str,
    model_ref: str,
    revision: str,
    primary_region: str,
    *,
    size_gb: int = 50,
) -> dict[str, str | None]:
    """
    Provision NFS weight-cache volumes in the primary region plus configured peers.

    Returns {region: volume_id} for all regions touched (primary included).
    """
    regions = [primary_region] + [
        r for r in cache_replicate_regions() if r and r != primary_region
    ]
    out: dict[str, str | None] = {}
    for region in regions:
        out[region] = resolve_cache_volume(
            repo,
            owner_id,
            model_ref,
            revision,
            region,
            size_gb=size_gb,
        )
    return out