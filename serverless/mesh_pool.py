"""Two-host (or N-host) LMCache mesh configuration for preset vLLM pools."""

from __future__ import annotations

import os
from typing import Any


def mesh_host_urls() -> list[str]:
    """Comma-separated LMCache/vLLM mesh peers from env."""
    raw = os.environ.get("XCELSIOR_LMCACHE_MESH_HOSTS", "").strip()
    if not raw:
        return []
    return [u.strip() for u in raw.split(",") if u.strip()]


def mesh_pool_status() -> dict[str, Any]:
    hosts = mesh_host_urls()
    live_peers = int(os.environ.get("XCELSIOR_LIVE_MESH_PEER_COUNT", "0") or 0)
    return {
        "configured_hosts": hosts,
        "host_count": len(hosts),
        "two_host_mesh_ready": len(hosts) >= 2,
        "live_mesh_peer_count": live_peers,
        "live_two_host_mesh": live_peers >= 2,
        "mesh_mode": "live" if live_peers >= 2 else "configured_env_only",
        "lmcache_remote_url": os.environ.get("XCELSIOR_LMCACHE_REMOTE_URL", ""),
        "dynamo_required": False,
    }


def lmcache_mesh_env() -> dict[str, str]:
    """Env vars for vLLM workers joining a multi-host LMCache mesh."""
    hosts = mesh_host_urls()
    if len(hosts) < 2:
        return {}
    remote = os.environ.get("XCELSIOR_LMCACHE_REMOTE_URL", "").strip()
    if not remote and hosts:
        remote = hosts[0]
    out: dict[str, str] = {
        "XCELSIOR_LMCACHE_MESH_HOSTS": ",".join(hosts),
        "LMCACHE_USE_EXPERIMENTAL": os.environ.get("LMCACHE_USE_EXPERIMENTAL", "True"),
    }
    if remote:
        out["LMCACHE_REMOTE_URL"] = remote
        out["LMCACHE_REMOTE_SERDE"] = os.environ.get(
            "XCELSIOR_LMCACHE_REMOTE_SERDE", "naive"
        )
    return out