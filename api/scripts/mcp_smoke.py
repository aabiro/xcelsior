#!/usr/bin/env python3
"""Staging smoke for Xcelsior MCP + upstream API.

Exercises: health → OAuth token → list GPUs → estimate cost → dry-run create_instance.

Env:
  MCP_BASE          default https://xcelsior.ca (MCP at {MCP_BASE}/mcp)
  XCELSIOR_API_URL  default MCP_BASE
  MCP_CLIENT_ID     OAuth machine client id
  MCP_CLIENT_SECRET OAuth machine client secret
"""

from __future__ import annotations

import json
import os
import sys

import requests

BASE = (os.environ.get("MCP_BASE") or os.environ.get("XCELSIOR_API_URL") or "https://xcelsior.ca").rstrip(
    "/"
)
MCP_URL = os.environ.get("MCP_URL") or f"{BASE}/mcp"
CLIENT_ID = os.environ.get("MCP_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("MCP_CLIENT_SECRET", "")


def _mcp_tool_call(token: str, name: str, arguments: dict) -> dict:
    """Minimal JSON-RPC tools/call against Streamable HTTP MCP."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    r = requests.post(
        MCP_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        json=payload,
        timeout=60,
    )
    return {"status": r.status_code, "body": r.text[:2000]}


def main() -> int:
    results: dict[str, object] = {}

    health = requests.get(f"{BASE}/mcp/health", timeout=15)
    if health.status_code == 404:
        health = requests.get(MCP_URL.replace("/mcp", "") + "/health", timeout=15)
    results["mcp_health"] = health.status_code

    if not CLIENT_ID or not CLIENT_SECRET:
        print("Set MCP_CLIENT_ID and MCP_CLIENT_SECRET for full smoke", file=sys.stderr)
        print(json.dumps(results, indent=2))
        return 0 if health.status_code == 200 else 1

    token_r = requests.post(
        f"{BASE}/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=30,
    )
    results["oauth_token"] = token_r.status_code
    if token_r.status_code != 200:
        print(json.dumps(results, indent=2))
        return 1

    token = token_r.json().get("access_token", "")
    gpus = requests.get(
        f"{BASE}/api/v2/gpu/available",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    results["list_gpus"] = gpus.status_code

    estimate = requests.post(
        f"{BASE}/api/pricing/estimate",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"gpu_model": "RTX 4090", "duration_hours": 1, "spot": False},
        timeout=30,
    )
    results["estimate_cost"] = estimate.status_code

    dry_create = _mcp_tool_call(
        token,
        "create_instance",
        {"name": "mcp-smoke-preview", "gpu_model": "RTX 4090", "confirm": False},
    )
    results["dry_create_instance"] = dry_create

    print(json.dumps(results, indent=2))
    ok = (
        results.get("mcp_health") == 200
        and results.get("oauth_token") == 200
        and results.get("list_gpus") == 200
        and results.get("estimate_cost") == 200
        and dry_create.get("status") in (200, 202)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())