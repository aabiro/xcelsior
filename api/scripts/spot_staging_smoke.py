#!/usr/bin/env python3
"""Staging smoke test for spot instances (Phase 10 §10.1).

Usage:
  python scripts/spot_staging_smoke.py
  python scripts/spot_staging_smoke.py --base-url https://staging.xcelsior.ca
  python scripts/spot_staging_smoke.py --email you@example.com --password '...'
  python scripts/spot_staging_smoke.py --launch   # optional spot instance launch
  # Default: public API checks only (no login)

Reads credentials from .env.audit (or AUDIT_* env vars). Exits 0 on success.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parent.parent
ENV_AUDIT = PROJECT / ".env.audit"

POLL_INTERVAL_SEC = 10
POLL_MAX_WAIT_SEC = 300
LAUNCH_IMAGE = "nvidia/cuda:12.0.0-base-ubuntu22.04"


def _load_cfg() -> dict[str, str]:
    cfg: dict[str, str] = {}
    if ENV_AUDIT.exists():
        for line in ENV_AUDIT.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    cfg["base"] = (
        os.environ.get("AUDIT_BASE") or cfg.get("AUDIT_BASE") or "https://xcelsior.ca"
    ).rstrip("/")
    cfg["email"] = os.environ.get("AUDIT_EMAIL") or cfg.get("AUDIT_EMAIL", "")
    cfg["password"] = os.environ.get("AUDIT_PASSWORD") or cfg.get("AUDIT_PASSWORD", "")
    return cfg


def _poll_instance(client: httpx.Client, hdrs: dict[str, str], job_id: str) -> dict:
    deadline = time.time() + POLL_MAX_WAIT_SEC
    last: dict = {}
    while time.time() < deadline:
        resp = client.get(f"/instance/{job_id}", headers=hdrs)
        if resp.status_code == 200:
            last = resp.json().get("instance") or {}
            if last.get("status") in ("running", "failed", "stopped", "completed", "assigned"):
                return last
        time.sleep(POLL_INTERVAL_SEC)
    return last


def _infra_checks(client: httpx.Client, results: dict[str, object], hdrs: dict[str, str] | None = None) -> bool:
    ok = True
    req_hdrs = hdrs or {}

    spot_flag = client.get("/api/pricing/spot-enabled", headers=req_hdrs, timeout=30)
    results["spot_enabled_status"] = spot_flag.status_code
    if spot_flag.status_code == 200:
        body = spot_flag.json()
        results["spot_enabled"] = body.get("enabled")
        if body.get("enabled") is False:
            results["spot_enabled_warning"] = body.get("message")
    else:
        ok = False

    prices = client.get("/spot-prices", headers=req_hdrs, timeout=30)
    results["spot_prices_status"] = prices.status_code
    if prices.status_code == 200:
        body = prices.json()
        price_map = body.get("prices") or body.get("spot_prices") or {}
        results["spot_price_models"] = len(price_map) if isinstance(price_map, dict) else 0
    else:
        ok = False

    openapi_path = PROJECT / "public" / "openapi.json"
    if openapi_path.exists():
        spec = json.loads(openapi_path.read_text())
        job_in = spec.get("components", {}).get("schemas", {}).get("JobIn", {})
        props = job_in.get("properties", {})
        results["openapi_has_pricing_mode"] = "pricing_mode" in props
        results["openapi_has_max_bid"] = "max_bid" in props
        if "max_bid" in props:
            ok = False
    else:
        results["openapi_missing"] = True

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Spot staging smoke test")
    parser.add_argument("--base-url", help="API base URL")
    parser.add_argument("--email", help="Login email")
    parser.add_argument("--password", help="Login password")
    parser.add_argument("--token", help="Bearer token (skips login)")
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Also launch a spot instance and verify pricing_mode in response",
    )
    args = parser.parse_args()

    cfg = _load_cfg()
    base = (args.base_url or cfg["base"]).rstrip("/")
    results: dict[str, object] = {"base": base}

    with httpx.Client(base_url=base, timeout=30.0) as client:
        token = args.token
        if not token:
            email = args.email or cfg["email"]
            password = args.password or cfg["password"]
            if email and password:
                login = client.post("/api/auth/login", json={"email": email, "password": password})
                results["login"] = login.status_code
                if login.status_code == 200:
                    token = login.json()["access_token"]
            elif args.launch:
                print("Missing AUDIT_EMAIL/AUDIT_PASSWORD for --launch")
                results["pass"] = False
                print(json.dumps(results, indent=2))
                return 1

        hdrs: dict[str, str] = {}
        if token:
            hdrs["Authorization"] = f"Bearer {token}"
        if args.launch:
            hdrs["Content-Type"] = "application/json"

        infra_ok = _infra_checks(client, results, hdrs=hdrs)
        if not args.launch:
            results["pass"] = infra_ok
            print(json.dumps(results, indent=2))
            return 0 if infra_ok else 1

        launch = client.post(
            "/instance",
            json={
                "name": f"spot-smoke-{int(time.time())}",
                "vram_needed_gb": 8.0,
                "pricing_mode": "spot",
                "gpu_model": "RTX 4090",
                "docker_image": LAUNCH_IMAGE,
            },
            headers=hdrs,
        )
        results["spot_launch_status"] = launch.status_code
        if launch.status_code not in (200, 201):
            results["spot_launch_body"] = launch.text[:500]
            results["pass"] = False
            print(json.dumps(results, indent=2))
            return 1

        body = launch.json()
        inst = body.get("instance") or {}
        job_id = inst.get("job_id")
        results["spot_job_id"] = job_id
        results["spot_launch_pricing_mode"] = inst.get("pricing_mode")
        if inst.get("pricing_mode") != "spot":
            results["pass"] = False
            print(json.dumps(results, indent=2))
            return 1

        if job_id:
            polled = _poll_instance(client, hdrs, job_id)
            results["spot_final_status"] = polled.get("status")
            results["spot_rate_cad"] = polled.get("spot_rate_cad")

        results["pass"] = infra_ok
        print(json.dumps(results, indent=2))
        return 0 if infra_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())