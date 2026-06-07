#!/usr/bin/env python3
"""Staging smoke test for persistent volumes API (metadata-only or NFS).

Usage:
  python scripts/volumes_e2e_smoke.py --email you@example.com --password '...'
  python scripts/volumes_e2e_smoke.py --token 'Bearer ...' --base-url https://xcelsior.ca
  python scripts/volumes_e2e_smoke.py --infra-only   # CRUD only; skips instance launch

Exits 0 on success, 1 on failure. Does not require a running GPU instance.
With --infra-only, launch is skipped (no wallet/GPU needed).
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parent.parent
ENV_AUDIT = PROJECT / ".env.audit"


def _load_audit_cfg() -> dict[str, str]:
    cfg: dict[str, str] = {}
    if ENV_AUDIT.exists():
        for line in ENV_AUDIT.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    cfg["base"] = (os.environ.get("AUDIT_BASE") or cfg.get("AUDIT_BASE") or "https://xcelsior.ca").rstrip(
        "/"
    )
    cfg["email"] = os.environ.get("AUDIT_EMAIL") or cfg.get("AUDIT_EMAIL", "")
    cfg["password"] = os.environ.get("AUDIT_PASSWORD") or cfg.get("AUDIT_PASSWORD", "")
    return cfg


def _headers(token: str | None) -> dict[str, str]:
    if not token:
        return {}
    if token.lower().startswith("bearer "):
        return {"Authorization": token}
    return {"Authorization": f"Bearer {token}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Volumes API smoke test")
    audit = _load_audit_cfg()
    parser.add_argument("--base-url", default=audit["base"] or "http://localhost:8000")
    parser.add_argument("--email", default=audit["email"] or None)
    parser.add_argument("--password", default=audit["password"] or None)
    parser.add_argument("--token", help="Bearer access token (skips login)")
    parser.add_argument(
        "--infra-only",
        action="store_true",
        help="Skip instance launch (NFS provision CRUD only; no wallet/GPU required)",
    )
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    client = httpx.Client(base_url=base, timeout=60.0)

    token = args.token
    if not token:
        if not args.email or not args.password:
            print("Provide --token or --email + --password", file=sys.stderr)
            return 1
        login = client.post(
            "/api/auth/login",
            json={"email": args.email, "password": args.password},
        )
        if login.status_code != 200:
            print(f"Login failed: {login.status_code} {login.text[:200]}", file=sys.stderr)
            return 1
        token = login.json().get("access_token")
        if not token:
            print("Login response missing access_token", file=sys.stderr)
            return 1

    hdrs = _headers(token)
    name = f"smoke-vol-{uuid.uuid4().hex[:8]}"

    created = client.post(
        "/api/v2/volumes",
        headers=hdrs,
        json={"name": name, "size_gb": 1, "encrypted": False},
    )
    if created.status_code != 200:
        print(f"Create failed: {created.status_code} {created.text[:300]}", file=sys.stderr)
        return 1
    body = created.json()
    vol = body.get("volume") or {}
    volume_id = vol.get("volume_id")
    if not volume_id:
        print(f"Create missing volume_id: {body}", file=sys.stderr)
        return 1
    if not vol.get("owner_id"):
        print(f"Create response missing owner_id: {vol}", file=sys.stderr)
        return 1
    print(f"created volume_id={volume_id} owner_id={vol.get('owner_id')}")

    fetched = client.get(f"/api/v2/volumes/{volume_id}", headers=hdrs)
    if fetched.status_code != 200:
        print(f"Get failed: {fetched.status_code}", file=sys.stderr)
        return 1

    listed = client.get("/api/v2/volumes", headers=hdrs)
    if listed.status_code != 200:
        print(f"List failed: {listed.status_code}", file=sys.stderr)
        return 1
    ids = [v["volume_id"] for v in listed.json().get("volumes") or []]
    if volume_id not in ids:
        print(f"Volume {volume_id} not in list", file=sys.stderr)
        return 1

    if not args.infra_only:
        launched = client.post(
            "/instance",
            headers=hdrs,
            json={"name": f"smoke-{name}", "vram_needed_gb": 1, "volume_ids": [volume_id]},
        )
        if launched.status_code == 402:
            print("launch skipped: insufficient wallet balance (use --infra-only for NFS CRUD smoke)")
        elif launched.status_code != 200:
            print(
                f"Launch with volume_ids failed: {launched.status_code} {launched.text[:300]}",
                file=sys.stderr,
            )
            return 1
        else:
            print("launch with volume_ids ok")
    else:
        print("infra-only: skipped instance launch")

    deleted = client.delete(f"/api/v2/volumes/{volume_id}", headers=hdrs)
    if deleted.status_code != 200:
        print(f"Delete failed: {deleted.status_code} {deleted.text[:200]}", file=sys.stderr)
        return 1
    print("deleted ok")

    readyz = client.get("/readyz")
    if readyz.status_code == 200:
        nfs = readyz.json().get("nfs_volumes") or {}
        print(f"readyz nfs_volumes: mode={nfs.get('mode')} configured={nfs.get('configured')}")

    print("volumes_e2e_smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())