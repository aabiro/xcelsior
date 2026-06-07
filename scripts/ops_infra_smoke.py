#!/usr/bin/env python3
"""Production infrastructure smoke — PayPal + NFS volumes readiness.

Reads AUDIT_* from .env.audit or env. Exits 0 when critical paths are healthy.

Checks:
  - PayPal enabled flag + create-order auth gate
  - /readyz nfs_volumes (mode full when NFS configured)
  - Provider PayPal status endpoint shape (if user is a provider)
  - Volume CRUD smoke (create/list/get/delete) when AUDIT credentials set
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import requests

PROJECT = Path(__file__).resolve().parent.parent
ENV_AUDIT = PROJECT / ".env.audit"


def _load_cfg() -> dict[str, str]:
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


def main() -> int:
    cfg = _load_cfg()
    base = cfg["base"]
    results: dict[str, object] = {"base": base}

    paypal = requests.get(f"{base}/api/billing/paypal/enabled", timeout=30)
    results["paypal_enabled_status"] = paypal.status_code
    paypal_body = paypal.json() if paypal.ok else {}
    results["paypal_enabled"] = paypal_body
    paypal_on = bool(paypal_body.get("enabled")) if isinstance(paypal_body, dict) else False

    readyz = requests.get(f"{base}/readyz", timeout=30)
    results["readyz_status"] = readyz.status_code
    nfs = {}
    if readyz.ok:
        body = readyz.json()
        nfs = body.get("nfs_volumes") or {}
        results["readyz"] = {
            "ok": body.get("ok"),
            "nfs_mode": nfs.get("mode"),
            "nfs_configured": nfs.get("configured"),
            "nfs_reachable": nfs.get("reachable"),
            "nfs_server": nfs.get("server"),
            "nfs_ssh_host": nfs.get("ssh_host"),
        }
    else:
        results["readyz_error"] = readyz.text[:300]

    nfs_ok = not nfs.get("configured") or (nfs.get("reachable") is True and nfs.get("mode") == "full")

    volumes_ok = True
    provider_paypal_ok = True
    if cfg["email"] and cfg["password"]:
        s = requests.Session()
        s.headers["Content-Type"] = "application/json"
        login = s.post(
            f"{base}/api/auth/login",
            json={"email": cfg["email"], "password": cfg["password"]},
            timeout=30,
        )
        results["login"] = login.status_code
        if login.ok:
            s.headers["Authorization"] = f"Bearer {login.json()['access_token']}"
            me = s.get(f"{base}/api/auth/me", timeout=30)
            if me.ok:
                provider_id = (me.json().get("user") or {}).get("provider_id")
                if provider_id:
                    pp = s.get(f"{base}/api/providers/{provider_id}/paypal", timeout=30)
                    results["provider_paypal_status"] = pp.status_code
                    if pp.ok:
                        results["provider_paypal"] = pp.json().get("paypal")
                    else:
                        provider_paypal_ok = False
            paypal_order = s.post(
                f"{base}/api/billing/paypal/create-order",
                json={"customer_id": cfg["email"], "amount_cad": 5.0},
                timeout=30,
            )
            results["paypal_create_order"] = paypal_order.status_code

            if nfs.get("configured") and nfs.get("mode") == "full":
                vol_name = f"ops-smoke-{uuid.uuid4().hex[:8]}"
                created = s.post(
                    f"{base}/api/v2/volumes",
                    json={"name": vol_name, "size_gb": 1, "encrypted": False},
                    timeout=60,
                )
                results["volume_create"] = created.status_code
                if created.ok:
                    vol = (created.json().get("volume") or {})
                    volume_id = vol.get("volume_id")
                    results["volume_id"] = volume_id
                    if not volume_id or not vol.get("owner_id"):
                        volumes_ok = False
                    else:
                        got = s.get(f"{base}/api/v2/volumes/{volume_id}", timeout=30)
                        results["volume_get"] = got.status_code
                        if not got.ok:
                            volumes_ok = False
                        deleted = s.delete(f"{base}/api/v2/volumes/{volume_id}", timeout=60)
                        results["volume_delete"] = deleted.status_code
                        if not deleted.ok:
                            volumes_ok = False
                else:
                    volumes_ok = False

    ok = paypal_on and readyz.status_code == 200 and nfs_ok and provider_paypal_ok and volumes_ok
    results["pass"] = ok
    print(json.dumps(results, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())