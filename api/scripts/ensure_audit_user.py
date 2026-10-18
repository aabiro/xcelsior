#!/usr/bin/env python3
"""Provision a dedicated MCP/dashboard audit user (verified, no MFA).

Usage:
  # Local test DB (auto-verified register path also works via API):
  XCELSIOR_ENV=test python scripts/ensure_audit_user.py --provision --write-env

  # Production (run inside api container on VPS, or via SSH wrapper):
  python scripts/ensure_audit_user.py --provision --write-env

  # Verify login against AUDIT_BASE:
  python scripts/ensure_audit_user.py --verify-login

Reads/writes repo-root `.env.audit` (gitignored). See `.env.audit.example`.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
import uuid
from pathlib import Path

import requests

PROJECT = Path(__file__).resolve().parent.parent
ENV_AUDIT = PROJECT / ".env.audit"
EXAMPLE = PROJECT / ".env.audit.example"

DEFAULT_EMAIL = "site-audit@xcelsior.ca"
DEFAULT_NAME = "Site Audit Bot"


def _load_env_audit() -> dict[str, str]:
    out: dict[str, str] = {}
    if not ENV_AUDIT.exists():
        return out
    for line in ENV_AUDIT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


def _write_env_audit(values: dict[str, str]) -> None:
    lines = [
        "# MCP / Playwright dashboard audit credentials — DO NOT COMMIT",
        f"# Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
    ]
    for key in (
        "AUDIT_BASE",
        "AUDIT_EMAIL",
        "AUDIT_PASSWORD",
        "AUDIT_USER_ID",
        "AUDIT_CUSTOMER_ID",
    ):
        if key in values and values[key]:
            lines.append(f"{key}={values[key]}")
    lines.append("")
    ENV_AUDIT.write_text("\n".join(lines), encoding="utf-8")
    os.chmod(ENV_AUDIT, 0o600)
    print(f"Wrote {ENV_AUDIT}")


def _cfg() -> dict[str, str]:
    file_env = _load_env_audit()
    base = os.environ.get("AUDIT_BASE") or file_env.get("AUDIT_BASE") or "https://xcelsior.ca"
    email = (os.environ.get("AUDIT_EMAIL") or file_env.get("AUDIT_EMAIL") or DEFAULT_EMAIL).lower()
    password = os.environ.get("AUDIT_PASSWORD") or file_env.get("AUDIT_PASSWORD") or ""
    return {"base": base.rstrip("/"), "email": email, "password": password}


def verify_login(cfg: dict[str, str]) -> dict:
    if not cfg["password"]:
        raise SystemExit("AUDIT_PASSWORD missing — run --provision or set in .env.audit")
    r = requests.post(
        f"{cfg['base']}/api/auth/login",
        json={"email": cfg["email"], "password": cfg["password"]},
        timeout=30,
    )
    body: dict = {}
    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text[:300]}
    ok = r.status_code == 200 and not body.get("mfa_required")
    if r.status_code == 403 and body.get("email_verification_required"):
        ok = False
        body["hint"] = "Run --provision on server to set email_verified=1"
    return {"ok": ok, "status": r.status_code, "body": body}


def provision_db(cfg: dict[str, str], *, write_env: bool) -> dict[str, str]:
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    from routes._deps import _hash_password
    from db import UserStore
    from billing import get_billing_engine

    email = cfg["email"]
    password = cfg["password"] or secrets.token_urlsafe(18)
    password_hash, salt = _hash_password(password)

    existing = UserStore.get_user(email)
    if existing:
        UserStore.update_user(
            email,
            {
                "password_hash": password_hash,
                "salt": salt,
                "email_verified": 1,
                "email_verification_token": None,
                "email_verification_expires": None,
                "mfa_enabled": 0,
                "name": DEFAULT_NAME,
                "role": "submitter",
                "is_admin": 0,
            },
        )
        user_id = existing["user_id"]
        customer_id = existing.get("customer_id") or f"cust-{uuid.uuid4().hex[:8]}"
        if not existing.get("customer_id"):
            UserStore.update_user(email, {"customer_id": customer_id})
        action = "updated"
    else:
        user_id = f"user-{uuid.uuid4().hex[:12]}"
        customer_id = f"cust-{uuid.uuid4().hex[:8]}"
        user = {
            "user_id": user_id,
            "email": email,
            "name": DEFAULT_NAME,
            "password_hash": password_hash,
            "salt": salt,
            "role": "submitter",
            "is_admin": 0,
            "customer_id": customer_id,
            "provider_id": None,
            "country": "CA",
            "province": "ON",
            "created_at": time.time(),
        }
        UserStore.create_user(user)
        UserStore.update_user(
            email,
            {
                "email_verified": 1,
                "email_verification_token": None,
                "email_verification_expires": None,
                "mfa_enabled": 0,
            },
        )
        action = "created"
        try:
            get_billing_engine().deposit(customer_id, 0.0, "Audit account bootstrap")
        except Exception:
            pass

    out = {
        "AUDIT_BASE": cfg["base"],
        "AUDIT_EMAIL": email,
        "AUDIT_PASSWORD": password,
        "AUDIT_USER_ID": user_id,
        "AUDIT_CUSTOMER_ID": customer_id,
        "action": action,
    }
    if write_env:
        _write_env_audit(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision MCP dashboard audit credentials")
    parser.add_argument("--provision", action="store_true", help="Create/update user in DB")
    parser.add_argument("--verify-login", action="store_true", help="POST /api/auth/login probe")
    parser.add_argument("--write-env", action="store_true", help="Write .env.audit")
    parser.add_argument("--json", action="store_true", help="Print JSON summary (no secrets unless --show-secret)")
    parser.add_argument("--show-secret", action="store_true", help="Include password in JSON output")
    args = parser.parse_args()

    cfg = _cfg()
    result: dict = {"email": cfg["email"], "base": cfg["base"]}

    if args.provision:
        prov = provision_db(cfg, write_env=args.write_env)
        result["provision"] = {k: v for k, v in prov.items() if k != "AUDIT_PASSWORD"}
        if args.show_secret:
            result["provision"]["password"] = prov["AUDIT_PASSWORD"]
        cfg["password"] = prov["AUDIT_PASSWORD"]

    if args.verify_login or not args.provision:
        if cfg["password"]:
            result["login"] = verify_login(cfg)
        elif not args.provision:
            raise SystemExit("No password — run with --provision first")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result.get("provision"):
            print(f"Provisioned {result['provision'].get('action')} {cfg['email']}")
        if result.get("login"):
            st = "OK" if result["login"]["ok"] else f"FAIL ({result['login']['status']})"
            print(f"Login: {st}")

    if result.get("login") and not result["login"]["ok"] and args.verify_login:
        sys.exit(1)


if __name__ == "__main__":
    main()