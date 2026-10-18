#!/usr/bin/env python3
"""Enterprise B2B team tenancy demo — API walkthrough for sales / acceptance.

Flow:
  1. Admin registers and creates a team (shared wallet)
  2. Admin invites member + viewer (direct add when emails already exist)
  3. Admin funds the team wallet
  4. Member launches an instance billed to the team
  5. Viewer can read team resources but is blocked from mutations

Reads AUDIT_* from .env.audit when present; otherwise registers ephemeral users.

Usage:
  python scripts/enterprise_team_demo.py
  python scripts/enterprise_team_demo.py --base-url https://xcelsior.ca
  XCELSIOR_ENV=test python scripts/enterprise_team_demo.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import requests

PROJECT = Path(__file__).resolve().parent.parent
ENV_AUDIT = PROJECT / ".env.audit"
STEP = 0


def _load_cfg() -> dict[str, str]:
    cfg: dict[str, str] = {}
    if ENV_AUDIT.exists():
        for line in ENV_AUDIT.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    port = os.environ.get("XCELSIOR_API_PORT", "").strip()
    default_base = f"http://127.0.0.1:{port}" if port else "http://localhost:8000"
    cfg["base"] = (os.environ.get("AUDIT_BASE") or cfg.get("AUDIT_BASE") or default_base).rstrip(
        "/"
    )
    return cfg


def _step(title: str, detail: str = "") -> None:
    global STEP
    STEP += 1
    print(f"\n[{STEP}] {title}")
    if detail:
        print(f"    {detail}")


def _fail(msg: str, resp: requests.Response | None = None) -> int:
    print(f"\nFAIL: {msg}", file=sys.stderr)
    if resp is not None:
        print(f"    HTTP {resp.status_code}: {resp.text[:400]}", file=sys.stderr)
    return 1


def _register(session: requests.Session, base: str, label: str) -> dict:
    email = f"enterprise-demo-{label}-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    password = "StrongPass123!"
    reg = session.post(
        f"{base}/api/auth/register",
        json={"email": email, "password": password, "name": f"Demo {label}"},
        timeout=30,
    )
    if reg.status_code != 200:
        raise RuntimeError(f"register {label}: {reg.status_code} {reg.text[:200]}")
    reg_body = reg.json()
    if reg_body.get("email_verification_required"):
        raise RuntimeError(
            f"register {label}: email verification required on {base}. "
            "Run with XCELSIOR_ENV=test (local/staging) or use pre-verified accounts."
        )
    login = session.post(
        f"{base}/api/auth/login",
        json={"email": email, "password": password},
        timeout=30,
    )
    if login.status_code != 200:
        raise RuntimeError(f"login {label}: {login.status_code} {login.text[:200]}")
    body = login.json()
    user = (reg_body.get("user") or body.get("user") or {})
    token = body.get("access_token") or reg_body.get("access_token")
    if not token:
        raise RuntimeError(f"register {label}: no access_token in register/login response")
    return {
        "email": email,
        "password": password,
        "customer_id": user.get("customer_id", ""),
        "headers": {"Authorization": f"Bearer {token}"},
    }


def _session_for(base: str, email: str, password: str) -> tuple[requests.Session, dict]:
    s = requests.Session()
    s.headers["Content-Type"] = "application/json"
    login = s.post(f"{base}/api/auth/login", json={"email": email, "password": password}, timeout=30)
    if login.status_code != 200:
        raise RuntimeError(f"login failed: {login.status_code} {login.text[:200]}")
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    return s, headers


def main() -> int:
    parser = argparse.ArgumentParser(description="Enterprise team tenancy API demo")
    parser.add_argument("--base-url", default=None, help="API base (default: AUDIT_BASE or localhost)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary on success")
    args = parser.parse_args()

    cfg = _load_cfg()
    base = (args.base_url or cfg["base"]).rstrip("/")

    print("=" * 60)
    print("Xcelsior Enterprise Team Demo")
    print(f"Target: {base}")
    print("=" * 60)

    _step("Register admin, member, and viewer accounts")
    admin_s = requests.Session()
    admin_s.headers["Content-Type"] = "application/json"
    admin = _register(admin_s, base, "admin")
    member = _register(requests.Session(), base, "member")
    viewer = _register(requests.Session(), base, "viewer")
    print(f"    admin={admin['email']}")
    print(f"    member={member['email']}")
    print(f"    viewer={viewer['email']}")

    _step("Admin creates team workspace")
    team_resp = admin_s.post(
        f"{base}/api/teams",
        json={"name": f"Enterprise Demo {uuid.uuid4().hex[:6]}", "plan": "free"},
        headers=admin["headers"],
        timeout=30,
    )
    if team_resp.status_code != 200:
        return _fail("create team", team_resp)
    team_id = team_resp.json()["team_id"]
    billing_customer_id = team_resp.json().get("billing_customer_id") or ""
    print(f"    team_id={team_id}")
    print(f"    billing_customer_id={billing_customer_id}")

    _step("Add member and viewer to team")
    for email, role in ((member["email"], "member"), (viewer["email"], "viewer")):
        add = admin_s.post(
            f"{base}/api/teams/{team_id}/members",
            json={"email": email, "role": role},
            headers=admin["headers"],
            timeout=30,
        )
        if add.status_code != 200:
            return _fail(f"add {role}", add)
        print(f"    {role}: {email}")

    _step("Fund shared team wallet (test/dev direct deposit)")
    deposit = admin_s.post(
        f"{base}/api/billing/wallet/{billing_customer_id}/deposit",
        json={"amount_cad": 50.0, "description": "Enterprise demo credits"},
        headers=admin["headers"],
        timeout=30,
    )
    if deposit.status_code != 200:
        return _fail(
            "deposit team wallet (requires XCELSIOR_ENV=test/dev or admin)",
            deposit,
        )
    balance = deposit.json().get("balance_cad")
    print(f"    balance_cad={balance}")

    _step("Member launches instance — billed to team wallet")
    member_s, member_headers = _session_for(base, member["email"], member["password"])
    launch = member_s.post(
        f"{base}/instance",
        json={"name": "enterprise-demo-instance", "vram_needed_gb": 1},
        headers=member_headers,
        timeout=30,
    )
    if launch.status_code != 200:
        return _fail("member launch", launch)
    job_id = (launch.json().get("instance") or {}).get("job_id") or ""
    owner = (launch.json().get("instance") or {}).get("owner") or ""
    print(f"    job_id={job_id}")
    print(f"    owner={owner}")
    if owner != billing_customer_id:
        return _fail(f"job owner {owner!r} != team billing {billing_customer_id!r}")

    _step("Viewer reads team instance (read-only)")
    viewer_s, viewer_headers = _session_for(base, viewer["email"], viewer["password"])
    read_job = viewer_s.get(f"{base}/instance/{job_id}", headers=viewer_headers, timeout=30)
    if read_job.status_code != 200:
        return _fail("viewer read instance", read_job)
    me = viewer_s.get(f"{base}/api/auth/me", headers=viewer_headers, timeout=30)
    if me.status_code != 200:
        return _fail("viewer auth/me", me)
    user = me.json().get("user") or {}
    print(f"    team_role={user.get('team_role')}")
    print(f"    billing_customer_id={user.get('billing_customer_id')}")

    _step("Viewer blocked from launch, stop, and team deposit")
    checks = [
        (
            "launch",
            viewer_s.post(
                f"{base}/instance",
                json={"name": "viewer-blocked", "vram_needed_gb": 1},
                headers=viewer_headers,
                timeout=30,
            ),
            403,
        ),
        (
            "stop",
            viewer_s.post(f"{base}/instances/{job_id}/stop", headers=viewer_headers, timeout=30),
            403,
        ),
        (
            "deposit",
            viewer_s.post(
                f"{base}/api/billing/wallet/{billing_customer_id}/deposit",
                json={"amount_cad": 5.0},
                headers=viewer_headers,
                timeout=30,
            ),
            403,
        ),
    ]
    for name, resp, expected in checks:
        if resp.status_code != expected:
            return _fail(f"viewer {name} expected {expected}", resp)
        print(f"    {name}: {resp.status_code} (blocked)")

    _step("Member blocked from team wallet deposit (admin-only)")
    member_deposit = member_s.post(
        f"{base}/api/billing/wallet/{billing_customer_id}/deposit",
        json={"amount_cad": 5.0},
        headers=member_headers,
        timeout=30,
    )
    if member_deposit.status_code != 403:
        return _fail("member deposit should be 403", member_deposit)
    print("    member deposit: 403 (admin-only billing)")

    print("\n" + "=" * 60)
    print("PASS — Enterprise team demo complete")
    print("=" * 60)
    print("Narrative for live demo:")
    print("  • Admin creates team → shared wallet for GPU spend")
    print("  • Member launches workloads against team balance")
    print("  • Viewer monitors instances/analytics without mutation rights")
    print("  • Only team admin can add credits to the shared wallet")

    if args.json:
        print(
            json.dumps(
                {
                    "ok": True,
                    "base": base,
                    "team_id": team_id,
                    "billing_customer_id": billing_customer_id,
                    "job_id": job_id,
                    "admin_email": admin["email"],
                    "member_email": member["email"],
                    "viewer_email": viewer["email"],
                },
                indent=2,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())