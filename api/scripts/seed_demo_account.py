#!/usr/bin/env python3
# ============================================================================
#  DEMO ACCOUNT SEED  —  run once per database, then leave it alone.
# ============================================================================
#  This plants the standing demo admin account (demo@xcelsior.ca) used by the
#  IP-gated "Demo account" login button and by Playwright / E2E automation.
#
#  It is DELIBERATELY idempotent and boring: run it as many times as you like,
#  it always converges the account to the same known state (admin, verified,
#  no MFA, fixed password) and never touches anything else. The credentials
#  just stay put — that's the whole point. This script is not part of any
#  normal request flow.
#
#  The email / password / name come from demo_account.py (the single source of
#  truth), so this can never drift from what the login endpoint hands out.
#
#  USAGE
#    # Test DB:
#    XCELSIOR_ENV=test python scripts/seed_demo_account.py
#    # Production (run inside the api container / on the VPS, real DB env):
#    python scripts/seed_demo_account.py
#    # Just check current state without writing:
#    python scripts/seed_demo_account.py --check
# ============================================================================

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from demo_account import DEMO_EMAIL, DEMO_NAME, DEMO_PASSWORD  # noqa: E402

# The exact fields we guarantee on the demo account every run.
_DESIRED = {
    "email_verified": 1,
    "email_verification_token": None,
    "email_verification_expires": None,
    "mfa_enabled": 0,
    "is_admin": 1,
    "role": "admin",
    "name": DEMO_NAME,
}


def _verify_password(user: dict) -> bool:
    import hmac

    from routes._deps import _hash_password  # local import: heavy deps

    try:
        expected, _ = _hash_password(DEMO_PASSWORD, user.get("salt", ""))
        return hmac.compare_digest(expected, user.get("password_hash", ""))
    except Exception:
        return False


def check() -> int:
    from db import UserStore

    user = UserStore.get_user(DEMO_EMAIL)
    if not user:
        print(f"[demo-seed] MISSING: {DEMO_EMAIL} does not exist in this DB.")
        return 1
    pw_ok = _verify_password(user)
    drift = {
        k: (user.get(k), v) for k, v in _DESIRED.items() if user.get(k) != v
    }
    print(f"[demo-seed] present: {DEMO_EMAIL}")
    print(f"[demo-seed]   user_id      = {user.get('user_id')}")
    print(f"[demo-seed]   is_admin     = {user.get('is_admin')}")
    print(f"[demo-seed]   role         = {user.get('role')}")
    print(f"[demo-seed]   verified     = {user.get('email_verified')}")
    print(f"[demo-seed]   mfa_enabled  = {user.get('mfa_enabled')}")
    print(f"[demo-seed]   password_ok  = {pw_ok}")
    if drift:
        print(f"[demo-seed]   DRIFT (field: got->want): {drift}")
    if not pw_ok or drift:
        print("[demo-seed] state is NOT converged — run without --check to fix.")
        return 1
    print("[demo-seed] fully converged. Nothing to do.")
    return 0


def seed() -> int:
    from routes._deps import _hash_password
    from db import UserStore

    password_hash, salt = _hash_password(DEMO_PASSWORD)
    existing = UserStore.get_user(DEMO_EMAIL)

    if existing:
        updates = {**_DESIRED, "password_hash": password_hash, "salt": salt}
        if not existing.get("customer_id"):
            updates["customer_id"] = f"cust-{uuid.uuid4().hex[:8]}"
        UserStore.update_user(DEMO_EMAIL, updates)
        user_id = existing["user_id"]
        action = "updated"
    else:
        user_id = f"user-demo-{uuid.uuid4().hex[:8]}"
        user = {
            "user_id": user_id,
            "email": DEMO_EMAIL,
            "name": DEMO_NAME,
            "password_hash": password_hash,
            "salt": salt,
            "role": "admin",
            "is_admin": 1,
            "customer_id": f"cust-{uuid.uuid4().hex[:8]}",
            "provider_id": None,
            "country": "CA",
            "province": "ON",
            "created_at": time.time(),
        }
        UserStore.create_user(user)
        UserStore.update_user(DEMO_EMAIL, _DESIRED)
        action = "created"
        # Give the demo account a little starting balance so demo/E2E flows that
        # look at billing don't render empty. Best-effort.
        try:
            from billing import get_billing_engine

            cust = UserStore.get_user(DEMO_EMAIL).get("customer_id")
            if cust:
                get_billing_engine().deposit(cust, 100.0, "Demo account seed credit")
        except Exception as e:  # noqa: BLE001
            print(f"[demo-seed] (non-fatal) starter credit skipped: {e}")

    print(f"[demo-seed] {action}: {DEMO_EMAIL} (user_id={user_id}, admin, verified, no MFA)")
    print("[demo-seed] credentials are fixed in demo_account.py — leave them put.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed/converge the demo admin account.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report current state and exit non-zero if not converged (no writes).",
    )
    args = parser.parse_args()
    sys.exit(check() if args.check else seed())


if __name__ == "__main__":
    main()
