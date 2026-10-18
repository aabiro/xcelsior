#!/usr/bin/env python3
"""Seed the connector-directory reviewer account (adoption plan X2.16, GAP 7).

An empty account is the most common reason a connector review fails: the
reviewer calls `list_instances`, gets `[]`, and cannot tell a working
integration from a broken one. This converges a standing account that is
deliberately *ordinary* and deliberately *populated*:

  * **Not an admin.** A reviewer must see the public customer profile, not the
    operator surface. Handing over the existing `demo@xcelsior.ca` admin would
    contradict the whole profile split — that account stays for our own E2E.
  * **No MFA, email already verified, no SMS or confirmation step.** A reviewer
    who has to click a link in an inbox they do not own is a failed review.
  * **No IP gate.** The demo-account button is whitelisted to our own networks,
    which is precisely wrong here — the reviewer connects from Anthropic's or
    OpenAI's egress. This account authenticates from anywhere, so the password
    is the only control and must be a real secret.
  * **Real sample data.** A wallet balance, instances in several states, and an
    invoice, all owned by the reviewer's own tenant.

Usage::

    XCELSIOR_REVIEWER_PASSWORD='...' python3 scripts/seed_reviewer_account.py
    python3 scripts/seed_reviewer_account.py --check     # report, write nothing

Idempotent: run it as often as you like. In production the password must be
supplied — this script will not invent a standing credential for an account
that anyone on the internet can reach.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import time
import uuid
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

REVIEWER_EMAIL = os.environ.get("XCELSIOR_REVIEWER_EMAIL", "reviewer@xcelsior.ca").strip().lower()
REVIEWER_NAME = os.environ.get("XCELSIOR_REVIEWER_NAME", "Directory Reviewer")
STARTING_CREDIT_CAD = float(os.environ.get("XCELSIOR_REVIEWER_CREDIT_CAD", "250"))

# What every run guarantees. `is_admin` is 0 on purpose and is the single most
# important field here.
DESIRED = {
    "email_verified": 1,
    "email_verification_token": None,
    "email_verification_expires": None,
    "mfa_enabled": 0,
    "is_admin": 0,
    "role": "submitter",
    "name": REVIEWER_NAME,
    "country": "CA",
    "province": "ON",
}

# Sample instances, chosen so a reviewer sees the states they will ask about:
# something running to inspect, something finished to read logs from, and
# something that failed so error handling is visible rather than theoretical.
SAMPLE_INSTANCES = [
    ("reviewer-finetune-demo", "RTX 4090", 24, "running"),
    ("reviewer-batch-inference", "RTX 3090", 24, "completed"),
    ("reviewer-failed-run", "RTX 4090", 24, "failed"),
]


def _password() -> str:
    supplied = os.environ.get("XCELSIOR_REVIEWER_PASSWORD", "").strip()
    if supplied:
        return supplied
    env = os.environ.get("XCELSIOR_ENV", "dev").strip().lower()
    if env in {"prod", "production"}:
        raise SystemExit(
            "XCELSIOR_REVIEWER_PASSWORD is required in production. This account has no "
            "IP gate, so the password is its only control — generate a strong one, store "
            "it in the secret manager, and hand it to the reviewer through the submission "
            "form, not over email."
        )
    generated = f"Rv-{secrets.token_urlsafe(18)}"
    print(f"[reviewer-seed] no password supplied; generated for this environment: {generated}")
    return generated


def _customer_id(user: dict) -> str:
    return str(user.get("customer_id") or "")


def check() -> int:
    from db import UserStore

    user = UserStore.get_user(REVIEWER_EMAIL)
    if not user:
        print(f"[reviewer-seed] MISSING: {REVIEWER_EMAIL} does not exist in this database.")
        return 1
    drift = {key: (user.get(key), want) for key, want in DESIRED.items() if user.get(key) != want}
    print(f"[reviewer-seed] present: {REVIEWER_EMAIL}")
    print(f"[reviewer-seed]   user_id     = {user.get('user_id')}")
    print(f"[reviewer-seed]   customer_id = {_customer_id(user)}")
    print(f"[reviewer-seed]   is_admin    = {user.get('is_admin')} (must be 0)")
    print(f"[reviewer-seed]   role        = {user.get('role')}")
    print(f"[reviewer-seed]   verified    = {user.get('email_verified')}")
    print(f"[reviewer-seed]   mfa_enabled = {user.get('mfa_enabled')}")

    instances = _reviewer_instances(_customer_id(user))
    print(f"[reviewer-seed]   instances   = {len(instances)}")
    balance = _wallet_balance(_customer_id(user))
    print(f"[reviewer-seed]   wallet      = ${balance:.2f} CAD")

    problems = []
    if drift:
        problems.append(f"field drift (got -> want): {drift}")
    if user.get("is_admin"):
        problems.append("account is an admin — a reviewer must not see the operator surface")
    if not instances:
        problems.append("no sample instances — list_instances would return an empty array")
    if balance <= 0:
        problems.append("no wallet balance — should_i_run_this would decline every request")
    for problem in problems:
        print(f"[reviewer-seed]   PROBLEM: {problem}")
    if problems:
        print("[reviewer-seed] not converged — run without --check to fix.")
        return 1
    print("[reviewer-seed] converged. A reviewer can sign in and see real data.")
    return 0


def _wallet_balance(customer_id: str) -> float:
    if not customer_id:
        return 0.0
    try:
        from billing import get_billing_engine

        wallet = get_billing_engine().get_wallet(customer_id)
        return float(wallet.get("balance_cad") or wallet.get("balance") or 0.0)
    except Exception as exc:  # pragma: no cover - reporting path
        print(f"[reviewer-seed] (non-fatal) could not read wallet: {exc}")
        return 0.0


def _reviewer_instances(customer_id: str) -> list:
    if not customer_id:
        return []
    try:
        import scheduler

        return [
            job
            for job in scheduler.list_jobs()
            if str(job.get("owner") or "") == customer_id
        ]
    except Exception as exc:  # pragma: no cover - reporting path
        print(f"[reviewer-seed] (non-fatal) could not list instances: {exc}")
        return []


def seed() -> int:
    from db import UserStore
    from routes._deps import _hash_password

    password = _password()
    password_hash, salt = _hash_password(password)
    existing = UserStore.get_user(REVIEWER_EMAIL)

    if existing:
        updates = {**DESIRED, "password_hash": password_hash, "salt": salt}
        if not existing.get("customer_id"):
            updates["customer_id"] = f"cust-{uuid.uuid4().hex[:8]}"
        UserStore.update_user(REVIEWER_EMAIL, updates)
        action = "updated"
    else:
        UserStore.create_user({
            "user_id": f"user-reviewer-{uuid.uuid4().hex[:8]}",
            "email": REVIEWER_EMAIL,
            "name": REVIEWER_NAME,
            "password_hash": password_hash,
            "salt": salt,
            "role": "submitter",
            "is_admin": 0,
            "customer_id": f"cust-{uuid.uuid4().hex[:8]}",
            "provider_id": None,
            "country": "CA",
            "province": "ON",
            "created_at": time.time(),
        })
        UserStore.update_user(REVIEWER_EMAIL, DESIRED)
        action = "created"

    user = UserStore.get_user(REVIEWER_EMAIL) or {}
    customer_id = _customer_id(user)
    print(f"[reviewer-seed] {action}: {REVIEWER_EMAIL} (customer_id={customer_id}, not an admin)")

    if _wallet_balance(customer_id) < STARTING_CREDIT_CAD:
        try:
            from billing import get_billing_engine

            get_billing_engine().deposit(
                customer_id, STARTING_CREDIT_CAD, "Connector reviewer sample credit"
            )
            print(f"[reviewer-seed] credited ${STARTING_CREDIT_CAD:.2f} CAD")
        except Exception as exc:  # noqa: BLE001
            print(f"[reviewer-seed] (non-fatal) credit skipped: {exc}")

    _seed_instances(customer_id)

    print("\n[reviewer-seed] hand the reviewer:")
    print(f"[reviewer-seed]   connector URL : {os.environ.get('XCELSIOR_MCP_RESOURCE_AUDIENCE', 'https://mcp.xcelsior.ca/mcp')}")
    print(f"[reviewer-seed]   email         : {REVIEWER_EMAIL}")
    print("[reviewer-seed]   password      : (the one you supplied; never email it)")
    print("[reviewer-seed] the account has no MFA, no email confirmation step, and no IP gate.")
    return 0


def _seed_instances(customer_id: str) -> None:
    """Give the reviewer instances to look at, without touching anyone else's."""
    if not customer_id:
        return
    try:
        import scheduler
    except Exception as exc:  # noqa: BLE001
        print(f"[reviewer-seed] (non-fatal) scheduler unavailable, no sample instances: {exc}")
        return

    existing = {str(job.get("name") or ""): job for job in _reviewer_instances(customer_id)}
    for name, gpu_model, vram, status in SAMPLE_INSTANCES:
        try:
            job = existing.get(name)
            if job is None:
                # submit_job returns the job record, not the id.
                job = scheduler.submit_job(
                    name=name,
                    vram_needed_gb=vram,
                    gpu_model=gpu_model,
                    owner=customer_id,
                    interactive=False,
                    command="python train.py --epochs 1",
                )
            # Converge the status on every run, not only at creation. An
            # instance that exists but sat in the wrong state is exactly the
            # drift a --check run is supposed to catch and a re-run to fix.
            landed = str(job.get("status") or "queued")
            if landed != status:
                # Set directly rather than by running real work — seeding a
                # reviewer account must never consume capacity or bill a host.
                updated = scheduler.update_job_status(job["job_id"], status)
                landed = status if updated is not None else landed
            print(f"[reviewer-seed]   sample instance {name} ({landed})")
        except Exception as exc:  # noqa: BLE001
            print(f"[reviewer-seed] (non-fatal) could not seed {name}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="report state, write nothing")
    args = parser.parse_args()
    sys.exit(check() if args.check else seed())


if __name__ == "__main__":
    main()
