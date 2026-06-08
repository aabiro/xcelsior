#!/usr/bin/env python3
"""Production billing path smoke checks.

Reads credentials from .env.audit (or AUDIT_* env vars). Verifies:
  - auth + wallet read
  - direct deposit blocked (403)
  - payment-intent auth + IDOR
  - PayPal/Lightning/crypto enabled flags
"""

from __future__ import annotations

import json
import os
import sys
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
    cfg["base"] = (os.environ.get("AUDIT_BASE") or cfg.get("AUDIT_BASE") or "https://xcelsior.ca").rstrip("/")
    cfg["email"] = os.environ.get("AUDIT_EMAIL") or cfg.get("AUDIT_EMAIL", "")
    cfg["password"] = os.environ.get("AUDIT_PASSWORD") or cfg.get("AUDIT_PASSWORD", "")
    cfg["customer_id"] = os.environ.get("AUDIT_CUSTOMER_ID") or cfg.get("AUDIT_CUSTOMER_ID") or cfg["email"]
    return cfg


def main() -> int:
    cfg = _load_cfg()
    if not cfg["email"] or not cfg["password"]:
        print("Missing AUDIT_EMAIL/AUDIT_PASSWORD — run scripts/provision_audit_user.sh")
        return 1

    base = cfg["base"]
    s = requests.Session()
    s.headers["Content-Type"] = "application/json"

    results: dict[str, object] = {}

    login = s.post(f"{base}/api/auth/login", json={"email": cfg["email"], "password": cfg["password"]}, timeout=30)
    results["login"] = login.status_code
    if login.status_code != 200:
        print(json.dumps(results, indent=2))
        return 1
    s.headers["Authorization"] = f"Bearer {login.json()['access_token']}"
    cid = cfg["customer_id"]

    wallet = s.get(f"{base}/api/billing/wallet/{cid}", timeout=30)
    results["wallet"] = wallet.status_code

    deposit = s.post(
        f"{base}/api/billing/wallet/{cid}/deposit",
        json={"amount_cad": 1.0, "description": "prod smoke"},
        timeout=30,
    )
    results["direct_deposit"] = deposit.status_code

    pi = s.post(f"{base}/api/billing/payment-intent", json={"customer_id": cid, "amount_cad": 5.0}, timeout=30)
    results["payment_intent"] = pi.status_code

    idor = s.post(
        f"{base}/api/billing/payment-intent",
        json={"customer_id": "victim@example.com", "amount_cad": 5.0},
        timeout=30,
    )
    results["payment_intent_idor"] = idor.status_code

    for path in (
        "/api/billing/paypal/enabled",
        "/api/billing/lightning/enabled",
        "/api/billing/crypto/enabled",
    ):
        r = requests.get(f"{base}{path}", timeout=30)
        try:
            results[path] = r.json()
        except Exception:
            results[path] = {"status": r.status_code, "raw": r.text[:200]}

    paypal_body = results.get("/api/billing/paypal/enabled", {})
    paypal_on = paypal_body.get("enabled") if isinstance(paypal_body, dict) else False
    results["paypal_enabled"] = paypal_on

    paypal_order = s.post(
        f"{base}/api/billing/paypal/create-order",
        json={"customer_id": cid, "amount_cad": 5.0},
        timeout=30,
    )
    results["paypal_create_order"] = paypal_order.status_code
    if paypal_order.status_code == 200:
        po_body = paypal_order.json()
        results["paypal_order_id"] = po_body.get("order_id") or po_body.get("id")

    if pi.status_code == 200:
        pi_body = pi.json()
        intent = pi_body.get("intent") if isinstance(pi_body.get("intent"), dict) else pi_body
        results["stripe_payment_intent_id"] = intent.get("stripe_intent_id") or intent.get("intent_id")
        results["stripe_client_secret"] = bool(intent.get("client_secret"))

    # Capture without buyer approval must fail safely (no wallet credit).
    if results.get("paypal_order_id"):
        bogus_capture = s.post(
            f"{base}/api/billing/paypal/capture-order",
            json={"order_id": results["paypal_order_id"], "customer_id": cid},
            timeout=30,
        )
        results["paypal_capture_unapproved"] = bogus_capture.status_code

    paypal_idor = s.post(
        f"{base}/api/billing/paypal/create-order",
        json={"customer_id": "victim@example.com", "amount_cad": 5.0},
        timeout=30,
    )
    results["paypal_create_order_idor"] = paypal_idor.status_code

    ok = (
        results["login"] == 200
        and results["wallet"] == 200
        and results["direct_deposit"] == 403
        and results["payment_intent"] != 401
        and results["payment_intent_idor"] == 403
        and paypal_on is True
        and results["paypal_create_order"] != 401
        and results["paypal_create_order_idor"] == 403
        and results.get("stripe_client_secret") is True
        and bool(results.get("paypal_order_id"))
    )
    results["pass"] = ok
    print(json.dumps(results, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())