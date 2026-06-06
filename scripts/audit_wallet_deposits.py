#!/usr/bin/env python3
"""Audit wallet deposits for payment-proof gaps.

Classifies deposit transactions as:
  - proven: Stripe/PayPal/crypto idempotency or description markers
  - promo: signup free-credits
  - bootstrap: $0 account-creation ledger rows
  - suspicious: positive amount without payment proof

Usage:
  python scripts/audit_wallet_deposits.py
  # On production VPS:
  docker compose exec -T api-blue python scripts/audit_wallet_deposits.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

PAYMENT_MARKERS = ("stripe", "paypal", "lightning", "btc", "bitcoin", "crypto", "ln ")
BOOTSTRAP_MARKERS = ("account created", "audit account bootstrap", "wallet created")


def classify(row: dict) -> str:
    amount = float(row.get("amount_cad") or 0)
    desc = (row.get("description") or "").lower()
    key = (row.get("idempotency_key") or "").lower()

    if key.startswith("free-credits-"):
        return "promo"
    if key.startswith("stripe:") or key.startswith("paypal-"):
        return "proven"
    if any(m in desc for m in PAYMENT_MARKERS):
        return "proven"
    if amount <= 0 or any(m in desc for m in BOOTSTRAP_MARKERS):
        return "bootstrap"
    return "suspicious"


def main() -> int:
    from billing import get_billing_engine

    be = get_billing_engine()
    with be._conn() as conn:
        rows = conn.execute(
            """
            SELECT tx_id, customer_id, amount_cad, description, idempotency_key, created_at
            FROM wallet_transactions
            WHERE tx_type = 'deposit'
            ORDER BY created_at DESC
            """
        ).fetchall()

    buckets: dict[str, list] = {"proven": [], "promo": [], "bootstrap": [], "suspicious": []}
    for r in rows:
        item = {
            "tx_id": r["tx_id"],
            "customer_id": r["customer_id"],
            "amount_cad": float(r["amount_cad"]),
            "description": r["description"],
            "idempotency_key": r["idempotency_key"] or "",
            "created_at": r["created_at"],
        }
        buckets[classify(item)].append(item)

    summary = {
        "total_deposits": len(rows),
        "proven_payment_deposits": len(buckets["proven"]),
        "promo_deposits": len(buckets["promo"]),
        "bootstrap_deposits": len(buckets["bootstrap"]),
        "suspicious_deposits": len(buckets["suspicious"]),
        "suspicious_total_cad": round(sum(x["amount_cad"] for x in buckets["suspicious"]), 2),
        "paypal_configured": bool(
            os.environ.get("PAYPAL_CLIENT_ID") and os.environ.get("PAYPAL_CLIENT_SECRET")
        ),
        "stripe_mode": os.environ.get("XCELSIOR_STRIPE_MODE", ""),
        "xcelsior_env": os.environ.get("XCELSIOR_ENV", ""),
    }
    print(
        json.dumps(
            {
                "summary": summary,
                "suspicious": buckets["suspicious"][:100],
                "recent_proven": buckets["proven"][:20],
            },
            indent=2,
        )
    )
    return 1 if buckets["suspicious"] else 0


if __name__ == "__main__":
    raise SystemExit(main())