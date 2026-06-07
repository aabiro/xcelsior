#!/usr/bin/env python3
"""Credit the MCP audit user's wallet for smoke/E2E tests.

Safe for production: uses idempotency_key so re-runs do not double-credit.

Usage (on VPS):
  docker compose exec -T api-blue python scripts/fund_audit_wallet.py
  docker compose exec -T api-blue python scripts/fund_audit_wallet.py --amount 50

Usage (local with .env.audit):
  python3 scripts/fund_audit_wallet.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
ENV_AUDIT = PROJECT / ".env.audit"


def _customer_id() -> str:
    if os.environ.get("AUDIT_CUSTOMER_ID"):
        return os.environ["AUDIT_CUSTOMER_ID"].strip()
    if ENV_AUDIT.exists():
        for line in ENV_AUDIT.read_text().splitlines():
            if line.startswith("AUDIT_CUSTOMER_ID="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("AUDIT_CUSTOMER_ID missing — run scripts/provision_audit_user.sh")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fund audit wallet for smoke tests")
    parser.add_argument("--amount", type=float, default=25.0, help="CAD credits to deposit")
    parser.add_argument(
        "--key",
        default="audit-smoke-fund",
        help="Idempotency key suffix (date or run id)",
    )
    args = parser.parse_args()

    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))

    from billing import get_billing_engine

    customer_id = _customer_id()
    idempotency_key = f"{args.key}-{customer_id}"
    be = get_billing_engine()
    result = be.deposit(
        customer_id,
        args.amount,
        "Audit smoke test credits",
        idempotency_key=idempotency_key,
    )
    print(
        {
            "customer_id": customer_id,
            "amount_cad": args.amount,
            "balance_cad": result.get("balance_cad"),
            "dedup": result.get("dedup", False),
            "tx_id": result.get("tx_id"),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())