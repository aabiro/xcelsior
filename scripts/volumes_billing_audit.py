#!/usr/bin/env python3
"""Audit production volume billing tick.

Verifies ``billing_cycles`` rows for the audit user's volumes use the
expected schema (``resource_type=volume``, ``gpu_model=storage``,
``tier=volume``, ``customer_id`` matches volume ``owner_id``).

Safe for production: read-only audit + one ``auto_billing_cycle()`` call
(idempotent; volumes already billed this cycle are skipped via row lock).

Usage (on VPS):
  docker compose exec -T api-blue python scripts/volumes_billing_audit.py

Usage (local with .env.audit):
  python3 scripts/volumes_billing_audit.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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


def _audit_volume_cycles(conn, volume: dict) -> dict:
    """Return audit result for one volume's billing_cycles rows."""
    vid = volume["volume_id"]
    owner = volume["owner_id"]
    rows = conn.execute(
        """SELECT cycle_id, customer_id, resource_type, gpu_model, tier, status,
                  amount_cad, period_start, period_end
           FROM billing_cycles
           WHERE job_id = %s AND resource_type = 'volume'
           ORDER BY period_end DESC
           LIMIT 10""",
        (vid,),
    ).fetchall()

    issues: list[str] = []
    for row in rows:
        if row["customer_id"] != owner:
            issues.append(
                f"customer_id mismatch: cycle {row['cycle_id']} "
                f"customer={row['customer_id']} owner={owner}"
            )
        if row["gpu_model"] != "storage":
            issues.append(f"gpu_model={row['gpu_model']!r} expected 'storage'")
        if row["tier"] != "volume":
            issues.append(f"tier={row['tier']!r} expected 'volume'")
        if row["resource_type"] != "volume":
            issues.append(f"resource_type={row['resource_type']!r} expected 'volume'")

    age_sec = time.time() - float(volume.get("created_at") or 0)
    return {
        "volume_id": vid,
        "owner_id": owner,
        "status": volume.get("status"),
        "size_gb": volume.get("size_gb"),
        "age_min": round(age_sec / 60, 1),
        "cycle_count": len(rows),
        "latest_cycle": dict(rows[0]) if rows else None,
        "issues": issues,
        "ok": len(issues) == 0 and (len(rows) > 0 or age_sec < 120),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit volume billing tick in prod")
    parser.add_argument(
        "--skip-tick",
        action="store_true",
        help="Only query existing billing_cycles; do not run auto_billing_cycle",
    )
    args = parser.parse_args()

    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))

    from billing import get_billing_engine
    from db import _get_pg_pool
    from psycopg.rows import dict_row
    from volumes import get_volume_engine

    customer_id = _customer_id()
    ve = get_volume_engine()
    volumes = [
        v
        for v in ve.list_volumes(customer_id)
        if v.get("status") in ("available", "attached")
    ]

    tick_result: dict | None = None
    if not args.skip_tick:
        be = get_billing_engine()
        tick_result = be.auto_billing_cycle()

    pool = _get_pg_pool()
    audited: list[dict] = []
    with pool.connection() as conn:
        conn.row_factory = dict_row
        for vol in volumes:
            audited.append(_audit_volume_cycles(conn, vol))

    all_ok = all(a["ok"] for a in audited)
    no_volumes_note = None
    if not volumes:
        no_volumes_note = "no active volumes for audit user — create one via volumes_e2e_smoke"
        all_ok = True  # structural audit N/A

    report = {
        "customer_id": customer_id,
        "active_volumes": len(volumes),
        "tick": tick_result,
        "volumes": audited,
        "pass": all_ok,
        "note": no_volumes_note,
    }
    print(json.dumps(report, indent=2, default=str))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())