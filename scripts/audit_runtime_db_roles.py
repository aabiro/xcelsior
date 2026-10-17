#!/usr/bin/env python3
"""Read-only deployment gate. Emits no DSNs, passwords, or user records.

Run under each service's actual environment. Exit 1 means its effective role
or DSN configuration does not meet the service contract. No fallback is used.
"""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg
from control_plane.db_roles import SERVICE_ROLES, service_dsn_env_var
from control_plane.db_role_audit import role_violations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--service", required=True, choices=[k for k, r in SERVICE_ROLES.items() if r.runtime]
    )
    args = parser.parse_args()
    dsn = os.environ.get(service_dsn_env_var(args.service))
    if not dsn:
        result = {
            "service": args.service,
            "ok": False,
            "issues": ["dedicated service DSN is missing"],
        }
    else:
        try:
            with psycopg.connect(
                dsn,
                connect_timeout=5,
                options="-c default_transaction_read_only=on -c statement_timeout=10000",
            ) as conn:
                issues = role_violations(conn, args.service)
                result = {"service": args.service, "ok": not issues, "issues": issues}
        except Exception:
            result = {
                "service": args.service,
                "ok": False,
                "issues": ["database role audit could not complete"],
            }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
