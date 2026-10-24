#!/usr/bin/env python3
"""Preview/apply qualified historical volume credits on an explicit database.

Plans contain financial identifiers and stay mode 0600 outside Git. Apply
requalifies every source row and caps the total credit; no source row is removed.
"""

import argparse
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-database", required=True)
    parser.add_argument("--expected-port", required=True, type=int)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--max-credit-micros", type=int)
    args = parser.parse_args()
    dsn = os.environ.get("XCELSIOR_POSTGRES_ADMIN_DSN")
    if not dsn:
        parser.error("XCELSIOR_POSTGRES_ADMIN_DSN must name the private one-shot login")
    import psycopg
    from control_plane.financial_history import apply_cluster, qualify_cluster

    os.umask(0o077)
    with psycopg.connect(dsn) as conn:
        if not args.apply:
            conn.execute("SET TRANSACTION READ ONLY")
        conn.execute("SET LOCAL statement_timeout='30s'")
        conn.execute("SET LOCAL lock_timeout='3s'")
        target = conn.execute("SELECT current_database(),current_setting('port')::int").fetchone()
        if target != (args.expected_database, args.expected_port):
            raise ValueError("database identity does not match the explicit target")
        if args.apply:
            if not args.max_credit_micros or args.max_credit_micros <= 0:
                parser.error("apply requires a positive --max-credit-micros")
            if args.plan.is_symlink() or args.plan.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                raise ValueError("the qualified plan must be a private non-symlink file")
            document = json.loads(args.plan.read_text())
            if (
                document["target"] != list(target)
                or document["policy_version"] != "volume_overlap/v1"
            ):
                raise ValueError("plan target or policy does not match")
            plans = document["clusters"]
            total = sum(plan["credit_micros"] for plan in plans)
            if not 0 < total <= args.max_credit_micros:
                raise ValueError("plan exceeds the total adjustment budget")
            outcomes = [
                apply_cluster(conn, plan, max_credit_micros=args.max_credit_micros)
                for plan in plans
            ]
            conn.commit()
            print(
                json.dumps(
                    {
                        "applied": True,
                        "clusters": len(outcomes),
                        "credited_micros": sum(
                            row["credit_micros"] for row in outcomes if not row["replayed"]
                        ),
                        "replayed": sum(row["replayed"] for row in outcomes),
                    }
                )
            )
        else:
            rows = conn.execute("""WITH spans AS (
                SELECT job_id,resource_type,period_start,max(period_end) OVER (
                PARTITION BY job_id,resource_type ORDER BY period_start,period_end,cycle_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prior_end
                FROM billing_cycles WHERE status='charged')
                SELECT DISTINCT job_id,period_start FROM spans
                WHERE resource_type='volume' AND period_start<prior_end
                ORDER BY job_id,period_start LIMIT 1001""").fetchall()
            if len(rows) > 1000:
                raise ValueError("historical qualification exceeds the bounded batch")
            plans = [qualify_cluster(conn, *row) for row in rows]
            with args.plan.open("x") as out:
                json.dump(
                    {
                        "target": list(target),
                        "policy_version": "volume_overlap/v1",
                        "clusters": plans,
                    },
                    out,
                    sort_keys=True,
                    indent=2,
                )
                out.write("\n")
            print(
                json.dumps(
                    {
                        "applied": False,
                        "qualified_clusters": len(plans),
                        "proposed_credit_micros": sum(plan["credit_micros"] for plan in plans),
                    }
                )
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Never echo DSNs, database rows, financial identities or private plans.
        print(f"Historical billing disposition refused ({type(exc).__name__})", file=sys.stderr)
        raise SystemExit(1)
