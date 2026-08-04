#!/usr/bin/env python3
"""READ-ONLY. Find OAuth clients holding platform-operator scopes.

Run this BEFORE deploying the fix. Patching changes what is observable: once
registration refuses operator scopes, a row created earlier still exists but you
lose the ability to distinguish "never happened" from "happened and was
cleaned up" by re-testing the route.

Usage
    XCELSIOR_POSTGRES_DSN='postgresql://user:pass@host:5432/xcelsior?sslmode=require' \
        python3 scripts/incident/audit_operator_clients.py

    # or pass it directly
    python3 scripts/incident/audit_operator_clients.py 'postgresql://...'

Needs only `psycopg` — no repo import, so it runs on the production host
against the production database without dragging the application in.

Emits no secrets: client_secret_hash, _salt and _preview are never selected.

The optional columns are discovered rather than assumed. Production and dev have
drifted — `registration_source` exists in one and not the other — and a query
that hard-codes the richer schema crashes on the poorer one, which previously
surfaced as an exit code identical to "incident found".

Exit codes
    0  no client holds an operator scope
    1  at least one does — read the output as an incident, not a finding
    2  could not run (bad DSN, missing table, unexpected error)
"""

from __future__ import annotations

import os
import sys

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    print("psycopg is not installed here — try the app venv, or pip install 'psycopg[binary]'", file=sys.stderr)
    sys.exit(2)

#: Scopes that confer platform-operator authority. `control_plane_v1.
#: _require_host_operator` authorizes a machine principal on scope alone, so any
#: client holding one of these can act as an operator with no role check.
#:
#: Hand-maintained, and therefore only as complete as this list. It is a third
#: copy alongside the hotfix literal and the branch's derived OPERATOR_SCOPES;
#: if those gain a member, this must too.
OPERATOR = [
    "control_plane:operate",
    "control_plane:read",
    "hosts:evict",
    "hosts:fleet",
    "hosts:operate",
    "transparency:read",
    "transparency:write",
]

#: Needed to interpret a row at all.
CORE = ["client_id", "client_name", "created_by_email", "created_at", "scopes"]

#: Useful for telling a system row from a user-minted one, but absent in some
#: deployments. Selected only where present.
OPTIONAL = ["is_first_party", "is_system_managed", "registration_source"]


def main() -> int:
    dsn = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("XCELSIOR_POSTGRES_DSN") or os.environ.get("DATABASE_URL")
    )
    if not dsn:
        print("no DSN: pass one as argv[1] or set XCELSIOR_POSTGRES_DSN", file=sys.stderr)
        return 2

    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        cols = {
            r["column_name"]: r["data_type"]
            for r in conn.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = 'oauth_clients'"
            ).fetchall()
        }
        if not cols:
            print("table oauth_clients not found — wrong database?", file=sys.stderr)
            return 2
        if "scopes" not in cols:
            print("oauth_clients.scopes not found — wrong schema?", file=sys.stderr)
            return 2

        missing_core = [c for c in CORE if c not in cols]
        if missing_core:
            print(f"oauth_clients is missing expected columns: {missing_core}", file=sys.stderr)
            return 2

        present = [c for c in OPTIONAL if c in cols]
        absent = [c for c in OPTIONAL if c not in cols]

        total = conn.execute("SELECT count(*) AS c FROM oauth_clients").fetchone()["c"]
        expand = (
            "jsonb_array_elements_text(scopes)"
            if cols["scopes"] == "jsonb"
            else "unnest(scopes)"
        )

        rows = conn.execute(
            f"""
            SELECT {", ".join(CORE + present)}
            FROM oauth_clients
            WHERE EXISTS (SELECT 1 FROM {expand} AS s WHERE s = ANY(%s))
            ORDER BY created_at
            """,
            (OPERATOR,),
        ).fetchall()

        print(f"database        : {dsn.split('@')[-1]}")
        print(f"scopes column   : {cols['scopes']}")
        print(f"total clients   : {total}")
        print(f"operator-scoped : {len(rows)}")
        if absent:
            print(f"columns absent  : {absent}  (schema drift — not fatal)")
        print()

        for r in rows:
            held = sorted(set(r["scopes"] or []) & set(OPERATOR))
            print(f"  client_id   = {r['client_id']}")
            print(f"    name      = {r['client_name']!r}")
            print(f"    created_by= {r['created_by_email']!r}   at {r['created_at']}")
            flags = "  ".join(f"{c}={r.get(c)!r}" for c in present) or "(no provenance columns in this schema)"
            print(f"    {flags}")
            print(f"    OPERATOR SCOPES HELD = {held}")
            print()

        if not rows:
            print("  (none)\n")
            print("Clean. No client can act as a platform operator through this path.")
            return 0

        print("=" * 72)
        print("At least one client holds an operator scope.")
        print()
        print("Legitimate holders are system paths only: first-party seeded defaults")
        print("(created_by_email NULL) and system-managed rows. Anything with a real")
        print("created_by_email is a client some user minted, and the two writing")
        print("routes did not check entitlement. Treat as compromised:")
        print("  - has it been exchanged for a token (check your token/audit store)?")
        print("  - what did that principal do (audit events for the client_id above)?")
        print("  - revoke before deploying, so the fix does not hide a live credential.")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - an audit that crashes must not look like a finding
        print(f"audit could not run: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
