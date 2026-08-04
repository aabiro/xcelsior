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

Exit codes
    0  no client holds an operator scope
    1  at least one does — read the output as an incident, not a finding
    2  could not run (bad DSN, missing table)
"""

from __future__ import annotations

import os
import sys

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    sys.exit("psycopg is not installed here — run on a host that has it, or `pip install psycopg[binary]`")

#: Scopes that confer platform-operator authority. `control_plane_v1.
#: _require_host_operator` authorizes a machine principal on scope alone, so any
#: client holding one of these can act as an operator with no role check.
OPERATOR = [
    "control_plane:operate",
    "control_plane:read",
    "hosts:evict",
    "hosts:fleet",
    "hosts:operate",
    "transparency:read",
    "transparency:write",
]

dsn = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("XCELSIOR_POSTGRES_DSN") or os.environ.get("DATABASE_URL")
if not dsn:
    sys.exit(2)

with psycopg.connect(dsn, row_factory=dict_row) as conn:
    col = conn.execute(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name='oauth_clients' AND column_name='scopes'"
    ).fetchone()
    if not col:
        sys.exit("oauth_clients.scopes not found — wrong database?")

    total = conn.execute("SELECT count(*) AS c FROM oauth_clients").fetchone()["c"]
    expand = "jsonb_array_elements_text(scopes)" if col["data_type"] == "jsonb" else "unnest(scopes)"

    rows = conn.execute(
        f"""
        SELECT client_id, client_name, created_by_email, created_at,
               is_first_party, is_system_managed, registration_source, scopes
        FROM oauth_clients
        WHERE EXISTS (SELECT 1 FROM {expand} AS s WHERE s = ANY(%s))
        ORDER BY created_at
        """,
        (OPERATOR,),
    ).fetchall()

    print(f"database        : {dsn.split('@')[-1]}")
    print(f"scopes column   : {col['data_type']}")
    print(f"total clients   : {total}")
    print(f"operator-scoped : {len(rows)}\n")

    for r in rows:
        held = sorted(set(r["scopes"] or []) & set(OPERATOR))
        print(f"  client_id   = {r['client_id']}")
        print(f"    name      = {r['client_name']!r}")
        print(f"    created_by= {r['created_by_email']!r}   at {r['created_at']}")
        print(f"    first_party={r['is_first_party']}  system_managed={r['is_system_managed']}  source={r['registration_source']!r}")
        print(f"    OPERATOR SCOPES HELD = {held}\n")

    if not rows:
        print("  (none)\n")
        print("Clean. No client can act as a platform operator through this path.")
        sys.exit(0)

    print("=" * 72)
    print("At least one client holds an operator scope.")
    print()
    print("Legitimate holders are system paths only: first-party seeded defaults")
    print("(created_by_email NULL, is_first_party=1) and system-managed rows.")
    print("Anything with a real created_by_email is a client some user minted, and")
    print("the two writing routes did not check entitlement. Treat as compromised:")
    print("  - has it been exchanged for a token (check your token/audit store)?")
    print("  - what did that principal do (audit events for the client_id above)?")
    print("  - revoke before deploying, so the fix does not hide a live credential.")
    sys.exit(1)
