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

#: Selected when present. The audit has to run against whatever schema the
#: target database is *at*, which is not necessarily head: `registration_source`
#: arrives in migration 091, and production sat at 079 when this was written, so
#: naming it unconditionally made the script exit 1 on an `UndefinedColumn` —
#: indistinguishable by exit code from "a client holds an operator scope."
OPTIONAL_COLUMNS = [
    "client_name",
    "created_by_email",
    "created_at",
    "is_first_party",
    "is_system_managed",
    "registration_source",
]

try:
    conn_ctx = psycopg.connect(dsn, row_factory=dict_row)
except psycopg.Error as exc:
    sys.exit(f"2: cannot connect: {exc}")

with conn_ctx as conn:
    present = {
        r["column_name"]
        for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='oauth_clients'"
        ).fetchall()
    }
    if "scopes" not in present:
        print("oauth_clients.scopes not found — wrong database?", file=sys.stderr)
        sys.exit(2)

    col = conn.execute(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name='oauth_clients' AND column_name='scopes'"
    ).fetchone()
    assert col is not None  # `scopes` is in `present`

    selected = [c for c in OPTIONAL_COLUMNS if c in present]
    missing = [c for c in OPTIONAL_COLUMNS if c not in present]

    total = conn.execute("SELECT count(*) AS c FROM oauth_clients").fetchone()["c"]
    expand = "jsonb_array_elements_text(scopes)" if col["data_type"] == "jsonb" else "unnest(scopes)"

    order_by = "ORDER BY created_at" if "created_at" in present else "ORDER BY client_id"
    try:
        rows = conn.execute(
            f"""
            SELECT client_id, scopes{"".join(", " + c for c in selected)}
            FROM oauth_clients
            WHERE EXISTS (SELECT 1 FROM {expand} AS s WHERE s = ANY(%s))
            {order_by}
            """,
            (OPERATOR,),
        ).fetchall()
    except psycopg.Error as exc:
        # Exit 2, never 1: "could not run" must not read as "found something".
        print(f"query failed against this schema: {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"database        : {dsn.split('@')[-1]}")
    print(f"scopes column   : {col['data_type']}")
    print(f"total clients   : {total}")
    print(f"operator-scoped : {len(rows)}")
    if missing:
        # Not cosmetic: without `registration_source` a self-registered client
        # cannot be told from a seeded one, so the operator must know the audit
        # ran degraded rather than assume it answered the whole question.
        print(f"columns absent  : {', '.join(missing)} (schema predates them —")
        print("                  provenance below is correspondingly incomplete)")
    print()

    for r in rows:
        held = sorted(set(r["scopes"] or []) & set(OPERATOR))
        print(f"  client_id   = {r['client_id']}")
        print(f"    name      = {r.get('client_name')!r}")
        print(f"    created_by= {r.get('created_by_email')!r}   at {r.get('created_at')}")
        print(
            f"    first_party={r.get('is_first_party')}  "
            f"system_managed={r.get('is_system_managed')}  "
            f"source={r.get('registration_source')!r}"
        )
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
