#!/usr/bin/env python3
"""Preview or apply application-only ownership transfer to xcelsior_migrator.

Uses XCELSIOR_POSTGRES_ADMIN_DSN. No REASSIGN OWNED, data changes, login
changes or credential output. Run only after runtime services use their
restricted logins. The old owner retains SELECT for offline pg_dump jobs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
    sys.path.insert(0, str(_p))

import psycopg
from psycopg import sql

from control_plane.db_roles import SERVICE_ROLES, domain_of, sequence_domains

MIGRATOR = SERVICE_ROLES["migrator"].name
SCHEMAS = ["public", "storage"]


def ownership_plan(
    conn, *, expected_database: str, expected_port: int, from_owner: str
) -> list[dict]:
    database, owner, port = conn.execute(
        "SELECT datname, pg_get_userbyid(datdba), current_setting('port') "
        "FROM pg_database WHERE datname=current_database()"
    ).fetchone()
    if database != expected_database or database in {"postgres", "template0", "template1"}:
        raise RuntimeError("ownership transfer connected to an unexpected database")
    if int(port) != expected_port:
        raise RuntimeError("ownership transfer connected to an unexpected PostgreSQL port")
    if from_owner in {r.name for r in SERVICE_ROLES.values()}:
        raise RuntimeError("legacy owner must be separate from the dedicated roles")
    if owner not in {from_owner, MIGRATOR}:
        raise RuntimeError("database owner does not match the reviewed legacy owner")
    attributes = conn.execute(
        "SELECT rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
        "FROM pg_roles WHERE rolname=%s",
        (MIGRATOR,),
    ).fetchone()
    if attributes is None or any(attributes):
        raise RuntimeError("migration role is absent or has escalating attributes")
    # Unknown owners and unclassified relations require review, not a broad
    # cluster-level transfer. Extension members retain their extension owner.
    relations = conn.execute(
        "SELECT n.nspname,c.relname,c.relkind,pg_get_userbyid(c.relowner), "
        "EXISTS (SELECT 1 FROM pg_depend d WHERE d.classid='pg_class'::regclass "
        "AND d.objid=c.oid AND d.refclassid='pg_class'::regclass AND d.deptype IN ('a','i')) "
        "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname=ANY(%s) AND c.relkind IN ('r','p','v','m','S','f') "
        "AND NOT EXISTS (SELECT 1 FROM pg_depend d WHERE d.classid='pg_class'::regclass "
        "AND d.objid=c.oid AND d.deptype='e') ORDER BY c.oid",
        (SCHEMAS,),
    ).fetchall()
    for schema, name, kind, old, _ in relations:
        if old not in {from_owner, MIGRATOR}:
            raise RuntimeError(f"unexpected relation owner: {schema}.{name}")
        if kind != "S" and domain_of(f"{schema}.{name}") is None:
            raise RuntimeError(f"unclassified application relation: {schema}.{name}")
    if any(domain is None for _, _, domain in sequence_domains(conn)):
        raise RuntimeError("unclassified application sequence")

    plan = []

    def add(kind, identifier, old):
        if old == MIGRATOR:
            return
        prefix = sql.SQL("ALTER " + kind + " {} OWNER TO ").format(identifier)
        plan.append(
            {
                "forward": (prefix + sql.Identifier(MIGRATOR)).as_string(conn),
                "rollback": (prefix + sql.Identifier(old)).as_string(conn),
            }
        )

    add("DATABASE", sql.Identifier(database), owner)
    for schema, old in conn.execute(
        "SELECT nspname,pg_get_userbyid(nspowner) FROM pg_namespace WHERE nspname=ANY(%s)",
        (SCHEMAS,),
    ).fetchall():
        if old not in {from_owner, MIGRATOR, "pg_database_owner"}:
            raise RuntimeError(f"unexpected schema owner: {schema}")
        add("SCHEMA", sql.Identifier(schema), old)
    for schema, name, kind, old, attached in relations:
        if kind == "S" and attached:
            continue  # Table ownership transfers serial/identity sequences.
        add(
            {"S": "SEQUENCE", "v": "VIEW", "m": "MATERIALIZED VIEW", "f": "FOREIGN TABLE"}.get(
                kind, "TABLE"
            ),
            sql.Identifier(schema, name),
            old,
        )
    routines = conn.execute(
        "SELECT n.nspname,p.proname,pg_get_function_identity_arguments(p.oid),"
        "p.prokind,pg_get_userbyid(p.proowner) FROM pg_proc p "
        "JOIN pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname=ANY(%s) "
        "AND NOT EXISTS (SELECT 1 FROM pg_depend d WHERE d.classid='pg_proc'::regclass "
        "AND d.objid=p.oid AND d.deptype='e')",
        (SCHEMAS,),
    ).fetchall()
    for schema, name, args, kind, old in routines:
        if old not in {from_owner, MIGRATOR}:
            raise RuntimeError(f"unexpected routine owner: {schema}.{name}")
        identifier = sql.SQL("{}({})").format(sql.Identifier(schema, name), sql.SQL(args))
        add({"p": "PROCEDURE", "a": "AGGREGATE"}.get(kind, "FUNCTION"), identifier, old)
    types = conn.execute(
        "SELECT n.nspname,t.typname,t.typtype,pg_get_userbyid(t.typowner) FROM pg_type t "
        "JOIN pg_namespace n ON n.oid=t.typnamespace WHERE n.nspname=ANY(%s) "
        "AND (t.typtype IN ('e','d','r') OR (t.typtype='c' AND EXISTS "
        "(SELECT 1 FROM pg_class c WHERE c.oid=t.typrelid AND c.relkind='c'))) "
        "AND NOT EXISTS (SELECT 1 FROM pg_depend d WHERE d.classid='pg_type'::regclass "
        "AND d.objid=t.oid AND d.deptype='e')",
        (SCHEMAS,),
    ).fetchall()
    for schema, name, kind, old in types:
        if old not in {from_owner, MIGRATOR}:
            raise RuntimeError(f"unexpected type owner: {schema}.{name}")
        add("DOMAIN" if kind == "d" else "TYPE", sql.Identifier(schema, name), old)
    return plan


def transfer(conn, plan: list[dict], *, backup_owner: str) -> None:
    """Caller owns the transaction; any failed statement rolls back the transfer."""
    conn.execute("SET LOCAL lock_timeout='5s'")
    conn.execute("SET LOCAL statement_timeout='120s'")
    for step in plan:
        conn.execute(step["forward"])
    # Backup credentials remain offline. Future tables and sequences need
    # explicit read grants too, or the first new partition breaks pg_dump.
    for schema in SCHEMAS:
        if not conn.execute("SELECT to_regnamespace(%s)", (schema,)).fetchone()[0]:
            continue
        conn.execute(
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                sql.Identifier(schema), sql.Identifier(backup_owner)
            )
        )
        for kind in ("TABLES", "SEQUENCES"):
            conn.execute(
                sql.SQL("GRANT SELECT ON ALL {} IN SCHEMA {} TO {}").format(
                    sql.SQL(kind), sql.Identifier(schema), sql.Identifier(backup_owner)
                )
            )
            conn.execute(
                sql.SQL(
                    "ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA {} GRANT SELECT ON {} TO {}"
                ).format(
                    sql.Identifier(MIGRATOR),
                    sql.Identifier(schema),
                    sql.SQL(kind),
                    sql.Identifier(backup_owner),
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-database", required=True)
    parser.add_argument("--expected-port", required=True, type=int)
    parser.add_argument("--from-owner", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    dsn = os.environ.get("XCELSIOR_POSTGRES_ADMIN_DSN")
    if not dsn:
        raise SystemExit("XCELSIOR_POSTGRES_ADMIN_DSN is required")
    with psycopg.connect(dsn, connect_timeout=10) as conn:
        plan = ownership_plan(
            conn,
            expected_database=args.expected_database,
            expected_port=args.expected_port,
            from_owner=args.from_owner,
        )
        if args.apply:
            transfer(conn, plan, backup_owner=args.from_owner)
        if args.evidence:
            # Refuse overwriting a prior rollback record or following symlinks.
            with args.evidence.open("x", encoding="utf-8") as stream:
                os.chmod(args.evidence, 0o600)
                json.dump(
                    {
                        "record": "ownership plan; execution requires a successful exit and postflight audit",
                        "database": args.expected_database,
                        "port": args.expected_port,
                        "ownership_plan": plan,
                    },
                    stream,
                    indent=2,
                )
                stream.write("\n")
    print(json.dumps({"applied": args.apply, "ownership_changes": len(plan)}))


if __name__ == "__main__":
    main()
