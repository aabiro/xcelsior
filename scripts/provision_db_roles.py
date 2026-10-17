#!/usr/bin/env python3
"""Provision the per-service PostgreSQL roles from companion §4.2.

Idempotent: safe to run on every deploy, before or after any service has
been cut over to its dedicated role. It only creates roles and converges
grants — it never drops a role, revokes a login, or touches data.

Usage
-----
    # create/converge all roles (no logins)
    python scripts/provision_db_roles.py --dsn "$XCELSIOR_POSTGRES_ADMIN_DSN"

    # additionally give the scheduler role a password so a service can log in
    XCELSIOR_DB_ROLE_PASSWORD_SCHEDULER=... \\
      python scripts/provision_db_roles.py --login scheduler

    # show the SQL without executing it
    python scripts/provision_db_roles.py --dry-run

Roles are created ``NOLOGIN`` by default. Passing ``--login <service>``
(repeatable, or ``--login all``) turns on ``LOGIN`` for that role using
``XCELSIOR_DB_ROLE_PASSWORD_<SERVICE>``; without the password variable the
role stays NOLOGIN and the script says so rather than creating a
password-less login.

The connecting role needs ``CREATEROLE`` (or superuser). The runtime
application role does not have it, which is the point: role management is
an administrative action, not something a compromised service can do.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control_plane.db_roles import (  # noqa: E402
    SERVICE_ROLES,
    STORAGE_SCHEMA,
    domain_of,
    sequence_domains,
    grant_statements,
)


def _fetch_inventory(conn) -> tuple[list[str], list[str]]:
    """(all schema-qualified tables, storage-schema table names)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT schemaname, tablename
              FROM pg_tables
             WHERE schemaname IN ('public', %s)
            """,
            (STORAGE_SCHEMA,),
        )
        rows = cur.fetchall()
    qualified = [f"{s}.{t}" for s, t in rows]
    storage = [t for s, t in rows if s == STORAGE_SCHEMA]
    return qualified, storage


class OverPrivilegedRole(RuntimeError):
    """An existing role already holds attributes a service role must never have."""


def _role_attributes(conn, name: str) -> tuple[bool, ...] | None:
    """Escalating role attributes for ``name``, or None if absent."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls "
            "FROM pg_roles WHERE rolname = %s",
            (name,),
        )
        row = cur.fetchone()
    return tuple(bool(value) for value in row) if row else None


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sequence_grants(conn, role) -> list[str]:
    """Sequence access follows its table or explicit standalone authority."""
    rows = sequence_domains(conn)
    unknown = [f"{schema}.{seq}" for schema, seq, domain in rows if domain is None]
    if unknown:
        raise OverPrivilegedRole("unclassified database sequences: " + ", ".join(unknown))
    return [
        f'GRANT USAGE, SELECT ON SEQUENCE {_quote(schema)}.{_quote(seq)} TO {_quote(role.name)}'
        for schema, seq, domain in rows if domain in role.write
    ]


def build_plan(
    conn,
    *,
    login_services: set[str],
    passwords: dict[str, str],
) -> list[tuple[str, tuple]]:
    """Return the ordered (sql, params) plan for the current database.

    Raises :class:`OverPrivilegedRole` when a role that already exists
    holds SUPERUSER/CREATEDB/CREATEROLE. Stripping those requires
    superuser, so the honest behaviour is to refuse loudly rather than
    provision a role whose name promises least privilege and whose
    attributes do not deliver it.
    """
    qualified, storage_tables = _fetch_inventory(conn)
    unclaimed = [table for table in qualified if domain_of(table) is None]
    if unclaimed:
        raise OverPrivilegedRole("unclassified database tables: " + ", ".join(sorted(unclaimed)))
    plan: list[tuple[str, tuple]] = []

    # PUBLIC grants otherwise defeat every per-role REVOKE below. These
    # schemas are Xcelsior-owned; unknown tables above stop convergence.
    schemas = ["public"] + ([STORAGE_SCHEMA] if storage_tables else [])
    for schema in schemas:
        plan.extend((stmt, ()) for stmt in (
            f'REVOKE CREATE ON SCHEMA {_quote(schema)} FROM PUBLIC',
            f'REVOKE ALL ON ALL TABLES IN SCHEMA {_quote(schema)} FROM PUBLIC',
            f'REVOKE ALL ON ALL SEQUENCES IN SCHEMA {_quote(schema)} FROM PUBLIC',
        ))

    for key, role in SERVICE_ROLES.items():
        attrs = _role_attributes(conn, role.name)
        if attrs is None:
            # CREATE ROLE defaults are already NOSUPERUSER NOCREATEDB
            # NOCREATEROLE; stating them explicitly requires superuser on
            # some builds, so rely on the defaults and verify afterwards.
            plan.append((f'CREATE ROLE "{role.name}" NOLOGIN', ()))
        elif any(attrs):
            names = [n for n, on in zip(("SUPERUSER", "CREATEDB", "CREATEROLE", "REPLICATION", "BYPASSRLS"), attrs) if on]
            raise OverPrivilegedRole(
                f"role {role.name} already has {', '.join(names)}; "
                "revoke with a superuser before provisioning"
            )
        if attrs is not None:
            from control_plane.db_role_audit import role_violations

            # Explicit grants can be converged. Ownership and memberships
            # require an operator-reviewed transfer/revocation, not GRANTs.
            unsafe = [issue for issue in role_violations(conn, key, check_session=False)
                      if " owns " in issue or " is a member " in issue]
            if unsafe:
                raise OverPrivilegedRole("; ".join(unsafe))
        password = passwords.get(key)
        if key in login_services and password:
            plan.append((f'ALTER ROLE "{role.name}" LOGIN PASSWORD %s', (password,)))
        for stmt in grant_statements(
            role, storage_tables=storage_tables, existing_tables=qualified
        ):
            plan.append((stmt, ()))
        plan.extend((stmt, ()) for stmt in sequence_grants(conn, role))

        # §4.3: pin the session limits on the role itself so they hold even
        # if a service connects without the pool's options string.
        from control_plane.db_roles import session_options

        for opt in session_options(key).split("-c ")[1:]:
            setting, _, value = opt.strip().partition("=")
            # ALTER ROLE ... SET takes no bind parameters; only emit values
            # we generated ourselves and that are provably numeric.
            if setting.isidentifier() and value.strip().isdigit():
                plan.append((f'ALTER ROLE "{role.name}" SET {setting} = {int(value)}', ()))
    # Future objects receive no blanket domain grants. Re-run convergence
    # after every migration/partition creation, once ownership is classified.
    owners = {row[0] for row in conn.execute(
        "SELECT DISTINCT pg_get_userbyid(c.relowner) FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname IN ('public','storage')"
    ).fetchall()}
    owners.add(SERVICE_ROLES["migrator"].name)
    for owner in sorted(owners):
        for schema in [None, *schemas]:
            for kind in ("TABLES", "SEQUENCES"):
                for grantee in ["PUBLIC", *(_quote(r.name) for r in SERVICE_ROLES.values() if r.runtime)]:
                    scope = f' IN SCHEMA {_quote(schema)}' if schema else ''
                    plan.append((f'ALTER DEFAULT PRIVILEGES FOR ROLE {_quote(owner)}{scope} REVOKE ALL ON {kind} FROM {grantee}', ()))
    return plan


def apply_plan(conn, plan):
    # PostgreSQL utility commands such as ALTER ROLE ... PASSWORD cannot use
    # server-side $1 parameters. ClientCursor quotes parameters safely without
    # putting the credential in the plan's SQL or dry-run output.
    from psycopg import ClientCursor

    with ClientCursor(conn) as cur:
        for sql, params in plan:
            cur.execute(sql, params or None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dsn",
        default=os.environ.get("XCELSIOR_POSTGRES_ADMIN_DSN")
        or os.environ.get("XCELSIOR_POSTGRES_DSN"),
        help="admin DSN (needs CREATEROLE); defaults to XCELSIOR_POSTGRES_ADMIN_DSN",
    )
    ap.add_argument(
        "--login",
        action="append",
        default=[],
        metavar="SERVICE",
        help="enable LOGIN for this role using XCELSIOR_DB_ROLE_PASSWORD_<SERVICE>; "
        "repeatable, or 'all'",
    )
    ap.add_argument("--dry-run", action="store_true", help="print SQL, execute nothing")
    args = ap.parse_args()

    if not args.dsn:
        print("error: no DSN (pass --dsn or set XCELSIOR_POSTGRES_ADMIN_DSN)", file=sys.stderr)
        return 2

    login_services: set[str] = set()
    for entry in args.login:
        if entry.strip().lower() == "all":
            login_services |= set(SERVICE_ROLES)
        else:
            login_services.add(entry.strip().lower())
    unknown = login_services - set(SERVICE_ROLES)
    if unknown:
        print(f"error: unknown service role(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    passwords = {
        key: (os.environ.get(f"XCELSIOR_DB_ROLE_PASSWORD_{key.upper()}") or "").strip()
        for key in SERVICE_ROLES
    }
    for key in sorted(login_services):
        if not passwords.get(key):
            print(
                f"warning: --login {key} requested but "
                f"XCELSIOR_DB_ROLE_PASSWORD_{key.upper()} is unset — role stays NOLOGIN",
                file=sys.stderr,
            )

    import psycopg

    with psycopg.connect(args.dsn, autocommit=False) as conn:
        plan = build_plan(conn, login_services=login_services, passwords=passwords)
        if args.dry_run:
            for sql, params in plan:
                rendered = sql
                if params:
                    rendered = sql.replace("%s", "'***'" if "PASSWORD" in sql else "%r")
                    if "PASSWORD" not in sql:
                        rendered = sql % tuple(repr(p) for p in params)
                print(rendered + ";")
            conn.rollback()
            return 0
        apply_plan(conn, plan)
        conn.commit()

    print(f"provisioned {len(SERVICE_ROLES)} roles ({len(plan)} statements)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
