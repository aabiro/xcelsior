"""Read-only checks of effective PostgreSQL runtime authority."""

from __future__ import annotations

from control_plane.db_roles import SERVICE_ROLES, domain_of, sequence_domains


def _only_declared_column_updates(cur, name: str, qualified: str, allowed) -> bool:
    if not allowed:
        return False
    cur.execute(
        "SELECT has_table_privilege(%s, %s, 'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER')",
        (name, qualified),
    )
    if cur.fetchone()[0]:
        return False
    cur.execute(
        "SELECT attname FROM pg_attribute WHERE attrelid=to_regclass(%s) "
        "AND attnum>0 AND NOT attisdropped AND ("
        "has_column_privilege(%s,attrelid,attnum,'INSERT,REFERENCES') OR "
        "(has_column_privilege(%s,attrelid,attnum,'UPDATE') AND NOT attname=ANY(%s)))",
        (qualified, name, name, sorted(allowed)),
    )
    return not cur.fetchall()


def role_violations(conn, service: str, *, check_session: bool = True) -> list[str]:
    role = SERVICE_ROLES[service]
    name = role.name
    issues = []
    with conn.cursor() as cur:
        cur.execute("SELECT current_user, session_user")
        current, login = cur.fetchone()
        if check_session and (current != name or login != name):
            issues.append(f"session identity must be {name}")
        cur.execute(
            "SELECT rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls "
            "FROM pg_roles WHERE rolname = %s",
            (name,),
        )
        attrs = cur.fetchone()
        if attrs is None:
            return issues + [f"missing role {name}"]
        for attr, enabled in zip(
            ("SUPERUSER", "CREATEDB", "CREATEROLE", "REPLICATION", "BYPASSRLS"), attrs
        ):
            if enabled:
                issues.append(f"{name} has {attr}")
        cur.execute(
            "SELECT parent.rolname FROM pg_auth_members m JOIN pg_roles parent ON parent.oid=m.roleid "
            "JOIN pg_roles child ON child.oid=m.member WHERE child.rolname=%s",
            (name,),
        )
        issues.extend(f"{name} is a member of {parent}" for (parent,) in cur.fetchall())
        if not role.runtime:
            return issues
        cur.execute("SELECT has_database_privilege(%s, current_database(), 'CREATE')", (name,))
        if cur.fetchone()[0]:
            issues.append(f"{name} can create schemas in the database")
        cur.execute(
            "SELECT datname FROM pg_database WHERE datdba=(SELECT oid FROM pg_roles WHERE rolname=%s)",
            (name,),
        )
        issues.extend(f"{name} owns database {db}" for (db,) in cur.fetchall())
        cur.execute(
            "SELECT nspname, has_schema_privilege(%s, oid, 'CREATE'), "
            "nspowner=(SELECT oid FROM pg_roles WHERE rolname=%s) "
            "FROM pg_namespace WHERE nspname NOT LIKE 'pg_%%' AND nspname <> 'information_schema'",
            (name, name),
        )
        for schema, create, owner in cur.fetchall():
            if owner:
                issues.append(f"{name} owns schema {schema}")
            elif create:
                issues.append(f"{name} can create in schema {schema}")
        cur.execute(
            "SELECT n.nspname, c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE c.relowner=(SELECT oid FROM pg_roles WHERE rolname=%s) "
            "AND n.nspname NOT LIKE 'pg_%%'",
            (name,),
        )
        issues.extend(f"{name} owns relation {schema}.{table}" for schema, table in cur.fetchall())
        cur.execute(
            "SELECT n.nspname, p.proname FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
            "WHERE p.proowner=(SELECT oid FROM pg_roles WHERE rolname=%s) "
            "AND n.nspname NOT LIKE 'pg_%%' AND n.nspname <> 'information_schema'",
            (name,),
        )
        issues.extend(f"{name} owns routine {schema}.{fn}" for schema, fn in cur.fetchall())
        cur.execute(
            "SELECT n.nspname, c.relname, "
            "(has_table_privilege(%s, c.oid, 'SELECT') OR has_any_column_privilege(%s, c.oid, 'SELECT')), "
            "(has_table_privilege(%s, c.oid, 'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') "
            "OR has_any_column_privilege(%s, c.oid, 'INSERT,UPDATE,REFERENCES')), "
            "has_table_privilege(%s, c.oid, 'TRUNCATE,REFERENCES,TRIGGER') "
            "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname IN ('public','storage') AND c.relkind IN ('r','p','v','m','f')",
            (name, name, name, name, name),
        )
        for schema, table, read, write, extra in cur.fetchall():
            domain = domain_of(f"{schema}.{table}")
            if (
                read
                and domain not in role.readable
                and table != "alembic_version"
                and f"{schema}.{table}" not in role.read_tables
            ):
                issues.append(f"{name} can read outside its domains: {schema}.{table}")
            column_only = (
                table != "alembic_version"
                and domain not in role.write
                and write
                and _only_declared_column_updates(
                    cur,
                    name,
                    f"{schema}.{table}",
                    role.column_updates.get(f"{schema}.{table}"),
                )
            )
            if (
                write
                and (domain not in role.write or table == "alembic_version")
                and not column_only
            ):
                issues.append(f"{name} can write outside its domains: {schema}.{table}")
            if extra:
                issues.append(f"{name} has non-DML authority on {schema}.{table}")
        cur.execute(
            "SELECT n.nspname, p.proname FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
            "WHERE p.prosecdef AND p.prorettype NOT IN ('trigger'::regtype, 'event_trigger'::regtype) "
            "AND n.nspname IN ('public','storage') AND has_function_privilege(%s,p.oid,'EXECUTE')",
            (name,),
        )
        issues.extend(
            f"{name} can execute security-definer routine {schema}.{fn}"
            for schema, fn in cur.fetchall()
        )
        from psycopg import sql

        for schema, sequence, domain in sequence_domains(conn):
            qualified = sql.Identifier(schema, sequence).as_string(conn)
            cur.execute(
                "SELECT has_sequence_privilege(%s, %s, 'USAGE,SELECT,UPDATE'), "
                "has_sequence_privilege(%s, %s, 'UPDATE')",
                (name, qualified, name, qualified),
            )
            access, reset = cur.fetchone()
            if reset or (access and domain not in role.write):
                issues.append(f"{name} has excess sequence authority: {schema}.{sequence}")
    return issues


def require_runtime_role(conn, service: str) -> None:
    if not SERVICE_ROLES[service].runtime:
        raise RuntimeError("migration credentials cannot be used by a runtime pool")
    issues = role_violations(conn, service)
    if issues:
        raise RuntimeError("database role validation failed: " + "; ".join(issues[:12]))
