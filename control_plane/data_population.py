"""Read-only counts of missing/sentinel values across every application table.

No row values, defaults, DSNs or credentials enter the report. Empty values
are observations, not automatic repair instructions: nullable lifecycle and
optional fields require their writer's semantic contract before backfill.
"""

from __future__ import annotations

from datetime import datetime, timezone

from psycopg import sql
from psycopg.rows import tuple_row


def audit_population(conn, *, expected_database: str, expected_port: int) -> dict:
    """Audit all non-system tables in one repeatable, read-only snapshot."""
    with conn.transaction():
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        conn.execute("SET LOCAL statement_timeout='15s'")
        conn.execute("SET LOCAL lock_timeout='1s'")
        with conn.cursor(row_factory=tuple_row) as cur:
            database, port = cur.execute(
                "SELECT current_database(),current_setting('port')::integer"
            ).fetchone()
            if (database, port) != (expected_database, expected_port):
                raise ValueError("population audit database/port does not match expected target")
            tables = cur.execute("""SELECT n.nspname,c.relname,c.relispartition
                FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                WHERE c.relkind IN ('r','p') AND n.nspname<>'information_schema'
                  AND n.nspname NOT LIKE 'pg\\_%' ESCAPE '\\'
                ORDER BY n.nspname,c.relname""").fetchall()
            report = {
                "database": database,
                "port": port,
                "read_only": True,
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "tables": [],
            }
            for schema, table, partition in tables:
                columns = cur.execute(
                    """SELECT column_name,data_type,is_nullable,column_default IS NOT NULL
                    FROM information_schema.columns
                    WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position""",
                    (schema, table),
                ).fetchall()
                fields = []
                expressions = [sql.SQL("count(*)")]
                for name, dtype, nullable, has_default in columns:
                    column = sql.Identifier(name)
                    metrics = ["nulls"]
                    expressions.append(sql.SQL("count(*) FILTER (WHERE {} IS NULL)").format(column))
                    if dtype in {"text", "character varying", "character"}:
                        metrics.append("blank")
                        expressions.append(
                            sql.SQL("count(*) FILTER (WHERE btrim({})='')").format(column)
                        )
                    elif dtype in {"json", "jsonb"}:
                        metrics.append("empty_json")
                        expressions.append(
                            sql.SQL(
                                "count(*) FILTER (WHERE {}::jsonb IN ('null'::jsonb,'{{}}'::jsonb,'[]'::jsonb))"
                            ).format(column)
                        )
                    elif dtype in {
                        "smallint",
                        "integer",
                        "bigint",
                        "numeric",
                        "real",
                        "double precision",
                    }:
                        metrics.append("zero")
                        expressions.append(sql.SQL("count(*) FILTER (WHERE {}=0)").format(column))
                    fields.append(
                        {
                            "name": name,
                            "type": dtype,
                            "nullable": nullable == "YES",
                            "has_default": has_default,
                            "metrics": metrics,
                        }
                    )
                entry = {
                    "schema": schema,
                    "table": table,
                    "partition": partition,
                    "columns": fields,
                }
                try:
                    # A blocked table must not prevent auditing every other one.
                    with conn.transaction():
                        counts = cur.execute(
                            sql.SQL("SELECT {} FROM {}.{}").format(
                                sql.SQL(",").join(expressions),
                                sql.Identifier(schema),
                                sql.Identifier(table),
                            )
                        ).fetchone()
                    entry["rows"] = counts[0]
                    position = 1
                    for field in fields:
                        for metric in field.pop("metrics"):
                            field[metric] = counts[position]
                            position += 1
                except Exception as exc:
                    # Do not serialize database exception text: it can include values.
                    entry["error"] = type(exc).__name__
                report["tables"].append(entry)
            report["complete"] = not any("error" in table for table in report["tables"])
            return report
