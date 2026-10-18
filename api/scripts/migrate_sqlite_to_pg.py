#!/usr/bin/env python3
"""
Migrate all SQLite data into PostgreSQL.

Run AFTER alembic upgrade head has created all PG tables.
Safe to re-run — uses INSERT ... ON CONFLICT DO NOTHING for idempotency.

Usage:
    python scripts/migrate_sqlite_to_pg.py [--dry-run]
"""

import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from psycopg.types.json import Jsonb

# ── Configuration ─────────────────────────────────────────────────────
DRY_RUN = "--dry-run" in sys.argv

SQLITE_DBS = {
    "auth":         os.environ.get("XCELSIOR_AUTH_DB_PATH", "data/auth.db"),
    "billing":      os.environ.get("XCELSIOR_BILLING_DB_PATH", "xcelsior_billing.db"),
    "reputation":   os.environ.get("XCELSIOR_REPUTATION_DB_PATH", "xcelsior_reputation.db"),
    "chat":         os.environ.get("XCELSIOR_CHAT_DB_PATH", "data/chat.db"),
    "sla":          os.environ.get("XCELSIOR_SLA_DB", "xcelsior_sla.db"),
    "privacy":      os.environ.get("XCELSIOR_PRIVACY_DB_PATH", "xcelsior_privacy.db"),
    "events":       "xcelsior_events.db",
    "stripe":       os.environ.get("XCELSIOR_STRIPE_DB", "xcelsior_stripe.db"),
    "transparency": os.environ.get("XCELSIOR_TRANSPARENCY_DB_PATH", "xcelsior_transparency.db"),
    "inference":    os.environ.get("XCELSIOR_INFERENCE_DB_PATH", "data/inference.db"),
    "btc":          os.environ.get("XCELSIOR_BTC_DB", "xcelsior_btc.db"),
}

# Columns that are JSONB in PG but TEXT in SQLite (need json.loads)
JSONB_COLUMNS = {
    "notifications": {"data"},
    "invoices": {"line_items"},
    "reputation_scores": {"verifications"},
    "reputation_events": {"metadata"},
    "retention_records": {"metadata"},
    "consent_records": {"details"},
    "privacy_configs": {"config"},
    "events": {"data", "metadata"},
    "host_verifications": {"checks"},
    "verification_history": {"checks"},
}

# ── Helpers ───────────────────────────────────────────────────────────

def sqlite_rows(db_path: str, table: str) -> tuple[list[str], list[tuple]]:
    """Return (column_names, rows) from a SQLite table."""
    if not os.path.exists(db_path):
        return [], []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(f"SELECT * FROM [{table}]")
        rows = cur.fetchall()
        if not rows:
            return [], []
        cols = rows[0].keys()
        return list(cols), [tuple(r) for r in rows]
    except Exception:
        return [], []
    finally:
        conn.close()


def convert_jsonb(table: str, cols: list[str], rows: list[tuple]) -> list[tuple]:
    """Convert TEXT JSON columns to Jsonb() for psycopg3."""
    jsonb_cols = JSONB_COLUMNS.get(table, set())
    if not jsonb_cols:
        return rows
    jsonb_idxs = [i for i, c in enumerate(cols) if c in jsonb_cols]
    if not jsonb_idxs:
        return rows
    converted = []
    for row in rows:
        row = list(row)
        for idx in jsonb_idxs:
            val = row[idx]
            if isinstance(val, str):
                try:
                    row[idx] = Jsonb(json.loads(val))
                except (json.JSONDecodeError, TypeError):
                    row[idx] = Jsonb(val)
            elif val is None:
                row[idx] = Jsonb({})
        converted.append(tuple(row))
    return converted


def pg_insert(conn, table: str, cols: list[str], rows: list[tuple]) -> int:
    """Insert rows into PG with ON CONFLICT DO NOTHING. Returns count."""
    if not rows:
        return 0
    placeholders = ", ".join(["%s"] * len(cols))
    col_list = ", ".join(f'"{c}"' for c in cols)
    sql = f'INSERT INTO {table} ({col_list}) VALUES ({placeholders}) ON CONFLICT DO NOTHING'
    if DRY_RUN:
        print(f"  [DRY RUN] Would insert {len(rows)} rows into {table}")
        return len(rows)
    count = 0
    for row in rows:
        try:
            conn.execute(sql, row)
            count += 1
        except Exception as e:
            print(f"  WARN: {table} row skip: {e}")
            conn.rollback()
            conn.execute("SELECT 1")  # reset connection state
    return count


def filter_columns(pg_conn, table: str, sqlite_cols: list[str], rows: list[tuple]) -> tuple[list[str], list[tuple]]:
    """Filter out SQLite columns that don't exist in PG table."""
    cur = pg_conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = %s AND table_schema = 'public'",
        (table,)
    )
    pg_cols = {r[0] for r in cur.fetchall()}
    if not pg_cols:
        return [], []
    # Find indices of cols that exist in PG
    valid = [(i, c) for i, c in enumerate(sqlite_cols) if c in pg_cols]
    if not valid:
        return [], []
    new_cols = [c for _, c in valid]
    idxs = [i for i, _ in valid]
    new_rows = [tuple(row[i] for i in idxs) for row in rows]
    return new_cols, new_rows


# ── Migration Map ─────────────────────────────────────────────────────
# (db_key, sqlite_table, pg_table)
MIGRATIONS = [
    # Auth
    ("auth", "users", "users"),
    ("auth", "sessions", "sessions"),
    ("auth", "api_keys", "api_keys"),
    ("auth", "teams", "teams"),
    ("auth", "team_members", "team_members"),
    ("auth", "notifications", "notifications"),
    ("auth", "user_ssh_keys", "user_ssh_keys"),
    # Billing
    ("billing", "usage_meters", "usage_meters"),
    ("billing", "invoices", "invoices"),
    ("billing", "payout_ledger", "payout_ledger"),
    ("billing", "wallets", "wallets"),
    ("billing", "wallet_transactions", "wallet_transactions"),
    # Reputation
    ("reputation", "reputation_scores", "reputation_scores"),
    ("reputation", "reputation_events", "reputation_events"),
    # Privacy
    ("privacy", "retention_records", "retention_records"),
    ("privacy", "consent_records", "consent_records"),
    ("privacy", "privacy_configs", "privacy_configs"),
    # SLA
    ("sla", "sla_downtime", "sla_downtime"),
    ("sla", "sla_monthly", "sla_monthly"),
    ("sla", "sla_violations", "sla_violations"),
    # Stripe
    ("stripe", "provider_accounts", "provider_accounts"),
    ("stripe", "payment_intents", "payment_intents"),
    ("stripe", "payout_splits", "payout_splits"),
    # Events
    ("events", "events", "events"),
    ("events", "leases", "leases"),
    # Verification
    ("events", "host_verifications", "host_verifications"),
    ("events", "verification_history", "verification_history"),
    ("events", "job_failure_log", "job_failure_log"),
    # Inference
    ("inference", "inference_jobs", "inference_jobs"),
    ("inference", "inference_results", "inference_results"),
    # Chat
    ("chat", "chat_conversations", "chat_conversations"),
    ("chat", "chat_messages", "chat_messages"),
    ("chat", "chat_feedback", "chat_feedback"),
    # Transparency
    ("transparency", "legal_requests", "legal_requests"),
    ("transparency", "data_disclosures", "data_disclosures"),
    # Bitcoin
    ("btc", "crypto_deposits", "crypto_deposits"),
]


# ── Main ──────────────────────────────────────────────────────────────

def main():
    from db import pg_connection

    mode = "DRY RUN" if DRY_RUN else "LIVE"
    print(f"\n{'='*60}")
    print(f"  SQLite → PostgreSQL Data Migration ({mode})")
    print(f"{'='*60}\n")

    total_migrated = 0
    total_skipped = 0

    with pg_connection() as conn:
        for db_key, sqlite_table, pg_table in MIGRATIONS:
            db_path = SQLITE_DBS.get(db_key, "")
            if not os.path.exists(db_path):
                continue

            cols, rows = sqlite_rows(db_path, sqlite_table)
            if not rows:
                continue

            # Filter to only columns that exist in PG
            cols, rows = filter_columns(conn, pg_table, cols, rows)
            if not rows:
                continue

            # Convert JSONB columns
            rows = convert_jsonb(pg_table, cols, rows)

            count = pg_insert(conn, pg_table, cols, rows)
            total_migrated += count
            skipped = len(rows) - count if not DRY_RUN else 0
            total_skipped += skipped

            status = f"{count} migrated"
            if skipped:
                status += f", {skipped} skipped (already exist)"
            print(f"  {pg_table}: {status}")

        if not DRY_RUN:
            conn.execute("COMMIT")
            print(f"\n  COMMITTED.")

    print(f"\n{'='*60}")
    print(f"  Total: {total_migrated} rows migrated, {total_skipped} skipped")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
