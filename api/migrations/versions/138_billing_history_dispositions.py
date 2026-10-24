"""Retain qualified historical financial disposition without rewriting evidence.

This migration creates the receipt relation only. It issues no credits and
deletes no history. The separate operator command requires a qualified plan.
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "138"
down_revision = "137"
branch_labels = None
depends_on = None


def _expand(conn):
    conn.execute(text("""CREATE TABLE IF NOT EXISTS billing_history_dispositions (
        disposition_key TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        job_id TEXT NOT NULL,
        policy_version TEXT NOT NULL CHECK(policy_version='volume_overlap/v1'),
        disposition TEXT NOT NULL CHECK(disposition='credited_duplicate_interval'),
        evidence JSONB NOT NULL CHECK(jsonb_typeof(evidence)='object'),
        evidence_sha256 TEXT NOT NULL CHECK(evidence_sha256 ~ '^[0-9a-f]{64}$'),
        credit_micros BIGINT NOT NULL CHECK(credit_micros>0),
        credit_tx_id TEXT NOT NULL UNIQUE REFERENCES wallet_transactions(tx_id),
        created_by TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
    )"""))
    conn.execute(text("""CREATE INDEX IF NOT EXISTS ix_billing_history_dispositions_tenant
        ON billing_history_dispositions(tenant_id,created_at)"""))


def upgrade():
    apply_in_own_transactions(
        [("financial disposition receipts", _expand)], tables=["billing_history_dispositions"]
    )


def _contract(conn):
    if conn.execute(text("SELECT 1 FROM billing_history_dispositions LIMIT 1")).first():
        raise RuntimeError("retain financial disposition evidence; downgrade refused")
    conn.execute(text("DROP TABLE billing_history_dispositions"))


def downgrade():
    apply_in_own_transactions(
        [("remove unused disposition relation", _contract)], tables=["billing_history_dispositions"]
    )
