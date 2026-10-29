"""Add composite indexes for all AI tool queries — dramatically speeds up
every new tool added in this batch: wallet transactions, invoices, payouts,
notifications, benchmarks, SLA, spot price history, billing cycles,
inference endpoints, reputation events, and SSH keys.

Revision ID: 012
Revises: 011
Create Date: 2026-04-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Use IF NOT EXISTS for every index so re-running after an interrupted
# migration does not fail with DuplicateTable.
_INDEXES = [
    ("wallet_transactions", "idx_wallet_tx_customer_created", "CREATE INDEX IF NOT EXISTS idx_wallet_tx_customer_created ON wallet_transactions (customer_id, created_at DESC)"),
    ("wallet_transactions", "idx_wallet_tx_customer_type", "CREATE INDEX IF NOT EXISTS idx_wallet_tx_customer_type ON wallet_transactions (customer_id, tx_type, created_at)"),
    ("invoices", "idx_invoices_customer_created", "CREATE INDEX IF NOT EXISTS idx_invoices_customer_created ON invoices (customer_id, created_at DESC)"),
    ("payout_ledger", "idx_payout_ledger_provider_created", "CREATE INDEX IF NOT EXISTS idx_payout_ledger_provider_created ON payout_ledger (provider_id, created_at DESC)"),
    ("notifications", "idx_notifications_email_read_created", "CREATE INDEX IF NOT EXISTS idx_notifications_email_read_created ON notifications (user_email, read, created_at DESC)"),
    ("benchmarks", "idx_benchmarks_host_run", "CREATE INDEX IF NOT EXISTS idx_benchmarks_host_run ON benchmarks (host_id, run_at DESC)"),
    ("benchmarks", "idx_benchmarks_gpu_run", "CREATE INDEX IF NOT EXISTS idx_benchmarks_gpu_run ON benchmarks (gpu_model, run_at DESC)"),
    ("sla_monthly", "idx_sla_monthly_host_month", "CREATE INDEX IF NOT EXISTS idx_sla_monthly_host_month ON sla_monthly (host_id, month DESC)"),
    ("sla_violations", "idx_sla_violations_host_ts", "CREATE INDEX IF NOT EXISTS idx_sla_violations_host_ts ON sla_violations (host_id, timestamp DESC)"),
    ("spot_price_history", "idx_spot_price_history_model_recorded", "CREATE INDEX IF NOT EXISTS idx_spot_price_history_model_recorded ON spot_price_history (gpu_model, recorded_at DESC)"),
    ("billing_cycles", "idx_billing_cycles_customer_created", "CREATE INDEX IF NOT EXISTS idx_billing_cycles_customer_created ON billing_cycles (customer_id, created_at DESC)"),
    ("billing_cycles", "idx_billing_cycles_job", "CREATE INDEX IF NOT EXISTS idx_billing_cycles_job ON billing_cycles (customer_id, job_id, created_at)"),
    ("inference_endpoints", "idx_inference_endpoints_owner_created", "CREATE INDEX IF NOT EXISTS idx_inference_endpoints_owner_created ON inference_endpoints (owner_id, created_at DESC)"),
    ("reputation_events", "idx_reputation_events_entity_created", "CREATE INDEX IF NOT EXISTS idx_reputation_events_entity_created ON reputation_events (entity_id, created_at DESC)"),
    ("user_ssh_keys", "idx_user_ssh_keys_email_created", "CREATE INDEX IF NOT EXISTS idx_user_ssh_keys_email_created ON user_ssh_keys (email, created_at DESC)"),
    ("crypto_deposits", "idx_crypto_deposits_customer_created", "CREATE INDEX IF NOT EXISTS idx_crypto_deposits_customer_created ON crypto_deposits (customer_id, created_at DESC)"),
    ("provider_accounts", "idx_provider_accounts_email", "CREATE INDEX IF NOT EXISTS idx_provider_accounts_email ON provider_accounts (email)"),
]

_INDEX_NAMES = [name for _, name, _ in _INDEXES]


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    for table_name, _, sql in _INDEXES:
        if inspector.has_table(table_name):
            op.execute(sql)


def downgrade() -> None:
    for name in reversed(_INDEX_NAMES):
        op.execute(f"DROP INDEX IF EXISTS {name}")
