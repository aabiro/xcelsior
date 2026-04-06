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

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── wallet_transactions ──────────────────────────────────────────
    op.create_index(
        "idx_wallet_tx_customer_created",
        "wallet_transactions",
        ["customer_id", "created_at"],
        postgresql_ops={"created_at": "DESC"},
    )
    op.create_index(
        "idx_wallet_tx_customer_type",
        "wallet_transactions",
        ["customer_id", "tx_type", "created_at"],
    )

    # ── invoices ─────────────────────────────────────────────────────
    op.create_index(
        "idx_invoices_customer_created",
        "invoices",
        ["customer_id", "created_at"],
        postgresql_ops={"created_at": "DESC"},
    )

    # ── payout_ledger ────────────────────────────────────────────────
    op.create_index(
        "idx_payout_ledger_provider_created",
        "payout_ledger",
        ["provider_id", "created_at"],
        postgresql_ops={"created_at": "DESC"},
    )

    # ── notifications ────────────────────────────────────────────────
    op.create_index(
        "idx_notifications_email_read_created",
        "notifications",
        ["user_email", "read", "created_at"],
        postgresql_ops={"created_at": "DESC"},
    )

    # ── benchmarks ───────────────────────────────────────────────────
    op.create_index(
        "idx_benchmarks_host_run",
        "benchmarks",
        ["host_id", "run_at"],
        postgresql_ops={"run_at": "DESC"},
    )
    op.create_index(
        "idx_benchmarks_gpu_run",
        "benchmarks",
        ["gpu_model", "run_at"],
        postgresql_ops={"run_at": "DESC"},
    )

    # ── sla_monthly ──────────────────────────────────────────────────
    op.create_index(
        "idx_sla_monthly_host_month",
        "sla_monthly",
        ["host_id", "month"],
        postgresql_ops={"month": "DESC"},
    )

    # ── sla_violations ───────────────────────────────────────────────
    op.create_index(
        "idx_sla_violations_host_ts",
        "sla_violations",
        ["host_id", "timestamp"],
        postgresql_ops={"timestamp": "DESC"},
    )

    # ── spot_price_history ───────────────────────────────────────────
    op.create_index(
        "idx_spot_price_history_model_recorded",
        "spot_price_history",
        ["gpu_model", "recorded_at"],
        postgresql_ops={"recorded_at": "DESC"},
    )

    # ── billing_cycles ───────────────────────────────────────────────
    op.create_index(
        "idx_billing_cycles_customer_created",
        "billing_cycles",
        ["customer_id", "created_at"],
        postgresql_ops={"created_at": "DESC"},
    )
    op.create_index(
        "idx_billing_cycles_job",
        "billing_cycles",
        ["customer_id", "job_id", "created_at"],
    )

    # ── inference_endpoints ──────────────────────────────────────────
    op.create_index(
        "idx_inference_endpoints_owner_created",
        "inference_endpoints",
        ["owner_id", "created_at"],
        postgresql_ops={"created_at": "DESC"},
    )

    # ── reputation_events ────────────────────────────────────────────
    op.create_index(
        "idx_reputation_events_entity_created",
        "reputation_events",
        ["entity_id", "created_at"],
        postgresql_ops={"created_at": "DESC"},
    )

    # ── user_ssh_keys ────────────────────────────────────────────────
    op.create_index(
        "idx_user_ssh_keys_email_created",
        "user_ssh_keys",
        ["email", "created_at"],
        postgresql_ops={"created_at": "DESC"},
    )

    # ── crypto_deposits ──────────────────────────────────────────────
    op.create_index(
        "idx_crypto_deposits_customer_created",
        "crypto_deposits",
        ["customer_id", "created_at"],
        postgresql_ops={"created_at": "DESC"},
    )

    # ── provider_accounts ────────────────────────────────────────────
    op.create_index(
        "idx_provider_accounts_email",
        "provider_accounts",
        ["email"],
    )


def downgrade() -> None:
    op.drop_index("idx_provider_accounts_email", table_name="provider_accounts")
    op.drop_index("idx_crypto_deposits_customer_created", table_name="crypto_deposits")
    op.drop_index("idx_user_ssh_keys_email_created", table_name="user_ssh_keys")
    op.drop_index("idx_reputation_events_entity_created", table_name="reputation_events")
    op.drop_index("idx_inference_endpoints_owner_created", table_name="inference_endpoints")
    op.drop_index("idx_billing_cycles_job", table_name="billing_cycles")
    op.drop_index("idx_billing_cycles_customer_created", table_name="billing_cycles")
    op.drop_index("idx_spot_price_history_model_recorded", table_name="spot_price_history")
    op.drop_index("idx_sla_violations_host_ts", table_name="sla_violations")
    op.drop_index("idx_sla_monthly_host_month", table_name="sla_monthly")
    op.drop_index("idx_benchmarks_gpu_run", table_name="benchmarks")
    op.drop_index("idx_benchmarks_host_run", table_name="benchmarks")
    op.drop_index("idx_notifications_email_read_created", table_name="notifications")
    op.drop_index("idx_payout_ledger_provider_created", table_name="payout_ledger")
    op.drop_index("idx_invoices_customer_created", table_name="invoices")
    op.drop_index("idx_wallet_tx_customer_type", table_name="wallet_transactions")
    op.drop_index("idx_wallet_tx_customer_created", table_name="wallet_transactions")
