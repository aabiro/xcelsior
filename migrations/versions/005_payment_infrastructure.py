"""Payment infrastructure: webhook inbox, billing cycles, auto-top-up, marketplace, inference, storage, observability

Phase 1: Stripe event inbox for idempotent webhook processing
Phase 1: Billing cycles for auto-billing running instances
Phase 1: Auto-top-up columns on wallets
Phase 1: Idempotency key on wallet_transactions
Phase 1: FINTRAC reporting table
Phase 2: GPU offers, allocations, spot price history, reservations
Phase 3: Inference endpoints, worker model cache
Phase 4: Volumes and attachments
Phase 5: Event snapshots, cloud burst instances

Revision ID: 005
Revises: 004
Create Date: 2026-03-30
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 1: Payment Infrastructure
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ── Stripe webhook event inbox (idempotent processing) ───────────
    op.create_table(
        "stripe_event_inbox",
        sa.Column("event_id", sa.Text(), primary_key=True),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("stripe_account", sa.Text(), server_default=""),
        sa.Column("livemode", sa.Boolean(), server_default="true"),
        sa.Column("api_version", sa.Text(), server_default=""),
        sa.Column("created_unix", sa.BigInteger(), nullable=False),
        sa.Column("received_at", sa.Float(), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.Text(), server_default="pending"),
        sa.Column("attempts", sa.Integer(), server_default="0"),
        sa.Column("last_error", sa.Text(), server_default=""),
        sa.Column("next_retry_at", sa.Float(), server_default="0"),
        sa.Column("processed_at", sa.Float(), server_default="0"),
    )
    op.create_index(
        "idx_event_inbox_status",
        "stripe_event_inbox",
        ["status", "next_retry_at"],
    )
    op.create_index(
        "idx_event_inbox_type",
        "stripe_event_inbox",
        ["event_type", "created_unix"],
    )

    # ── Billing cycles for running instances ─────────────────────────
    op.create_table(
        "billing_cycles",
        sa.Column("cycle_id", sa.Text(), primary_key=True),
        sa.Column("job_id", sa.Text(), nullable=False),
        sa.Column("customer_id", sa.Text(), nullable=False),
        sa.Column("host_id", sa.Text(), server_default=""),
        sa.Column("period_start", sa.Float(), nullable=False),
        sa.Column("period_end", sa.Float(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("rate_per_hour", sa.Float(), nullable=False),
        sa.Column("gpu_model", sa.Text(), server_default=""),
        sa.Column("tier", sa.Text(), server_default="free"),
        sa.Column("tier_multiplier", sa.Float(), server_default="1.0"),
        sa.Column("amount_cad", sa.Float(), nullable=False),
        sa.Column("status", sa.Text(), server_default="charged"),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("idx_billing_cycles_job", "billing_cycles", ["job_id"])
    op.create_index("idx_billing_cycles_customer", "billing_cycles", ["customer_id"])
    op.create_index(
        "idx_billing_cycles_pending",
        "billing_cycles",
        ["status"],
        postgresql_where=sa.text("status = 'pending'"),
    )

    # ── Wallet auto-top-up columns ───────────────────────────────────
    op.add_column("wallets", sa.Column("auto_topup_enabled", sa.Boolean(), server_default="false"))
    op.add_column("wallets", sa.Column("auto_topup_amount_cad", sa.Float(), server_default="50.0"))
    op.add_column("wallets", sa.Column("auto_topup_threshold_cad", sa.Float(), server_default="10.0"))
    op.add_column("wallets", sa.Column("stripe_payment_method_id", sa.Text(), server_default=""))
    op.add_column("wallets", sa.Column("auto_topup_failures", sa.Integer(), server_default="0"))
    op.add_column("wallets", sa.Column("last_topup_attempt_at", sa.Float(), server_default="0"))

    # ── Idempotency key on wallet_transactions ───────────────────────
    op.add_column("wallet_transactions", sa.Column("idempotency_key", sa.Text(), server_default=""))
    op.create_index(
        "idx_wallet_tx_idempotency",
        "wallet_transactions",
        ["idempotency_key"],
        unique=True,
        postgresql_where=sa.text("idempotency_key != ''"),
    )

    # ── FINTRAC reporting table ──────────────────────────────────────
    op.create_table(
        "fintrac_reports",
        sa.Column("report_id", sa.Text(), primary_key=True),
        sa.Column("customer_id", sa.Text(), nullable=False),
        sa.Column("report_type", sa.Text(), nullable=False),  # LVCTR, STR
        sa.Column("trigger_amount_cad", sa.Float(), nullable=False),
        sa.Column("trigger_currency", sa.Text(), server_default="BTC"),
        sa.Column("aggregate_window_start", sa.Float(), nullable=False),
        sa.Column("aggregate_window_end", sa.Float(), nullable=False),
        sa.Column("transaction_ids", sa.dialects.postgresql.JSONB(), server_default="[]"),
        sa.Column("status", sa.Text(), server_default="pending"),  # pending, filed, reviewed
        sa.Column("filed_at", sa.Float(), server_default="0"),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("notes", sa.Text(), server_default=""),
    )
    op.create_index("idx_fintrac_customer", "fintrac_reports", ["customer_id"])
    op.create_index("idx_fintrac_status", "fintrac_reports", ["status"])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 2: Marketplace & Pricing
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ── GPU offers (provider-set pricing) ────────────────────────────
    op.create_table(
        "gpu_offers",
        sa.Column("offer_id", sa.Text(), primary_key=True),
        sa.Column("provider_id", sa.Text(), nullable=False),
        sa.Column("host_id", sa.Text(), nullable=False),
        sa.Column("gpu_model", sa.Text(), nullable=False),
        sa.Column("gpu_count_total", sa.Integer(), server_default="1"),
        sa.Column("gpu_count_available", sa.Integer(), server_default="1"),
        sa.Column("vram_gb", sa.Integer(), nullable=False),
        sa.Column("ask_cents_per_hour", sa.BigInteger(), nullable=False),
        sa.Column("spot_multiplier", sa.Float(), server_default="0.6"),
        sa.Column("currency", sa.Text(), server_default="CAD"),
        sa.Column("region", sa.Text(), server_default=""),
        sa.Column("province", sa.Text(), server_default=""),
        sa.Column("verified_level", sa.Text(), server_default="unverified"),
        sa.Column("reliability_score", sa.Float(), server_default="0"),
        sa.Column("dlperf_score", sa.Float(), server_default="0"),
        sa.Column("available", sa.Boolean(), server_default="true"),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.Float(), nullable=False),
    )
    op.create_index("idx_gpu_offers_provider", "gpu_offers", ["provider_id"])
    op.create_index("idx_gpu_offers_host", "gpu_offers", ["host_id"])
    op.create_index("idx_gpu_offers_gpu", "gpu_offers", ["gpu_model", "vram_gb"])
    op.create_index(
        "idx_gpu_offers_available",
        "gpu_offers",
        ["gpu_model", "ask_cents_per_hour"],
        postgresql_where=sa.text("available = true"),
    )

    # ── GPU allocations (prevents double-sell) ───────────────────────
    op.create_table(
        "gpu_allocations",
        sa.Column("allocation_id", sa.Text(), primary_key=True),
        sa.Column("offer_id", sa.Text(), nullable=False),
        sa.Column("job_id", sa.Text(), nullable=False),
        sa.Column("gpu_count", sa.Integer(), server_default="1"),
        sa.Column("price_cents_per_hour", sa.BigInteger(), nullable=False),
        sa.Column("allocation_type", sa.Text(), server_default="on_demand"),  # on_demand, spot, reserved
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("released_at", sa.Float(), server_default="0"),
    )
    op.create_index("idx_gpu_alloc_offer", "gpu_allocations", ["offer_id"])
    op.create_index("idx_gpu_alloc_job", "gpu_allocations", ["job_id"], unique=True)

    # ── Spot price history ───────────────────────────────────────────
    op.create_table(
        "spot_price_history",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("gpu_model", sa.Text(), nullable=False),
        sa.Column("region", sa.Text(), server_default=""),
        sa.Column("clearing_price_cents", sa.BigInteger(), nullable=False),
        sa.Column("supply_count", sa.Integer(), server_default="0"),
        sa.Column("demand_count", sa.Integer(), server_default="0"),
        sa.Column("recorded_at", sa.Float(), nullable=False),
    )
    op.create_index("idx_spot_history_gpu_time", "spot_price_history", ["gpu_model", "recorded_at"])

    # ── Reservations (term commitments) ──────────────────────────────
    op.create_table(
        "reservations",
        sa.Column("reservation_id", sa.Text(), primary_key=True),
        sa.Column("customer_id", sa.Text(), nullable=False),
        sa.Column("gpu_model", sa.Text(), nullable=False),
        sa.Column("gpu_count", sa.Integer(), server_default="1"),
        sa.Column("period_months", sa.Integer(), nullable=False),
        sa.Column("discount_pct", sa.Float(), nullable=False),
        sa.Column("monthly_rate_cad", sa.Float(), nullable=False),
        sa.Column("starts_at", sa.Float(), nullable=False),
        sa.Column("ends_at", sa.Float(), nullable=False),
        sa.Column("status", sa.Text(), server_default="active"),  # active, expired, cancelled
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("idx_reservations_customer", "reservations", ["customer_id"])
    op.create_index(
        "idx_reservations_active",
        "reservations",
        ["customer_id", "gpu_model"],
        postgresql_where=sa.text("status = 'active'"),
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 3: Serverless Inference
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ── Inference endpoints ──────────────────────────────────────────
    op.create_table(
        "inference_endpoints",
        sa.Column("endpoint_id", sa.Text(), primary_key=True),
        sa.Column("owner_id", sa.Text(), nullable=False),
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column("model_revision", sa.Text(), server_default="main"),
        sa.Column("gpu_type", sa.Text(), server_default=""),
        sa.Column("vram_required_gb", sa.Float(), server_default="0"),
        sa.Column("max_batch_size", sa.Integer(), server_default="8"),
        sa.Column("max_concurrent", sa.Integer(), server_default="4"),
        sa.Column("min_workers", sa.Integer(), server_default="0"),
        sa.Column("max_workers", sa.Integer(), server_default="4"),
        sa.Column("scaledown_window_sec", sa.Integer(), server_default="300"),
        sa.Column("status", sa.Text(), server_default="active"),  # active, paused, deleted
        sa.Column("total_requests", sa.BigInteger(), server_default="0"),
        sa.Column("total_tokens_generated", sa.BigInteger(), server_default="0"),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.Float(), nullable=False),
    )
    op.create_index("idx_inference_ep_owner", "inference_endpoints", ["owner_id"])
    op.create_index("idx_inference_ep_model", "inference_endpoints", ["model_id"])

    # ── Worker model cache (warm/cold tracking) ──────────────────────
    op.create_table(
        "worker_model_cache",
        sa.Column("worker_id", sa.Text(), nullable=False),
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column("model_revision", sa.Text(), server_default="main"),
        sa.Column("state", sa.Text(), server_default="loading"),  # loading, ready, evicting, error
        sa.Column("vram_bytes", sa.BigInteger(), server_default="0"),
        sa.Column("loaded_at", sa.Float(), server_default="0"),
        sa.Column("last_used_at", sa.Float(), server_default="0"),
        sa.PrimaryKeyConstraint("worker_id", "model_id", "model_revision"),
    )
    op.create_index(
        "idx_worker_model_ready",
        "worker_model_cache",
        ["model_id", "state", "last_used_at"],
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 4: Persistent Storage
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ── Persistent volumes ───────────────────────────────────────────
    op.create_table(
        "volumes",
        sa.Column("volume_id", sa.Text(), primary_key=True),
        sa.Column("owner_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("storage_type", sa.Text(), server_default="nfs"),  # nfs, block, object_ref
        sa.Column("size_gb", sa.Integer(), nullable=False),
        sa.Column("region", sa.Text(), server_default=""),
        sa.Column("province", sa.Text(), server_default=""),
        sa.Column("encrypted", sa.Boolean(), server_default="true"),
        sa.Column("encryption_key_id", sa.Text(), server_default=""),
        sa.Column("mount_path_host", sa.Text(), server_default=""),
        sa.Column("status", sa.Text(), server_default="creating"),  # creating, available, attached, deleting, deleted
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("deleted_at", sa.Float(), server_default="0"),
    )
    op.create_index("idx_volumes_owner", "volumes", ["owner_id"])
    op.create_index(
        "idx_volumes_available",
        "volumes",
        ["owner_id", "status"],
        postgresql_where=sa.text("status != 'deleted'"),
    )

    # ── Volume attachments ───────────────────────────────────────────
    op.create_table(
        "volume_attachments",
        sa.Column("attachment_id", sa.Text(), primary_key=True),
        sa.Column("volume_id", sa.Text(), nullable=False),
        sa.Column("instance_id", sa.Text(), nullable=False),
        sa.Column("mount_path", sa.Text(), server_default="/workspace"),
        sa.Column("mode", sa.Text(), server_default="rw"),  # ro, rw
        sa.Column("attached_at", sa.Float(), nullable=False),
        sa.Column("detached_at", sa.Float(), server_default="0"),
    )
    op.create_index("idx_vol_attach_volume", "volume_attachments", ["volume_id"])
    op.create_index("idx_vol_attach_instance", "volume_attachments", ["instance_id"])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 5: Auto-Scaling & Observability
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ── Cloud burst instances ────────────────────────────────────────
    op.create_table(
        "cloud_burst_instances",
        sa.Column("instance_id", sa.Text(), primary_key=True),
        sa.Column("cloud_provider", sa.Text(), nullable=False),  # aws, gcp
        sa.Column("cloud_instance_id", sa.Text(), server_default=""),
        sa.Column("instance_type", sa.Text(), nullable=False),
        sa.Column("region", sa.Text(), nullable=False),
        sa.Column("gpu_model", sa.Text(), server_default=""),
        sa.Column("gpu_count", sa.Integer(), server_default="1"),
        sa.Column("cost_per_hour_cad", sa.Float(), nullable=False),
        sa.Column("host_id", sa.Text(), server_default=""),  # xcelsior host_id once registered
        sa.Column("status", sa.Text(), server_default="provisioning"),  # provisioning, running, draining, terminated
        sa.Column("started_at", sa.Float(), nullable=False),
        sa.Column("terminated_at", sa.Float(), server_default="0"),
        sa.Column("budget_spent_cad", sa.Float(), server_default="0"),
    )
    op.create_index("idx_cloud_burst_status", "cloud_burst_instances", ["status"])

    # ── Event snapshots (for event sourcing at scale) ────────────────
    op.create_table(
        "event_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("entity_type", sa.Text(), nullable=False),
        sa.Column("entity_id", sa.Text(), nullable=False),
        sa.Column("snapshot_data", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column("sequence_number", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index(
        "idx_event_snap_entity",
        "event_snapshots",
        ["entity_type", "entity_id", "sequence_number"],
    )

    # ── Events archive (cold storage for events >90 days) ────────────
    op.create_table(
        "events_archive",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("event_id", sa.Text(), nullable=False),
        sa.Column("entity_type", sa.Text(), nullable=False),
        sa.Column("entity_id", sa.Text(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("data", sa.dialects.postgresql.JSONB()),
        sa.Column("actor", sa.Text(), server_default=""),
        sa.Column("chain_hash", sa.Text(), server_default=""),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("archived_at", sa.Float(), nullable=False),
    )
    op.create_index("idx_events_archive_entity", "events_archive", ["entity_type", "entity_id"])
    op.create_index("idx_events_archive_created", "events_archive", ["created_at"])

    # ── SLA auto-credit tracking ─────────────────────────────────────
    op.add_column("sla_monthly", sa.Column("credit_issued_at", sa.Float(), server_default="0"))
    op.add_column("sla_monthly", sa.Column("credit_tx_id", sa.Text(), server_default=""))


def downgrade() -> None:
    # Drop in reverse order
    op.drop_column("sla_monthly", "credit_tx_id")
    op.drop_column("sla_monthly", "credit_issued_at")

    op.drop_table("events_archive")
    op.drop_table("event_snapshots")
    op.drop_table("cloud_burst_instances")
    op.drop_table("volume_attachments")
    op.drop_table("volumes")
    op.drop_table("worker_model_cache")
    op.drop_table("inference_endpoints")
    op.drop_table("reservations")
    op.drop_table("spot_price_history")
    op.drop_table("gpu_allocations")
    op.drop_table("gpu_offers")
    op.drop_table("fintrac_reports")

    op.drop_index("idx_wallet_tx_idempotency", "wallet_transactions")
    op.drop_column("wallet_transactions", "idempotency_key")

    op.drop_column("wallets", "last_topup_attempt_at")
    op.drop_column("wallets", "auto_topup_failures")
    op.drop_column("wallets", "stripe_payment_method_id")
    op.drop_column("wallets", "auto_topup_threshold_cad")
    op.drop_column("wallets", "auto_topup_amount_cad")
    op.drop_column("wallets", "auto_topup_enabled")

    op.drop_table("billing_cycles")
    op.drop_table("stripe_event_inbox")
