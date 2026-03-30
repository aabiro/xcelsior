"""Consolidate all SQLite tables into PostgreSQL

Migrates every remaining SQLite database into PostgreSQL:
- Auth (users, sessions, api_keys, teams, team_members, notifications, user_ssh_keys)
- Billing (usage_meters, invoices, payout_ledger, wallets, wallet_transactions)
- Reputation (reputation_scores, reputation_events)
- Privacy (retention_records, consent_records, privacy_configs)
- SLA (sla_downtime, sla_monthly, sla_violations)
- Stripe Connect (provider_accounts, payment_intents, payout_splits)
- Events (events, leases)
- Verification (host_verifications, verification_history, job_failure_log)
- Inference (inference_jobs, inference_results)
- Chat (chat_conversations, chat_messages, chat_feedback)
- Transparency (legal_requests, data_disclosures)

Revision ID: 004
Revises: 003
Create Date: 2026-03-29
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Auth: users ──────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("email", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False, unique=True),
        sa.Column("name", sa.Text(), nullable=False, server_default=""),
        sa.Column("password_hash", sa.Text(), nullable=False, server_default=""),
        sa.Column("salt", sa.Text(), nullable=False, server_default=""),
        sa.Column("role", sa.Text(), nullable=False, server_default="submitter"),
        sa.Column("customer_id", sa.Text()),
        sa.Column("provider_id", sa.Text()),
        sa.Column("country", sa.Text(), server_default="CA"),
        sa.Column("province", sa.Text(), server_default="ON"),
        sa.Column("oauth_provider", sa.Text()),
        sa.Column("team_id", sa.Text()),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("reset_token", sa.Text()),
        sa.Column("reset_token_expires", sa.Float()),
        sa.Column("notifications_enabled", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("canada_only_routing", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("idx_users_team", "users", ["team_id"])
    op.create_index("idx_users_user_id", "users", ["user_id"])
    op.create_index("idx_users_customer_id", "users", ["customer_id"])

    # ── Auth: sessions ───────────────────────────────────────────────
    op.create_table(
        "sessions",
        sa.Column("token", sa.Text(), primary_key=True),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False, server_default="submitter"),
        sa.Column("name", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("expires_at", sa.Float(), nullable=False),
    )
    op.create_index("idx_sessions_email", "sessions", ["email"])
    op.create_index("idx_sessions_expires", "sessions", ["expires_at"])

    # ── Auth: api_keys ───────────────────────────────────────────────
    op.create_table(
        "api_keys",
        sa.Column("key", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False, server_default="default"),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False, server_default="submitter"),
        sa.Column("scope", sa.Text(), nullable=False, server_default="full-access"),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("last_used", sa.Float()),
    )
    op.create_index("idx_api_keys_email", "api_keys", ["email"])

    # ── Auth: teams ──────────────────────────────────────────────────
    op.create_table(
        "teams",
        sa.Column("team_id", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("owner_email", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("plan", sa.Text(), nullable=False, server_default="free"),
        sa.Column("max_members", sa.Integer(), nullable=False, server_default="5"),
    )

    # ── Auth: team_members ───────────────────────────────────────────
    op.create_table(
        "team_members",
        sa.Column("team_id", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False, server_default="member"),
        sa.Column("joined_at", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("team_id", "email"),
    )
    op.create_index("idx_team_members_email", "team_members", ["email"])

    # ── Auth: notifications ──────────────────────────────────────────
    op.create_table(
        "notifications",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_email", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False, server_default=""),
        sa.Column("data", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("read", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index(
        "idx_notif_user", "notifications",
        [sa.text("user_email"), sa.text("read"), sa.text("created_at DESC")],
    )
    op.create_index("idx_notif_created", "notifications", ["created_at"])
    # PG optimisation: partial index for unread-only (hot path)
    op.create_index(
        "idx_notif_unread", "notifications",
        [sa.text("user_email"), sa.text("created_at DESC")],
        postgresql_where=sa.text("read = 0"),
    )
    # PG optimisation: GIN index on JSONB data for containment queries
    op.create_index("idx_notif_data_gin", "notifications", ["data"], postgresql_using="gin")

    # ── Auth: user_ssh_keys ──────────────────────────────────────────
    op.create_table(
        "user_ssh_keys",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False, server_default="default"),
        sa.Column("public_key", sa.Text(), nullable=False),
        sa.Column("fingerprint", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("idx_ssh_keys_email", "user_ssh_keys", ["email"])

    # ── Billing: usage_meters ────────────────────────────────────────
    op.create_table(
        "usage_meters",
        sa.Column("meter_id", sa.Text(), primary_key=True),
        sa.Column("job_id", sa.Text(), nullable=False),
        sa.Column("host_id", sa.Text(), server_default=""),
        sa.Column("owner", sa.Text(), server_default=""),
        sa.Column("started_at", sa.Float(), server_default="0"),
        sa.Column("completed_at", sa.Float(), server_default="0"),
        sa.Column("duration_sec", sa.Float(), server_default="0"),
        sa.Column("gpu_seconds", sa.Float(), server_default="0"),
        sa.Column("gpu_model", sa.Text(), server_default=""),
        sa.Column("vram_gb", sa.Float(), server_default="0"),
        sa.Column("gpu_utilization_pct", sa.Float(), server_default="0"),
        sa.Column("xcu_score", sa.Float(), server_default="0"),
        sa.Column("country", sa.Text(), server_default=""),
        sa.Column("province", sa.Text(), server_default=""),
        sa.Column("is_canadian_compute", sa.Integer(), server_default="0"),
        sa.Column("trust_tier", sa.Text(), server_default="community"),
        sa.Column("base_rate_per_hour", sa.Float(), server_default="0"),
        sa.Column("tier_multiplier", sa.Float(), server_default="1.0"),
        sa.Column("spot_discount", sa.Float(), server_default="0"),
        sa.Column("total_cost_cad", sa.Float(), server_default="0"),
        sa.Column("created_at", sa.Float(), server_default="0"),
    )
    op.create_index("idx_meters_job", "usage_meters", ["job_id"])
    op.create_index("idx_meters_owner", "usage_meters", ["owner"])
    op.create_index("idx_meters_time", "usage_meters", ["started_at"])

    # ── Billing: invoices ────────────────────────────────────────────
    op.create_table(
        "invoices",
        sa.Column("invoice_id", sa.Text(), primary_key=True),
        sa.Column("customer_id", sa.Text(), nullable=False),
        sa.Column("customer_name", sa.Text(), server_default=""),
        sa.Column("currency", sa.Text(), server_default="CAD"),
        sa.Column("period_start", sa.Float(), server_default="0"),
        sa.Column("period_end", sa.Float(), server_default="0"),
        sa.Column("line_items", JSONB, server_default=sa.text("'[]'::jsonb")),
        sa.Column("subtotal_cad", sa.Float(), server_default="0"),
        sa.Column("tax_rate", sa.Float(), server_default="0"),
        sa.Column("tax_amount_cad", sa.Float(), server_default="0"),
        sa.Column("total_cad", sa.Float(), server_default="0"),
        sa.Column("canadian_compute_total_cad", sa.Float(), server_default="0"),
        sa.Column("non_canadian_compute_total_cad", sa.Float(), server_default="0"),
        sa.Column("fund_eligible_reimbursement_cad", sa.Float(), server_default="0"),
        sa.Column("effective_cost_after_fund_cad", sa.Float(), server_default="0"),
        sa.Column("created_at", sa.Float(), server_default="0"),
        sa.Column("status", sa.Text(), server_default="draft"),
        sa.Column("notes", sa.Text(), server_default=""),
    )
    op.create_index("idx_invoices_customer", "invoices", ["customer_id"])
    op.create_index("idx_invoices_status", "invoices", ["status"])
    # PG optimisation: GIN on line_items JSONB
    op.create_index("idx_invoices_items_gin", "invoices", ["line_items"], postgresql_using="gin")

    # ── Billing: payout_ledger ───────────────────────────────────────
    op.create_table(
        "payout_ledger",
        sa.Column("payout_id", sa.Text(), primary_key=True),
        sa.Column("provider_id", sa.Text(), nullable=False),
        sa.Column("job_id", sa.Text(), server_default=""),
        sa.Column("amount_cad", sa.Float(), server_default="0"),
        sa.Column("platform_fee_cad", sa.Float(), server_default="0"),
        sa.Column("provider_payout_cad", sa.Float(), server_default="0"),
        sa.Column("status", sa.Text(), server_default="pending"),
        sa.Column("created_at", sa.Float(), server_default="0"),
    )
    op.create_index("idx_payouts_provider", "payout_ledger", ["provider_id"])
    # PG optimisation: partial index for pending payouts
    op.create_index(
        "idx_payouts_pending", "payout_ledger", ["provider_id"],
        postgresql_where=sa.text("status = 'pending'"),
    )

    # ── Billing: wallets ─────────────────────────────────────────────
    op.create_table(
        "wallets",
        sa.Column("customer_id", sa.Text(), primary_key=True),
        sa.Column("balance_cad", sa.Float(), server_default="0"),
        sa.Column("total_deposited_cad", sa.Float(), server_default="0"),
        sa.Column("total_spent_cad", sa.Float(), server_default="0"),
        sa.Column("total_refunded_cad", sa.Float(), server_default="0"),
        sa.Column("grace_until", sa.Float(), server_default="0"),
        sa.Column("status", sa.Text(), server_default="active"),
        sa.Column("created_at", sa.Float(), server_default="0"),
        sa.Column("updated_at", sa.Float(), server_default="0"),
    )

    # ── Billing: wallet_transactions ─────────────────────────────────
    op.create_table(
        "wallet_transactions",
        sa.Column("tx_id", sa.Text(), primary_key=True),
        sa.Column("customer_id", sa.Text(), nullable=False),
        sa.Column("tx_type", sa.Text(), nullable=False),
        sa.Column("amount_cad", sa.Float(), server_default="0"),
        sa.Column("balance_after_cad", sa.Float(), server_default="0"),
        sa.Column("description", sa.Text(), server_default=""),
        sa.Column("job_id", sa.Text(), server_default=""),
        sa.Column("created_at", sa.Float(), server_default="0"),
    )
    op.create_index("idx_wallet_tx_customer", "wallet_transactions", ["customer_id"])
    op.create_index("idx_wallet_tx_time", "wallet_transactions", ["created_at"])

    # ── Reputation: reputation_scores ────────────────────────────────
    op.create_table(
        "reputation_scores",
        sa.Column("entity_id", sa.Text(), primary_key=True),
        sa.Column("entity_type", sa.Text(), server_default="host"),
        sa.Column("verification_points", sa.Float(), server_default="0"),
        sa.Column("activity_points", sa.Float(), server_default="0"),
        sa.Column("penalty_points", sa.Float(), server_default="0"),
        sa.Column("reliability_score", sa.Float(), server_default="1.0"),
        sa.Column("raw_score", sa.Float(), server_default="0"),
        sa.Column("final_score", sa.Float(), server_default="0"),
        sa.Column("tier", sa.Text(), server_default="bronze"),
        sa.Column("jobs_completed", sa.Integer(), server_default="0"),
        sa.Column("jobs_failed_host", sa.Integer(), server_default="0"),
        sa.Column("jobs_failed_user", sa.Integer(), server_default="0"),
        sa.Column("days_active", sa.Integer(), server_default="0"),
        sa.Column("last_activity_at", sa.Float(), server_default="0"),
        sa.Column("verifications", JSONB, server_default=sa.text("'[]'::jsonb")),
        sa.Column("search_boost", sa.Float(), server_default="1.0"),
        sa.Column("pricing_premium_pct", sa.Float(), server_default="0"),
        sa.Column("updated_at", sa.Float(), server_default="0"),
    )

    # ── Reputation: reputation_events ────────────────────────────────
    op.create_table(
        "reputation_events",
        sa.Column("event_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("entity_id", sa.Text(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("points_delta", sa.Float(), server_default="0"),
        sa.Column("reason", sa.Text(), server_default=""),
        sa.Column("metadata", JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.Float(), server_default="0"),
    )
    op.create_index("idx_rep_events_entity", "reputation_events", ["entity_id"])
    op.create_index("idx_rep_events_time", "reputation_events", ["created_at"])

    # ── Privacy: retention_records ────────────────────────────────────
    op.create_table(
        "retention_records",
        sa.Column("record_id", sa.Text(), primary_key=True),
        sa.Column("data_category", sa.Text(), nullable=False),
        sa.Column("entity_id", sa.Text(), nullable=False),
        sa.Column("entity_type", sa.Text(), server_default="job"),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("expires_at", sa.Float(), nullable=False),
        sa.Column("purged_at", sa.Float(), server_default="0"),
        sa.Column("purge_reason", sa.Text(), server_default=""),
        sa.Column("metadata", JSONB, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index("idx_retention_expiry", "retention_records", ["expires_at"])
    op.create_index("idx_retention_entity", "retention_records", ["entity_id"])
    op.create_index("idx_retention_category", "retention_records", ["data_category"])

    # ── Privacy: consent_records ─────────────────────────────────────
    op.create_table(
        "consent_records",
        sa.Column("consent_id", sa.Text(), primary_key=True),
        sa.Column("entity_id", sa.Text(), nullable=False),
        sa.Column("consent_type", sa.Text(), nullable=False),
        sa.Column("granted_at", sa.Float(), nullable=False),
        sa.Column("revoked_at", sa.Float(), server_default="0"),
        sa.Column("is_active", sa.Integer(), server_default="1"),
        sa.Column("details", JSONB, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index("idx_consent_entity", "consent_records", ["entity_id"])

    # ── Privacy: privacy_configs ─────────────────────────────────────
    op.create_table(
        "privacy_configs",
        sa.Column("org_id", sa.Text(), primary_key=True),
        sa.Column("config", JSONB, nullable=False),
        sa.Column("updated_at", sa.Float(), nullable=False),
    )

    # ── SLA: sla_downtime ────────────────────────────────────────────
    op.create_table(
        "sla_downtime",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("host_id", sa.Text(), nullable=False),
        sa.Column("start_ts", sa.Float(), nullable=False),
        sa.Column("end_ts", sa.Float(), server_default="0"),
        sa.Column("reason", sa.Text(), server_default=""),
        sa.Column("resolved", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.Float()),
    )
    op.create_index("idx_downtime_host", "sla_downtime", ["host_id", "start_ts"])

    # ── SLA: sla_monthly ─────────────────────────────────────────────
    op.create_table(
        "sla_monthly",
        sa.Column("host_id", sa.Text(), nullable=False),
        sa.Column("month", sa.Text(), nullable=False),
        sa.Column("tier", sa.Text(), nullable=False, server_default="community"),
        sa.Column("total_seconds", sa.Float(), server_default="0"),
        sa.Column("downtime_seconds", sa.Float(), server_default="0"),
        sa.Column("incidents", sa.Integer(), server_default="0"),
        sa.Column("credit_pct", sa.Float(), server_default="0"),
        sa.Column("credit_cad", sa.Float(), server_default="0"),
        sa.Column("enforced", sa.Integer(), server_default="0"),
        sa.PrimaryKeyConstraint("host_id", "month"),
    )

    # ── SLA: sla_violations ──────────────────────────────────────────
    op.create_table(
        "sla_violations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("host_id", sa.Text(), nullable=False),
        sa.Column("violation_type", sa.Text(), nullable=False),
        sa.Column("severity", sa.Text(), nullable=False),
        sa.Column("metric_value", sa.Float()),
        sa.Column("threshold", sa.Float()),
        sa.Column("timestamp", sa.Float()),
        sa.Column("details", sa.Text(), server_default=""),
        sa.Column("created_at", sa.Float()),
    )
    op.create_index("idx_violations_host", "sla_violations", ["host_id", "timestamp"])

    # ── Stripe: provider_accounts ────────────────────────────────────
    op.create_table(
        "provider_accounts",
        sa.Column("provider_id", sa.Text(), primary_key=True),
        sa.Column("provider_type", sa.Text(), server_default="individual"),
        sa.Column("stripe_account_id", sa.Text(), server_default=""),
        sa.Column("status", sa.Text(), server_default="pending"),
        sa.Column("corporation_name", sa.Text(), server_default=""),
        sa.Column("business_number", sa.Text(), server_default=""),
        sa.Column("incorporation_file_id", sa.Text(), server_default=""),
        sa.Column("gst_hst_number", sa.Text(), server_default=""),
        sa.Column("email", sa.Text(), server_default=""),
        sa.Column("legal_name", sa.Text(), server_default=""),
        sa.Column("country", sa.Text(), server_default="CA"),
        sa.Column("province", sa.Text(), server_default=""),
        sa.Column("created_at", sa.Float()),
        sa.Column("onboarded_at", sa.Float(), server_default="0"),
        sa.Column("default_currency", sa.Text(), server_default="cad"),
        sa.Column("payout_schedule", sa.Text(), server_default="weekly"),
    )

    # ── Stripe: payment_intents ──────────────────────────────────────
    op.create_table(
        "payment_intents",
        sa.Column("intent_id", sa.Text(), primary_key=True),
        sa.Column("customer_id", sa.Text(), nullable=False),
        sa.Column("amount_cents", sa.Integer(), nullable=False),
        sa.Column("currency", sa.Text(), server_default="cad"),
        sa.Column("status", sa.Text(), server_default="created"),
        sa.Column("stripe_intent_id", sa.Text(), server_default=""),
        sa.Column("description", sa.Text(), server_default=""),
        sa.Column("created_at", sa.Float()),
    )

    # ── Stripe: payout_splits ────────────────────────────────────────
    op.create_table(
        "payout_splits",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.Text(), nullable=False),
        sa.Column("provider_id", sa.Text(), nullable=False),
        sa.Column("total_cad", sa.Float(), nullable=False),
        sa.Column("provider_share_cad", sa.Float(), nullable=False),
        sa.Column("platform_share_cad", sa.Float(), nullable=False),
        sa.Column("gst_hst_cad", sa.Float(), server_default="0"),
        sa.Column("stripe_transfer_id", sa.Text(), server_default=""),
        sa.Column("created_at", sa.Float()),
    )
    op.create_index("idx_payout_splits_provider", "payout_splits", ["provider_id", "created_at"])

    # ── Events: events ───────────────────────────────────────────────
    op.create_table(
        "events",
        sa.Column("event_id", sa.Text(), primary_key=True),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("entity_type", sa.Text(), nullable=False),
        sa.Column("entity_id", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.Float(), nullable=False),
        sa.Column("actor", sa.Text(), server_default=""),
        sa.Column("data", JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("metadata", JSONB, server_default=sa.text("'{}'::jsonb")),
        sa.Column("prev_hash", sa.Text(), server_default=""),
        sa.Column("event_hash", sa.Text(), server_default=""),
    )
    op.create_index("idx_events_entity", "events", ["entity_type", "entity_id"])
    op.create_index("idx_events_type", "events", ["event_type"])
    op.create_index("idx_events_ts", "events", ["timestamp"])
    # PG optimisation: GIN on JSONB data & metadata for containment queries
    op.create_index("idx_events_data_gin", "events", ["data"], postgresql_using="gin")
    op.create_index("idx_events_meta_gin", "events", ["metadata"], postgresql_using="gin")

    # ── Events: leases ───────────────────────────────────────────────
    op.create_table(
        "leases",
        sa.Column("lease_id", sa.Text(), primary_key=True),
        sa.Column("job_id", sa.Text(), nullable=False, unique=True),
        sa.Column("host_id", sa.Text(), nullable=False),
        sa.Column("granted_at", sa.Float(), nullable=False),
        sa.Column("expires_at", sa.Float(), nullable=False),
        sa.Column("last_renewed", sa.Float(), nullable=False),
        sa.Column("duration_sec", sa.Integer(), server_default="300"),
        sa.Column("status", sa.Text(), server_default="active"),
    )
    op.create_index("idx_leases_job", "leases", ["job_id"])
    op.create_index("idx_leases_status", "leases", ["status"])
    # PG optimisation: partial index for active leases (hot path)
    op.create_index(
        "idx_leases_active", "leases", ["job_id", "host_id"],
        postgresql_where=sa.text("status = 'active'"),
    )

    # ── Verification: host_verifications ─────────────────────────────
    op.create_table(
        "host_verifications",
        sa.Column("host_id", sa.Text(), primary_key=True),
        sa.Column("verification_id", sa.Text(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False, server_default="unverified"),
        sa.Column("verified_at", sa.Float()),
        sa.Column("deverified_at", sa.Float()),
        sa.Column("deverify_reason", sa.Text(), server_default=""),
        sa.Column("last_check_at", sa.Float()),
        sa.Column("next_check_at", sa.Float()),
        sa.Column("failure_count", sa.Integer(), server_default="0"),
        sa.Column("gpu_fingerprint", sa.Text(), server_default=""),
        sa.Column("overall_score", sa.Float(), server_default="0.0"),
        sa.Column("checks", JSONB, server_default=sa.text("'[]'::jsonb")),
        sa.Column("updated_at", sa.Float()),
    )
    # PG optimisation: GIN on checks JSONB for verification queries
    op.create_index("idx_hv_checks_gin", "host_verifications", ["checks"], postgresql_using="gin")

    # ── Verification: verification_history ───────────────────────────
    op.create_table(
        "verification_history",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("host_id", sa.Text(), nullable=False),
        sa.Column("verification_id", sa.Text(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("checks", JSONB, server_default=sa.text("'[]'::jsonb")),
        sa.Column("score", sa.Float(), server_default="0.0"),
        sa.Column("reason", sa.Text(), server_default=""),
        sa.Column("timestamp", sa.Float(), nullable=False),
    )
    op.create_index("idx_vh_host", "verification_history", ["host_id"])

    # ── Verification: job_failure_log ────────────────────────────────
    op.create_table(
        "job_failure_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("host_id", sa.Text(), nullable=False),
        sa.Column("job_id", sa.Text(), nullable=False),
        sa.Column("failed_at", sa.Float(), nullable=False),
        sa.Column("reason", sa.Text(), server_default=""),
    )
    op.create_index("idx_jfl_host", "job_failure_log", ["host_id"])
    op.create_index("idx_jfl_time", "job_failure_log", ["failed_at"])

    # ── Inference: inference_jobs ─────────────────────────────────────
    op.create_table(
        "inference_jobs",
        sa.Column("job_id", sa.Text(), primary_key=True),
        sa.Column("customer_id", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("inputs", sa.Text(), nullable=False),
        sa.Column("max_tokens", sa.Integer(), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("timeout_sec", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="queued"),
        sa.Column("submitted_at", sa.Float(), nullable=False),
        sa.Column("completed_at", sa.Float()),
    )
    op.create_index("idx_inf_jobs_status", "inference_jobs", ["status", "submitted_at"])
    op.create_index("idx_inf_jobs_submitted", "inference_jobs", ["submitted_at"])
    # PG optimisation: partial index for queued jobs (scheduler hot path)
    op.create_index(
        "idx_inf_jobs_queued", "inference_jobs", [sa.text("submitted_at ASC")],
        postgresql_where=sa.text("status = 'queued'"),
    )

    # ── Inference: inference_results ──────────────────────────────────
    op.create_table(
        "inference_results",
        sa.Column("job_id", sa.Text(), sa.ForeignKey("inference_jobs.job_id"), primary_key=True),
        sa.Column("outputs", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("completed_at", sa.Float(), nullable=False),
    )

    # ── Chat: chat_conversations ─────────────────────────────────────
    op.create_table(
        "chat_conversations",
        sa.Column("conversation_id", sa.Text(), primary_key=True),
        sa.Column("ip_hash", sa.Text()),
        sa.Column("user_email", sa.Text()),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.Float(), nullable=False),
    )
    op.create_index("idx_chat_conv_updated", "chat_conversations", ["updated_at"])

    # ── Chat: chat_messages ──────────────────────────────────────────
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("conversation_id", sa.Text(), sa.ForeignKey("chat_conversations.conversation_id"), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("idx_chat_msgs_conv", "chat_messages", ["conversation_id", "created_at"])

    # ── Chat: chat_feedback ──────────────────────────────────────────
    op.create_table(
        "chat_feedback",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("message_id", sa.Text(), nullable=False),
        sa.Column("vote", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
    )

    # ── Transparency: legal_requests ─────────────────────────────────
    op.create_table(
        "legal_requests",
        sa.Column("request_id", sa.Text(), primary_key=True),
        sa.Column("received_at", sa.Float(), nullable=False),
        sa.Column("request_type", sa.Text(), nullable=False),
        sa.Column("jurisdiction", sa.Text(), server_default=""),
        sa.Column("authority", sa.Text(), server_default=""),
        sa.Column("scope", sa.Text(), server_default=""),
        sa.Column("status", sa.Text(), server_default="received"),
        sa.Column("responded_at", sa.Float(), server_default="0"),
        sa.Column("complied", sa.Integer(), server_default="0"),
        sa.Column("challenged", sa.Integer(), server_default="0"),
        sa.Column("notes", sa.Text(), server_default=""),
    )

    # ── Transparency: data_disclosures ───────────────────────────────
    op.create_table(
        "data_disclosures",
        sa.Column("disclosure_id", sa.Text(), primary_key=True),
        sa.Column("request_id", sa.Text()),
        sa.Column("disclosed_at", sa.Float(), nullable=False),
        sa.Column("data_category", sa.Text(), server_default=""),
        sa.Column("record_count", sa.Integer(), server_default="0"),
        sa.Column("entities_affected", sa.Integer(), server_default="0"),
        sa.Column("was_mandatory", sa.Integer(), server_default="0"),
        sa.Column("notes", sa.Text(), server_default=""),
    )


def downgrade() -> None:
    # Drop in reverse order
    for table in [
        "data_disclosures", "legal_requests",
        "chat_feedback", "chat_messages", "chat_conversations",
        "inference_results", "inference_jobs",
        "job_failure_log", "verification_history", "host_verifications",
        "leases", "events",
        "payout_splits", "payment_intents", "provider_accounts",
        "sla_violations", "sla_monthly", "sla_downtime",
        "privacy_configs", "consent_records", "retention_records",
        "reputation_events", "reputation_scores",
        "wallet_transactions", "wallets", "payout_ledger", "invoices", "usage_meters",
        "user_ssh_keys", "notifications", "team_members", "teams",
        "api_keys", "sessions", "users",
    ]:
        op.drop_table(table)
